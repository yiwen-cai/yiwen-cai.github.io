"""Z-Cache S3 单卡实测：Wan2.1 DiT 全量步耗时 + graft 重放步耗时 + 缓存显存。

随机权重（kernel shape 与真实模型一致，耗时有效；不评估画质）。
用法: CUDA_VISIBLE_DEVICES=0 python zcache_s3_bench.py --model 1.3b
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

CFGS = {
    "1.3b": dict(num_attention_heads=12, attention_head_dim=128, ffn_dim=8960, num_layers=30),
    "14b": dict(num_attention_heads=40, attention_head_dim=128, ffn_dim=13824, num_layers=40),
}
COMMON = dict(patch_size=(1, 2, 2), in_channels=16, out_channels=16, text_dim=4096,
              freq_dim=256, cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6)


def cuda_time(fn, warmup=2, iters=10):
    # 用 CUDA event 计时，返回 ms
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.3b", choices=list(CFGS))
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--full-iters", type=int, default=3)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    assert args.height % 16 == 0 and args.width % 16 == 0, "height/width must be divisible by 16"
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    dt = torch.bfloat16
    res = {"model": args.model, "gpu": torch.cuda.get_device_name(0)}

    # latent shape: VAE 时间 4x、空间 8x 压缩；patch (1,2,2)
    T = (args.frames - 1) // 4 + 1
    H, W = args.height // 8, args.width // 8
    L = T * (H // 2) * (W // 2)
    D = CFGS[args.model]["num_attention_heads"] * CFGS[args.model]["attention_head_dim"]
    res.update(latent=[16, T, H, W], tokens=L, hidden=D)

    # ---------- 1. 带宽标定：大张量拷贝 ----------
    a = torch.empty(1024 * 1024 * 1024, dtype=dt, device=dev)  # 2 GiB
    b = torch.empty_like(a)
    ms = cuda_time(lambda: b.copy_(a), iters=10)
    res["copy_bw_TBps"] = 2 * a.numel() * 2 / (ms * 1e-3) / 1e12  # 读 + 写
    del a, b
    torch.cuda.empty_cache()

    # ---------- 2. graft 重放步 ----------
    per_copy = L * D * 2
    base = torch.cuda.memory_allocated()
    Ha, D1, D2 = (torch.randn(1, L, D, dtype=dt, device=dev) for _ in range(3))
    res["cache_3copies_GB"] = (torch.cuda.memory_allocated() - base) / 1e9
    res["per_copy_GB"] = per_copy / 1e9
    x = torch.randn(1, 16, T, H, W, dtype=dt, device=dev)
    shift = torch.randn(1, 1, D, dtype=dt, device=dev)
    scale = torch.randn(1, 1, D, dtype=dt, device=dev)
    proj = torch.nn.Linear(D, 16 * 4, dtype=dt, device=dev)
    pt, ph, pw = 1, 2, 2

    def fuse(dL):
        # 预合并二次式: H̃ = H_a + d_L·D1 + d_L²·D2
        return Ha + dL * D1 + (dL * dL) * D2

    def head_and_step(Ht, x_in, dtau):
        # 输出头: AdaLN 调制 LayerNorm + 线性投影 + unpatchify + 欧拉步
        h = F.layer_norm(Ht.float(), (D,), eps=1e-6).to(dt) * (1 + scale) + shift
        v = proj(h)  # [1, L, 64]
        v = v.view(1, T, H // 2, W // 2, pt, ph, pw, 16).permute(0, 7, 1, 4, 2, 5, 3, 6)
        v = v.reshape(1, 16, T, H, W)
        return x_in + dtau * v

    def replay_step():
        return head_and_step(fuse(2.0), x, -0.02)

    res["replay_fuse_only_eager_ms"] = cuda_time(lambda: fuse(2.0))
    res["replay_step_eager_ms"] = cuda_time(replay_step)
    try:
        fuse_c = torch.compile(fuse)
        step_c = torch.compile(replay_step)
        res["replay_fuse_only_compiled_ms"] = cuda_time(lambda: fuse_c(2.0), warmup=3)
        res["replay_step_compiled_ms"] = cuda_time(step_c, warmup=3)
    except Exception as ex:  # noqa: BLE001
        res["compile_error"] = repr(ex)[:300]
    # 最小访存模型：读 3 份系数（融合进输出头时不写回 H̃）
    res["replay_min_bytes_GB"] = 3 * per_copy / 1e9
    del Ha, D1, D2, x
    torch.cuda.empty_cache()

    # ---------- 3. 全量 DiT 前向 ----------
    if not args.skip_full:
        from diffusers import WanTransformer3DModel

        torch.set_default_dtype(dt)
        with torch.device(dev):
            model = WanTransformer3DModel(**CFGS[args.model], **COMMON).eval()
        torch.set_default_dtype(torch.float32)
        n_param = sum(p.numel() for p in model.parameters())
        res["params_B"] = n_param / 1e9
        res["weights_GB"] = n_param * 2 / 1e9
        hs = torch.randn(1, 16, T, H, W, dtype=dt, device=dev)
        enc = torch.randn(1, 512, 4096, dtype=dt, device=dev)
        ts = torch.tensor([500], device=dev)
        torch.cuda.reset_peak_memory_stats()

        def full():
            return model(hidden_states=hs, timestep=ts, encoder_hidden_states=enc, return_dict=False)[0]

        full()  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.full_iters):
            full()
        torch.cuda.synchronize()
        full_s = (time.perf_counter() - t0) / args.full_iters
        res["full_step_s"] = full_s
        res["peak_mem_full_GB"] = torch.cuda.max_memory_allocated() / 1e9
        nl = CFGS[args.model]["num_layers"]
        lin = 2 * n_param * L
        attn = 4 * L * L * D * nl
        res["flops_linear_P"] = lin / 1e15
        res["flops_attn_P"] = attn / 1e15
        res["achieved_TFLOPS"] = (lin + attn) / full_s / 1e12
        res["attn_share_est"] = attn / (lin + attn)

    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
