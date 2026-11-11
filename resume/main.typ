#import "@preview/basic-resume:0.2.3": *
// Put your personal information here, replacing mine
#let name = "Cai Yiwen"
#let location = "Beijing, China"
#let email = "caiyiwen.cs@foxmail.com"
#let github = "github.com/yiwen-cai"
#let phone = "(+86) 189-2876-9793"
#let personal-site = "yiwen-cai.github.io"


#show: resume.with(
  author: name,
  // All the lines below are optional.
  // For example, if you want to to hide your phone number:
  // feel free to comment those lines out and they will not show.
  location: location,
  email: email,
  github: github,
  phone: phone,
  personal-site: personal-site,
  accent-color: "#26428b",
  // CI runner 上没有 Times New Roman，回退到 Libertinus Serif
  font: ("Times New Roman", "Libertinus Serif"),
  paper: "us-letter",
  author-position: center,
  personal-info-position: center,
)

/*
 * Lines that start with == are formatted into section headings
 * You can use the specific formatting functions if needed
 * The following formatting functions are listed below
 * #edu(dates: "", degree: "", gpa: "", institution: "", location: "")
 * #work(company: "", dates: "", location: "", title: "")
 * #project(dates: "", name: "", role: "", url: "")
 * certificates(name: "", issuer: "", url: "", date: "")
 * #extracurriculars(activity: "", dates: "")
 * There are also the following generic functions that don't apply any formatting
 * #generic-two-by-two(top-left: "", top-right: "", bottom-left: "", bottom-right: "")
 * #generic-one-by-two(left: "", right: "")
 */
#align(center)[Zhihu: #link("https://www.zhihu.com/people/cywww-14/")[zhihu.com/people/cywww-14]]

== Education

#edu(
  institution: "Beijing University of Posts and Telecommunications (BUPT)",
  location: "Beijing, China",
  dates: dates-helper(start-date: "Sep 2026", end-date: "Jun 2029"),
  degree: "M.Eng. in Computer Technology, Recommended Admission (ParCIS Lab, Advisor: Prof. Shigang Li)",
)

#edu(
  institution: "Beijing University of Posts and Telecommunications (BUPT)",
  location: "Beijing, China",
  dates: dates-helper(start-date: "Sep 2022", end-date: "Jun 2026"),
  degree: "B.Eng. in Computer Science and Technology",
)
- GPA: 3.76 / 4.0 · Grade: 89.37 / 100.0 · Rank: 59 / 389 (15.16%) · English: CET6 552 / 710

== Work Experience

#work(
  title: "AI Infra Intern (Algorithm–Infra Co-design & Video Diffusion Serving Optimization)",
  location: "Beijing, China",
  company: "Beijing Yongyue Intelligence (Loopit)",
  dates: dates-helper(start-date: "Oct 2026", end-date: "Present"),
)
- *Zing H3 Full-FP8 3-Tier Sparse Attention & Resident Compressed KV Pool (Algorithm–Kernel–System Co-design)*:
  - *Algorithm Design*: Proposed a *Refine (1/1 exact) / Skim (1/16 quadrant low-rank estimation) / Skip (1/64 pruning)* 3-tier sparse attention algorithm with total-variance/CosGate dynamic routing (typical distribution `16.05% / 31.83% / 52.12%`, achieving *0.9964* cosine similarity vs. full exact attention); replaced 1D raster row-major blocking (which mixed cross-row tokens and yielded only *0.4741* intra-block cosine coherence) with a *single-frame-aligned 2D spatial square Tile permutation algorithm*, boosting Skim quadrant coherence to *0.9036*; leveraged 3D RoPE $"SO"(2)$ group homomorphism and orthogonal adjoint properties to design an *$O(1)$ zero-KV-recomputation Query inverse-temporal rotation alignment algorithm* under sliding-window eviction, achieving *23.14 dB* full-horizon Mean PSNR on 15,921-frame (11-min) ultra-long sequences (*+1.67 dB* over Dense FP8 baseline).
  - *Kernel & System Co-design*: Designed a resident compressed KV pool (`ZingCompressedKVStore`) storing zero-mean FP8 residuals + block means + quadrant means, shrinking per-block per-head KV footprint from `49,152 B` to `18,248 B` (*62.85% KV memory savings*: 480p SP1 from `41.8 GiB` to `15.5 GiB`, 736p SP2 from `40.5 GiB` to `15.0 GiB/rank`); precomputed `qk_dc_table` via Tensor Cores outside the SM90 Triton kernel to eliminate FP8 WGMMA register layout conflicts, emitting 18 `QGMMA.E4M3` instructions with *0 register spills* and cutting kernel latency from `1.842 ms` to *`0.646 ms` (2.85x speedup, 74.47% logical MBU)*; reached *23.82 FPS (+26.2% vs. BF16)* on H200 short sequences and *20.40 FPS (26.7% E2E speedup / 210.9s saved)* on 11-min ultra-long sequences.
- *SM90 FP8 GEMM Routing, Kernel Fusion & FA4 Tuning*: Implemented SM90 native FP8 GEMM tuning and dynamic routing for Zing H3 (peaking at *71.7% MFU*); developed Fused Pre-Linear RMSNorm + Indexed Scale/Shift + Online Per-Token FP8 Quant and Gated Residual / SwiGLU fusion kernels; integrated FlashAttention-4 SplitKV tuning and per-forward metadata reuse, achieving *+14.03%* speedup on SP2/3 calls under L20D 1100W sustained power cap and reducing single-GPU median denoise latency by *5.85%–7.62%*.
- *Disaggregated Pooled Serving, Dedicated Lynn Decoder Split, Offline AdaLN & Full-Chunk CUDA Graph*:
  - *Encoder / Denoiser / Remote Lynn Decoder GPU Disaggregation (D1/D2 Split)*: Offloaded VAE decoding, fused preparation, RGB `uint8` quantization, D2H copy, and `/dev/shm` shared-memory publishing to dedicated Decoder GPUs with SP-rank Lynn tile partitioning; boosted `1280×736 (E1+SP2+D1)` steady-state throughput from `15.14` to *`16.24 FPS` (+7.29%)* and reduced DiT latency from `525.66` to *`487.13 ms/chunk` (-38.53 ms/chunk, -7.33%, 243.57 ms/latent)*.
  - *Offline `weights_free` AdaLN Precomputed Table Mechanism*: Materialized 96,768-dim dynamic TimeMLP modulations into an offline resident table with hardware/software contract verification, eliminating runtime weight residency and projection overhead.
  - *`after_step0` Deferred KV Eviction & SP>1 Full-Chunk CUDA Graph*: Stabilized graph input shapes via `after_step0` deferred cross-chunk KV eviction to enable default full-chunk CUDA Graph replay for SP>1 denoising; diagnosed and fixed `cuda_graph_max_bs` capture memory regression via a `max_graphs=16` generational pool drain policy and GC suppression during capture, eliminating memory fragmentation leaks.

#work(
  title: "LLM High-Performance Inference Framework Scheduling & Operator Optimization Research",
  location: "Beijing, China",
  company: "Tencent Hunyuan AI Infra",
  dates: dates-helper(start-date: "Jul 2026", end-date: "Sep 2026"),
)
- Optimized inference framework scheduling and high-performance operators for LLM serving, developing Blackwell-architecture BF16 paged KV attention prefill, `nvfp4_blockwise_gemm`, `per_group` MXFP8 quantization, and `fused_silu_per_token_quant` fusion kernels
- Main open-source repository: #link("https://github.com/Tencent/hpc-ops")[Tencent/hpc-ops], with partial contributions merged into the vLLM main branch

== Open-Source Contributions & Projects

#project(
  name: link("https://github.com/sgl-project/sglang-omni")[SGLang-Omni Open-Source Contribution (Issue #2941 P3.6 / P7.4, #3583 & PR #3782)],
  role: text[Algorithm–Infra Co-design],
  dates: "Nov 2026",
)
- *MiniCPM-o 4.5 Dual-Stream Speculative Async Decode Overlap & Per-Request Hidden FIFO (Issue #2941 P7.4 / PR #3782)*: Resolved write-after-write (WAW) hazards for the DSpark thinker drafter under `event_loop_spec_overlap` ($W_max = 2$) across the `Audio Encoder -> Thinker -> Talker -> Code2Wav` pipeline by upgrading the single-slot cache to a per-request 2-deep FIFO (`dict[str, deque[torch.Tensor]]`), filtering non-final chunked-prefill slices, injecting stream-synchronized hooks and `on_request_retracted` cleanup into `OmniScheduler`, and removing the forced synchronous fallback; validated on H100 PCIe across P0.6 ABBA sweeps (`c1..c64`) and 33 forced decode retractions (100% valid WAV output), eliminating the `c16` sync-draft `0.90x` regression on `MMSU Text Out` (restored to *`1.00x`*, p95 *`-9.6%`*) and achieving *`1.25x–1.40x`* speedup on `MMSU Speech Out` (`c2..c32`, p95 *`-16.7%` to `-39.0%`*) and *`1.26x–1.46x`* on `Seed-TTS EN Speech Out` (`c1..c32`, zero regressions across all concurrencies).
- *Qwen3-TTS Dual-AR Speech Speculative Decoding Algorithm Design & H100 Dual-Dimension Lossless Falsification (Issue #2941 P3.6 & #3583, Adopted by Maintainers)*: Led official Roadmap *P3.6*, profiling `Qwen3-TTS-12Hz-1.7B` E2E bottlenecks on H100 PCIe (dual-AR accounts for *>85%*: 28-layer `Main Talker` `2.45 ms/frame` at 48%, 15-step `CodePredictor` `2.60 ms/frame` at 52%); trained an inner grouped drafter (Route A) and an outer 1.1M-param `Gated-ResMLP` cross-frame drafter (Route B, `0.082 ms` CUDA Graph latency, *85.44%* Top-50 Shaped lossless acceptance on primary code $c_(t+1,0)$); proved Route A is bounded by deep residual codebook entropy ($g >= 5, H_g > 5.2 "bits"$, `~1.2%` 15-step pass rate, *2.46 accepted steps*, *0.330x slowdown*) and Route B is bottlenecked by a cross-frame bidirectional causal lock (*`~1.01` frames/round* lossless advancement, while approximate residuals induce *21.9%–38.5% TV drift*), establishing a dual-dimension *No-Go architectural verdict* for lossless speculative decoding on dual-AR RVQ models.

#work(
  title: "The 10th Huawei ICT Competition China Finals Challenge Track",
  location: "Huazhong University of Science and Technology, Wuhan",
  company: "Huawei",
  dates: dates-helper(start-date: "Apr 2026", end-date: "Apr 2026"),
)
- Ranked 8th among 64 university teams in preliminaries and 3rd overall among 16 finalists to win *National First Prize*; received the *Best Operator Migration Optimization Award* (best optimization across all teams) as the sole developer
- Selected projects: #link("https://github.com/yiwen-cai/fused_add_rmsnorm_kernel")[*fused_add_rmsnorm_kernel*] (Triton fused RMSNorm + Add kernel with multi-row program mapping and `hidden_size` bucketed scheduling, *19.28x* speedup vs. PyTorch); #link("https://github.com/yiwen-cai/xllm-ictfinal")[*xllm-ictfinal*] (adapted Qwen3.5 on the xllm engine with GDN linear attention and MTP speculative decoding, resolving KVCache memory anomalies to serve 32k-token sequences)

#project(
  name: link("https://github.com/yiwen-cai/cuda-gemm")[cuda-gemm],
  dates: "Jun 2025",
)
- Hand-wrote 8 CUDA GEMM kernels from scratch (covering FP32 CUDA Core and FP16 Tensor Core WMMA), progressively applying Shared Memory Tiling, Bank Conflict Avoidance, Double Buffering, and `cp.async`; SGEMM v3 reaches *95.1%* of cuBLAS (4096²: 37,898 GFLOPS; 8192²: 36,799 GFLOPS), HGEMM v3 Tensor Core reaches *213 TFLOPS* (4096²), with Nsight Compute/Systems profiling

#project(name: link("https://github.com/yiwen-cai/fastGNN")[FastGNN], dates: "May 2026")
- Full-batch GNN training system on H100 Tensor Cores with a custom sparse operator backend (Voltrix block-sparse SpMM, Fused3S sparse attention, Rabbit node reordering), achieving *1.8x–3.3x* E2E speedup over DGL/PyG

#work(
  title: "NUS Summer Workshop",
  location: "Singapore",
  company: "National University of Singapore",
  dates: dates-helper(start-date: "Jul 2024", end-date: "Aug 2024"),
)
- Earned Grade *A+* (*Rank 1st*) in Cloud Computing and Big Data; built #link("https://github.com/uplion")[*UPLION*], a cloud-native multi-agent orchestration and Kubernetes scheduling platform

== Awards
#project(
  name: "The 10th Huawei ICT Competition China Finals Challenge Track",
  role: text[First Prize, Best Operator Migration Optimization Award],
  dates: "Apr 2026",
)

#project(
  name: "NUS Summer Workshop",
  role: text[A+ (1#super[st] Place)],
  dates: "Jul 2024",
)

#project(
  name: "15th Blue Bridge Cup National Software and Information Technology Competition (Beijing Region)",
  role: text[2#super[nd] Prize],
  dates: "Apr 2024",
)

#project(name: "BUPT University-Level Scholarship (Freshman, Sophomore & Junior Years)", dates: "2023 – 2025")

== Skills
- *Programming Languages*: Python, C/C++, CUDA, Triton, Rust, Go
- *Research Directions*: High-Performance GPU Kernels, LLM Training/Inference Systems & Algorithm–Infra Co-design, Distributed & High-Performance Computing
