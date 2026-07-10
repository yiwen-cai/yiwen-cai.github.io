+++
title = 'Kernels、Triton 与 Profiling：CS336 Lecture 6 笔记'
date = 2026-06-14
draft = false
summary = '从硬件抽象进入 kernel 编程实践：warp/occupancy/bank conflict/coalescing 如何映射到性能，benchmarking 与 profiling 方法论，以及用 Triton 实现 GeLU、softmax、row sum、matmul+ReLU 四个 kernel。'
tags = ['Triton', 'CUDA', '性能优化', 'Profiling', 'CS336']
showReadingTime = true
showTableOfContents = true
+++

> 基于 Stanford CS336 Spring 2026 Lecture 6（Kernels, Triton, XLA）整理。官方材料：[lecture_06.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)。

Lecture 5 从硬件角度建立 GPU 直觉，Lecture 6 从实践角度进入 kernel 编程和 profiling。三条线：

1. GPU 硬件回顾 + 编程模型与硬件的交互（warp/occupancy/bank conflict/coalescing）
2. 性能分析：benchmarking + profiling，理解瓶颈在哪
3. 手写 kernel：用 Triton 实现 GeLU、softmax、row sum、matmul+ReLU

## 硬件参数对比

| Accelerator | A100 | H100 | B200 |
|---|---|---|---:|
| # SMs | 108 | 132 | 148 |
| Register (per SM) | 256 KB | 256 KB | 256 KB |
| L1 + shared (per SM) | 192 KB | 256 KB | 256 KB |
| L2 cache | 40 MB | 50 MB | 96-126 MB |
| HBM | 80 GB | 80 GB | 192 GB |
| Register bandwidth | ~116 TB/s | ~401 TB/s | ~447 TB/s |
| L1 + shared bandwidth | ~19 TB/s | ~33 TB/s | ~19 TB/s |
| L2 bandwidth | ~5-8 TB/s | ~12 TB/s | ~9 TB/s |
| HBM bandwidth | 2 TB/s | 3.35 TB/s | 8 TB/s |

B200 还有一个对 programmer 不可见的 tensor memory（TMEM），位于 register 和 shared memory 之间，专门用于 Tensor Core。

Programming model：

```
GPU kernel
└── Grid (所有 block)
    └── Thread block / CTA (一组 threads，共享 shared memory)
        └── Thread (最小执行单元，私有 register)
```

- **Thread**：执行一小部分数据的代码
- **Thread block / CTA (Concurrent Thread Array)**：一组共享 shared memory 的 threads
- **Grid**：所有 thread blocks 的集合

H100/B200 还有 thread block clusters，支持 distributed shared memory。

## 编程模型与硬件的交互

核心思想：**程序模型提供了正确的抽象，但性能极度依赖对硬件的理解**。

### Warps

- 一个 thread block 内的 threads 被分为 warps，每个 warp 32 个 threads。
- 所有 warp 内 threads 在 SM 上以 lockstep 执行同一条指令。
- **Control divergence**：如果同一个 warp 内不同线程需要走不同分支，GPU 会串行执行两个路径（先 mask 掉一部分线程执行 path A，再 mask 掉另一部分执行 path B），总时间 ≈ 两个分支时间之和。
- **Zero-cost warp switching**：SM 可以在多个 warp 之间切换（例如当某个 warp 等待 HBM 读写时），切换开销几乎为零。

### Warp Occupancy

- 每个 thread 可使用 0-255 个 registers。
- threads 用越多 registers，SM 上能同时调度的 threads 就越少（低 occupancy）。
- **Low occupancy 不一定是坏事**：如果每个 thread 做更多工作（thread coarsening），总吞吐可能更高。

示例计算：

```
thread block = 128 threads
每个 thread 用 160 registers
SM 最多 65536 registers → 每 SM 最多 65536 / (128 × 160) = 3 个 blocks
warps = 3 × 128 / 32 = 12
occupancy = 12 / 64 = 18.75%
```

较高的 register 使用降低 occupancy，但可能意味着每个 thread 做了更多有用的工作（例如处理更多元素），最终可能更快。

### Bank Conflicts (Shared Memory)

Shared memory 分为 32 个 banks，每个 4 字节宽。每个周期，每个 bank 只能被一个 thread 访问（除非访问完全相同的位置）。

- 如果多个 threads 访问同一个 bank（不同位置），访问被串行化 → **bank conflict**。
- 最坏情况：矩阵每行跨所有 banks，32 个 threads 访问第一列 → 32-way bank conflict。
- 解决方式：**swizzling**，通过重新排列 shared memory 布局（如 row xor col）来避免冲突。

### Memory Coalescing (HBM)

当一个 warp 的 32 个 threads 访问 HBM 时，内存访问会合并为 128 字节的缓存行（cache line）事务。

最佳情况：**全部 coalesced**，所有 threads 访问同一 cache line 内的连续地址（32 threads × 4 bytes = 128 bytes）。

### Block Occupancy 与 Wave Quantization

Thread blocks 按 wave 调度到 SM 上。B200 有 148 个 SMs，如果 launch 160 个 thread blocks，第一 wave 148 个，第二 wave 12 个。**Wave quantization 问题**：最后一 wave 的 block 太少，部分 SM 空闲。解决：让 thread blocks 数量尽量被 SM 数量整除。

## Benchmarking 与 Profiling

方法论三步循环：

```
1. Benchmark 和 profile 你的代码
2. 做优化
3. 再次 benchmark 和 profile
```

### Benchmarking

测量端到端的 wall-clock time。**为什么要用 CUDA events 而不是 Python 的 `time.time()`？**

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
run()
end_event.record()
torch.cuda.synchronize()
time_ms = start_event.elapsed_time(end_event)
```

CUDA events 直接在 GPU 侧记录时间，避免了 CPU 调度开销和 Python 解释器干扰。三个关键步骤：Warmup（不计入编译/JIT 开销）、同步（`torch.cuda.synchronize()`）、多次运行取平均。

### Profiling

Benchmarking 只告诉你"慢不慢"，profiling 告诉你"慢在哪"。用 `torch.profiler`：

```python
with torch.profiler.profile(activities=[ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
    run()
    torch.cuda.synchronize()

table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
```

关键发现：
- `add(dim=2048)`：少量简单 kernel，memory-bound。
- `matmul(dim=2048)`：`ampere_sgemm*` 系列 GEMM kernel，compute-bound。
- `matmul(dim=128)`：小矩阵时 kernel 名字不同，可能不是 GEMM 而是更小的 kernel template。

## Naive vs Builtin vs Compiled GeLU

用 GeLU 演示 **kernel fusion** 的重要性。

1. **Naive GeLU**：从公式拼出，每一步产生独立 kernel
   ```python
   0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
   ```
   多个元素操作 → 多个 kernel launches → 多次 HBM 读写。
2. **Builtin GeLU**：`torch.nn.functional.gelu(x, approximate="tanh")`，PyTorch 内置的 fused 实现。
3. **Compiled GeLU**：`torch.compile(naive_gelu)`，PyTorch JIT 自动分析和融合，生成单个 Triton kernel。

| 实现 | kernel 数量 | HBM 读写 | 速度 |
|---|---|---|---|
| naive | 多个 | 多次 read/write | 最慢 |
| builtin | 单个 fused | 一次 read + 一次 write | 快速 |
| compiled | 单个 Triton kernel | 一次 read + 一次 write | 接近 builtin |

**核心 insight**：未融合的 elementwise chain 中，每个中间结果都要写入 HBM 再读回。Fusion 后中间结果留在 register 中，大幅减少 HBM traffic。

## Triton Kernel 编程

Triton（OpenAI 开发）的编程模型与 CUDA 不同：

```
CUDA: 程序员指定每个 thread 做什么
Triton: 程序员指定每个 thread block 做什么
```

思维框架：`load 数据到 shared memory → 在片上操作 → 写回 global memory`。

### Triton GeLU（Elementwise）

```python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)      # 当前 block 的 ID
    start = pid * BLOCK_SIZE         # 该 block 的起始偏移
    offsets = start + tl.arange(0, BLOCK_SIZE)  # 该 block 的索引范围

    mask = offsets < num_elements     # 边界检查
    x = tl.load(x_ptr + offsets, mask=mask)

    # 计算 GeLU
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    tl.store(y_ptr + offsets, y, mask=mask)
```

关键点：`tl.program_id(axis=0)` 获取 block 索引；`tl.arange(0, BLOCK_SIZE)` 生成线程索引范围；`mask` 用于边界检查；声明 `BLOCK_SIZE: tl.constexpr` 使编译器编译时确定 block size。

### Triton Softmax（单行 reduction）

```python
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE  # 一行必须能放入一个 block

    row_idx = tl.program_id(0)           # 每行由一个 block 处理
    col_offsets = tl.arange(0, BLOCK_SIZE)

    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_row = tl.load(x_start_ptr + col_offsets, mask=col_offsets < num_cols, other=float("-inf"))

    # softmax: x - max → exp → sum → normalize
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    y_start_ptr = y_ptr + row_idx * y_row_stride
    tl.store(y_start_ptr + col_offsets, y_row, mask=col_offsets < num_cols)
```

每行一个 block，`tl.max`、`tl.sum` 是 Triton 的 reduction 操作符，内部处理跨线程通信。相比 naive 实现，Triton 版本在 shared memory 中完成所有操作，只有一次 read 和一次 write。

### Triton Row Sum（大行需要循环的 reduction）

如果一行比 BLOCK_SIZE 大，需要 tiling：

```python
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # 每个线程的累加器

    for start in range(0, N, BLOCK_SIZE):            # 循环处理 tiles
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
        acc += x

    result = tl.sum(acc, axis=0)                     # 最终 reduction
    tl.store(out_ptr + row, result)
```

这是 Triton 的 "baby tiling" 模式：当行太大无法放入单个 block 时，每个 block 依次处理多个 tiles，在 `acc` 中累加，最后用 `tl.sum` 做跨线程 reduction。

### Triton Matmul + ReLU（Tiling + Kernel Fusion）

Naive matmul 的问题：每次读 $A[m,k]$ 和 $B[k,n]$ 从 HBM，计算后写 $C$ 到 HBM → $M \times K \times N$ 次 HBM 读，总共只做 $O(1)$ arithmetic intensity。

Tiling 方案：

```
1. 把 C 分成输出 tile（如 64×64），每个 block 负责一个 tile
2. 每次从 HBM 加载一对 A tile（如 64×32）和 B tile（如 32×64）到 shared memory
3. 在片上做这小块 matmul，累加到部分和
4. 重复直到覆盖整个 K 维度
5. 写回 HBM

Arithmetic intensity ≈ O(tile_size)，远高于 naive 的 O(1)
```

```python
@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K, strides...,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    indices_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + indices_m[:, None] * stride_am + indices_k[None, :] * stride_ak
    b_ptrs = b_ptr + indices_k[:, None] * stride_bk + indices_n[None, :] * stride_bn

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=..., other=0.0)
        b = tl.load(b_ptrs, mask=..., other=0.0)
        acc += tl.dot(a, b)             # GPU Tensor Core 加速
        a_ptrs += BLOCK_K * stride_ak    # 推进到下一块
        b_ptrs += BLOCK_K * stride_bk

    acc = tl.maximum(acc, 0.0)           # fused ReLU
    c_ptrs = ...
    tl.store(c_ptrs, acc, mask=...)
```

Kernel fusion 的优势：在写回 HBM 之前，ReLU（elementwise）在 register 中完成，不需要额外的 HBM 往返。

## 理解 CUDA Kernel 名字

profiling 输出的 kernel 名字里有大量信息：

```
cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_...
```

| 片段 | 含义 |
|---|---|
| `sm100` | Blackwell B200 架构 |
| `simt` | SIMT 路径（非 Tensor Core 路径） |
| `sgemm` | Single precision GEMM |
| `f32` | float32 精度 |
| `64x64x16` | tile shape: 输出 C tile 64×64, K tile 16 |
| `nnn` | 矩阵转置标记（n = not transposed） |

`ampere_sgemm_128x128_*` 则是用于 Ampere A100 的 GEMM kernel，128×128 是 tile size。

## 从 Triton 到 PTX

Triton 编译生成的 PTX（Parallel Thread Execution）是 GPU 的中间表示/汇编层。Triton GeLU 生成的 PTX 包含：`ld.global.*`（从 global memory 读取）、`st.global.*`（写入 global memory）、`%ctaid.x`（block index）、`%tid.x`（thread index）。Triton 自动做了 thread coarsening——一个 thread 处理 8 个元素。

## 术语表

| 术语 | 含义 |
|---|---|
| CTA | Concurrent Thread Array，与 thread block 同义 |
| Warp | GPU 调度单位，32 个线程 |
| SIMT | Single Instruction Multiple Threads |
| Occupancy | SM 上活跃 warp 占理论上限的比例 |
| Bank conflict | 多个线程访问 shared memory 同一 bank 导致串行化 |
| Memory coalescing | warp 内线程访问连续 HBM 地址，合并为单个事务 |
| Wave quantization | 最后一批 thread block 数量少于 SM 数，造成部分 SM 空闲 |
| Swizzling | 重新排列 shared memory 布局以规避 bank conflict |
| Thread coarsening | 每个线程处理多个元素以提高利用率 |
| Kernel fusion | 合并多个 kernel 以减少 HBM 读写 |
| PTX | Parallel Thread Execution，GPU 汇编级 IR |
| Triton | OpenAI 的 GPU 编程语言，以 thread block 为单位编程 |
| CUDA events | GPU 侧计时器，用于精确的 GPU 时间测量 |
| cutlass | NVIDIA 的 CUDA 线性代数模板库 |

## 总结

```
- 了解编程模型（PyTorch, Triton, PTX）以保证正确性
- 理解硬件（SMs, warps, occupancy, bank conflicts 等）以优化性能
- Benchmark 以了解 scaling 行为
- Profile 以查看在运行什么、跑多久
- Triton 以 thread block 为单位思考（load到shared memory → 操作/fusion → 写回HBM）
- 示例：GeLU (elementwise), softmax (row-wise), row sum (tiling), matmul (tiling + fusion)
```
