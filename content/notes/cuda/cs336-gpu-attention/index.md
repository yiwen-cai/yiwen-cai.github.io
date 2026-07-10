+++
title = 'GPU 架构与 FlashAttention：CS336 Lecture 5 笔记'
date = 2026-06-13
draft = false
summary = '从模型结构转入系统视角：为什么 LLM 必须依赖 GPU、GPU 的执行模型与内存层次、arithmetic intensity 与 roofline model，以及如何用这套性能模型解释 FlashAttention 这类 IO-aware 算法。'
tags = ['CUDA', 'GPU', 'FlashAttention', '性能优化', 'CS336']
showReadingTime = true
showTableOfContents = true
+++

> 基于 Stanford CS336 Spring 2026 Lecture 5（GPUs, TPUs, and Efficient Attention）整理。官方材料：[lecture_05.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf)。

本讲从模型结构转入系统视角，目标是理解：为什么现代 LLM 训练和推理必须依赖 GPU/TPU，以及为什么同样的数学计算在不同 kernel、不同内存访问模式下性能可能相差巨大。

## 主线

本讲可以概括为三条线：

1. GPU/TPU 为什么适合 LLM
2. GPU 的执行模型和内存层次
3. 如何用性能模型解释 FlashAttention 这类 IO-aware 算法

核心不是学习 CUDA 语法，而是建立系统直觉：

```
快的深度学习程序 = 足够多的并行性 + 高效的数据复用 + 避免昂贵的显存访问
```

对于 LLM，大量计算集中在矩阵乘法、attention、MLP、normalization 和 elementwise 操作。矩阵乘法能很好利用 Tensor Cores，但 attention 和许多小算子容易受内存带宽、kernel launch、数据搬运影响。

## 为什么 LLM scaling 依赖 GPU scaling

LLM 的能力提升通常来自：更多参数、更多训练 token、更多训练 FLOPs、更高硬件利用率、更强并行化能力。

过去 CPU 单核性能增长依赖频率提升、工艺缩放和 Dennard scaling。Dennard scaling 放缓后，单核性能无法继续高速增长。现代深度学习 scaling 更依赖 parallel scaling——通过大量并行计算单元提升吞吐。

GPU 的优势在于：

```
CPU: 优化少量复杂线程的低延迟执行
GPU: 优化海量简单线程的高吞吐执行
```

这和 LLM 的计算模式高度匹配，因为 Transformer 中大部分算子都可以表达为大批量矩阵或张量操作。

## CPU vs GPU：latency-oriented 与 throughput-oriented

CPU 更关注：少量线程、复杂控制流、强分支预测、大缓存、低延迟响应。

GPU 更关注：大量线程、简单控制逻辑、高并行吞吐、更高算术单元密度、用线程切换隐藏内存延迟。

GPU 并不是在每个线程上都比 CPU 快，而是能同时运行大量线程，让整体吞吐非常高。

```
CPU 像少数很强的工人，每个人能处理复杂任务。
GPU 像大量专门工人，每个人做简单操作，但总吞吐极高。
```

在深度学习中，我们通常不关心单个 token 或单个元素的延迟，而关心整个 batch、整个矩阵乘法、整个训练 step 的吞吐。

## GPU 的基本硬件结构

GPU 由多个 Streaming Multiprocessors（SM）组成：

```
GPU
└── many SMs
    ├── CUDA cores / SPs
    ├── Tensor Cores
    ├── registers
    ├── shared memory / L1 cache
    └── warp schedulers
```

| 术语 | 含义 |
|---|---|
| SM | Streaming Multiprocessor，执行 thread block 的主要计算单元 |
| SP / CUDA core | 执行普通标量或向量浮点运算的计算单元 |
| Tensor Core | 专门加速矩阵乘法的硬件单元 |
| register | 每个线程私有的最快存储 |
| shared memory | 一个 block 内线程共享的片上内存 |
| L2 cache | GPU 全局共享缓存，连接 SM 和 HBM |
| HBM / global memory | 显存，容量大但访问慢 |

关键事实：**Tensor Core 上的矩阵乘法吞吐远高于普通 CUDA core 上的一般浮点操作**。因此高性能实现会努力把计算组织成 Tensor Core 友好的矩阵乘法形式。

## CUDA 执行模型：thread、block、warp

三层执行抽象：

```
thread: 最小逻辑执行单元
block: 一组 threads，可以共享 shared memory
grid: 一个 kernel launch 中的所有 blocks
```

硬件调度中还有 **warp**：

```
warp = GPU 实际调度执行的线程组，通常包含 32 个连续 threads
```

GPU 使用 SIMT 模型（Single Instruction, Multiple Threads）：同一个 warp 中的线程通常执行同一条指令，但操作不同数据。

这解释了为什么分支会拖慢 GPU——如果同一个 warp 中一部分线程走 path A，另一部分走 path B，GPU 会 serially 执行两个路径：先执行 path A 并 mask 掉走 path B 的线程，再执行 path B 并 mask 掉走 path A 的线程。总时间 ≈ 两个分支时间之和。这叫 **branch divergence**。

关键细节：diverge 时线程不是 idle，而是被 mask 掉不执行当前指令。所以同一个 warp 内应尽量避免数据相关的分支。

```
GPU 适合规则、密集、统一的计算模式；不适合大量不规则分支。
```

## GPU 内存层次

从快到慢：

```
registers
  ↓
shared memory / L1 cache
  ↓
L2 cache
  ↓
HBM / global memory
```

| 层级 | 位置 | 特点 |
|---|---|---|
| register | thread 私有 | 最快，但容量最小 |
| shared memory | block 内共享 | 很快，适合 tile 复用 |
| L1 cache | SM 附近 | 缓存局部访问 |
| L2 cache | GPU 全局 | 所有 SM 共享 |
| HBM/global memory | 显存 | 容量大，带宽高但延迟远高于片上内存 |

重要原则：**一次从 HBM 读入的数据，最好在 register/shared memory 中被尽可能多次复用**。

如果一个算子反复从 HBM 读写中间结果，即使 FLOPs 不多，也可能非常慢。这也是 FlashAttention 的核心动机：减少 attention matrix 在 HBM 中的读写。

## Compute scaling 快于 memory scaling

现代 GPU 计算能力增长非常快（尤其 Tensor Core），但显存带宽和内存访问速度没有以同样速度增长。

```
很多程序不是算不动，而是数据搬不够快。
```

| 类型 | 含义 |
|---|---|
| compute-bound | 主要受计算单元峰值 FLOPs 限制 |
| memory-bound | 主要受内存带宽或数据搬运限制 |

如果一个 kernel 每读入很多 bytes 只做很少 FLOPs，它很可能是 memory-bound。反过来，如果每读入一份数据能做大量计算并反复复用，就更可能接近 compute-bound。

## Arithmetic intensity 与 roofline model

Arithmetic intensity 衡量每搬运一个 byte 数据能做多少计算：

```
Arithmetic Intensity = FLOPs / Bytes moved
```

可达到的吞吐可以用 roofline model 估计：

```
Attainable FLOP/s = min(Peak FLOP/s, Memory Bandwidth × Arithmetic Intensity)
```

- 低 arithmetic intensity：受内存带宽限制，memory-bound
- 高 arithmetic intensity：受峰值计算限制，compute-bound

转折点称为 ridge point：

```
ridge point = Peak FLOP/s / Memory Bandwidth
```

如果某个算子的 arithmetic intensity 低于 ridge point，提高计算单元数量不一定有帮助；更应该减少数据搬运、做 operator fusion、tiling 或 recomputation。

数值直觉：A100 的 FP32 peak 约 19.5 TFLOP/s，HBM bandwidth 约 2 TB/s，ridge point ≈ 10 FLOPs/byte。BF16 Tensor Core peak 约 312 TFLOP/s，ridge point ≈ 156 FLOPs/byte。所以低精度 matmul 更容易 compute-bound，而 elementwise/softmax 更容易 memory-bound。

## Matmul 为什么容易快

矩阵乘法 $C = AB$（$A,B,C \in \mathbb{R}^{N \times N}$）：FLOPs ≈ $2N^3$，理想读写数据规模约为 $O(N^2)$，所以理想 arithmetic intensity 约为 $O(N)$。当 N 足够大时，矩阵乘法可以有很高的数据复用，容易接近 compute-bound。

但 naive matmul 不一定快：如果每次计算都从 global memory 重复读取 A 和 B 的元素，实际 bytes moved 会大幅增加。解决方法是 tiling：

```
1. 把 A 和 B 切成 tile
2. 把 tile 加载到 shared memory / registers
3. 在片上反复复用 tile
4. 计算出 C 的一个 tile
```

## 让 GPU workload 变快的六类技巧

**避免 control divergence**：让同一个 warp 中的线程执行尽可能一致的控制流，避免数据相关的复杂分支。

**使用低精度计算**：FP16/BF16/FP8/INT8 减少内存读写、提高缓存有效容量、利用 Tensor Cores。但会带来数值稳定性问题，需要 loss scaling、FP32 master weights、混合精度策略。

**Operator fusion**：未融合的 elementwise chain 每步都读写 HBM；fusion 后中间结果留在 register 中，大幅减少 HBM traffic。

**Recomputation**：用额外计算换更少存储——前向不保存某些中间激活，反向时重新计算。核心权衡：多做一点 FLOPs，少占很多 HBM。memory-bound 时这个交换往往值得。

**Coalesced memory access**：相邻线程访问连续内存，一个 warp 的访问可以合并成较少的 memory transactions。

**Tiling**：把一小块数据搬到快内存中，并在写回 HBM 前尽可能多次复用。在 matmul 中复用 A/B 子矩阵；在 attention 中分块计算 QK^T 和 softmax，避免完整 attention matrix 落到 HBM。

## Occupancy 的直觉

Occupancy 指一个 SM 上实际活跃 warp 数量相对于理论最大数量的比例。高 occupancy 的意义：当某些 warp 等待内存时，SM 可以切换到其他 ready warp 继续执行，从而隐藏延迟。

但 occupancy 不是越高越好。影响 occupancy 的资源：每个 thread 使用的 registers 数量、每个 block 使用的 shared memory、block size、硬件最大 active warps/blocks 限制。如果为了提高 occupancy 而牺牲数据复用，可能反而变慢。

## Attention 为什么容易 memory-bound

标准 attention：$\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d}) V$。如果序列长度为 n，attention score matrix 大小为 $n \times n$。

普通实现：

```
1. 计算 S = QK^T
2. 把 S 写入 HBM
3. 从 HBM 读出 S，做 softmax
4. 把 P = softmax(S) 写入 HBM
5. 从 HBM 读出 P 和 V，计算 O = P V
```

这会产生大量 HBM traffic，尤其当 n 很大时，中间矩阵 S 和 P 都是 $O(n^2)$。问题不只是 FLOPs，而是中间 attention matrix 太大、频繁读写 HBM。此外 softmax 包含跨 key 维度的 reduction（求 max、求 sum），需要 warp 内线程通信，进一步增加延迟。

## FlashAttention 的核心思想

FlashAttention 是 IO-aware exact attention。它不改变 attention 的数学定义，而是改变计算顺序：

```
把 Q, K, V 分块
在 SRAM/shared memory/registers 中计算局部 QK^T
用 online softmax 维护正确的归一化统计量
逐块累积输出 O
避免把完整 S 或 P 写入 HBM
```

关键词：tiling、online softmax、recomputation、HBM traffic reduction、exact attention。

重要性：**它不是近似 attention，而是通过系统优化得到数学等价的结果**。

## Online softmax 回顾

标准 stable softmax 对一行 score：

```
m = max_i s_i
l = sum_i exp(s_i - m)
p_i = exp(s_i - m) / l
```

分块 attention 中不能一次性拿到整行 score，需要 online softmax。假设旧状态为 $m_{old}, l_{old}$，新 block 的最大值和分母为 $m_B = \max(S_B)$，$l_B = \sum \exp(S_B - m_B)$，合并：

$$m_{new} = \max(m_{old}, m_B)$$

$$l_{new} = l_{old} \cdot \exp(m_{old} - m_{new}) + l_B \cdot \exp(m_B - m_{new})$$

如果还维护未归一化输出 $a$：

$$a_{new} = a_{old} \cdot \exp(m_{old} - m_{new}) + a_B \cdot \exp(m_B - m_{new})$$

$$\text{output} = a_{new} / l_{new}$$

当新的最大值出现时，旧结果不是被忽略，而是整体乘以 $\exp(m_{old} - m_{new})$——等价于把所有旧项从减去旧最大值重新换成减去新最大值。

## 为什么 FlashAttention 能减少内存

普通 attention 的瓶颈：需要显式存储 $S = QK^T$ 和 $P = \text{softmax}(S)$，二者都是 $O(n^2)$。

FlashAttention 的策略：不把完整 S/P 写入 HBM，只保存每行的 $m$、$l$ 和最终 $O$，每个 block 内临时 score 用完即丢弃。反向传播时根据保存的少量统计量重算局部 score 和 softmax（recomputation）。

核心 tradeoff：**多做一些计算，显著减少 HBM 读写和中间存储**。在 attention 这种 memory-bound 场景中，这个 tradeoff 很划算。

## 如何用这些直觉分析 kernel

做 profiling 或写 kernel 时，应该反复问：

1. 这个 kernel 是 compute-bound 还是 memory-bound？
2. 数据是否 coalesced 访问？
3. 有没有多次读写 HBM 的中间结果？
4. 是否可以 fusion？
5. 是否可以 tiling 到 shared memory/registers？
6. 是否可以用 recomputation 换内存？
7. Tensor Core 是否被充分利用？
8. occupancy 低是瓶颈还是合理的资源权衡？

几个典型判断：
- forward pass 中 GEMM kernel 占比高，因为 matmul 的 arithmetic intensity 高，容易 compute-bound。
- 完整 training step 中 elementwise/reduction/optimizer kernel 显著增加，因为 backward 需要逐元素梯度计算，AdamW 需要逐元素更新一阶/二阶动量，这些操作 arithmetic intensity 低，更偏 memory-bound。
- `attention_softmax` 的 runtime 可能比 `attention_scores_matmul` 更长，因为 softmax 受 memory bandwidth 和 reduction 通信限制，而 matmul 被 cuBLAS 高度优化后 compute throughput 更高。
- mixed precision 中 GEMM 加速最多（Tensor Core），elementwise/reduction 可能变化不大（带宽限制 + 数值稳定性要求）。

GPU 优化本质上是围绕硬件约束重新组织数学计算——这比单纯看代码更重要。

## 术语表

| 术语 | 中文理解 |
|---|---|
| GPU | 面向高吞吐并行计算的处理器 |
| TPU | 面向矩阵/张量计算的专用加速器 |
| SM | GPU 中执行 thread block 的主要计算单元 |
| Warp | GPU 调度单位，通常 32 个 threads |
| SIMT | 单指令多线程执行模型 |
| Register | 线程私有最快存储 |
| Shared memory | block 内共享片上内存 |
| HBM | 高带宽显存，也常被称为 global memory |
| Tensor Core | 专门加速矩阵乘法的硬件 |
| Occupancy | SM 上活跃 warp 占理论上限的比例 |
| Arithmetic intensity | FLOPs / bytes moved |
| Roofline model | 用计算峰值和内存带宽估计性能上限的模型 |
| Memory-bound | 性能主要受数据搬运限制 |
| Compute-bound | 性能主要受计算峰值限制 |
| Operator fusion | 合并多个算子减少中间读写 |
| Recomputation | 用额外计算换更少存储 |
| Coalescing | 相邻线程访问连续内存以合并 transaction |
| Tiling | 分块计算并在快内存中复用数据 |
| FlashAttention | IO-aware exact attention 实现 |

## 一句话总结

```
LLM 系统性能不只由 FLOPs 决定，更由数据如何在 GPU 内存层次中移动决定。
```

GPU 优化的关键是：让大量线程规则并行执行，让 Tensor Cores 吃满矩阵乘法，让数据尽量留在 register/shared memory 中复用，减少 HBM 往返。FlashAttention 是这一思想的代表——用 tiling、online softmax 和 recomputation 重新组织 attention 计算，在保持数学等价的同时显著降低 memory I/O。
