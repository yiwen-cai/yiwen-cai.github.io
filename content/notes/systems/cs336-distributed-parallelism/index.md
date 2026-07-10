+++
title = '分布式训练并行策略：CS336 Lecture 7 笔记'
date = 2026-06-15
draft = false
summary = '从单 GPU 扩展到多 GPU/多机并行：集合通信原语（all-reduce = reduce-scatter + all-gather）、NVLink/InfiniBand 互联，以及 DDP、FSDP/ZeRO、Tensor/Pipeline/Sequence Parallelism 的取舍与实践法则。'
tags = ['分布式训练', '并行策略', 'DDP', 'FSDP', 'CS336']
showReadingTime = true
showTableOfContents = true
+++

> 基于 Stanford CS336 Lecture 7（Parallelism），Tatsunori Hashimoto & Percy Liang，Stanford Spring 2025。视频：[YouTube](https://www.youtube.com/watch?v=l1RJcDjzK8M)，Slides：[lecture_07.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py)。

上周讲单 GPU 内部的并行（kernel fusion、tiling、shared memory）。本周扩展到**多 GPU / 多机并行**。核心主题：**编排计算以避免数据传输瓶颈**。无论单 GPU 内部还是多 GPU 之间，根本问题都是——算力离数据太远。

广义的存储层级：单 GPU 内 L1/shared memory（最快）→ 单 GPU 内 HBM → 单节点多 GPU NVLink/NVSwitch → 多节点 InfiniBand/Ethernet（最慢）。

## 分布式通信 / 计算的基础构件

### 为什么需要多 GPU？

1. **内存不够**：参数量 + 优化器状态 + 梯度 + 激活值，单卡放不下
2. **算力不够**：想要更多 FLOPs 缩短训练时间

### 集合通信操作（Collective Operations）

基本概念：**Rank** = 一个设备/GPU（如 0,1,2,3）；**World Size** = 总设备数（如 4）。

| 类别 | 操作 | 说明 |
|------|------|------|
| **基础** | Broadcast | rank 0 复制到所有 rank |
| | Scatter | rank 0 切分后分发，每人拿 1/world_size |
| | Gather | 所有 rank 汇集到 rank 0（scatter 的逆）|
| | Reduce | 所有 rank 聚合（SUM/MAX/MIN）到 rank 0 |
| **主力** | All-gather | 每人把自己的 shard 广播给所有人，每人最终持有完整拼接 |
| | Reduce-scatter | 先 reduce 再 scatter，每人只拿到结果的 1/world_size |
| | All-reduce | 先 reduce 再 broadcast 给所有人，每人拿完整聚合结果 |
| **特殊** | All-to-all | 每人对每人发一份数据（MoE 路由的关键操作），等价于转置 |

**关键恒等式**：

$$\text{all-reduce} = \text{reduce-scatter} + \text{all-gather}$$

```
All-reduce 一步到位：
  rank0: [0,1,2,3]    →    [6, 10, 14, 18]
  rank1: [1,2,3,4]    →    [6, 10, 14, 18]
  rank2: [2,3,4,5]    →    [6, 10, 14, 18]
  rank3: [3,4,5,6]    →    [6, 10, 14, 18]

等价于 reduce-scatter + all-gather：
  Reduce-scatter:     rank0=[6], rank1=[10], rank2=[14], rank3=[18]
  All-gather:         所有人=[6,10,14,18]
```

带宽效率上两者等价，但拆成两步给了 FSDP/ZeRO 灵活性的空间。

命名记忆法：Reduce = 聚合（求和/取最大/取最小）；Scatter 是 Gather 的逆；All = 结果发给所有设备。

### 硬件互联

| 层级 | 介质 | 带宽（典型值）| 物理拓扑 |
|------|------|-------------|----------|
| 单节点多 GPU | NVLink 5.0 + NVSwitch | ~1.8 TB/s（B200）| 全互联（all-to-all）|
| 多节点（同 pod）| InfiniBand | ~0.05 TB/s | 全互联 ≤256 GPU |
| 跨 pod / 跨数据中心 | Ethernet | ~200 MB/s ~ 0.05 TB/s | 叶脊交换机 |

**绕过 CPU 的技术**：
- **RDMA**（Remote Direct Memory Access）：GPU 直接读写远端 GPU 内存，不经 CPU
- **RoCE**（RDMA over Converged Ethernet）：让 Ethernet 也支持 RDMA（Meta 在用）
- InfiniBand 原生支持 RDMA，标准 Ethernet 不支持

**GPU vs TPU 网络拓扑差异**：GPU 节点内 8 GPU 全互联（NVSwitch），≤256 GPU 内任意通信都快；TPU 是 3D 环面网格（Toroidal Mesh），芯片只和邻居连接，可轻松扩展但只能邻居通信。对于集合通信，两者在理论上效率相同。

### NCCL 与 PyTorch Distributed

NCCL（NVIDIA Collective Communication Library）把集合通信操作翻译成 GPU 间传输的低层数据包：检测硬件拓扑 → 优化通信路径 → 启动 GPU kernel 做收发。

```python
import torch.distributed as dist

# 初始化
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "15623"
dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 核心操作
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)      # 修改 in-place
dist.reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM)
dist.all_gather_into_tensor(output_tensor, input_tensor)
dist.broadcast(tensor, src=0)

# 清理
dist.destroy_process_group()
```

后端：**gloo**（CPU）、**nccl**（GPU）。

All-reduce 的有效带宽公式：$\text{bandwidth} \approx \frac{2 \times \text{size\_bytes}}{\text{duration}}$，与 world_size 无关，与拓扑无关。

## 分布式训练算法

### Data Parallelism（DDP）

每张卡持有完整模型副本，把 batch 切分到各卡。

```python
# 每个 rank 拿自己的数据切片
data = data[rank * local_bs : (rank+1) * local_bs]

for step in range(num_steps):
    loss = forward(data, params)
    loss.backward()

    # 唯一区别于单卡训练的地方
    for param in params:
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)  # 注意是 AVG

    optimizer.step()
```

| 维度 | 评分 | 说明 |
|------|------|------|
| 计算扩展 | ✅ 好 | 每卡拿到 B/world_size 样本，batch 够大就能打满算力 |
| 通信开销 | ⚠️ 中等 | 每步 all-reduce 2×参数量，batch 大可以隐藏 |
| 内存扩展 | ❌ 差 | **每张卡都要存完整模型 + 优化器状态** |

使用 `ReduceOp.AVG` 而非 `SUM`：AVG 自动除以 world_size。

### 为什么 DDP 的内存问题这么严重？

以 AdamW 优化器为例，一个参数需要存储：

| 组件 | 精度 | 字节数 |
|------|------|--------|
| 参数（weights）| BF16 | 2 |
| 梯度（gradients）| BF16 | 2 |
| 主权重（master weights）| FP32 | 4 |
| Adam m（一阶矩）| FP32 | 4 |
| Adam v（二阶矩）| FP32 | 4 |
| **总计** | | **16 bytes/param** |

**内存大头是优化器状态**（m + v + master weights = 12 bytes），占 75%。这就是为什么需要 ZeRO / FSDP。

### FSDP / ZeRO（Fully Sharded Data Parallel）

渐进式内存节省：

| 阶段 | 切分内容 | 内存节省（相对 DDP）|
|------|----------|---------------------|
| ZeRO-1 | 优化器状态 | ~4× |
| ZeRO-2 | 优化器状态 + 梯度 | ~8× |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | 线性（world_size 倍）|

**数据流（ZeRO-3 / FSDP）**：

```
Forward:
  all-gather params → 拼出完整参数（临时）→ 前向计算 → 释放完整参数

Backward:
  反向计算 → 每卡算出完整梯度
  reduce-scatter grads → 每卡只保留自己的梯度 shard

Optimizer:
  每卡只对自己的参数 shard 做 optimizer.step()
  每卡只存自己的优化器状态
```

**核心操作对**：`all-gather`（forward 拼参数）+ `reduce-scatter`（backward 收梯度）。代价：每个 layer 都要做 all-gather 和 reduce-scatter，有同步 barrier，比 DDP 更多通信量和同步点。这就是开头强调 `all-reduce = reduce-scatter + all-gather` 的原因——DDP 用 all-reduce 一步完成，FSDP 拆成两步，换来了内存线性扩展。

### Tensor Parallelism

横切——把每层的权重矩阵沿列/行方向切开，每张卡只保留矩阵的一「列」。

```python
# 每张卡用自己的参数分片做局部计算
x = x @ params[layer]          # (B, local_dim)

# All-gather 把各卡的激活值拼回完整维度
activations = [torch.empty(B, local_dim) for _ in range(world_size)]
dist.all_gather(tensor_list=activations, tensor=x)
x = torch.cat(activations, dim=1)  # (B, num_dim)
```

| 维度 | 评分 | 说明 |
|------|------|------|
| 计算扩展 | ✅ 好 | 算力随 GPU 数线性增长 |
| 通信开销 | ❌ 高 | **每层都要 all-gather**，需要极快互联 |
| 内存扩展 | ✅ 线性 | 参数/激活都能切分 |
| 对 batch size 影响 | ✅ 无 | 唯一不消耗 batch size 的并行 |

**实践规则**：TP 只在单节点内做（NVLink 带宽够），通常 TP=8。

### Pipeline Parallelism

纵切——把模型的不同层放到不同 GPU 上，数据像流水线一样穿过各 GPU。

**Pipeline Bubble 问题**：GPU 间有空闲时间。**解决方案：Micro-batches**——把一个大 batch 切成多个 micro-batch 交错执行减少空闲。micro-batch 越多 bubble 越小，但需要更多内存存中间激活值。核心是用 P2P 通信（`dist.send` / `dist.recv`），不需要集合通信。

| 维度 | 评分 | 说明 |
|------|------|------|
| 计算扩展 | ⚠️ 中等 | 有 pipeline bubble |
| 通信开销 | ✅ 低 | 点对点传输，带宽要求低 |
| 内存扩展 | ✅ 线性 | 每卡只存一个 stage 的参数 |
| 对 batch size 影响 | ❌ 消耗 | micro-batch 数消耗有效 batch size |
| 工程复杂度 | ❌ 高 | 需要精细调度 micro-batch |

### Sequence Parallelism & Activation Memory

即使用了 TP，激活内存中仍有一些项没法被 TP 切分——LayerNorm、Dropout 等 point-wise 操作。

**激活内存公式**（Transformers 单层）：

$$\text{activation memory} = SBH \times 34 + \frac{5AS^2B}{H}$$

- 左边（34 SBH）：MLP 和点操作的激活（取决于 H）
- 右边（5AS²B / H）：Attention softmax 的中间结果（和 S² 成正比）

用了 TP（切分到 T 个设备）后：

$$\text{memory after TP} = \frac{SBH \times 34}{T} + \frac{5AS^2B}{H}_{\text{被 FlashAttention 消掉}} + SBH \times 10$$

- 第二项被 FlashAttention 消掉（recomputation）
- 第三项 `SBH × 10` 是 LayerNorm、Dropout 等不受 TP 影响的残留

**Sequence Parallelism** 沿序列维度切分这些 point-wise 操作，最终收敛到 $\text{minimal activation memory} \approx \frac{SBH \times 34}{T}$，这是 TP + FlashAttention + Sequence Parallelism 的极限。

### 其他并行策略

| 策略 | 思路 | 适用场景 |
|------|------|----------|
| **Ring Attention / Context Parallel** | 切分长序列的 attention，KV 在设备间循环传递 | 超长上下文训练 |
| **Expert Parallelism** | MoE 中不同 expert 放不同设备，用 all-to-all 路由 | MoE 模型 |

## 组合使用

### 并行策略对比总表

| 策略 | 切分维度 | 通信开销 | 内存扩展 | 消耗 batch size | 适用网络 |
|------|----------|----------|----------|:---:|----------|
| **DDP** | Batch | 每步 1 次 all-reduce | ❌ 无 | 否 | 任意 |
| **FSDP (ZeRO-3)** | Batch+参数 | 每层 all-gather + reduce-scatter | ✅ 线性 | 否 | IB/Ethernet |
| **Tensor Parallel** | Width | 每层 all-gather | ✅ 线性 | 否（唯一！）| NVLink |
| **Pipeline Parallel** | Depth | 低（P2P）| ✅ 线性 | **是** | IB/Ethernet |
| **Sequence Parallel** | Sequence | all-gather + reduce-scatter | 辅助 | 否 | NVLink |

### 三种有限资源

1. **Memory**：决定模型能不能跑
2. **Bandwidth + Compute**：决定跑得快不快
3. **Batch Size**：影响通信隐藏效率——batch 太小则通信 overhead 占比高

### 实践法则（Rule of Thumb）

```
第一步：让模型放进内存（硬约束）
  → Tensor Parallelism：先铺满单节点内 GPU（通常 TP=8），利用 NVLink 的高带宽
  → FSDP (ZeRO-3) 或 Pipeline Parallelism：跨节点扩展直到模型能装下

第二步：用剩余的 GPU 做 Data Parallelism 放大吞吐
  → DP 带宽要求低，最灵活

第三步（可选）：如果 batch size 太小
  → Gradient Accumulation：多步再同步一次梯度，等价于增大有效 batch size
```

带宽从高到低的并行策略层：

```
TP (NVLink)  →  CP (NVLink)  →  PP (IB)  →  DP (IB/Ethernet)
高带宽需求 ←─────────────────────────────→ 低带宽容忍
```

### 真实案例

| 模型 | 并行策略 | 细节 |
|------|----------|------|
| **Megatron-LM** (530B) | TP=8 + PP 递增 + DP 调整 | 1.7B→1T 参数，TP 在 8 封顶，大模型加 PP |
| **DeepSeek-V3** | 16-way PP + 64-way Expert Parallel + ZeRO-1 DP | 用 Expert Parallel 替代 TP |
| **Llama 3** (405B) | TP=8 + CP（长上下文）+ PP + DP | 严格按 TP→CP→PP→DP 带宽递减顺序 |
| **Gemma 2** (TPU) | ZeRO-3 + Model Parallelism | TPU 3D Torus 允许更大的 Model Parallel 范围 |

**Llama 3 的血泪教训**：训练中断共 **466 次**，其中 **148 次** 来自 GPU 硬件故障（占 30%）；计划外维护导致 **32 次** 中断；**静默数据损坏**（silent data corruption）比显式故障更可怕——GPU 可能输出错误的数值而不报任何错误。

## 关键恒等式速查

| 恒等式 | 含义 |
|--------|------|
| all-reduce = reduce-scatter + all-gather | DDP 一步 = FSDP 两步，带宽等价 |
| 有效 BW ≈ 2×size/duration | all-reduce 带宽公式（与 world_size 无关）|
| memory/param = 16 bytes (AdamW) | 参数 2 + 梯度 2 + master 4 + m4 + v4 |
| activation ≈ SBH×34/T | TP + FlashAttn + SeqParallel 后的极限 |

## 参考

- 课程网站：[cs336.stanford.edu](https://cs336.stanford.edu/)
- Slides 源码：[github.com/stanford-cs336/lectures](https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py)
- 视频：[YouTube](https://www.youtube.com/watch?v=l1RJcDjzK8M)
