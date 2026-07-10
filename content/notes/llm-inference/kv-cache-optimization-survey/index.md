+++
title = '长上下文 LLM 推理中的 KV Cache 优化综述：系统管理、缓存压缩与架构协同'
date = 2026-07-10
draft = false
summary = '以 KV Cache Size = 2×L×T×H_kv×D_h×bytes 为锚点，系统梳理 2023–2026 年 KV Cache 优化全景：PagedAttention 与 KV-aware serving 的系统管理、DapQ/LaProx/TurboQuant/LongFlow 等压缩新方法、MHA→MLA→DeepSeek V4 CSA/HCA→GDN→MLRA 的架构演进，并讲清四条路线（压维度/压序列/少访问/固定状态）的边界。'
tags = ['KV Cache', 'LLM 推理', 'PagedAttention', 'DeepSeek V4', 'MLA', '长上下文']
showReadingTime = true
showTableOfContents = true
+++

{{< katex >}}

> 本文基于 2026 年组会综述汇报整理，覆盖 2023–2026 年 KV Cache 优化的系统管理、缓存压缩与架构协同三条主线，并梳理最新工作（DeepSeek V4 CSA/HCA、MLRA、GDN 等）。

长上下文与长输出正在成为 LLM 推理的常态——长文档问答、RAG、Agent 工具调用、多轮对话、Reasoning Model 的长链推理，都在把 KV Cache 从「推理加速机制」推向「核心资源瓶颈」。本文不堆砌论文，而是用一条公式作为思维骨架，把纷繁的方法组织成一个有层次的优化图谱。

## 核心公式

$$\text{KV Cache Size} = 2 \times L \times T \times H_{kv} \times D_h \times \text{bytes}$$

| 符号 | 含义 |
|---|---|
| $2$ | key + value 两份 |
| $L$ | 层数 |
| $T$ | 上下文长度（含已生成 token） |
| $H_{kv}$ | KV head 数 |
| $D_h$ | 每个 head 的维度 |
| $\text{bytes}$ | 每个元素的字节数 |

全文每个优化方法，都锚定在这个公式的某个变量上：

- **系统管理**优化 $T$ 的调度与放置（不减少 $T$ 本身）
- **压缩 / 量化**减少有效 $T$ 或 $\text{bytes}$
- **架构协同**减少 $H_{kv}$、改变缓存表示，或压缩有效 $T$

---

## 一、为什么 KV Cache 成为新瓶颈？

本章建立动机：在核心公式里，**$T$ 正在不断增长**。

### 自回归推理的两个阶段：Prefill 与 Decode

LLM 推理分为 prefill 和 decode 两个阶段，它们的资源瓶颈截然不同：

- **Prefill**：把完整 prompt 一次性送入模型，并行处理，构建初始 KV Cache。这一阶段算力密集，更偏 **compute-bound**。
- **Decode**：每次只生成一个 token，每一步都要读取全部历史 KV Cache，同时追加新 token 的 K/V。访存密集，更偏 **memory-bandwidth-bound**。

一句话：**Prefill builds the cache; Decode consumes and extends the cache.**

{{< mermaid >}}
graph TD
    P["Prompt tokens"] --> PF["Prefill<br/>并行处理 / 构建 KV Cache<br/>compute-bound"]
    PF --> D1["Decode step 1<br/>读 cache → 生成 token 1 → 追加 KV"]
    D1 --> D2["Decode step 2<br/>读更新后的 cache → 生成 token 2 → 追加 KV"]
    D2 --> DN["...<br/>memory-bandwidth-bound"]
{{< /mermaid >}}

长上下文与长输出下，decode 阶段会不断访问越来越大的 cache，显存带宽压力非常突出。

### KV Cache 缓存什么：动态推理状态，不是模型权重

Transformer 每一层都会为 token 计算 key 和 value（$K = XW_K$，$V = XW_V$）。自回归生成时，历史 token 的 K/V 对后续 step 不会变化，所以可以缓存起来。KV Cache 和模型权重是两类完全不同的东西：

| 项目 | 模型权重 | KV Cache |
|---|---|---|
| 性质 | 固定参数 | 动态状态 |
| 是否请求相关 | 否 | 是 |
| 是否可共享 | 多请求共享 | 通常每请求独立 |
| 增长方式 | 固定大小 | 随上下文和输出增长 |
| 常见优化 | 权重量化、剪枝 | 管理、压缩、量化、架构协同 |

**KV Cache 不是模型权重，而是每个请求在推理过程中动态增长的状态。** 这个区分决定了后续优化路线的根本差异。

### 显存公式与线性增长

回到核心公式 $\text{KV Cache Size} = 2 \times L \times T \times H_{kv} \times D_h \times \text{bytes}$。最重要的变量是 $T$——cache 随 $T$ **线性增长**。serving 场景还要乘上并发请求数，瓶颈从两个方向逼近：

- **容量压力**：cache 能不能存下（GPU HBM 容量有限）
- **带宽压力**：decode 每一步读取历史 KV 的 HBM 带宽

上下文从 4K → 32K → 128K → 1M，KV Cache 同步线性膨胀，容量和带宽双双吃紧。

### 2025–2026 的新压力：三种 Long

传统长上下文主要关注 long input（长文档、RAG）。但 2025–2026 的 workload 带来了三种新的「Long」：

- **Long Input**：长文档问答、RAG、多文档检索——压力来自 prefill 后形成的大规模初始 KV Cache。
- **Long Output**：Chain-of-thought reasoning、self-reflection、verification——压力来自 decode 中不断追加新 token 的 K/V。输入可能不长，但 CoT 很长。
- **Long-lived State**：multi-turn dialogue、Agent tool calls——cache 生命周期更长，涉及恢复、迁移、复用。

{{< mermaid >}}
graph TD
    LI["Long Input<br/>长文档 / RAG"] --> B["更大、更长生命周期、<br/>workload-dependent KV Cache"]
    LO["Long Output<br/>CoT / verification"] --> B
    LS["Long-lived State<br/>多轮对话 / Agent"] --> B
    B --> P["GPU 显存容量 + 带宽瓶颈"]
{{< /mermaid >}}

**KV Cache 优化正在从「长输入管理」扩展到「长输出推理」和「长生命周期状态管理」。** 这是后文第五部分前沿趋势的伏笔，也是 2025–2026 KV Cache 研究升温的重要原因。

## 二、系统管理：KV Cache 怎么放、怎么调度、怎么恢复

本章的发力点是 **$T$ 的管理与调度，但不减少 $T$ 本身**。明确边界：系统管理提高的是利用率，不改变每个 token 的 KV 表示。

### PagedAttention：像虚拟内存一样管理 KV Cache

[PagedAttention](https://arxiv.org/abs/2309.06180)（vLLM, SOSP 2023）借鉴操作系统的分页思想：请求在逻辑上看到连续的 token 序列，但 KV blocks 在物理 GPU memory 中可以不连续，由一张 **block table** 做映射。

{{< mermaid >}}
graph LR
    subgraph Logical["逻辑 KV blocks（请求视角：连续）"]
        L0["block 0"] --> L1["block 1"] --> L2["block 2"] --> L3["block 3"]
    end
    subgraph BT["Block table（页表）"]
        direction LR
        T0["0 → 物理 7"]
        T1["1 → 物理 2"]
        T2["2 → 物理 9"]
        T3["3 → 物理 4"]
    end
    subgraph Physical["GPU 物理块（分散）"]
        P0["0"] --- P7["7"] --- P2["2"] --- P9["9"] --- P4["4"]
    end
    L0 -.-> T0 -.-> P7
    L1 -.-> T1 -.-> P2
    L2 -.-> T2 -.-> P9
    L3 -.-> T3 -.-> P4
{{< /mermaid >}}

它解决的是：连续显存分配的碎片、输出长度未知导致的预分配浪费、decode 动态增长，从而支撑更大的有效 batch size 和 continuous batching。

**PagedAttention is memory management, not KV compression.** 它不改变 K/V 表示本身。

### KV-aware scheduling：把 cache 容量变成一等调度约束

PagedAttention 解决了 KV blocks 在 GPU 里怎么组织，但真实服务中多个请求同时到达、长度不同、输出预算不同。传统按到达时间 batching 只关注 batch size 和 compute，忽略未来 KV Cache 占用，可能导致 OOM 或高延迟。

[Online Scheduling for LLM Inference with KV Cache Constraints](https://arxiv.org/abs/2502.07115)（2025）一类的工作把 KV Cache capacity 纳入调度模型：估算 prompt 长度 + 输出预算 + KV footprint，在 KV cache 约束下做在线 batching，换取更低延迟和更好资源利用。

**KV Cache 不只是内存对象，而是 online serving scheduler 必须显式建模的一等约束。**

### Agent 与多轮对话：cache 生命周期变长了

Agent workload 和普通单轮问答不同。Agent 会多次规划、调用工具、等待返回，然后继续推理。这个过程中 KV Cache 可能长时间 idle，但又不能随便丢掉——否则恢复时 TTFT（首 token 延迟）会飙升。

{{< mermaid >}}
graph LR
    ACTIVE["active<br/>正在推理"] -->|tool call| IDLE["idle<br/>等待返回"]
    IDLE -->|resume| ACTIVE
    IDLE -->|空间竞争| EVICT["evicted"]
    EVICT -->|重算/加载| RESTORE["restored"]
    ACTIVE -.->|prefix 共享| SHARED["shared / reused"]
{{< /mermaid >}}

新挑战包括：cache 在工具调用期间 idle、critical agent cache 被挤出会显著增加 TTFT、多 agent 并发导致空间竞争、shared prefix / session cache 需要复用策略。代表方向如 TokenCake（KV-Cache-Centric Serving for Multi-Agent）、Continuum（带 KV Cache TTL 的 multi-turn agent 调度）。

**Agent 场景使 KV Cache 管理从「单次请求资源分配」变成「跨时间的状态生命周期管理」。**

### KV Cache restoration：重算还是加载？

多轮对话、RAG、Agent 场景经常需要恢复已有 KV Cache。恢复有两种方式，各有代价：

- **重算（Recompute）**：从原始 prompt 重新计算，消耗 GPU compute，长上下文下代价高。
- **加载（Load）**：从 CPU/SSD/remote 加载，消耗 I/O 带宽，可能增加 TTFT。

CacheFlow 把这个问题变成 token / layer / multi-GPU 三维并行恢复：token 级（早期 chunk 重算、后期 chunk 加载）、layer 级（低层重算、高层加载）、multi-GPU 级（并发恢复 shard）。Kareto 则从 multi-objective tiered storage 配置角度，在 latency ↔ throughput ↔ cost 之间权衡，跨 GPU HBM / DRAM / SSD / remote 做分层。

**2026 年的系统问题不只是「cache 放在哪里」，还包括「cache 如何在计算和 I/O 之间高效恢复」。**

### 三个常被混淆的层次：FlashAttention / PagedAttention / KV Compression

这是一个常被混淆的关键区分，三者位于不同层次，不是替代关系：

| 层次 | 代表方法 | 解决的问题 |
|---|---|---|
| Attention kernel | [FlashAttention](https://arxiv.org/abs/2205.14135) | 减少 attention 计算中的 HBM/SRAM IO，避免 materialize attention matrix |
| KV block management | [PagedAttention](https://arxiv.org/abs/2309.06180) | 减少碎片，支持动态 batching |
| Cache policy | eviction / quantization / retrieval | 减少保存或访问的 KV |

FlashAttention 优化 attention 计算路径（尤其 prefill/训练的 IO）；PagedAttention 管理 decode serving 中的 KV blocks；KV compression 才真正减少或近似历史 K/V。

### 小结：系统管理的边界

系统管理能做的：显存碎片与预分配浪费 ✓、动态增长与 continuous batching ✓、KV-aware scheduling ✓、multi-turn/agent cache 生命周期 ✓、restoration/offloading/multi-tier storage ✓。

系统管理做不到的：单个 token 的 KV 表示大小 ✗、KV Cache 随上下文线性增长 ✗、long output 中 decode cache 持续增长 ✗。

**System management improves utilization; compression reduces what must be stored or accessed.** 这就自然引出下一部分——哪些 KV 值得保留。

## 三、缓存压缩：哪些 KV 值得保留？

本章的发力点是 **减少有效 $T$ 或 $\text{bytes}$**。系统管理回答「KV Cache 怎么放」，压缩回答「哪些 KV 保留、如何表示、何时压缩」。重点是 2025–2026 对「重要性」定义的修正，而不是旧的 SnapKV/PyramidKV。

### 压缩方法的四条路线

{{< mermaid >}}
graph TD
    ROOT["KV Cache Compression<br/>哪些 KV 值得保留？"]
    ROOT --> A["Token selection<br/>eviction / pruning"]
    ROOT --> B["Representation compression<br/>quantization"]
    ROOT --> C["Semantic compression<br/>chunk-level selection"]
    ROOT --> D["Decode-time compression<br/>long-output / reasoning-aware"]
    A -.-> A1["SnapKV / DapQ / LaProx / KVP"]
    B -.-> B1["KIVI / KVQuant / TurboQuant"]
    C -.-> C1["ChunkKV"]
    D -.-> D1["LongFlow / Moment-KV"]
{{< /mermaid >}}

### 旧基线：attention-based heuristic

早期方法奠定基础，但大多依赖 attention pattern，且主要面向 long input：

| 方法 | 核心原理 | 优势 | 局限 |
|---|---|---|---|
| [StreamingLLM](https://arxiv.org/abs/2309.17453) | sink tokens + recent window | 稳定流式生成 | 语义选择弱 |
| [SnapKV](https://arxiv.org/abs/2404.14469) | observation window attention | training-free，简单 | input-side 与 decode-side 可能错位 |
| PyramidKV | layer-wise budget | 考虑层间差异 | 仍依赖 attention pattern |

关键铺垫：**Attention score is useful, but not always sufficient.** attention sink 说明「高 attention」不一定等于语义重要——开头 token 可能只是承担概率质量或位置稳定作用。这正是后面 DapQ 和 LaProx 对「重要性」定义做修正的出发点。

### position-aware：DapQ（位置比内容更重要）

SnapKV 的问题：用 prompt 末尾的 observation window 近似未来 decode attention，但 input-side query 可能与 decode-time query 错位。

DapQ 的关键是**位置对齐**：它认为构造 future query 时，位置比内容更重要。流程是追加 pseudo tokens → 赋予未来 decoding 位置 → pseudo queries attend to prompt KV → Top-K 选择 prompt KV。

{{< mermaid >}}
graph LR
    A["Append pseudo tokens"] --> B["赋予未来 decode 位置"]
    B --> C["pseudo queries attend to prompt KV"]
    C --> D["Top-K 选择 prompt KV"]
{{< /mermaid >}}

DapQ training-free，比 observation window 更 decode-aligned；局限是主要压缩 prompt KV，pseudo position 是超参数。

### output-aware：LaProx（attention score ≠ 真实输出贡献）

LaProx 指出 token 是否重要不只取决于 attention weight，还取决于 value 和 output projection。它把 eviction 重构为 output-aware 的矩阵乘法近似：

$$\text{Attention output} = A \cdot V \cdot W_O, \qquad \text{importance}_i \propto \|A_{:,i}\| \times \|(V W_O)_i\|$$

即 token 对输出的真实贡献由 attention map $A$、value $V$、output projection $W_O$ 三者共同决定。这比纯 attention 启发式更接近真实输出贡献，代价是计算和实现复杂度更高，端到端收益依赖 kernel。

### learning-based：KVP（从手工规则到学习式 policy）

2026 年出现了 learned eviction：把 $(k_i, v_i, \text{position}_i)$ 喂给一个轻量 policy / RL agent，输出 token ranking，保留 top-budget KV。KVP 学习的是 token 对未来 decoding 的 utility，不再依赖固定启发式，可适配不同 head/budget；代价是需要离线训练，泛化和部署成本更高。

### 量化：从 scalar 到 vector quantization

量化的背景是 KIVI / KVQuant 这类 activation-specific 低比特量化，处理 key/value 不对称、outlier、RoPE 效应。新进展 TurboQuant（QJL + PolarQuant）走向 online vector quantization：

| 维度 | scalar quant（KIVI / KVQuant） | vector quant（TurboQuant） |
|---|---|---|
| 压缩单位 | 单个标量 | 向量 |
| 关注点 | 低比特表示 | near-optimal distortion、保持内积结构 |
| token | 可能丢 token | 不丢 token |

核心动机：attention score 取决于 $q^\top k$，所以量化要尽量保持向量内积结构，而不是单纯追求每个元素的低比特。TurboQuant 很新，实际收益依赖 kernel 和 benchmark 范围。

### semantic-aware：ChunkKV（以语义 chunk 为压缩单位）

token-level pruning 的反例：可能保住了关键词，却丢了条件/否定这种依赖语义连续性的内容。ChunkKV 以语义 chunk 而非孤立 token 为压缩单位：semantic chunking → chunk importance estimation → preserve coherent chunks，保留语义连续性和证据链。局限是 chunk 边界和粒度影响压缩率与质量。

### reasoning-aware：LongFlow（decode-time 压缩）

要区分 long input 和 long output：前者是 prefill 后形成大 prompt KV，后者是 decode 中 CoT 持续增长。[LongFlow](https://arxiv.org/abs/2605.29873)（2026）面向 reasoning 的 long output，复用 attention 中间结果 $\alpha_t$，用 $\text{score}_i = \|\alpha_t^i v_i\|$ 估计 token 重要性，把 attention + estimation + eviction 融合进一个 kernel。卖点是 zero-history + zero-cost estimation——复用已有 attention 中间量，几乎不额外算；局限是依赖 query 相似性，需要定制 kernel。

### 动态压缩趋势与核心 trade-off

动态压缩还有 RocketKV（coarse permanent eviction + dynamic sparse attention）、Moment-KV（momentum-based decode-time importance tracking）等方向。但压缩不是单纯追压缩率，共性 trade-off 有五条：

| 共性 trade-off | 含义 |
|---|---|
| Memory saving ≠ latency improvement | 省显存不等于端到端变快 |
| Attention score ≠ 真实重要性 | 见 LaProx |
| Long input ≠ Long output | 是不同问题，见 LongFlow |
| 高压缩率可能损害 retrieval/reasoning/code | 任务敏感 |
| 真实收益依赖 kernel/scheduler/workload | 离开系统谈压缩率无意义 |

**压缩的灵魂是 memory-latency-quality-system 的平衡，而不是更高的压缩率。**

## 四、架构协同：从设计源头减少 KV Cache

（待写）

## 五、前沿趋势：任务、检索、验证、系统共同感知

（待写）

## 六、统一框架与未来工作

（待写）

## 参考

（待写）
