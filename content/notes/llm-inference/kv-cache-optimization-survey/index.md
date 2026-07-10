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

本章是全文重点。发力点是 **$H_{kv}$ 与缓存表示，并进一步攻击有效 $T$**——问题前移到模型设计阶段，而不是推理时再压缩。

### 架构协同的路线全景

回到公式 $\text{KV Cache Size} = 2 \times L \times T \times H_{kv} \times D_h \times \text{bytes}$，架构协同抓三个量：$H_{kv}$（减少 KV heads）、cached state dimension（latent 压维度）、有效 $T$（token-axis compressed KV）。

{{< mermaid >}}
graph LR
    A["MHA<br/>H_kv=H_q"] --> B["MQA<br/>H_kv=1"]
    B --> C["GQA<br/>1<H_kv<H_q"]
    C --> D["MLA<br/>latent state"]
    D --> E["TransMLA<br/>GQA→MLA 泛化"]
    E --> F["Sparse/Compressed<br/>Top-k / CSA / HCA"]
    F --> G["MLRA / GDN<br/>低秩可切分 / 固定状态"]
{{< /mermaid >}}

### MHA → MQA → GQA：减少 KV heads

最直接的降 KV Cache 方式是减少 KV head 数，这是「共享 K/V」的折中：

| 架构 | KV heads | 特点 |
|---|---|---|
| MHA | $H_{kv} = H_q$ | 表达能力强，cache 最大 |
| [MQA](https://arxiv.org/abs/1911.02150) | $H_{kv} = 1$ | cache 最小，可能影响质量 |
| [GQA](https://arxiv.org/abs/2305.13245) | $1 < H_{kv} < H_q$ | 质量与显存折中 |

MQA 保留多个 query heads，共享一组 K/V；GQA 在 MQA 的极端共享和 MHA 的完全独立之间折中。

### MLA：缓存 latent state，压维度

[MLA](https://arxiv.org/abs/2405.04434)（DeepSeek-V2）把 KV Cache 优化前移到架构层：缓存的是低维 latent state，而不是完整 K/V。

{{< mermaid >}}
graph LR
    H["Hidden state"] -->|"down projection"| L["低维 latent state<br/>（缓存这个）"]
    L -->|"attention 时 up projection"| KV["还原出 K/V 用于 attention"]
{{< /mermaid >}}

从架构层减少 KV Cache，不需要 token eviction；但需要训练阶段支持，不是任意模型即插即用。

### TransMLA：让 MLA 成为更一般的注意力表达

[TransMLA](https://arxiv.org/abs/2502.07864)（2025）建立 GQA 与 MLA-like latent attention 的结构联系，让已有 GQA/MHA 模型通过 transformation / reparameterization 向低 KV 架构迁移成为可能。代价是需要转换、训练或校准，不是无成本替换。

### DeepSeek V4 CSA/HCA：压序列长度（token-axis compressed KV）★

这是全文的技术核心。先讲清 MLA 与 V4 的边界：

- **MLA**：每个 token 的 cache entry 更窄（压 hidden dimension），但 cache length 仍随 $T$ 增长——一个历史 token 对应一个 cache entry。
- **DeepSeek V4 CSA/HCA**：把历史 token group 压缩成更少的 compressed entries（压 sequence length），让远距离历史 cache entries 变少——**直接攻击有效 $T$**。

{{< mermaid >}}
graph TD
    subgraph MLA["MLA / latent KV：压维度"]
        direction LR
        M1["token 1"] --> ME1["1 latent entry"]
        M2["token 2"] --> ME2["1 latent entry"]
        MN["token T"] --> MEN["1 latent entry"]
        MNOTE["cache length 仍随 T 增长<br/>每个 entry 更窄"]
    end
    subgraph V4["V4 CSA/HCA：压序列"]
        direction LR
        G1["token group 1"] --> CE1["compressed entries"]
        G2["远距离 token group"] --> CE2["更少 compressed entries"]
        V4NOTE["cache sequence 更短<br/>直接攻击有效 T"]
    end
    MNOTE -.-> V4NOTE
{{< /mermaid >}}

V4 的三条路径：

| 路径 | 机制 | 用途 |
|---|---|---|
| **CSA**（Compressed Sparse Attention） | 温和块压缩 → compressed entries + indexer/top-k 选择 | 减少远距离历史的存储条目数和每步访问量 |
| **HCA**（Heavily Compressed Attention） | 更强的历史块压缩 → 在短 cache 上做 dense/global attention | 用较低成本保留全局覆盖 |
| **SWA** / local branch | 保留最近窗口未压缩 KV | 弥补压缩历史的 token-level 细节损失 |

> 报告中 1M context 场景下 V4-Pro 相对 V3.2 的单 token FLOPs 约 27%、KV cache 约 10% 是**资料/公开报告口径，不是本文作者的复现实验**。CSA/HCA 不是无损删除历史，而是用压缩历史换效率；实际 serving 收益依赖推理引擎如何分别管理 CSA/HCA/SWA 和 prefix cache。

**MLA 让每个 token 的 KV entry 更窄；CSA/HCA 让远距离历史的 cache entries 更少——两者作用于公式的不同维度。**

### 四条路线的边界：压维度 / 压序列 / 少访问 / 固定状态

这是必须讲清的核心区分（四条路线经常被混为一谈）：

| 路线 | 改变什么 | 是否减少总存储 | 代表方法 |
|---|---|---|---|
| **MLA** | 压 hidden dimension（每个 entry 更窄） | 是 | DeepSeek-V2 |
| **CSA/HCA** | 压 sequence length（cache entries 更少） | 是 | DeepSeek V4 |
| **Top-k / page sparse** | 少访问（每步只读 $k$ 个，完整 KV 可能仍在） | 不一定 | Quest / MInference / LServe / NSA |
| **GDN** | 固定状态（不再为每个 token 存 KV） | 是 | Gated DeltaNet |

Top-k / page sparse attention 从 $O(T^2)$ → $O(Tk)$，$k$ 固定则线性。但要注意边界：**完整 KV 仍在 GPU 时，sparse 主要降带宽和 kernel 计算，不一定降总显存**；只有 full KV 移到 CPU/SSD、按需取 critical pages，才直接缓解 HBM 容量压力。

### GDN：从显式 KV Cache 到固定 recurrent state

softmax attention decode 时，cache $K_1, V_1 \dots K_T, V_T$，每步 query $q_t$ 读全部历史 KV，memory $O(T)$、per-step read $O(T)$。

GDN（[Gated Delta Networks](https://arxiv.org/abs/2412.06464)，ICLR 2025）走线性注意力路线，利用结合律把历史 key-value 外积累积为固定状态 $S_t$：

$$S_t = \text{gated\_decay}(S_{t-1}) + \text{delta\_update}(k_t, v_t), \qquad o_t = S_t^\top q_t$$

memory $\approx O(d^2)$，不随 $T$ 线性增长。GDN 在简单累加上加入 gating（控制旧记忆衰减）和 delta rule（用在线更新修正 key→value 关联）；GDN-2 解耦 erase 和 write 的 channel-wise gate；FG²-GDN 把标量 $\beta_t$ 改成 channel-wise，提高长上下文 associative recall。

{{< mermaid >}}
graph TD
    subgraph Soft["Softmax attention decode"]
        SK["cache K_1,V_1 ... K_T,V_T"] --> SQ["每步 q_t 读全部历史 KV"]
        SQ --> SO["memory O(T)<br/>per-step read O(T)"]
    end
    subgraph GDN["Linear / GDN attention"]
        GS["维护固定状态 S_t"] --> GUP["S_t = gated_decay(S_{t-1}) + delta_update(k_t,v_t)"]
        GUP --> GOUT["o_t = S_t^T q_t"]
        GOUT --> GM["memory ≈ O(d²)<br/>不随 T 增长"]
    end
{{< /mermaid >}}

边界：固定状态容量有限，精确 retrieval 可能遗忘。因此实际大模型常采用 **hybrid stack**——多数层用 GDN/linear attention，少数层保留 full/sliding-window attention。

### MLRA：面向 tensor parallel 的低秩 latent attention

MLA 的瓶颈：tensor parallel 下 latent cache 不易切分。MLRA（2026）将 latent state 分成多个可分区的 low-rank branches，保留低 cache 的同时改善多 GPU 并行效率。

**这体现的是 architecture-system co-design：注意力架构不仅要省 cache，还要适合多 GPU serving。**

## 五、前沿趋势：任务、检索、验证、系统共同感知

前三部分是相对成熟的分类，这一部分是从单点方法走向 workload-aware 的新趋势。

### Reasoning-aware：long output 成为新瓶颈

| 时代 | workload |
|---|---|
| Long input era | 长 prompt → 短回答 |
| Reasoning era | 适中 prompt → 长 CoT / verification / reflection |

reasoning model 使 long output 成为新瓶颈：输入可能不长，但 CoT 很长，decoding 中 KV Cache 持续膨胀。代表工作如 LongFlow、Moment-KV、Hold Onto That Thought、DesireKV。核心是关注 decode-time KV 增长而非只压缩 prompt KV；难点是中间推理 token 是否可丢弃很难判断。

### Retrieval-aware：每步访问哪些 KV

KV 优化不一定只是压缩存储，也可以优化每步 decode 访问哪些 KV：current query → retrieve 相关 KV 子集 → attend to selected → 避免读 full cache。

ParisKV 的关键设计：drift-robust retrieval、GPU-native retrieval path、sink + recent + CPU backing store、4-bit reranking。**KV 优化可以发生在访问侧，而不只是存储侧。**

### Verifiable / Lossless：compressed draft + full verify

[VeriCache](https://arxiv.org/)（2026）走 speculative 风格：compressed KV cache 出 draft tokens，full KV cache（在 GPU 外）做 verify，accept或纠正。兼顾加速和输出一致性；代价是 verification 和 full cache 管理有额外开销。

### System-aware multi-tier serving

未来 KV Cache 优化会把 compression、offloading、scheduler、kernel 和 workload 统一考虑。分层存储 GPU HBM ↔ CPU DRAM ↔ SSD/remote ↔ multi-GPU/distributed 成为标配，代表工作如 CacheFlow（3D-parallel restoration）、Kareto（tiered storage trade-off）、TokenCake/Continuum（agent cache lifecycle）、AsymCache。

## 六、统一框架与未来工作

### 五层统一框架

把全文串起来，KV Cache 优化是一个 Workload-aware Policy 统领下、三大支柱协同的问题：

{{< mermaid >}}
graph TD
    TOP["Workload-aware Policy<br/>reasoning · agents · RAG · serving"]
    TOP --> S["System Management<br/>where to place & restore KV?<br/>PagedAttention · CacheFlow · Kareto"]
    TOP --> C["Cache Compression<br/>which KV to keep / represent?<br/>DapQ · LaProx · TurboQuant · LongFlow"]
    TOP --> A["Architecture<br/>how to reduce KV by design?<br/>MLA · CSA/HCA · GDN · MLRA"]
{{< /mermaid >}}

**KV Cache 优化是算法、架构和系统共同作用的问题**——没有任何单一方法是银弹。

### 如何公平评测 KV Cache 方法

压缩率不是唯一指标。公平评测要看四个维度：

| 维度 | 指标 |
|---|---|
| Memory | 峰值显存、cache size、可支持 batch/context |
| Latency | TTFT、TPOT、decode throughput、恢复开销 |
| Quality | LongBench、RULER、perplexity、retrieval/reasoning/code tasks |
| System | kernel 是否规则、scheduler 是否能利用、offloading 是否稳定 |

四个常见误区：

| 误区 | 说明 |
|---|---|
| Needle-in-a-Haystack 做得好 ≠ 长上下文强 | 单点检索不等于完整长上下文能力 |
| 高压缩率 ≠ 端到端加速 | memory saving ≠ latency improvement |
| attention score 高 ≠ 语义重要 | 见 LaProx |
| 只看 prompt compression | 忽略 reasoning long-output 的 decode-time growth |

### 未来工作：从压缩率到 workload-aware co-design

1. Reasoning-aware cache policy
2. Training-time cache compressibility
3. Retrieval + compression hybrid KV memory
4. Hardware/kernel-friendly sparse KV access
5. Lossless or verifiable KV compression
6. Realistic multi-turn/agent serving benchmarks
7. Evaluation beyond Needle-in-a-Haystack（LongBench + RULER + serving metrics）

**The next stage is not just higher compression ratio, but memory-latency-quality-hardware-workload co-design.**

### 一句话总结

> KV Cache 优化不是单纯追求更高压缩率，而是在具体 workload 下共同平衡显存容量、显存带宽、延迟、输出质量、kernel 友好性和 serving 调度复杂度。

## 参考

### 系统管理
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) — Kwon et al., SOSP 2023
- [Online Scheduling for LLM Inference with KV Cache Constraints](https://arxiv.org/abs/2502.07115) — 2025
- CacheFlow: 3D-Parallel KV Cache Restoration — 2026
- Kareto: Multi-Objective Tiered Storage — 2026
- TokenCake: KV-Cache-Centric Serving for Multi-Agent — 2025–2026
- Continuum: Multi-Turn LLM Agent Scheduling with KV Cache TTL — 2026

### 缓存压缩
- [StreamingLLM / Attention Sinks](https://arxiv.org/abs/2309.17453) — 2023
- [SnapKV](https://arxiv.org/abs/2404.14469) — 2024
- PyramidKV — 2024
- KIVI — 2024
- KVQuant — NeurIPS 2024
- DapQ — 2026
- LaProx — 2026
- KVP — 2026
- TurboQuant — 2025/2026
- [ChunkKV](https://arxiv.org/abs/2502.00299) — 2025
- [LongFlow](https://arxiv.org/abs/2605.29873) — 2026
- [RocketKV](https://arxiv.org/abs/2502.14051) — 2025
- Moment-KV — 2026

### 架构协同
- [One Write-Head / MQA](https://arxiv.org/abs/1911.02150) — Shazeer, 2019
- [GQA](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [DeepSeek-V2 MLA](https://arxiv.org/abs/2405.04434) — 2024
- [TransMLA](https://arxiv.org/abs/2502.07864) — 2025
- [Native Sparse Attention / NSA](https://arxiv.org/abs/2502.11089) — ACL 2025
- [Quest](https://arxiv.org/abs/2406.10774) — ICML 2024
- [MInference](https://arxiv.org/abs/2407.02490) — NeurIPS 2024
- [LServe](https://arxiv.org/abs/2502.14866) — MLSys 2025
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) — ICLR 2025
- DeepSeek V4 CSA/HCA Compressed Attention — 2026（资料来源：DeepSeek V4 architecture report, 2026；CSA/HCA 解读参考 Sebastian Raschka (2026)、Together AI (2026)）
- MLRA — 2026

### 前沿趋势
- VeriCache — 2026
- ParisKV — 2026
- Hold Onto That Thought — 2026
- DesireKV — 2026

### 评测与基础
- [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al., NeurIPS 2022
- [LongBench](https://arxiv.org/abs/2308.14508)
- [RULER](https://arxiv.org/abs/2404.06654)
