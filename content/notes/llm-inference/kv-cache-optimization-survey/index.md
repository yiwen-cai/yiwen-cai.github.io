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

（待写）

## 三、缓存压缩：哪些 KV 值得保留？

（待写）

## 四、架构协同：从设计源头减少 KV Cache

（待写）

## 五、前沿趋势：任务、检索、验证、系统共同感知

（待写）

## 六、统一框架与未来工作

（待写）

## 参考

（待写）
