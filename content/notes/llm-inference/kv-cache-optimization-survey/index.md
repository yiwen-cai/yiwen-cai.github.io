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

（待写）

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
