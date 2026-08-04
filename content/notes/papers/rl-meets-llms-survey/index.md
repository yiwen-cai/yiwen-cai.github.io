+++
title = 'RL Meets LLMs：大语言模型全生命周期强化学习综述'
date = 2026-08-04
draft = false
summary = '梳理 RL 如何贯穿 LLM 全生命周期——预训练/Mid-training、RLHF 对齐微调、RLVR 强化推理三条主线；从策略梯度、PPO 到 GRPO 的算法演进，以及 RLVR 是否真正扩展推理能力、熵坍缩与性能上限等争议。'
tags = ['强化学习', 'RLHF', 'RLVR', 'GRPO', 'PPO', 'LLM']
showReadingTime = true
showTableOfContents = true
+++

{{< katex >}}

> 原论文：*Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle*，Keliang Liu 等，[arXiv:2509.16679](https://arxiv.org/abs/2509.16679)（2025-09）。

## 阅读地图

这篇综述把 RL 贯穿 LLM **全生命周期**，主线是三分法：

1. **预训练（Pre-training / Mid-training）**：为后续 RL 做数据与风格适配，或把 next-token prediction 改造成可验证奖励的 RL 任务
2. **对齐微调（Alignment）**：RLHF、偏好优化、奖励模型设计，对齐人类意图与安全
3. **强化推理（RLVR）**：用可自动验证的客观奖励（数学判分、单测等）推高推理上限——综述强调这是近年主战场

相对只谈 RLHF 的旧综述，本文额外覆盖了数据集与基准、开源工具框架，以及 RLVR 能力边界、熵坍缩等争议话题。

## Preliminaries：Policy 与 Value 的地基

### MDP 与两大范式

RL 通常建模为 MDP：状态 $s$、动作 $a$、转移、奖励 $r$。目标是学到策略 $\pi$，最大化期望累积回报。

- **Policy-based**：直接优化 $\pi_\theta(a|s)$（策略梯度族）
- **Value-based**：估计 $V$ 或 $Q$，再由值函数导出策略（Q-learning / SARSA / DQN）

LLM 场景的动作空间约等于词表（乃至整段序列），很难为每一个可能输出维护显式的 $Q$ 值——这正是 value-based 方法很少作为 LLM RLHF 主框架的原因：像表格型或小离散动作环境那样为每个动作维护 Q 值，在词表级动作空间下不现实。因此 **RLHF / RLVR 的主流是 policy gradient（PPO、GRPO 等）**，而非 DQN 主框架；value 思想仍会以 critic 或 baseline 的形式出现。

### Policy Gradient、Baseline、优势

目标是 $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$，写成期望形式：

$$
J(\theta)=\sum_\tau \pi_\theta(\tau)R(\tau)
\quad\Rightarrow\quad
\nabla J=\sum_\tau \nabla\pi_\theta(\tau)R(\tau)
$$

直接对 $\pi_\theta(\tau)$ 求导不好蒙特卡洛估计，这里用 **log-derivative trick** 这一关键恒等式：

$$
\nabla_\theta\pi_\theta(\tau)=\pi_\theta(\tau)\nabla_\theta\log\pi_\theta(\tau)
$$

代入后

$$
\nabla J=\mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla\log\pi_\theta(\tau)R(\tau)\big]
$$

轨迹概率取对数后乘积变求和，就得到综述公式 (1)：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)R_t\Big]
$$

这里 $\log$ **不改变优化目标**，而是把「对概率求导」变成「对已采样动作的 log 概率求导」，从而可以用采样轨迹做蒙特卡洛估计。

引入一个仅依赖状态的 baseline $b(s)$（公式 (2)）：

$$
\nabla_\theta J(\theta)=\mathbb{E}\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\,(R_t-b(s_t))\Big]
$$

$A_t=R_t-b(s_t)$ 即优势。为什么减去 $b(s)$ 不引入偏差？对固定 $s$，$b(s)$ 与动作无关：

$$
\sum_a \pi(a|s)\nabla\log\pi(a|s)b(s)=b(s)\nabla\Big(\sum_a\pi(a|s)\Big)=b(s)\nabla 1=0
$$

即 $\mathbb{E}_{a\sim\pi}[\nabla\log\pi\cdot b(s)]=0$，减去 baseline 不改变梯度期望；若 $b(s)$ 与回报正相关，则能显著降低方差——这是**无偏但可降方差**的直觉来源。

Actor-Critic 结构里，actor 更新策略，critic 估计 $V$ 或 $Q$，以提供低方差的优势估计（或 TD 误差）。

### TRPO → PPO

TRPO 在 KL 信任域约束下最大化优势期望（公式 (3)）。PPO 用 **clipped surrogate** 做工程近似（公式 (4)）：

$$
L^{\mathrm{PPO}}(\theta)=\mathbb{E}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\;\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\big)\Big]
$$

其中 $r_t(\theta)=\pi_\theta/\pi_{\mathrm{old}}$。目标外层取 $\min$ 是一个悲观下界，保证不会因为「涨分」而鼓励过大的更新：

- $\hat{A}_t>0$：想提高这个动作的概率；当 $r>1+\epsilon$ 后 clip 封顶，$\min$ **阻止好动作概率涨过头**
- $\hat{A}_t<0$：想降低这个动作的概率；当 $r<1-\epsilon$ 后 clip 封底，$\min$ **阻止差动作概率砍过头**

### GRPO：去掉 Critic，用组内相对奖励

LLM 推理场景下 PPO 有几个痛点：(1) 额外的 value 网络带来显存和算力开销；(2) 长序列上 value 估计不准；(3) 传统设置常一次只打一条回复，学习效率低。

**GRPO**（DeepSeekMath 提出）的做法是：每道题采样 $G$ 条回复成组，用奖励模型或规则打分；**去掉独立的 critic**，用**组内平均奖励**作动态 baseline：$A_i=R_i-\bar{R}_{\mathrm{group}}$。

综述正文明确写的是 group average，但印刷公式 (5) 排版成了 $\max$，这与正文叙述以及 DeepSeekMath 原式都不一致——应按 mean / std 理解：

$$
\hat{A}_{i,t}=\frac{r_i-\mathrm{mean}(\{R_i\}_{i=1}^{G})}{\mathrm{std}(\{R_i\}_{i=1}^{G})}
$$

分子是该条奖励相对组统计量的偏离，分母是组内标准差（做尺度归一，近似 z-score）。若真的按 max 理解，多数样本的优势会 $\le 0$，这与「组内相对」的设计初衷不符，因此印 max 几乎可以肯定是笔误。全组奖励相同时 std 会趋近于 0，此时应让优势为 0（没有相对信号可言），实现上通常加一个小 $\varepsilon$ 防止除零。

公式 (6) 在 clip 目标上还做了两处调整：对 token 长度 $|o_i|$ 做平均，以及**直接在目标中加**上 $-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$（而经典 PPO-RLHF 更常把 KL 折进 reward 里）。这两种放置方式的差异值得展开：把 KL 折进 reward 会改变有效回报，进而影响优势与 baseline 的关系，容易「污染」优势的尺度；GRPO 把 KL 作为独立正则项加进 loss，优势仍完全由任务奖励（组内相对）计算，更「干净」，也更容易和组内相对的设计保持自洽。

**PPO vs GRPO 对照**

| 维度 | PPO（经典 RLHF） | GRPO |
|------|------------------|------|
| 采样 | 常一次一条 | 每题 $G$ 条成组 |
| 优势 | value/critic（+GAE） | 组内相对奖励，无独立 critic |
| KL | 常进 reward | 常直接进损失 |
| 目标骨架 | $\min(r\hat{A},\mathrm{clip}\cdot\hat{A})$ | 同结构 + 组平均 + 长度归一 |

除以 std 这个设计本身也有取舍：好处是尺度无关、能对难度自适应；风险是 std 很小时信号会被放大到不稳定，稀疏 0/1 可验证奖励下组内方差结构比较特殊，整组全对或全错时更是完全没有相对信号。常见的改法包括 std 过小时把优势直接置零、只减均值不除 std（部分 R1-Zero / Dr.GRPO 一脉的讨论）、过滤掉零方差的组，或者混入过程奖励增加区分度。

### Value Learning 速记

- **Q-learning**（公式 (7)）：用 $\max_{a'}Q(s',a')$，行为策略可以不同于目标策略 → **off-policy**
- **SARSA**（公式 (8)）：用实际执行的 $a_{t+1}$ → **on-policy**
- **DQN**（公式 (9)）：神经网络逼近 $Q$，配合目标网络与 replay buffer

## 预训练与对齐

### 预训练阶段的 RL 与 Mid-training

多数 RL 工作仍落在对齐与后训练，但预训练侧也有一些探索：

- **Reinforcement Pre-Training**：把 next-token 预测改造成「正确预测下一个 token 得可验证奖励」的 RL 推理任务，资源较重，常需要已具备推理能力的底座
- **视觉预训练**：把无标注图像预训练也框成 RL 问题（如 Annotation Bootstrapping）
- **OctoThinker 的 Mid-training**：两阶段 mid-training 提升底座与后续 RL 的兼容性，例如让原本不太适合 RL 的 Llama 在数学推理上追平同量级的 Qwen

**Mid-training**（综述 §3.1）在训练方式上仍是 next-token / next-word 自监督，和预训练没有区别；不同的是目标——把预训练模型改造得更适合后续 RL，数据也从海量杂文转向高质量、任务相关的语料。数据质量、风格与课表调度，对 RL 阶段能否 scaling 起来很关键。

### 经典对齐：RLHF 与偏好优化

**InstructGPT 范式**常分三阶段：

1. **SFT**：在演示数据上做监督微调，得到可遵从指令的初始策略
2. **Reward Modeling**：在偏好对上训练打分模型 $r_\phi(q,o)$
3. **RL（常用 PPO）**：用 $r_\phi$ 打分，加上相对 reference 的 KL 约束，优化策略

相关工作还涵盖信息论视角的迭代对齐、Constitutional AI / RLAIF（用 AI 反馈替代人类标注），以及缓解 **reward hacking** 的方法——策略学会利用奖励函数的漏洞刷高代理奖励，但真实的有用性或正确性并未提升，在 RM 不完美或规则存在漏洞时尤其常见，是对齐与 RLVR 共同的核心风险。

**DPO** 在一定假设下**绕过**了显式的奖励建模与 RL 优化，直接用偏好数据把策略微调到偏好诱导的最优策略；传统 RLHF 才是「先训 RM 再跑 PPO」的两段式。后续还有 KTO、ORPO、$\beta$-DPO、ΨPO 等变体。

### 奖励模型的新方向

- **推理式 / 生成式 RM**：在打分时引入测试时算力、CoT，甚至代码验证（RRM、GenPRM、RM-R1 等）
- **过程奖励（PRM）**：对推理**中间步骤**打分，而不是只给终局一个稀疏结果奖励——信号更密集，但标注和「步骤对错」的定义都更难
- **原则 / 规则奖励**：用自然语言原则、或从偏好数据中抽取规则，再经 verifier 度量满足程度（RewardAnything、AUTORULE 等）

训练环中，奖励模型（或规则）只负责产出标量或过程奖励 $r$，本身并不直接对策略做监督梯度更新——真正驱动策略更新的是优势与 PPO/GRPO 目标：

```text
prompt → π 采样回复(s) → RM/规则打分得 r → 算 Â（PPO: critic；GRPO: 组内相对）→ clip 目标更新 π
```

DPO 路径则可以绕开这个显式 RM + RL 的训练环。

## RLVR：强化推理

### 定义与能力边界之争

**RLVR（Reinforcement Learning with Verifiable Rewards）**用程序检查、数学答案验证、形式化证明等**可自动验证的客观奖励**做 RL 微调，不必依赖人类偏好训练出的奖励模型（也可以和规则奖励并存）。这是 o1 / R1 一路方法的核心范式之一。

综述里一个尚无定论的争议是：RL 究竟是真正扩展了推理能力，还是只是放大了底座模型分布里本就存在的高奖励路径？

- **Yue et al.** 用 **pass@k**（采样 $k$ 次、至少一次正确的概率）做评估，发现小 $k$（如 $k=1$）时 RLVR 更好，但 **$k$ 增大后 base 模型往往反而更好**；结合覆盖度和困惑度分析，他们认为 RLVR 的正确轨迹大多落在 base 分布内，更像是提高了采样效率，能力边界甚至有所收窄，模型的反思行为也常能在 base 模型里找到源头。
- **Liu / ProRL 等**的工作则显示，在足够的训练时间和新任务上，RL 能发现 base 模型完全没有的新解法路径，据此主张 RL 确实可以扩展能力边界。
- **Wu 等**认为 RLVR 主要还是一个高效采样器，偶尔会越界，但训练中多样性坍缩和遗忘现象同样会发生。

综述本身**并列了双方证据，没有给出单方面的定论**。一个可能的折中判断是：区分「pass@1 的实用提升」和「能力边界 / 新颖推理路径」这两层不同的主张——前者证据相当扎实，后者仍存在争议，取决于训练时长、任务新颖度等具体条件。

### 熵、性能上限与训练动态

Cui et al. 给出了熵 $H$ 与下游性能 $R$ 之间的一个经验关系（公式 (10)）：

$$
R=-a\exp(H)+b
$$

$H$ 下降时，$\exp(H)$ 随之下降，$-a\exp(H)$ 上升，$R$ 也随之上升（在拟合成立的区间内）；当 $H\to 0$ 时，$R\to -a+b$，给出了一个可预测的性能上限。这意味着**性能提升常常以消耗熵为代价**，靠不断压低熵来换性能不可能无限持续——熵一旦耗尽，就会限制「只靠堆算力做 RL」的收益，需要熵管理，或者更好的探索 / 目标函数设计，而不只是简单加一个熵正则项。

**熵坍缩（entropy collapse）**指策略分布越来越尖锐、生成多样性下降的现象。它和「遗忘预训练知识」相关但不完全等同——综述在这里强调的重点是探索耗尽与扩展上限的关系。高协方差、高熵的 token（例如逻辑连接词）对整体熵和推理路径的影响尤其大，限制更新对象、设计优于朴素熵损失的目标函数，都是目前的活跃方向。

### 算法进展：DAPO 等

DAPO（基于 GRPO）针对长 CoT 场景做了几处改动：Clip-Higher、Dynamic Sampling、Token-Level Policy Gradient Loss、Overlong Reward Shaping。

标准的对称 clip 会限制低概率动作的更新幅度，容易促成熵坍缩；**Clip-Higher** 的做法是放宽 clip 的上界，以保留更多探索空间。此外还包括对极端奖励的 prompt 做动态采样、按序列长度加权、对过长输出做惩罚或截断，用来缓解样本浪费和输出冗长失控的问题。

另外还有树结构 RL、对抗 / 多智能体训练、序列级重要性采样（GSPO）等分支，这里从略。

### 多模态、自适应长度与 Agent

- **多模态**：视觉、视频、具身等场景常有显式 ground truth，比较适合规则奖励；难点在于可验证性弱于数学、容易出现文本偏见而忽略图像信息，以及过程稠密奖励的设计更复杂
- **自适应推理**：控制 CoT 长度以及是否需要思考，避免简单题过度思考、难题预算不足
- **Agent**：多轮交互下延迟或稀疏的终局奖励（如 Echo Trap 现象）、工具调用带来的动作空间、长程信用分配、记忆管理；过程奖励和结果奖励哪个更合适也更难定义（如 SPA-RL、LARM）

把 RLVR 从数学 / 代码推广到多模态推理或多轮 Agent 时，奖励设计会遇到几类新困难：可验证性上，视觉、空间、具身类的结果很难像数学答案那样规则化判分；稀疏和延迟上，多轮工具调用、长程任务的终局奖励天然稀疏且滞后；过程与结果的取舍上，中间的工具调用步骤对错本身就难标注，因此需要更细粒度的密集奖励、过程奖励或进度分解设计（例如宏动作层面的优势估计、检索 token 的 mask、专门的记忆管理机制）。

## 数据、工具与开放问题

### 数据与基准类型

- **对齐 / 对话偏好**：人类标注偏好、AI 反馈偏好
- **可验证语料**：数学（GSM8K、MATH、AIME 等）、代码（LiveCodeBench、SWE-bench 等）、程序 / 形式化验证式奖励
- **合成数据**：任务定义驱动的合成 RL 数据、多步工具合成等
- **评测**：对齐基准 + 推理基准（含 Agent / 多轮工具）

### 工具与框架

综述整理了主流开源 RLHF / RLVR 训练框架与库（trlX、各厂商开源训练栈等），实践中按底座模型与分布式 rollout 的需求选型即可；具体细节随生态更新较快，以综述原文和各框架官方文档为准。

### 开放问题

综述列出的开放问题包括：能力边界是否真正扩展、熵管理与探索、奖励黑客、过程奖励质量、Agent 长程信用分配、RLIF（内部反馈）后期退化、多模态可验证奖励设计等，这些都可以进一步拆解成具体、可检验的研究假说。

## 公式速查

| 公式 | 内容 |
|------|------|
| (1) | $\nabla\log\pi\cdot R_t$（REINFORCE） |
| (2) | baseline / advantage：$R_t-b(s_t)$ |
| (4) | PPO $\min(r\hat{A},\mathrm{clip}\cdot\hat{A})$ |
| (5) | 印刷为 $r_i-\max$ over std；正文与原论文应理解为 mean |
| (6) | GRPO clip + 组相对优势 + KL 项 + 长度平均 |
| (10) | $R=-a\exp(H)+b$（熵–性能） |
