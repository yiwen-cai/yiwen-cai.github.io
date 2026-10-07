+++
title = 'MIT 6.S184: Flow Matching and Diffusion Models — 第 5 章：Guidance 与无分类器引导'
date = 2026-10-07
draft = false
summary = '从 Vanilla Guidance 的训练目标出发，用贝叶斯公式把条件向量场拆成无条件向量场加分类器梯度，推出 Classifier-Free Guidance 的线性组合形式，说明 label dropping 如何让单一网络同时学条件与无条件向量场，以及 guidance scale 只存在于采样阶段。'
tags = ['Flow Matching', 'Diffusion Models', 'Guidance', 'Classifier-Free Guidance', '生成模型']
showReadingTime = true
showTableOfContents = true
+++

{{< katex >}}

> **[MIT 6.S184: Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/)** 课程笔记。对应讲义第 5 章 *Guidance: How To Condition on a Prompt*，讲义印刷页码 34–40。本文沿用课程从噪声到数据的时间方向（$t=0$ 是噪声，$t=1$ 是数据）。公式与编号已对照讲义核对；代码是公式的教学实现，未作为训练或生成质量的复现实验。

## 一、本章要解决什么问题

前几章的模型都是**无条件**的：给一个噪声，生成"某个"样本。实际使用时，我们想要的是**按提示词 $y$ 生成**，也就是从 $p_{\mathrm{data}}(z\mid y)$ 采样。[第 3 章：流匹配](/notes/diffusion/flow-matching/)解决了怎么训练，[第 4 章：Score 函数与 Score Matching](/notes/diffusion/score-based-diffusion/)解决了 score 与速度场怎么互相转换。第 5 章要回答的是：**怎样让生成结果真的服从提示词？**

讲义用 *guided* 专指"以 $y$ 为条件"，避免与前面"条件概率路径 / 条件向量场"（以数据点 $z$ 为条件）混淆（讲义注 25）。本文沿用这个区分。

| 方法 | 做法 | 主要问题 |
| --- | --- | --- |
| Vanilla Guidance | 训练和采样时都把 $y$ 直接喂给网络 | 实践中生成结果对提示词的服从度不够，且无法在采样时调节 |
| Classifier Guidance | 无条件向量场外加分类器梯度，并放大它 | 需要额外训练一个作用在带噪数据上的分类器 |
| Classifier-Free Guidance（CFG） | 用同一个网络学条件与无条件向量场，采样时线性组合 | 采样需要两次网络前向 |

```text
无条件模型无法响应提示词
        │
        ▼
Vanilla Guidance：把 y 作为网络输入        ──►  服从度不够，且不可调
        │
        ▼
Classifier Guidance：放大分类器梯度         ──►  要另训带噪分类器
        │
        ▼
CFG：label dropping 训一个网络，采样时组合  ──►  现代主流做法
```

## 二、符号约定

| 符号 | 含义 |
| --- | --- |
| $z\sim p_{\mathrm{data}}$ | 干净数据；有提示词时 $(z,y)\sim p_{\mathrm{data}}(z,y)$ |
| $y\in\mathcal Y$ | 提示词或类别标签，$\mathcal Y$ 可以是所有文本的集合，也可以是离散标签集合 |
| $\varnothing$ | 表示"没有条件"的额外标签，$\mathcal Y$ 在此基础上扩充 |
| $p_t(x\mid z)=\mathcal N(\alpha_tz,\beta_t^2I_d)$ | 高斯条件概率路径，$\alpha_0=\beta_1=0$，$\alpha_1=\beta_0=1$ |
| $u_t^{\mathrm{target}}(x)$ | 无条件边缘向量场，等于 $u_t^{\mathrm{target}}(x\mid\varnothing)$ |
| $u_t^{\mathrm{target}}(x\mid y)$ | 以 $y$ 为条件的边缘向量场 |
| $\tilde u_t(x\mid y)$ | 经过引导修正后的向量场 |
| $w$ | guidance scale，只出现在采样阶段 |
| $\eta$ | label dropping 概率，只出现在训练阶段 |

高斯路径下，向量场与 score 的转换沿用第 4 章命题 1：

$$
u_t^{\mathrm{target}}(x\mid y)=a_t\nabla\log p_t(x\mid y)+b_tx,\qquad
a_t=\frac{\beta_t^2\dot\alpha_t}{\alpha_t}-\dot\beta_t\beta_t,\quad b_t=\frac{\dot\alpha_t}{\alpha_t}.
$$

## 三、Vanilla Guidance：把提示词直接喂给网络

**模型。** 引导模型由神经网络 $u_t^\theta(x\mid y)$ 和固定的扩散系数 $\sigma_t$ 组成：

$$
u^\theta:\mathbb R^d\times\mathcal Y\times[0,1]\to\mathbb R^d,\qquad (x,y,t)\mapsto u_t^\theta(x\mid y).
$$

对任意给定的 $y$，从 $X_0\sim p_{\mathrm{init}}$ 出发模拟 $dX_t=u_t^\theta(X_t\mid y)\,dt+\sigma_t\,dW_t$，目标是 $X_1\sim p_{\mathrm{data}}(\cdot\mid y)$。$\sigma_t=0$ 时称为 guided flow model，讲义后文也只讨论这种情形。

> 记号提醒：这里 $u^\theta$ 若是 ODE 向量场，加入噪声项 $\sigma_t\,dW_t$ 后，漂移还需要补上第 4 章的 score 修正 $\frac{\sigma_t^2}{2}\nabla\log p_t$。$\sigma_t\equiv0$ 时不需要。

**训练。** 固定一个 $y$，把数据分布取成 $p_{\mathrm{data}}(\cdot\mid y)$，问题就退化成无条件问题，可以直接用条件流匹配目标。对所有 $y$ 取期望，得到引导条件流匹配目标（讲义式 (58)）：

$$
\mathcal L_{\mathrm{CFM}}^{\mathrm{guided}}(\theta)=
\mathbb E_{(z,y)\sim p_{\mathrm{data}},\,t\sim\mathrm{Unif}[0,1],\,x\sim p_t(\cdot\mid z)}
\left\|u_t^\theta(x\mid y)-u_t^{\mathrm{target}}(x\mid z)\right\|^2.
$$

与无条件目标（式 (26)）的唯一区别是：采样的是 $(z,y)\sim p_{\mathrm{data}}$，而不是只采 $z$。条件概率路径 $p_t(\cdot\mid z)$ 和条件向量场 $u_t^{\mathrm{target}}(\cdot\mid z)$ 仍然不依赖 $y$。代码层面，这意味着数据加载器每个 batch 同时返回 $z$ 和 $y$。

**为什么不够。** 讲义（§5.2 开头，图 11）指出，理论上这套流程会忠实地采样 $p_{\mathrm{data}}(\cdot\mid y)$，但经验上生成的样本不够贴合提示词。可能的原因有两个：模型欠拟合，没有真正学到边缘向量场；训练数据本身不完美，例如网络上的图文对有很多错误。所以要想得到更贴合提示词的样本，需要在采样时**人为强化**条件变量 $y$。

## 四、Classifier Guidance：放大分类器的梯度

只考虑高斯概率路径。对条件密度用贝叶斯公式：

$$
p_t(x\mid y)=\frac{p_t(x)\,p_t(y\mid x)}{p_t(y)}
\quad\Longrightarrow\quad
\nabla\log p_t(x\mid y)=\nabla\log p_t(x)+\nabla\log p_t(y\mid x).
$$

梯度是对 $x$ 求的，$p_t(y)$ 与 $x$ 无关，所以 $\nabla\log p_t(y)=0$。代入向量场与 score 的转换公式，$b_tx$ 项两边相同，得到

$$
u_t^{\mathrm{target}}(x\mid y)=u_t^{\mathrm{target}}(x)+a_t\nabla\log p_t(y\mid x).
$$

条件向量场 = 无条件向量场 + 一个由"带噪数据 $x$ 属于 $y$ 的似然"决定的修正。既然观察到样本不够贴合提示词，一个自然的想法是放大这一项（讲义式 (62)）：

$$
\tilde u_t(x\mid y)=u_t^{\mathrm{target}}(x)+w\,a_t\nabla\log p_t(y\mid x),\qquad w>1.
$$

$w$ 称为 guidance scale。$\log p_t(y\mid x)$ 可以看成作用在**带噪数据**上的分类器，用监督学习即可得到。直觉上，$\nabla_x\log p_t(y\mid x)$ 指向"让 $x$ 更像类别 $y$"的方向，$w>1$ 就是把这股力放大。

这是一种启发式：$w\neq1$ 时 $\tilde u_t(x\mid y)\neq u_t^{\mathrm{target}}(x\mid y)$，它不再是"真正的"条件向量场。讲义指出 Classifier Guidance 已基本被 CFG 取代，所以只把它当作 CFG 的基础来讲（相关工作见 Dhariwal 与 Nichol 的 [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)）。

它的困难也很具体：要在流 / 扩散模型旁边再训一个网络，一共两个；而且当 $y$ 是高维的文本提示词而不是类别时，$p_t(y\mid x)$ 很难学，梯度 $\nabla\log p_t(y\mid x)$ 也很难得到。

## 五、Classifier-Free Guidance

### 5.1 把分类器消掉

把 $\nabla\log p_t(y\mid x)=\nabla\log p_t(x\mid y)-\nabla\log p_t(x)$ 代回式 (62)：

$$
\begin{aligned}
\tilde u_t(x\mid y)
&=u_t^{\mathrm{target}}(x)+w\,a_t\big(\nabla\log p_t(x\mid y)-\nabla\log p_t(x)\big)\\
&=u_t^{\mathrm{target}}(x)+w\big(u_t^{\mathrm{target}}(x\mid y)-u_t^{\mathrm{target}}(x)\big)\\
&=(1-w)\,u_t^{\mathrm{target}}(x)+w\,u_t^{\mathrm{target}}(x\mid y).
\end{aligned}
$$

第二行用到 $a_t\nabla\log p_t(x\mid y)-a_t\nabla\log p_t(x)=u_t^{\mathrm{target}}(x\mid y)-u_t^{\mathrm{target}}(x)$，两个 $b_tx$ 恰好相消。结果是**无条件向量场与条件向量场的线性组合**，里面不再出现分类器。

几个特例：

| $w$ | $\tilde u_t(x\mid y)$ | 含义 |
| --- | --- | --- |
| $0$ | $u_t^{\mathrm{target}}(x)$ | 完全无条件，忽略提示词 |
| $1$ | $u_t^{\mathrm{target}}(x\mid y)$ | 精确的条件向量场，等价于 Vanilla Guidance |
| $>1$ | 沿"条件减无条件"的方向外推 | 放大提示词的作用 |

讲义注 26 补充了一点：这个线性组合作为**定义**对任意概率路径都成立，$w=1$ 时容易验证它等于条件向量场；高斯路径的推导只是为了说明"放大一个假想分类器的贡献"这个直觉。

### 5.2 一个网络同时学两个向量场：label dropping

线性组合需要 $u_t^{\mathrm{target}}(x)$ 与 $u_t^{\mathrm{target}}(x\mid y)$ 两个向量场。如果分别训练两个网络，代价和 Classifier Guidance 差不多。CFG 的做法是：给标签集合增加一个表示"没有条件"的 $\varnothing$，令 $u_t^{\mathrm{target}}(x)=u_t^{\mathrm{target}}(x\mid\varnothing)$，用**同一个网络**同时学两者。

问题在于，从数据集采样 $(z,y)\sim p_{\mathrm{data}}$ 永远不会得到 $y=\varnothing$。所以要人为引入：以概率 $\eta$ 丢弃原标签，替换成 $\varnothing$。这就是 label dropping，对应 CFG 条件流匹配目标：

$$
\mathcal L_{\mathrm{CFM}}^{\mathrm{CFG}}(\theta)=\mathbb E_{\square}\left\|u_t^\theta(x\mid y)-u_t^{\mathrm{target}}(x\mid z)\right\|^2,
$$

$$
\square=(z,y)\sim p_{\mathrm{data}},\ t\sim\mathrm{Unif}[0,1],\ x\sim p_t(\cdot\mid z),\ \text{以概率 }\eta\text{ 把 }y\text{ 换成 }\varnothing.
$$

讲义的 Algorithm 5 给出高斯路径下的训练流程（讲义里丢弃概率记作 $p$，目标函数里记作 $\eta$，是同一个量）：

```python
# CFG 训练，高斯路径 p_t(x|z) = N(alpha_t z, beta_t^2 I)
for z, y in dataloader:                 # 配对数据 (z, y)
    t = torch.rand(batch)               # t ~ Unif[0, 1]
    eps = torch.randn_like(z)           # eps ~ N(0, I)
    x = alpha(t) * z + beta(t) * eps    # 加噪
    drop = torch.rand(batch) < eta      # 以概率 eta 丢弃标签
    y = torch.where(drop, NULL_LABEL, y)
    target = alpha_dot(t) * z + beta_dot(t) * eps
    loss = ((u_theta(x, y, t) - target) ** 2).mean()
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

> 易错点：回归目标是 $\dot\alpha_tz+\dot\beta_t\epsilon$，用的是**导数**。误写成 $\alpha_tz+\beta_t\epsilon$ 就等于输入 $x$ 本身，网络会学成恒等映射。

讲义没有给出 $\eta$ 的推荐数值，只说它是一个超参数。直觉上 $\eta$ 太大会让网络把大部分容量花在无条件任务上，$\eta$ 为 0 则网络从没见过 $\varnothing$，采样时无法给出无条件向量场。具体取值需要按任务调。

### 5.3 采样：guidance scale 只在这里出现

固定 $y$，每一步算两次网络前向，再线性组合（讲义总结 27，式 (65)）：

$$
\tilde u_t^\theta(x\mid y)=(1-w)\,u_t^\theta(x\mid\varnothing)+w\,u_t^\theta(x\mid y).
$$

```python
# CFG 采样（ODE，Euler 离散）
x = torch.randn(shape)                          # X_0 ~ p_init
for t, h in time_grid:                          # t 从 0 走向 1
    u_uncond = u_theta(x, NULL_LABEL, t)        # 无条件
    u_cond   = u_theta(x, y, t)                 # 条件
    u_guided = (1 - w) * u_uncond + w * u_cond
    x = x + h * u_guided
```

因此 $w$ 是**推理阶段的超参数**，训练目标里没有 $w$，同一个训练好的网络可以随时换 $w$ 重新采样。代价是每一步需要两次网络前向，推理开销约为无引导时的两倍。

对扩散模型（讲义注 28），做法是把原来的 $u_t^\theta(x\mid y)$ 换成 $\tilde u_t^\theta(x\mid y)$，再按第 4 章的方式用 SDE 采样。这里有一个本文补充：因为组合是线性的，$\tilde u_t-b_tx$ 也是对应 score 的同样线性组合，所以引导后的 score 为 $\tilde s_t=(1-w)s_t(x\mid\varnothing)+w\,s_t(x\mid y)$，SDE 漂移里的 score 修正项用 $\tilde s_t$ 即可。

## 六、$w>1$ 时到底在采样什么

$w>1$ 时，$X_1$ 的分布**不再**是 $p_{\mathrm{data}}(\cdot\mid y)$。讲义明确说这是一种启发式，主要靠出色的经验效果来支撑：几乎所有你见到的 AI 生成的图像或视频，都大量依赖 $w\ge4$ 的 CFG。

讲义里有两处直观例子：

- 图 11：在 128×128 的 ImageNet 上，以 "corgi dog" 为提示。Vanilla Guidance 的样本不太像提示词，使用 $w=4$ 的引导后明显更贴合（图取自 Ho 与 Salimans 的 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)）。
- 图 13：在 MNIST 上取 $w=1.0,\,2.0,\,4.0$，对比不同引导强度的效果，课程第 3 个实验会让你自己复现。

讲义自己没有展开多样性的讨论。一般的经验认识是：放大条件信号会以样本多样性为代价换取对提示词的服从度，所以 $w$ 并不是越大越好，需要在两者之间权衡；具体数值也随任务而变。比如讲义在 §6.3 的案例分析中提到，Stable Diffusion 3 采样时使用的 CFG 权重在 2.0–5.0 之间。

还要注意，$w$ 的尺度在不同实现之间不一定可比：有的实现把组合写成 $u_\varnothing+s\,(u_y-u_\varnothing)$，有的写成 $(1+w')\,u_y-w'\,u_\varnothing$，对应 $s=w=1+w'$。比较别人给出的"guidance scale = 7.5"之类数值之前，先对齐定义。

## 七、常见误解

| 误解 | 纠正 |
| --- | --- |
| CFG 需要训练两个模型 | 只训一个网络，靠 label dropping 同时学条件与无条件向量场 |
| CFG 推理只需要一次前向 | 每一步要对 $y$ 和 $\varnothing$ 各前向一次，约两倍开销；只是不需要第二个模型 |
| guidance scale $w$ 是训练超参数 | $w$ 只在采样时出现；训练里出现的是 $\eta$ |
| $w>1$ 时采样的是真实条件分布 | 只有 $w=1$ 精确等于 $u_t^{\mathrm{target}}(x\mid y)$；$w>1$ 是启发式修正 |
| CFG 只适用于文本到图像 | 组合式对任意概率路径都成立，只要有条件信号与 $\varnothing$ 即可；扩散模型见注 28 |
| Classifier Guidance 已无用 | 讲义称它基本被取代；但它不需要重训条件模型，可以外挂在已有无条件模型上，而 CFG 需要用配对数据训练 |

## 八、自测题

### 1. Vanilla Guidance 的训练目标和无条件目标差在哪里？

只差数据采样：前者采 $(z,y)\sim p_{\mathrm{data}}$ 并让网络以 $y$ 为输入，后者只采 $z$。条件概率路径和条件向量场本身不依赖 $y$。

### 2. 推导 Classifier Guidance 的向量场分解。

用贝叶斯公式得 $\nabla\log p_t(x\mid y)=\nabla\log p_t(x)+\nabla\log p_t(y\mid x)$。代入 $u_t^{\mathrm{target}}=a_t\nabla\log p_t+b_tx$，两侧的 $b_tx$ 一致，得 $u_t^{\mathrm{target}}(x\mid y)=u_t^{\mathrm{target}}(x)+a_t\nabla\log p_t(y\mid x)$。再把分类器项乘上 $w$ 即式 (62)。

### 3. 为什么 CFG 里不再出现分类器？

因为 $\nabla\log p_t(y\mid x)$ 可以写成 $\nabla\log p_t(x\mid y)-\nabla\log p_t(x)$，乘上 $a_t$ 就是"条件向量场减无条件向量场"。这两个向量场都能由同一个网络给出，不再需要独立的分类器。

### 4. label dropping 为什么必须是训练时人为加入的？

因为从配对数据集采样永远得不到 $y=\varnothing$。若不人为替换，网络从未见过 $\varnothing$，采样时没法给出无条件向量场。

### 5. 训练好的 CFG 模型，换一个 $w$ 需要重训吗？

不需要。$w$ 只出现在采样公式里，训练目标里没有它。

### 6. $w=1$ 与 $w>1$ 的采样分布有什么区别？

$w=1$ 时 $\tilde u=u_t^{\mathrm{target}}(x\mid y)$，理论上采样 $p_{\mathrm{data}}(\cdot\mid y)$。$w>1$ 时 $X_1$ 的分布不再与之一致，经验上与提示词更对齐，所以 CFG 是以经验效果为依据的启发式。

## 九、公式速查与阅读位置

| 对象 | 公式或结论 | 讲义位置 |
| --- | --- | --- |
| 术语 | guided 专指以 $y$ 为条件 | 注 25 |
| 引导模型采样 | $dX_t=u_t^\theta(X_t\mid y)dt+\sigma_tdW_t$ | §5.1 |
| 引导条件流匹配 | 对 $(z,y)\sim p_{\mathrm{data}}$ 回归 $u_t^{\mathrm{target}}(x\mid z)$ | 式 (58) |
| Classifier Guidance | $\tilde u=u_t(x)+wa_t\nabla\log p_t(y\mid x)$ | 式 (62)，图 12 |
| CFG 的一般性 | 线性组合对任意概率路径成立，$w=1$ 时为条件向量场 | 注 26 |
| CFG 训练 | 以概率 $\eta$ 把 $y$ 换成 $\varnothing$ | Algorithm 5 |
| CFG 向量场 | $\tilde u=(1-w)u_t(x\mid\varnothing)+wu_t(x\mid y)$ | 总结 27，式 (65) |
| 扩散模型的引导 | 把 $u^\theta$ 换成 $\tilde u^\theta$，按第 4 章用 SDE 采样 | 注 28 |

推导以课程讲义为主；数值上我在一维高斯混合的玩具例子里核对过：用 $a_t,b_t$ 转换得到的条件向量场与解析速度场一致，Classifier Guidance 形式与 CFG 线性组合形式一致，$w=1$ 时退化为条件向量场。采样中的 score 组合与代码属于本文补充，没有做图像生成实验。完整讲义可从[课程主页](https://diffusion.csail.mit.edu/)获取。

前置阅读：[第 2 章：流模型与扩散模型](/notes/diffusion/flow-and-diffusion-models/)、[第 3 章：流匹配](/notes/diffusion/flow-matching/)、[第 4 章：Score 函数与 Score Matching](/notes/diffusion/score-based-diffusion/)。下一章讲义转向如何构建大规模图像与视频生成器（DiT、VAE 与潜空间）。
