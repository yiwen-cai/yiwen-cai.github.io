+++
title = 'MIT 6.S184: Flow Matching and Diffusion Models — 第 7 章：离散扩散模型：用 CTMC 构建语言模型'
date = 2026-11-03
draft = false
summary = '用连续时间马尔可夫链（CTMC）的跳转速率把简单离散分布逐步演变为文本分布：从速率矩阵、小步 Euler 转移与因子化 CTMC 出发，经条件混合路径、Kolmogorov Forward Equation 与后验加权边缘化，将 Discrete Flow Matching 与掩码扩散语言模型（MDLM）的训练化为逐位置干净 token 的交叉熵分类问题。'
tags = ['Flow Matching', 'Diffusion Models', 'Discrete Diffusion', 'CTMC', 'MDLM', '生成模型']
showReadingTime = true
showTableOfContents = true
+++

{{< katex >}}

> **[MIT 6.S184: Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/)** 课程笔记。对应讲义第 7 章 *Discrete Diffusion: Building Language Models with CTMCs*，讲义印刷页码 54–65。本文沿用课程从噪声到数据的时间方向（$t=0$ 是噪声，$t=1$ 是数据）。公式、定理与算法编号已对照讲义核对；文中的三状态数值例子、Euler 非负性条件、终点正则条件说明等属于辅助理解讲义公式的补充推导，未额外开展训练或生成复现实验。

> 🎯 **本章一句话总结：** 用连续时间马尔可夫链（CTMC）的跳转速率，把简单的离散分布逐渐变成文本分布；通过学习“给定含噪序列时原始 token 是什么”，将生成模型训练化为逐位置的分类问题。

## 一、学习目标

- 区分连续时间与离散状态，理解为什么本章用 CTMC 描述 token 的变化。
- 从速率矩阵的定义推导小步转移概率，解释指示函数、负对角项和步长约束。
- 理解因子化 CTMC 的稀疏结构，区分“一次跳转只改一个位置”和“各位置统计独立”。
- 构造条件概率路径，并用 Kolmogorov Forward Equation（KFE）解释条件速率到边缘速率的推导。
- 说明为什么只需预测逐位置的终点后验，就能获得采样所需的速率矩阵。
- 写出 Discrete Flow Matching 的训练与采样步骤，解释掩码扩散语言模型（MDLM）的生成过程。

## 二、章节主线

| 核心问题 | 讲义中的方法 | 得到什么 |
| --- | --- | --- |
| token 无法沿欧氏空间中的方向连续移动，怎样定义生成过程？ | §7.1：CTMC 与速率矩阵 | 用随机跳转描述序列变化 |
| 序列状态数随长度指数增长，怎样输出速率？ | §7.1：只允许单位置跳转 | 每次网络输出缩减为逐位置的词表大小数组 |
| 怎样把初始噪声分布接到数据分布？ | §7.2.1：因子化混合概率路径 | 可直接采样训练用的含噪序列 |
| 怎样让跳转过程遵循指定路径？ | §7.2.2：条件速率、KFE、边缘化技巧 | 得到遵循边缘概率路径的速率矩阵 |
| 边缘速率含未知后验，怎样学习？ | §7.2.3：终点 token 分类与交叉熵 | 无须在训练时模拟整条 CTMC 轨迹 |

阅读顺序：先弄清“速率如何变成概率”，再看“什么速率能产生目标分布”，最后看“神经网络实际预测什么”。

## 三、核心内容

### 7.1 连续时间马尔可夫链模型（CTMC Models）

对应讲义第 54–58 页。

#### 7.1.1 时间连续，状态离散

设词表为 $\mathcal V=\{v_1,\ldots,v_V\}$，其中 $V$ 是词表大小；固定序列长度为 $d$。状态空间为

$$
\mathcal S=\mathcal V^d,\qquad |\mathcal S|=V^d.
$$

这里的一个状态 $x=(x_1,\ldots,x_d)$ 是**整条序列**，$x_j$ 才是第 $j$ 个位置的 token。随机变量 $X_t$ 表示时刻 $t\in[0,1]$ 的序列。

CTMC 的时间参数连续，但取值只能是离散状态。轨迹在两次跳转之间保持不变，到跳转时刻才切换状态。例如在三状态空间中，可以先停在 $a$，再跳到 $c$，之后跳到 $b$；不存在从 $a$ 到 $c$ 途中的“半个 token”。

马尔可夫性质是：给定当前状态，过去历史不再改变未来的条件分布。对于 $t_1<\cdots<t_k<t$ 和 $h>0$，

$$
\Pr(X_{t+h}=y\mid X_t=x,X_{t_1},\ldots,X_{t_k})
=\Pr(X_{t+h}=y\mid X_t=x).
$$

本章“离散扩散”沿用机器学习中的名称，具体数学对象是 CTMC（在生成模型中的系统化表述可参考 Campbell 等人的 [A Continuous Time Framework for Discrete Denoising Models](https://arxiv.org/abs/2205.14987) 及离散扩散早期工作 [D3PM](https://arxiv.org/abs/2107.03006)）。它没有采用前面章节那种由布朗运动驱动、在欧氏空间中具有连续轨迹的 SDE。

#### 7.1.2 速率矩阵：描述概率变化的速度

转移概率记为 $p_{t+h\mid t}(y\mid x)=\Pr(X_{t+h}=y\mid X_t=x)$。速率矩阵（rate matrix）定义为它对时间间隔 $h$ 在零点的右导数：

$$
Q_t(y\mid x)
=\left.\frac{\mathrm d}{\mathrm dh}p_{t+h\mid t}(y\mid x)\right|_{h=0^+}.
$$

沿用讲义约定，$x$ 是出发状态、$y$ 是到达状态；矩阵中列对应出发状态，行对应到达状态。固定 $x$ 时，合法速率必须满足

$$
Q_t(y\mid x)\ge 0\quad(y\ne x),
\qquad
Q_t(x\mid x)=-\sum_{y\ne x}Q_t(y\mid x).
$$

把总离开速率记为

$$
\lambda_t(x):=\sum_{y\ne x}Q_t(y\mid x)\ge 0.
$$

于是对角项为 $Q_t(x\mid x)=-\lambda_t(x)$，每列之和为零。

- **非对角项**描述向其他状态跳转的瞬时速率，可以大于 1。
- **对角项**描述“仍在当前状态”的概率在零时间间隔处如何减少，不能把它当成负的跳转概率。
- **每列和为零**保证概率总量守恒：离开一个状态的概率质量进入其他状态。

讲义定理 33 说明，在有限状态空间、速率有界且随时间连续的条件下，速率矩阵确定唯一的转移概率族；再指定初始分布，就确定了 CTMC 的分布规律。

#### 7.1.3 阅读疑问：为什么在 $y=x$ 时要加 1？

讲义第 56 页将转移概率在 $h=0$ 处展开：

$$
p_{t+h\mid t}(y\mid x)
=p_{t\mid t}(y\mid x)+hQ_t(y\mid x)+R_t(h)
=\mathbf 1_{y=x}+hQ_t(y\mid x)+R_t(h).
$$

余项 $R_t(h)$ 实际也依赖固定的 $x,y$，讲义省略了这些下标；一阶可微展开保证 $R_t(h)/h\to0$，即 $R_t(h)=o(h)$。若另有足够的光滑性，才可进一步讨论 $O(h^2)$ 的界。

**1 来自零时间间隔时的初始概率。** 已知 $X_t=x$，仍在同一时刻问 $X_t=y$，答案只能是

$$
p_{t\mid t}(y\mid x)=\mathbf 1_{y=x}
=\begin{cases}
1,&y=x,\\
0,&y\ne x.
\end{cases}
$$

因此原式的含义是“之后的概率 = 初始概率 + 一阶变化量 + 余项”。$hQ_t$ 是变化量，不能单独替代概率。

忽略余项后，分两种情况看最清楚：

$$
p_{t+h\mid t}(y\mid x)\approx
\begin{cases}
1-h\lambda_t(x),&y=x,\\
hQ_t(y\mid x),&y\ne x.
\end{cases}
$$

**数值例子（依据定义自拟）**：若从 $a$ 到 $b$ 的速率为 2、到 $c$ 的速率为 1，则 $Q_t(a\mid a)=-3$。取 $h=0.01$，从 $a$ 出发的小步近似分布为

$$
\widetilde p(a\mid a)=0.97,\qquad
\widetilde p(b\mid a)=0.02,\qquad
\widetilde p(c\mid a)=0.01.
$$

三者之和为 1。不加初始的 1，就会把留在 $a$ 的概率写成 $-0.03$；这只是概率的一阶变化量。

#### 7.1.4 Euler 近似与步长限制

定义小步 Euler 转移分布

$$
\widetilde p_{t+h\mid t}(y\mid x)
=\mathbf 1_{y=x}+hQ_t(y\mid x),
\qquad
X_{t+h}\sim\widetilde p_{t+h\mid t}(\cdot\mid X_t).
$$

它的总和自动为 1，但非负性还需要

$$
h\lambda_t(x)\le 1.
$$

这只是该步近似分布合法的条件；要获得准确模拟，仍需让步长足够小。不能用“概率和为 1”代替非负性检查，也不能任意增大 $h$ 后仍把一阶式当成精确转移概率。

讲义例 34 的双状态 CTMC 提供了一个精确对照：若 $a,b$ 之间的跳转速率都为常数 $\lambda>0$，则

$$
\Pr(X_{t+h}=a\mid X_t=a)=\frac{1+e^{-2\lambda h}}2,
\qquad
\Pr(X_{t+h}=b\mid X_t=a)=\frac{1-e^{-2\lambda h}}2.
$$

小 $h$ 时两者分别为 $1-\lambda h+O(h^2)$ 与 $\lambda h+O(h^2)$；长时间后则都趋向 $1/2$。有限时间内可能跳走又跳回来，Euler 一阶式不完整描述这些多次跳转。

#### 7.1.5 因子化 CTMC：限制跳转结构，降低输出规模

若网络要对当前序列 $x$ 输出所有 $Q_t^\theta(y\mid x)$，输出数量为 $V^d$，其中 $\theta$ 表示网络参数。这随序列长度指数增长。

讲义采用因子化 CTMC（factorized CTMC），要求

$$
Q_t^\theta(y\mid x)=0
\quad\text{当 }x,y\text{ 在超过一个位置上不同时}.
$$

也就是只给单位置替换设置直接跳转速率。讲义将“在位置 $j$ 换成 token $v$”的局部速率写成 $Q_t(v,j\mid x)$，网络输出形状变为 $d\times V$。

对 $v\ne x_j$，该数值就是从整条序列 $x$ 到对应邻居的速率。局部对角项定义为

$$
Q_t(x_j,j\mid x)=-\sum_{v\ne x_j}Q_t(v,j\mid x).
$$

**整条序列的对角项是所有位置局部对角项之和**：

$$
Q_t(x\mid x)=\sum_{j=1}^d Q_t(x_j,j\mid x).
$$

这避免将每个位置的“保持不变”项误认为同一个完整对角项。实际不同的非自身邻居共有 $d(V-1)$ 个，$d\times V$ 表示还包含每个位置的局部对角项。

> 💡 **因子化不等于 token 彼此独立。** 一次直接跳转只改一个位置，但该位置的速率可以依赖当前整条序列 $x$。Transformer 可以利用其他位置的信息，决定当前位置如何改变。

#### 7.1.6 并行逐 token 采样的含义

讲义算法 7 在每个小步开始时，用同一个旧序列 $X_t=x$ 计算所有位置的速率，然后并行采样各位置。令 $q_j(v)=Q_t^\theta(v,j\mid x)$，则

$$
\widetilde p_{j,t}(v\mid x)=
\begin{cases}
hq_j(v),&v\ne x_j,\\
1-h\sum_{u\ne x_j}q_j(u),&v=x_j.
\end{cases}
$$

这里 $u$ 是用于求和的词表元素，每个位置都需满足 $h\sum_{u\ne x_j}q_j(u)\le1$。

并行采样可能在同一数值步改变多个位置；当速率有界时，两个指定位置同时改变的概率是 $O(h^2)$。所以该方法与整条序列的 Euler 近似在一阶上相同，但不是完全相同的离散转移分布。

还要区分：CTMC 的一次跳转只改一个位置，不代表在一个有限时间区间内只能跳一次；连续时间过程中可以先后改变多个位置。

```text
X ← 从 p_init 采样整条初始序列
t ← 0
重复直到到达终点：
    用当前 X 和 t 计算全部位置的速率 q_j(v)
    选择足够小的 h，并检查每个位置的转移概率非负
    用同一个旧 X 构造各位置的分类分布
    并行采样所有新 token，组成 X_new
    X ← X_new
    t ← t + h
返回 X
```

这是对算法 7 的说明性改写；讲义使用固定步长 $h=1/n$。实际选用固定步长时，也必须满足相应的非负性条件。

### 7.2 训练 CTMC 模型（Training CTMC Models）

对应讲义第 59–65 页。设真实数据分布为 $p_{\mathrm{data}}$，训练时能取得样本 $z\sim p_{\mathrm{data}}$，但通常不知道整个概率质量函数。目标是学习速率，使从初始分布 $p_{\mathrm{init}}$ 出发的过程在终点产生数据分布（离散流匹配框架详见 Gat 等人的 [Discrete Flow Matching](https://arxiv.org/abs/2407.15595) 与 Campbell 等人的 [Generative Flows on Discrete State-Spaces](https://arxiv.org/abs/2402.04997)）。

理想目标与实际结果要分开：若学到精确边缘速率并精确模拟，终点分布是 $p_{\mathrm{data}}$；有限网络、有限数据和有限步长通常会带来近似误差。

#### 7.2.1 条件与边缘概率路径

**先固定一条干净序列 $z$。** 条件概率路径 $p_t(x\mid z)$ 描述与这个终点配对的时刻 $t$ 状态分布，满足

$$
p_0(\cdot\mid z)=p_{\mathrm{init}},
\qquad p_1(\cdot\mid z)=\delta_z.
$$

这里 $\delta_z(x)=\mathbf 1_{x=z}$ 是离散空间上的点质量：所有概率都集中在 $z$。再对训练数据求平均，得到边缘概率路径

$$
p_t(x)=\sum_{z\in\mathcal S}p_t(x\mid z)p_{\mathrm{data}}(z),
\qquad p_0=p_{\mathrm{init}},\quad p_1=p_{\mathrm{data}}.
$$

**概率路径规定各个时刻的分布，尚未规定一条样本轨迹如何跳转。** 后面需要另外构造与这条路径相容的速率矩阵。

**因子化混合路径（讲义例 35）**

假设初始分布按位置分解为 $p_{\mathrm{init}}(x)=\prod_{j=1}^d p_{\mathrm{init}}^{(j)}(x_j)$。取调度函数 $\kappa_t$，满足

$$
0\le\kappa_t\le1,\qquad
\kappa_0=0,\quad\kappa_1=1,\quad
\dot\kappa_t:=\frac{\mathrm d\kappa_t}{\mathrm dt}\ge0.
$$

构造

$$
p_t(x\mid z)=\prod_{j=1}^d
\left[(1-\kappa_t)p_{\mathrm{init}}^{(j)}(x_j)
+\kappa_t\delta_{z_j}(x_j)\right].
$$

单个位置从初始分布与真实 token 的点质量之间做概率混合。它混合的是两种选择的概率，不是在 token 数值之间做连续插值。

直接采样方法是：对每个位置独立采样 $m_j\sim\operatorname{Bernoulli}(\kappa_t)$ 和 $\xi_j\sim p_{\mathrm{init}}^{(j)}$，然后

$$
x_j=\begin{cases}
z_j,&m_j=1,\\
\xi_j,&m_j=0.
\end{cases}
$$

讲义的 $x_j=m_jz_j+(1-m_j)\xi_j$ 应理解为这个选择操作，不能把 token 编号相加解释成新的语义 token。

当 $t=0$，所有位置都采用初始噪声；当 $t=1$，全部采用干净数据。若初始噪声也可能抽中 $z_j$，则“选择了噪声分支”不一定真的改变 token：

$$
\Pr(x_j\ne z_j\mid z)
=(1-\kappa_t)\left[1-p_{\mathrm{init}}^{(j)}(z_j)\right].
$$

上式是由混合路径直接得到的补充解释。只有噪声不可能等于目标 token 时，实际改变概率才恰好为 $1-\kappa_t$。

**一个需要区分的条件**：给定整条 $z$ 时，各位置的加噪选择独立；对相关的真实序列 $z$ 求平均后，边缘分布 $p_t(x)$ 一般不再按位置分解。一般也不能将上面的乘积写成整条序列层面的 $(1-\kappa_t)p_{\mathrm{init}}(x)+\kappa_t\delta_z(x)$。

#### 7.2.2 条件与边缘速率矩阵

**条件速率的任务**

对每个固定终点 $z$，构造条件速率矩阵 $Q_t^z$，使从 $p_{\mathrm{init}}$ 出发的 CTMC 满足 $X_t\sim p_t(\cdot\mid z)$。它在推导中可以利用已知的干净序列 $z$。

实际生成时没有预先给定的真实 $z$，所以还需要边缘速率 $Q_t$，让过程遵循 $p_t$。这一步靠后验加权的边缘化技巧完成。

**Kolmogorov Forward Equation：概率流入减去流出**

KFE（讲义命题 2）连接了速率矩阵与分布随时间的变化：

$$
\frac{\mathrm d}{\mathrm dt}p_t(x)
=\sum_{y\in\mathcal S}Q_t(x\mid y)p_t(y).
$$

这里 $y$ 是出发状态、$x$ 是目标状态。展开对角项，得到更直观的形式：

$$
\frac{\mathrm d}{\mathrm dt}p_t(x)
=\underbrace{\sum_{y\ne x}Q_t(x\mid y)p_t(y)}_{\text{从其他状态流入}}
-\underbrace{p_t(x)\sum_{y\ne x}Q_t(y\mid x)}_{\text{从当前状态流出}}.
$$

推导只需全概率公式和速率定义：

$$
p_{t+h}(x)=\sum_y p_{t+h\mid t}(x\mid y)p_t(y).
$$

对 $h$ 在零点求导，就得到 KFE。在讲义所用的正则条件下，再由具有同一初值的线性 ODE 解的唯一性，可知满足 KFE 的概率路径就是该 CTMC 的边缘分布。

**样本状态是离散的，概率向量的变化可以由 ODE 描述**，两者不矛盾。这里 ODE 的未知量是 $p_t$，不是 token 轨迹 $X_t$。

**离散边缘化技巧（讲义定理 36）**

先定义干净终点的后验。令随机变量 $Z\sim p_{\mathrm{data}}$，并按 $p_t(x\mid Z)$ 构造当前状态，则在 $p_t(x)>0$ 的状态上

$$
p_{1\mid t}(z\mid x)
:=\Pr(Z=z\mid X_t=x)
=\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}.
$$

它回答：“观察到当前含噪序列 $x$，哪个干净终点 $z$ 更有可能产生它？”边缘速率为

$$
Q_t(y\mid x)
=\sum_z Q_t^z(y\mid x)p_{1\mid t}(z\mid x).
$$

因此，要对所有可能终点的条件速率取平均，权重是**看到当前 $x$ 后的后验**，不能直接使用先验数据概率 $p_{\mathrm{data}}(z)$。

速率非负性和列和为零在这种加权平均下仍成立。将条件 KFE 代入边缘路径导数，可以验证它也满足 KFE：

$$
\begin{aligned}
\frac{\mathrm d}{\mathrm dt}p_t(x)
&=\sum_z\sum_y Q_t^z(x\mid y)p_t(y\mid z)p_{\mathrm{data}}(z)\\
&=\sum_y p_t(y)\sum_z Q_t^z(x\mid y)p_{1\mid t}(z\mid y)\\
&=\sum_y Q_t(x\mid y)p_t(y).
\end{aligned}
$$

这里的权重条件是求和中的出发状态 $y$。在 $p_t(y)=0$ 处，上式中的后验分式不直接定义；推导在正概率支持上使用，零质量项对该时刻的概率流贡献为零。

符号 $p_{1\mid t}$ 在这里按“数据终点与含噪状态的联合分布”定义。边缘化定理保证单时刻边缘分布相符，不自动保证新构造的边缘 CTMC 保留原条件过程混合后的全部多时刻联合分布；不能仅因记号相似，就把它无条件认作该 CTMC 的完整终点转移核。

**混合路径的条件速率（讲义例 37）**

记

$$
a_t:=\frac{\dot\kappa_t}{1-\kappa_t},\qquad \kappa_t<1.
$$

逐位置的条件速率可以取为

$$
Q_t^z(v,j\mid x)
=a_t\left[\delta_{z_j}(v)-\delta_{x_j}(v)\right].
$$

本式沿用讲义的逐位置表示；右侧实际只需 $x_j,z_j$。其含义分三种情况：

- 若 $x_j=z_j$，该位置所有速率均为零，已经到达目标。
- 若 $x_j\ne z_j$，唯一非零的向外跳转指向 $z_j$，速率为 $a_t$。
- 此时局部对角项为 $-a_t$，其余非对角项为零。

用单位置 KFE 可以直接验证。记 $r_t(v)=(1-\kappa_t)p_{\mathrm{init}}^{(j)}(v)+\kappa_t\delta_{z_j}(v)$，则

$$
\begin{aligned}
\dot r_t(v)
&=\dot\kappa_t\left[\delta_{z_j}(v)-p_{\mathrm{init}}^{(j)}(v)\right]\\
&=a_t\left[\delta_{z_j}(v)-r_t(v)\right]\\
&=\sum_{u\in\mathcal V}a_t\left[\delta_{z_j}(v)-\delta_u(v)\right]r_t(u).
\end{aligned}
$$

最后一行正是该局部速率作用下的 KFE。给定 $z$ 时各位置因子化，因此可组合成整条序列的条件过程。

**终点与正则条件的补充说明**

若取线性调度 $\kappa_t=t$，就有 $a_t=1/(1-t)$。它在 $t\to1$ 时发散，所以不能直接把第 7.1 节“整个闭区间上有界”的存在唯一性条件原样套到这个端点。

对该构造，应在每个 $[0,T]$、$T<1$ 上讨论速率与动力学，再通过路径的 $t\uparrow1$ 极限理解终点。数值采样时在当前 $t<1$ 处计算速率，并检查步长条件；到达 1 后停止，不计算 $Q_1$ 中的除零表达式。讲义算法 7 的最后一次速率求值也在 $t=1-h$。

#### 7.2.3 学习边缘速率：从生成问题到 token 分类

**把未知量收缩到逐位置终点后验**

将条件速率代入边缘化公式（讲义定理 38），得到

$$
\begin{aligned}
Q_t(v,j\mid x)
&=a_t\sum_z\left[\delta_{z_j}(v)-\delta_{x_j}(v)\right]p_{1\mid t}(z\mid x)\\
&=a_t\left[p_{1\mid t}(Z_j=v\mid x)-\delta_{x_j}(v)\right].
\end{aligned}
$$

其中 $Z_j$ 是干净终点的第 $j$ 个 token。第一项的化简来自

$$
\sum_z\delta_{z_j}(v)p_{1\mid t}(z\mid x)
=\Pr(Z_j=v\mid X_t=x).
$$

我们不必输出所有 $V^d$ 种完整干净序列的后验概率；每个位置输出一个词表分类分布，就足以构造这一类边缘速率。各位置的预测仍以**整条含噪序列**为条件。

**网络输入和输出**

用 Transformer 等序列模型 $f_\theta$ 输入 $(x,t)$，输出每个位置的 logits（未归一化分类分数）$\ell_j(v)$，再对词表维度做 softmax：

$$
\ell=f_\theta(x,t)\in\mathbb R^{d\times V},
\qquad
p_{1\mid t}^\theta(Z_j=v\mid x)
=\frac{e^{\ell_j(v)}}{\sum_{u\in\mathcal V}e^{\ell_j(u)}}.
$$

据此得到模型速率

$$
Q_t^\theta(v,j\mid x)
=a_t\left[p_{1\mid t}^\theta(Z_j=v\mid x)-\delta_{x_j}(v)\right].
$$

当 $v\ne x_j$，速率为非负的 $a_t p_{1\mid t}^\theta(Z_j=v\mid x)$；当 $v=x_j$，局部对角项为 $a_t[p_{1\mid t}^\theta(Z_j=x_j\mid x)-1]\le0$。softmax 的归一化保证局部速率之和为零。

**训练目标：逐位置交叉熵**

讲义第 64 页给出的 Discrete Flow Matching 损失是

$$
\mathcal L_{\mathrm{DFM}}(\theta)
=\mathbb E_{\substack{z\sim p_{\mathrm{data}}\\t\sim\operatorname{Unif}[0,1]\\x\sim p_t(\cdot\mid z)}}
\left[\sum_{j=1}^d-\log p_{1\mid t}^\theta(Z_j=z_j\mid x)\right].
$$

监督标签是原始干净序列的 token $z_j$，不是当前含噪 token $x_j$，也不是下一小步采样得到的 token。

为什么它学习的是所需后验？固定 $x,t,j$ 时，真实标签的条件分布就是 $p_{1\mid t}(Z_j\mid x)$；分类交叉熵在预测分布等于真实条件分布时最小。因此，在容量与优化足够的理想情形下，这个目标学到构造边缘速率所需的逐位置后验。

讲义这里对全部位置求和，没有额外的时间权重。不要在对照本章公式时，未经说明改成“只训练被掩码位置”或其他文献中的加权目标。

**训练算法（依据讲义算法 8 改写）**

```text
每次训练迭代：
    采样一条干净序列 z 和一个时间 t
    对每个位置独立选择：保留 z_j，或从初始噪声分布采样
    拼成含噪序列 x
    logits ← f_theta(x, t)
    loss ← 各位置对真实标签 z_j 的交叉熵之和
    反向传播并更新 theta
```

这里“simulation-free”指训练时直接取得一个随机时刻的含噪样本，**无须运行 CTMC 来生成整条训练轨迹**。训练仍有随机采样，生成时仍需通过速率迭代采样。

**采样怎样使用分类输出？**

令 $c_{t,h}:=ha_t$。把模型速率代入逐位置 Euler 式，可写成

$$
\widetilde p_{j,t}(v\mid x)
=(1-c_{t,h})\delta_{x_j}(v)
+c_{t,h}p_{1\mid t}^\theta(Z_j=v\mid x).
$$

当 $0\le c_{t,h}\le1$ 时，它是“保持当前 token”与“按终点后验分类分布采样”的凸组合，因此肯定是合法概率分布。这是方便的充分条件；更一般的精确非负性条件是 $c_{t,h}[1-p_{1\mid t}^\theta(Z_j=x_j\mid x)]\le1$。

所以分类器提供的是**干净终点预测**，采样器还要结合当前 token、时间调度与步长，才能得到**下一小步的转移概率**。

#### 7.2.4 掩码扩散语言模型（MDLM，讲义例 39）

这是讲义 §7.2.3 末尾的具体例子（掩码扩散语言模型系统研究见 Sahoo 等人的 [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) 及 Lou 等人的 [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834)）。注意在扩散语言模型中常用此类离散 CTMC 建模，但整个自然语言生成领域的主流范式仍是自回归语言模型。

将词表扩充一个特殊 token `[MASK]`，并令初始分布集中在全掩码序列上：

$$
p_{\mathrm{init}}=\delta_m,\qquad m=(\text{[MASK]},\ldots,\text{[MASK]})\in\mathcal V^d.
$$

扩充后，网络的词表维度也包括该特殊 token。假设真实文本不含这个特殊符号，则条件路径在每个位置以概率 $1-\kappa_t$ 放置 `[MASK]`，以概率 $\kappa_t$ 放置真实 token。

训练时，输入随机掩码后的句子及时间，预测各位置的原始 token；采样时，从全掩码开始，反复计算后验与速率，逐步得到文本。下方是机制示意，不表示确定的采样顺序：

```text
t = 0      [MASK] [MASK] [MASK] [MASK]
中间时刻    [MASK] 喜欢   [MASK] [MASK]
更晚时刻    我     喜欢   [MASK] 学习
t → 1      我     喜欢   机器   学习
```

各位置可以在不同时间被揭示，无须按从左到右的顺序填入。每次网络预测可以利用所有当前可见 token。

由这一特定混合路径还可推出：对**精确后验**，如果当前位置已是一个非掩码 token，那么它必定等于对应干净 token，该位置继续改变的速率为零。因此讲义所构造的理想掩码过程可以逐步揭示并保持已揭示的 token。普通近似网络不自动严格满足这个性质；也不应把它推广为所有离散扩散模型都禁止修改已生成 token。

| 比较项 | 连续 Flow Matching | 本章离散 Flow Matching |
| --- | --- | --- |
| 状态 | 欧氏空间向量 | 离散 token 序列 |
| 动力学对象 | 向量场与 ODE | 速率矩阵与 CTMC |
| 概率路径 | 条件路径再边缘化 | 条件路径再边缘化 |
| 本章对应的学习任务 | 向量场回归 | 逐位置干净 token 分类 |
| 生成步骤 | 数值积分 | 按小步转移分布随机采样 |

这张表比较连续 Flow Matching 与本章的具体构造；前面关于 score matching 的其他参数化不包含在这一简化对照中。

## 四、易混淆点

| 容易误读的地方 | 正确理解 |
| --- | --- |
| 连续时间意味着 token 连续变化 | 时间连续，状态以跳转方式变化 |
| 速率矩阵每列应该和为 1 | 速率每列和为 0；转移概率每列和为 1 |
| 对角项为负会产生负概率 | 概率从 1 开始减小；还需选择合法步长 |
| 指示函数是人为添加的修正项 | 它就是零时间间隔的精确转移概率 |
| 忽略余项后得到了精确 CTMC | 得到的是数值近似；有限步长会有误差 |
| 因子化 CTMC 无法建模 token 依赖 | 速率可以依赖整条序列，只限制直接跳转的邻接结构 |
| 每个小步并行更新等于一次 CTMC 跳转 | 并行步可能更改多个位置，是一阶一致的数值近似 |
| 给定干净序列独立加噪，意味着边缘序列独立 | 对相关的干净序列求平均后，一般仍有依赖 |
| 后验分类分布就是下一步转移分布 | 后者还需结合当前 token、调度函数与步长 |
| simulation-free 意味着生成时不用迭代 | 它描述训练数据构造方式；生成仍要模拟 |
| 所有掩码扩散都只能揭示、不能修改 | 单调揭示结论依赖本章的特定路径与精确后验 |

## 五、重点公式速查

以下公式服务于“路径 → 速率 → 学习 → 采样”这一条主线。

**1. 合法速率矩阵：用于检查概率守恒。**

$$
Q_t(y\mid x)\ge0\ (y\ne x),\qquad
Q_t(x\mid x)=-\sum_{y\ne x}Q_t(y\mid x).
$$

**2. 小步转移：用于采样，不能忽略步长条件。**

$$
\widetilde p_{t+h\mid t}(y\mid x)=\mathbf 1_{y=x}+hQ_t(y\mid x),
\qquad h\lambda_t(x)\le1.
$$

**3. KFE：用于判断速率是否产生给定概率路径。**

$$
\partial_t p_t(x)=\sum_yQ_t(x\mid y)p_t(y).
$$

**4. 条件混合路径：用于直接构造训练样本。**

$$
p_t(x\mid z)=\prod_j\left[(1-\kappa_t)p_{\mathrm{init}}^{(j)}(x_j)+\kappa_t\delta_{z_j}(x_j)\right].
$$

条件是初始分布因子化、$\kappa_0=0$、$\kappa_1=1$ 且调度单调不减。

**5. 后验加权边缘速率：用于连接条件过程与生成过程。**

$$
Q_t(y\mid x)=\sum_zQ_t^z(y\mid x)
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)},\qquad p_t(x)>0.
$$

**6. 模型速率参数化：用于把分类输出变成跳转规则。**

$$
Q_t^\theta(v,j\mid x)=\frac{\dot\kappa_t}{1-\kappa_t}
\left[p_{1\mid t}^\theta(Z_j=v\mid x)-\delta_{x_j}(v)\right],\qquad \kappa_t<1.
$$

**7. DFM 损失：用于学习干净 token 后验。**

$$
\mathcal L_{\mathrm{DFM}}(\theta)
=\mathbb E_{z,t,x}\left[-\sum_{j=1}^d\log p_{1\mid t}^\theta(Z_j=z_j\mid x)\right],
\quad z\sim p_{\mathrm{data}},\quad t\sim\operatorname{Unif}[0,1],\quad x\sim p_t(\cdot\mid z).
$$

## 六、自测题

### 1. CTMC 中“连续”的是什么？一条样本轨迹连续吗？

<details>
<summary>展开答案与解析</summary>

连续的是时间参数。状态属于离散集合，轨迹在跳转间隔内保持不变，跳转时改变，因此一般不是连续轨迹。状态离散不妨碍每个状态的概率质量随时间连续、甚至可微地变化。

</details>

### 2. 为什么 $p_{t+h\mid t}(x\mid x)$ 的一阶近似必须从 1 开始？

<details>
<summary>展开答案与解析</summary>

因为已知当前状态为 $x$，零时间间隔后仍是 $x$ 的概率为 1。Taylor 展开的常数项就是这个初值，$hQ_t(x\mid x)$ 只描述概率的变化量。对其他状态，初始概率为 0，所以只剩 $hQ_t(y\mid x)$。

</details>

### 3. 从 $a$ 到 $b,c$ 的速率分别为 2 和 1，取 $h=0.1$ 与 $h=0.4$ 会怎样？

<details>
<summary>展开答案与解析</summary>

总离开速率为 3，对角项为 $-3$。当 $h=0.1$，按 $(a,b,c)$ 顺序得到 Euler 分布 $(0.7,0.2,0.1)$，合法但仍是近似。当 $h=0.4$，得到 $(-0.2,0.8,0.4)$，尽管总和是 1，仍因负概率而非法。该状态要求 $h\le1/3$。

</details>

### 4. 因子化 CTMC 的输出为什么是 $d\times V$？这意味着各位置独立吗？

<details>
<summary>展开答案与解析</summary>

网络对每个位置输出词表上的局部速率，直接跳转只能替换一个位置，因此不必枚举全部 $V^d$ 个序列。各位置速率可以依赖整条输入序列，因而不意味着各位置统计独立。整条序列的对角项等于各局部对角项之和。

</details>

### 5. 一次跳转只改一个位置，为什么算法 7 可以同一步改多个 token？

<details>
<summary>展开答案与解析</summary>

算法的“一个数值步”对应一小段时间，不是一场单独跳转。逐位置并行 Euler 采样允许多个位置同时被更新；在有界速率、小步长下，多位置更新从二阶开始出现，因此与完整 Euler 转移在一阶上相符。精确 CTMC 在同一有限时间区间内也可能发生先后多次跳转。

</details>

### 6. 为什么因子化混合路径满足 $p_0=p_{\mathrm{init}}$ 和 $p_1=p_{\mathrm{data}}$？

<details>
<summary>展开答案与解析</summary>

在 $t=0$，$\kappa_0=0$，各位置都是初始分布，乘积为 $p_{\mathrm{init}}$。在 $t=1$，$\kappa_1=1$，给定 $z$ 时每个位置都等于 $z_j$，故整个条件分布为 $\delta_z$。对 $z\sim p_{\mathrm{data}}$ 求平均，终点边缘分布就是 $p_{\mathrm{data}}$。

</details>

### 7. KFE 为什么既有流入项，又有流出项？

<details>
<summary>展开答案与解析</summary>

其他状态 $y$ 的概率质量通过 $Q_t(x\mid y)$ 流入 $x$；当前状态 $x$ 的质量也可能通过 $Q_t(y\mid x)$ 流出。KFE 中的对角项 $Q_t(x\mid x)p_t(x)$ 恰好记录总流出量的负值。如果只累加非对角流入而漏掉对角项，总概率会错误地增加。

</details>

### 8. 边缘速率为什么用 $p_{1\mid t}(z\mid x)$ 加权，不能直接用 $p_{\mathrm{data}}(z)$？

<details>
<summary>展开答案与解析</summary>

当前已经观察到状态 $x$，不同干净终点产生这个状态的可能性不同，必须按 Bayes 公式更新权重。条件路径对边缘路径的贡献是 $p_t(x\mid z)p_{\mathrm{data}}(z)$，除以 $p_t(x)$ 后正是终点后验。使用这个权重代入 KFE 才能得到指定的边缘路径。

</details>

### 9. 网络预测的是完整序列后验、下一个 token，还是各位置的干净 token？

<details>
<summary>展开答案与解析</summary>

网络输出给定整条含噪序列和时间后，各位置干净 token 的后验分布，形状为 $d\times V$。它不输出完整序列的 $V^d$ 类联合后验，也不只是自回归意义上的下一个位置。交叉熵的标签是 $z_j$，采样时再由后验、当前 token 与调度系数构造转移分布。

</details>

### 10. 若 $\kappa_t=t$，在 $t=0.8$、$h=0.02$，当前位置是 `[MASK]`，网络给“猫、狗、`[MASK]`”的概率为 $(0.7,0.3,0)$，下一步概率是多少？

<details>
<summary>展开答案与解析</summary>

此时 $a_t=1/(1-0.8)=5$，$ha_t=0.1$。跳到“猫”的概率为 $0.1\times0.7=0.07$，跳到“狗”的概率为 $0.03$，保持 `[MASK]` 的概率为 $0.9$。这与网络的终点后验 $(0.7,0.3,0)$ 不同。这里零概率用于理想化的数值演示。

</details>

### 11. DFM 的训练为什么称为 simulation-free？MDLM 怎样从全掩码产生句子？

<details>
<summary>展开答案与解析</summary>

训练时直接从混合路径采样某个随机时间的含噪序列，再用原始 token 做分类监督，不必先模拟整条 CTMC 轨迹。MDLM 生成时则从全掩码出发，在每一步用当前序列预测干净 token，转成速率和小步转移概率，随机更新并重复直到终点。训练无轨迹模拟不代表生成无迭代。

</details>

### 12. 为什么不能把 $t=1$ 直接代入线性调度的速率公式？

<details>
<summary>展开答案与解析</summary>

因为系数 $1/(1-t)$ 在 $t=1$ 无定义。动力学在 $t<1$ 处计算，终点分布通过极限理解；数值算法在终点之前计算最后一步速率，到达 1 后停止。接近终点时还需检查步长与速率的乘积，不能因为时间接近结束而忽略概率非负性。

</details>

## 七、学习检查清单

- [ ] 我能解释连续时间、离散状态与马尔可夫性质分别约束什么。
- [ ] 我能从零时间间隔的转移概率推导指示函数，并解释为什么 $y=x$ 时必须有 1。
- [ ] 我能区分速率、概率变化量和转移概率，检查 Euler 步长是否合法。
- [ ] 我能说明因子化 CTMC 的输出规模、局部对角项和整条序列对角项的关系。
- [ ] 我能解释并行逐 token 更新为什么与连续时间的单位置跳转相容到一阶。
- [ ] 我能构造因子化混合路径，并核验初始与终点边界条件。
- [ ] 我能用“流入减流出”解释 KFE，并复述后验加权边缘化的推导。
- [ ] 我能从条件速率推导逐位置后验参数化的边缘速率。
- [ ] 我能写出交叉熵训练步骤，说明标签、输入、输出和期望分别是什么。
- [ ] 我能区分干净 token 后验与下一小步转移分布。
- [ ] 我能解释 MDLM 的训练、从全掩码开始的采样，以及理想单调揭示的适用条件。
- [ ] 我能说明线性调度在终点附近的速率行为与数值处理原则。

## 八、资料来源与阅读定位

本笔记依据 Peter Holderrieth 与 Ezra Erives 的 **[An Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/)**（MIT 6.S184，2026 年课程讲义，亦见 [arXiv:2506.02070](https://arxiv.org/abs/2506.02070)）第 7 章，印刷页码 **54–65**。内容以讲义原文的定义、定理和算法为准，中文部分为重新组织的学习解释。

| 定位 | 页码 | 对应内容 |
| --- | --- | --- |
| §7.1 | 54–55 | 离散状态空间、马尔可夫性质、速率矩阵，公式 (84)–(87) |
| 定理 33、例 34、公式 (88) | 56–57 | 存在唯一性、双状态精确解、小步 Euler 采样 |
| 因子化 CTMC、算法 7 | 57–58 | 单位置跳转结构与并行数值采样 |
| §7.2.1、例 35、图 19 | 59–60 | 条件与边缘路径、因子化混合路径 |
| §7.2.2、定理 36、命题 2 | 60–62 | 条件速率、KFE 与边缘化技巧 |
| 例 37 | 62–63 | 混合路径条件速率及验证 |
| §7.2.3、定理 38 | 63–64 | 逐位置后验参数化、交叉熵损失 |
| 例 39、算法 8、图 20 | 64–65 | 掩码扩散语言模型、训练算法与生成示意 |

**进一步学习资源：**

- [Discrete Flow Matching](https://arxiv.org/abs/2407.15595) —— Gat 等人提出的离散流匹配框架，系统讨论离散状态空间上的概率路径与生成建模。
- [Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design](https://arxiv.org/abs/2402.04997) —— Campbell 等人关于离散状态空间 CTMC 生成流的代表性工作。
- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) —— Sahoo 等人关于掩码扩散语言模型（MDLM）训练目标化简与采样特性的研究。
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834) —— Lou 等人提出的 SEDD，从 Concrete Score 角度建模离散扩散语言模型。
- [A Continuous Time Framework for Discrete Denoising Models](https://arxiv.org/abs/2205.14987) —— Campbell 等人将离散去噪扩散推广到连续时间马尔可夫链（CTMC）的奠基工作。

**补充推导的范围**：三状态数字例子、Euler 非负性条件、噪声抽中原 token 的概率、整条序列对角项、后验与下一步转移的凸组合表达式、终点正则条件说明以及自测中的数字，均由本章公式直接推得或用于解释这些公式，不是讲义中的额外实验结论。

**排版校读**：例 35 的调度条件按“$\dot\kappa_t=\mathrm d\kappa_t/\mathrm dt\ge0$”明确书写；例 37 的分段表达按其紧凑公式解释，当前 token 不等于目标 token 时，当前位置的对角项应为 $-a_t$，不能归入“其他目标速率为零”而漏掉。

前置阅读：[第 2 章：流模型与扩散模型](/notes/diffusion/flow-and-diffusion-models/)、[第 3 章：流匹配](/notes/diffusion/flow-matching/)、[第 6 章：构建大规模图像与视频生成器](/notes/diffusion/large-scale-generators/)。本章是 MIT 6.S184 课程讲义的最后一章。
