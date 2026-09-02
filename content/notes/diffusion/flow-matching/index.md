+++
title = 'MIT 6.S184: Flow Matching and Diffusion Models — 第 3 章：流匹配'
date = 2026-09-02
draft = false
summary = '从条件概率路径、边缘化技巧到 Conditional Flow Matching，系统推导如何在不模拟 ODE 的情况下学习生成所需的向量场，并落到 Gaussian CondOT 的训练与采样实现。'
tags = ['Diffusion Models', 'Flow Matching', 'ODE', '生成建模']
showReadingTime = true
showTableOfContents = true
+++

{{< katex >}}

> **[MIT 6.S184: Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/)** 第 3 章课程笔记，依据讲义印刷页 14–24 整理。

> **本章导览：** 本章解决“流模型究竟该回归哪个向量场”这一训练问题：先用条件概率路径规定分布如何从噪声演化到单个数据点，再通过边缘化技巧把可解析的条件向量场合成为真正负责生成的边缘向量场，最后证明不可计算的边缘 Flow Matching 损失与可计算的条件 Flow Matching 损失只差一个与参数无关的常数，从而得到 simulation-free 的训练算法。


## 一、本章一句话总结
Flow Matching 通过回归一个可解析采样的**条件向量场**，间接学到能把初始噪声分布连续输运为数据分布的**边缘向量场**，而且训练时不需要模拟 ODE。

## 二、学习目标
完成本章学习后，你应当能够：
**理解层面**
- 解释条件概率路径 $p_t(\cdot\mid z)$ 与边缘概率路径 $p_t$ 分别描述什么，以及二者的端点条件
- 区分“分布的轨迹”“单个粒子的轨迹”和“产生轨迹的向量场”三个不同对象
- 说明连续性方程如何表达概率质量守恒，以及式中负号的含义
- 理解 simulation-free 指的是训练阶段，而不是生成阶段
**推导层面**
- 从高斯路径 $X_t=\alpha_tz+\beta_t\epsilon$ 推导条件目标向量场
- 用后验加权（边缘化技巧）写出边缘向量场，并解释它为何等于条件期望
- 证明 $L_{\mathrm{FM}}(\theta)=L_{\mathrm{CFM}}(\theta)+C$，其中 $C$ 与 $\theta$ 无关
- 由 Gaussian CondOT 调度 $\alpha_t=t,\ \beta_t=1-t$ 推出监督速度 $z-\epsilon$
**区分层面**
- 区分条件向量场与边缘向量场各自的作用与终点
- 区分概率路径（分布如何变化）与向量场（粒子如何运动）
- 对比本章的确定性 ODE 视角与第 4 章 Score-Based Models 的 SDE 视角
**应用层面**
- 写出 Gaussian CondOT 的完整训练循环（采样 $z,t,\epsilon$ → 构造 $x$ → 回归 $z-\epsilon$）
- 用 Euler、Heun 等 ODE solver 从 $p_{\mathrm{init}}$ 积分到 $t=1$ 生成样本
- 判断给定的 noise scheduler $(\alpha_t,\beta_t)$ 是否满足端点条件与单调性要求

## 三、章节主线
### 核心问题驱动链

| 要解决的问题 | 本章的方法 | 得到的结论 |
| --- | --- | --- |
| ODE 在中间时刻应该产生什么分布？ | 指定条件概率路径 $p_t(x\mid z)$，再对数据点 $z$ 边缘化 | 得到连接 $p_{\mathrm{init}}$ 与 $p_{\mathrm{data}}$ 的边缘概率路径 $p_t(x)$ |
| 什么速度场能实现这条分布路径？ | 先为每个数据点构造条件向量场 $u_t^{\mathrm{target}}(x\mid z)$ | 通过边缘化技巧得到真正负责生成的边缘向量场 $u_t^{\mathrm{target}}(x)$ |
| 边缘向量场包含不可计算的积分，怎么训练？ | 用条件向量场作为逐样本回归目标 | 条件损失与边缘损失只差与参数无关的常数，梯度完全相同 |
| 训练后如何生成？ | 从 $p_{\mathrm{init}}$ 采样并求解学习到的 ODE | 理想情况下终点满足 $X_1\sim p_{\mathrm{data}}$ |

### 概念依赖图
```text
第 2 章：流模型 dX_t = u_t(X_t) dt
    ↓
条件概率路径 p_t(·|z)：p_init → δ_z
    ↓                          ↓
条件向量场 u_target(x|z)    对 z ~ p_data 边缘化
    ↓                          ↓
    └──→ 边缘化技巧（定理 9）←── 边缘概率路径 p_t：p_init → p_data
                ↓
    边缘向量场 u_target(x)：真正负责生成，但含不可计算积分
                ↓
    定理 12：L_FM = L_CFM + C，∇_θ 完全相同
                ↓
    条件 Flow Matching 损失：可计算、simulation-free
                ↓
    Gaussian CondOT：x = tz + (1-t)ε，监督速度 z - ε
                ↓
    训练完成 → 求解 ODE → X_1 ~ p_data
```

## 四、核心概念详解
### 3.0 问题设定：如何训练流模型
[第 2 章：流模型与扩散模型](/notes/diffusion/flow-and-diffusion-models/) 已经定义了由神经网络向量场 $u_t^\theta:\mathbb R^d\to\mathbb R^d$ 参数化的流模型：
$$
X_0\sim p_{\mathrm{init}},
\qquad
dX_t=u_t^\theta(X_t)\,dt.
$$
其中：
- $t\in[0,1]$ 是连续时间；
- $X_t\in\mathbb R^d$ 是时刻 $t$ 的随机状态；
- $p_{\mathrm{init}}$ 是容易采样的初始分布；
- $p_{\mathrm{data}}$ 是希望模型学习的数据分布；
- $u_t^\theta(x)$ 是神经网络在时刻 $t$、位置 $x$ 给出的速度。
目标是选择参数 $\theta$，使 ODE 的终点满足：
$$
X_1\sim p_{\mathrm{data}}.
$$
困难在于：我们只有数据样本，既不知道应该直接回归哪个向量场，也不想在每次训练更新时都完整模拟一条 ODE 轨迹。Flow Matching 的核心贡献，就是构造一个可以像监督学习一样训练的速度回归目标。

### 3.1 条件概率路径与边缘概率路径
#### 3.1.1 条件概率路径：从噪声分布走向一个数据点
固定一个数据点 $z\in\mathbb R^d$。条件概率路径（conditional probability path）是随时间变化的一族分布：
$$
t\longmapsto p_t(\cdot\mid z),
$$
并满足端点条件：
$$
p_0(\cdot\mid z)=p_{\mathrm{init}},
\qquad
p_1(\cdot\mid z)=\delta_z.
$$
$\delta_z$ 是位于 $z$ 的 Dirac delta 分布：从中采样永远得到 $z$。因此，固定 $z$ 后，条件路径描述的是一个完整初始分布如何逐渐收缩并移动到这个数据点。

> **直觉：** 条件概率路径是“概率分布空间中的轨迹”，不是某个粒子在 $\mathbb R^d$ 中的运动轨迹。前者规定每个时刻整群粒子应当呈现的分布，后者才描述单个粒子具体怎么走。

概率路径只指定每个时刻的分布，通常不唯一决定粒子的运动方式。负责指定粒子速度的是后面定义的向量场。
#### 3.1.2 边缘概率路径：从噪声分布走向整个数据分布
对随机数据点 $Z\sim p_{\mathrm{data}}$，先采样 $Z$，再从其条件路径采样：
$$
Z\sim p_{\mathrm{data}},
\qquad
X_t\sim p_t(\cdot\mid Z).
$$
忽略具体采到了哪个 $Z$ 后，$X_t$ 的分布称为边缘概率路径（marginal probability path）：
$$
p_t(x)
=
\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz.
$$
它的端点为：
$$
p_0=p_{\mathrm{init}},
\qquad
p_1=p_{\mathrm{data}}.
$$
验证终点很直接：
$$
p_1(x)
=
\int \delta_z(x)p_{\mathrm{data}}(z)\,dz
=
p_{\mathrm{data}}(x).
$$
这里有一个贯穿全章的“可采样但不可求密度”现象：
- 从 $p_t$ 采样很容易：采样 $z$，再采样 $x\sim p_t(\cdot\mid z)$；
- 直接计算 $p_t(x)$ 往往很难，因为需要对所有可能的数据点 $z$ 积分。
#### 3.1.3 最重要的例子：高斯条件概率路径
令 $\alpha_t,\beta_t$ 是连续可微的 noise scheduler，其中 $\alpha_t$ 单调递增、$\beta_t$ 单调递减，并满足端点条件：
$$
\alpha_0=0,\quad \alpha_1=1,
\qquad
\beta_0=1,\quad \beta_1=0.
$$
定义：
$$
p_t(\cdot\mid z)
=
\mathcal N(\alpha_tz,\beta_t^2I_d).
$$
若 $\epsilon\sim\mathcal N(0,I_d)$，则可以用重参数化方式采样：
$$
X_t=\alpha_tz+\beta_t\epsilon
\sim
p_t(\cdot\mid z).
$$
在 $t=0$ 时 $X_0=\epsilon$，在 $t=1$ 时 $X_1=z$。因此 $\alpha_t$ 控制数据成分，$\beta_t$ 控制噪声成分，端点条件正好给出 $p_0(\cdot\mid z)=\mathcal N(0,I_d)=p_{\mathrm{init}}$ 与 $p_1(\cdot\mid z)=\delta_z$。
教材重点使用高斯路径，是因为它容易采样，而且条件分布、flow map 和目标速度都能解析写出。概率路径的抽象定义本身并不要求初始分布一定是高斯；高斯只是本章最重要、最方便的具体实例。

### 3.2 条件向量场与边缘向量场
#### 3.2.1 从“希望的分布”到“实现它的速度场”
概率路径只表达希望 $X_t$ 服从什么分布。为了让粒子真正沿这条分布路径演化，需要找到一个向量场。
对每个固定数据点 $z$，条件目标向量场 $u_t^{\mathrm{target}}(x\mid z)$ 满足：
$$
X_0\sim p_{\mathrm{init}},
\qquad
\frac{dX_t}{dt}
=
u_t^{\mathrm{target}}(X_t\mid z)
\quad\Longrightarrow\quad
X_t\sim p_t(\cdot\mid z).
$$
这个条件 ODE 的终点是 $X_1=z$，所以它本身只会重新生成一个已知数据点。它的作用不是直接生成新数据，而是作为构造边缘向量场和训练目标的中间对象。
#### 3.2.2 边缘化技巧：把条件速度合成为生成速度
教材定理 9 定义边缘目标向量场：
$$
u_t^{\mathrm{target}}(x)
=
\int
u_t^{\mathrm{target}}(x\mid z)
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}
\,dz.
$$
根据 Bayes 法则，权重
$$
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}
$$
正是“观察到时刻 $t$ 的带噪位置 $x$ 后，它来自数据点 $z$ 的后验权重”。因此也可以写成条件期望：
$$
u_t^{\mathrm{target}}(x)
=
\mathbb E\!\left[
u_t^{\mathrm{target}}(x\mid Z)
\;\middle|\;
X_t=x
\right].
$$

> **直觉：** 对当前带噪点 $x$，每个潜在数据点 $z$ 都给出一个“应该往哪里走”的条件速度；边缘向量场按照“$x$ 由这个 $z$ 产生的可能性”对所有建议方向加权平均。

定理 9 的结论是：这个边缘向量场会产生边缘概率路径：
$$
X_0\sim p_{\mathrm{init}},
\qquad
\frac{dX_t}{dt}=u_t^{\mathrm{target}}(X_t)
\quad\Longrightarrow\quad
X_t\sim p_t.
$$
特别地：
$$
X_1\sim p_1=p_{\mathrm{data}}.
$$
#### 3.2.3 高斯概率路径的条件目标向量场
对高斯路径
$$
X_t=\alpha_tz+\beta_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I_d),
$$
固定初始噪声 $\epsilon$，直接对时间求导：
$$
\frac{dX_t}{dt}
=
\dot\alpha_tz+\dot\beta_t\epsilon,
$$
其中 $\dot\alpha_t=\partial_t\alpha_t$，$\dot\beta_t=\partial_t\beta_t$。
为了把速度写成当前位置 $x$ 和数据点 $z$ 的函数，由
$$
x=\alpha_tz+\beta_t\epsilon
$$
解出：
$$
\epsilon=\frac{x-\alpha_tz}{\beta_t}.
$$
代回速度公式：
$$
\begin{aligned}
u_t^{\mathrm{target}}(x\mid z)
&=
\dot\alpha_tz
+
\dot\beta_t\frac{x-\alpha_tz}{\beta_t}\\
&=
\left(
\dot\alpha_t-\frac{\dot\beta_t}{\beta_t}\alpha_t
\right)z
+
\frac{\dot\beta_t}{\beta_t}x.
\end{aligned}
$$
这就是教材公式 (20)。训练时使用重参数化样本 $x=\alpha_tz+\beta_t\epsilon$，目标速度可以直接使用更简单的形式：
$$
u_t^{\mathrm{target}}(X_t\mid z)
=
\dot\alpha_tz+\dot\beta_t\epsilon.
$$
#### 3.2.4 连续性方程：概率质量守恒
若 ODE 向量场 $u_t$ 使随机变量 $X_t$ 的密度为 $p_t$，则二者满足连续性方程：
$$
\partial_t p_t(x)
=
-\operatorname{div}\!\left(p_tu_t\right)(x).
$$
散度定义为：
$$
\operatorname{div}(v_t)(x)
=
\sum_{i=1}^{d}
\frac{\partial v_t^i(x)}{\partial x_i}.
$$
其中 $v_t^i$ 是向量场的第 $i$ 个坐标分量。$p_t(x)u_t(x)$ 是概率通量：密度乘以速度。散度为正表示一个微小区域存在净流出，所以区域内密度下降；这解释了连续性方程前面的负号。
定理 9 的证明思路是：
1. 每条条件概率路径与其条件向量场满足条件连续性方程；
2. 对数据点 $z$ 按 $p_{\mathrm{data}}(z)$ 积分；
3. 利用积分、时间导数和散度的线性性交换运算顺序；
4. 得到边缘密度 $p_t$ 与边缘向量场 $u_t^{\mathrm{target}}$ 的连续性方程。
因此，边缘化后的向量场确实会搬运出我们指定的边缘概率路径，而不只是一个形式上的加权平均。

### 3.3 学习边缘向量场
#### 3.3.1 理想但不可计算的边缘 Flow Matching 损失
希望神经网络直接拟合边缘目标向量场：
$$
L_{\mathrm{FM}}(\theta)
=
\mathbb E_{\substack{t\sim\mathrm{Unif}[0,1]\\x\sim p_t}}
\left[
\left\|u_t^\theta(x)-u_t^{\mathrm{target}}(x)\right\|^2
\right].
$$
如果能够最小化这个损失，理想情况下便有
$$
u_t^\theta(x)=u_t^{\mathrm{target}}(x),
$$
从而学习到正确的边缘概率路径。但 $u_t^{\mathrm{target}}(x)$ 需要对所有数据点做后验加权积分，通常无法高效计算。
#### 3.3.2 可计算的条件 Flow Matching 损失
改为使用可解析的条件向量场作为监督目标：
$$
L_{\mathrm{CFM}}(\theta)
=
\mathbb E_{\substack{
t\sim\mathrm{Unif}[0,1]\\
z\sim p_{\mathrm{data}}\\
x\sim p_t(\cdot\mid z)
}}
\left[
\left\|
u_t^\theta(x)-u_t^{\mathrm{target}}(x\mid z)
\right\|^2
\right].
$$
一次训练样本只需要：
1. 从数据集中采样 $z$；
2. 采样时间 $t$；
3. 从条件路径采样 $x$；
4. 计算解析的条件目标速度；
5. 对网络输出和目标速度做均方误差回归。
#### 3.3.3 为什么回归条件速度能学到边缘速度
教材定理 12 给出：
$$
L_{\mathrm{FM}}(\theta)
=
L_{\mathrm{CFM}}(\theta)+C,
$$
其中 $C$ 与模型参数 $\theta$ 无关，因此：
$$
\nabla_\theta L_{\mathrm{FM}}(\theta)
=
\nabla_\theta L_{\mathrm{CFM}}(\theta).
$$
证明的关键不是死记完整的均方误差展开，而是认出下面的交叉项恒等式：
$$
\mathbb E_{t,x}
\left[
u_t^\theta(x)^\top u_t^{\mathrm{target}}(x)
\right]
=
\mathbb E_{t,z,x}
\left[
u_t^\theta(x)^\top u_t^{\mathrm{target}}(x\mid z)
\right].
$$
它成立是因为边缘向量场就是给定 $X_t=x$ 后条件向量场的条件期望。展开两个平方损失后：
- 网络输出的平方项相同；
- 网络输出与目标速度的交叉项相同；
- 剩余的目标速度平方项不依赖 $\theta$，只构成常数差。
所以，虽然单个训练样本的条件目标速度可能指向某个特定数据点，许多样本的均方误差最优预测会自动变成条件平均，也就是所需的边缘向量场。
#### 3.3.4 Gaussian CondOT 的训练公式
对一般高斯路径：
$$
X_t=\alpha_tz+\beta_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I_d),
$$
条件 Flow Matching 损失化为：
$$
L_{\mathrm{CFM}}(\theta)
=
\mathbb E_{t,z,\epsilon}
\left[
\left\|
u_t^\theta(\alpha_tz+\beta_t\epsilon)
-
(\dot\alpha_tz+\dot\beta_t\epsilon)
\right\|^2
\right].
$$
最常用的 Gaussian CondOT 路径取：
$$
\alpha_t=t,
\qquad
\beta_t=1-t.
$$
于是：
$$
X_t=tz+(1-t)\epsilon,
$$
并且（因为 $\dot\alpha_t=1$、$\dot\beta_t=-1$）：
$$
u_t^{\mathrm{target}}(X_t\mid z)
=
z-\epsilon.
$$
最终训练损失为：
$$
\boxed{
L_{\mathrm{CFM}}(\theta)
=
\mathbb E_{t,z,\epsilon}
\left[
\left\|
u_t^\theta\!\left(tz+(1-t)\epsilon\right)
-(z-\epsilon)
\right\|^2
\right]
}
$$
对应教材算法 3：
```text
重复执行：
1. 采样数据 z ~ p_data
2. 采样时间 t ~ Unif[0,1]
3. 采样噪声 ε ~ N(0, I_d)
4. 构造带噪样本 x = tz + (1-t)ε
5. 计算目标速度 v = z - ε
6. 最小化 ||u_θ(x, t) - v||² 并更新 θ
```
写成训练循环的形式：
```python
# 训练一个 epoch（Gaussian CondOT）
for batch in dataloader:
    z = batch                           # 真实数据 z ~ p_data
    t = uniform(0, 1, size=batch_size)  # 时间 t ~ Unif[0,1]
    eps = randn_like(z)                 # 噪声 ε ~ N(0, I)

    x = t * z + (1 - t) * eps           # 带噪样本，无需模拟 ODE
    v_target = z - eps                  # 解析目标速度

    v_pred = u_theta(x, t)
    loss = mean((v_pred - v_target) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
#### 3.3.5 训练与生成是两个不同阶段
Flow Matching 训练具有 simulation-free 特性：训练时直接构造任意时刻的 $(t,x)$ 和目标速度，不需要从 $t=0$ 开始逐步求解 ODE。
训练完成后，生成阶段才需要模拟 ODE：
$$
X_0\sim p_{\mathrm{init}},
\qquad
dX_t=u_t^\theta(X_t)\,dt.
$$
使用 Euler、Heun 或其他 ODE solver 从 $t=0$ 积分到 $t=1$，得到生成样本 $X_1$：
```python
# 采样（Euler 法，N 步）
x = randn(shape)      # X_0 ~ p_init = N(0, I)
dt = 1.0 / N
for i in range(N):
    t = i * dt
    x = x + u_theta(x, t) * dt
return x              # X_1 ≈ p_data 的样本
```

| 阶段 | 是否需要真实数据 $z$ | 是否采样条件噪声 | 是否求解 ODE | 输出 |
| --- | --- | --- | --- | --- |
| **训练** | 是 | 是 | 否 | 学到向量场参数 $\theta$ |
| **生成** | 否 | 只采样初始状态 $X_0$ | 是 | 生成样本 $X_1$ |


## 五、与相邻章节的对比
### 5.1 与第 2 章的关系
第 2 章只定义了流模型“长什么样”（ODE 与向量场），没有说明如何训练。本章补上的正是**训练目标的构造**：
- 第 2 章：给定 $u_t$，如何演化并采样；
- 第 3 章：给定数据，如何确定应该回归哪个 $u_t$。
### 5.2 与第 4 章 Score-Based Models 的对比

| 维度 | Flow Matching（第 3 章） | Score-Based Models（第 4 章） |
| --- | --- | --- |
| **生成过程** | 确定性 ODE | 随机 SDE（或对应的 Probability Flow ODE） |
| **学习目标** | 向量场 $u_t^\theta(x)$ | Score function $\nabla_x\log p_t(x)$ |
| **监督信号** | 条件向量场 $u_t^{\mathrm{target}}(x\mid z)$ | 条件 score $\nabla_x\log p_t(x\mid z)$ |
| **训练方式** | 条件流匹配 CFM | 去噪评分匹配 DSM |
| **核心定理** | 边缘化技巧 + 连续性方程（定理 9、12） | Anderson 时间反演定理 |
| **随机性来源** | 仅初始点 $X_0$ | 初始点 + 每一步布朗噪声 |

**共同的证明骨架**：两章都使用同一个论证——“边缘量 = 条件量的后验期望”，因此“回归条件目标”与“回归边缘目标”只差一个与参数无关的常数。理解本章的定理 12，第 4 章的 DSM 推导几乎可以直接复用。
### 5.3 两者之间的桥梁（延伸）
对高斯路径 $p_t(\cdot\mid z)=\mathcal N(\alpha_tz,\beta_t^2I_d)$，边缘向量场与边缘 score 可以互相换算：
$$
u_t^{\mathrm{target}}(x)
=
\frac{\dot\alpha_t}{\alpha_t}x
+
\left(
\frac{\dot\alpha_t\beta_t^2}{\alpha_t}-\dot\beta_t\beta_t
\right)
\nabla_x\log p_t(x).
$$
这说明在同一族高斯路径下，学向量场和学 score 携带的是等价信息；第 4 章会从 SDE 角度重新导出这一关系。

## 六、自测与检查清单
### 自测题
<details>
<summary><b>Q1：条件概率路径和边缘概率路径分别连接什么？</b></summary>
**答案**：
固定数据点 $z$ 后，条件概率路径连接 $p_{\mathrm{init}}$ 与 $\delta_z$；对 $z\sim p_{\mathrm{data}}$ 边缘化后，边缘概率路径连接 $p_{\mathrm{init}}$ 与完整的 $p_{\mathrm{data}}$。
**相关章节**：3.1.1、3.1.2 节
</details>
<details>
<summary><b>Q2：为什么说概率路径是“分布的轨迹”，而不是粒子的轨迹？</b></summary>
**答案**：
概率路径 $t\mapsto p_t$ 只规定每个时刻所有粒子的总体分布；单个粒子的轨迹是 $t\mapsto X_t$。大量粒子沿向量场运动后在每个时刻形成的总体分布，才构成概率路径。同一条概率路径通常可以由不止一个向量场实现。
**相关章节**：3.1.1 节
</details>
<details>
<summary><b>Q3：边缘向量场为什么是条件向量场的后验加权平均？</b></summary>
**答案**：
观察到带噪点 $X_t=x$ 后，我们不知道它对应哪个干净数据点 $z$。每个 $z$ 给出一个条件速度，而 $p_t(x\mid z)p_{\mathrm{data}}(z)/p_t(x)$ 表示该 $z$ 的后验可信度；按此权重平均便得到只依赖 $(x,t)$ 的边缘速度。
**相关章节**：3.2.2 节
</details>
<details>
<summary><b>Q4：从 $X_t=\alpha_tz+\beta_t\epsilon$ 推导条件目标速度。</b></summary>
**答案**：
固定 $z$ 和初始噪声 $\epsilon$，对时间求导：
$$
\frac{dX_t}{dt}=\dot\alpha_tz+\dot\beta_t\epsilon.
$$
若需要写成 $x,z$ 的函数，再代入 $\epsilon=(x-\alpha_tz)/\beta_t$，得到
$$
u_t^{\mathrm{target}}(x\mid z)
=
\left(\dot\alpha_t-\frac{\dot\beta_t}{\beta_t}\alpha_t\right)z
+
\frac{\dot\beta_t}{\beta_t}x.
$$
**相关章节**：3.2.3 节
</details>
<details>
<summary><b>Q5：连续性方程中的负号表示什么？</b></summary>
**答案**：
$\operatorname{div}(p_tu_t)>0$ 表示某个微小区域存在概率质量净流出，因此该位置的概率密度随时间下降，即 $\partial_tp_t<0$，所以二者之间有负号。注意散度作用在概率通量 $p_tu_t$ 上，而不是单独作用在 $u_t$ 上。
**相关章节**：3.2.4 节
</details>
<details>
<summary><b>Q6：为什么可以用 $L_{\mathrm{CFM}}$ 替代不可计算的 $L_{\mathrm{FM}}$？</b></summary>
**答案**：
两者展开后，依赖网络参数的平方项和交叉项相同，差异只来自目标向量场自身的平方项，而该项不依赖 $\theta$。因此两个损失只差常数，关于 $\theta$ 的梯度相同。交叉项相等的原因是：边缘向量场恰好是给定 $X_t=x$ 后条件向量场的条件期望。
**相关章节**：3.3.1–3.3.3 节
</details>
<details>
<summary><b>Q7：写出 Gaussian CondOT 的带噪样本和监督速度。</b></summary>
**答案**：
$$
x=tz+(1-t)\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I_d),
$$
监督速度为：
$$
z-\epsilon.
$$
它来自 $\alpha_t=t,\ \beta_t=1-t$，于是 $\dot\alpha_t=1,\ \dot\beta_t=-1$。
**相关章节**：3.3.4 节
</details>
<details>
<summary><b>Q8：Flow Matching 的 simulation-free 指什么？生成时也不需要 ODE solver 吗？</b></summary>
**答案**：
simulation-free 只指训练时可以直接采样任意 $(t,x)$ 并回归目标速度，不需要展开 ODE 轨迹。生成时仍需从 $p_{\mathrm{init}}$ 采样 $X_0$，再用 ODE solver 积分到 $t=1$。
**相关章节**：3.3.5 节
</details>
### 检查清单
完成本章学习后，确认你能够：
**概念理解**
- [ ] 解释 $p_t(\cdot\mid z)$ 与 $p_t$ 的区别及端点条件
- [ ] 区分分布轨迹、粒子轨迹和向量场
- [ ] 用“后验加权平均”解释边缘化技巧
- [ ] 解释连续性方程如何表达概率质量守恒
**数学推导**
- [ ] 从高斯路径推导 $\dot\alpha_tz+\dot\beta_t\epsilon$
- [ ] 把条件速度改写成 $(x,z)$ 的函数（教材公式 20）
- [ ] 说明 $L_{\mathrm{FM}}$ 为什么不可直接计算
- [ ] 说明 $L_{\mathrm{CFM}}$ 为什么仍能学到边缘向量场
**算法实现**
- [ ] 写出 CondOT 的 $x=tz+(1-t)\epsilon$ 和目标速度 $z-\epsilon$
- [ ] 实现教材算法 3 的训练循环
- [ ] 用 Euler 法实现从 $t=0$ 到 $t=1$ 的采样
**对比分析**
- [ ] 区分 Flow Matching 的训练阶段与 ODE 生成阶段
- [ ] 对比条件向量场与边缘向量场的作用
- [ ] 说明本章与第 4 章 Score-Based Models 在目标和随机性上的差异

## 七、重点公式速查
### 条件概率路径的端点
$$
p_0(\cdot\mid z)=p_{\mathrm{init}},
\qquad
p_1(\cdot\mid z)=\delta_z.
$$
### 边缘概率路径
$$
p_t(x)=\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz,
\qquad
p_0=p_{\mathrm{init}},\quad p_1=p_{\mathrm{data}}.
$$
### 高斯条件路径的采样
$$
X_t=\alpha_tz+\beta_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I_d).
$$
### 边缘化技巧
$$
u_t^{\mathrm{target}}(x)
=
\mathbb E\!\left[
u_t^{\mathrm{target}}(x\mid Z)
\;\middle|\;
X_t=x
\right].
$$
### 高斯路径的条件目标速度
$$
u_t^{\mathrm{target}}(X_t\mid z)
=
\dot\alpha_tz+\dot\beta_t\epsilon.
$$
### 连续性方程
$$
\partial_t p_t(x)
=
-\operatorname{div}(p_tu_t)(x).
$$
### 条件 Flow Matching 损失
$$
L_{\mathrm{CFM}}(\theta)
=
\mathbb E_{t,z,x}
\left[
\left\|u_t^\theta(x)-u_t^{\mathrm{target}}(x\mid z)\right\|^2
\right].
$$
### Gaussian CondOT 训练目标
$$
x=tz+(1-t)\epsilon,
\qquad
u_t^{\mathrm{target}}=z-\epsilon.
$$

## 八、与课程其他章节的连接
**输入依赖**（需要先掌握的章节）：
- 第 1 章：生成建模即采样，明确目标是学习并采样 $p_{\mathrm{data}}$
- 第 2 章：ODE / SDE、流模型与向量场的定义，以及数值模拟方法
**输出支持**（本章为后续章节奠定基础）：
- 第 4 章：Score-Based Diffusion Models 把同一套“条件目标 → 边缘目标”的论证搬到 score 与 SDE 上
- 第 5 章：Guidance 在已有向量场 / score 的基础上加入条件控制
- 第 6 章：DiT、VAE 与潜空间，把本章的训练目标放进大规模架构中
**横向对比**：
- 第 4 章：随机 SDE 视角与本章确定性 ODE 视角互为补充，在 Probability Flow ODE 处汇合

## 九、进一步学习资源
**原始论文**
1. [**Flow Matching for Generative Modeling**](https://arxiv.org/abs/2210.02747)（Lipman et al., ICLR 2023）
- 提出 Flow Matching 与 Conditional Flow Matching 框架
- 引入本章使用的 CondOT（最优传输式）条件路径
2. [**Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow**](https://arxiv.org/abs/2209.03003)（Liu et al., ICLR 2023）
- 与 CondOT 等价的“直线插值”视角
- Reflow 迭代拉直轨迹，支持极少步采样
3. [**Building Normalizing Flows with Stochastic Interpolants**](https://arxiv.org/abs/2209.15571)（Albergo & Vanden-Eijnden, ICLR 2023）
- 随机插值视角，把 ODE 与 SDE 统一在同一族路径下
4. [**Neural Ordinary Differential Equations**](https://arxiv.org/abs/1806.07366)（Chen et al., NeurIPS 2018）
- 连续归一化流的起点，理解 ODE 生成模型的背景
**推荐阅读顺序**
1. 先读本章笔记 + 课程讲义第 3 章（建立整体框架）
2. 精读 Flow Matching 论文（理解条件路径的一般构造）
3. 对照 Rectified Flow（理解直线路径与少步采样）
4. 再进入第 4 章，把 ODE 视角扩展到 SDE 与 score
**代码实现**
- [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)：Meta 官方 Flow Matching 库
- [conditional-flow-matching (torchcfm)](https://github.com/atong01/conditional-flow-matching)：条件流匹配的教学与研究实现
- [MIT 6.S184 课程网站](https://diffusion.csail.mit.edu/)：配套 lab notebook

## 十、常见误解澄清
**误解 1：条件概率路径就是单个样本的轨迹**
- **纠正**：$p_t(\cdot\mid z)$ 是每个时刻的**分布**；$t\mapsto X_t$ 才是单粒子轨迹。
**误解 2：条件路径和边缘路径的终点相同**
- **纠正**：固定 $z$ 的条件路径终止于 $\delta_z$；混合所有 $z$ 的边缘路径才终止于 $p_{\mathrm{data}}$。
**误解 3：条件向量场直接负责生成新样本**
- **纠正**：它只把样本送向指定数据点；真正把噪声分布送往数据分布的是边缘向量场。
**误解 4：边缘化就是简单的等权平均**
- **纠正**：权重是给定当前带噪点 $x$ 后关于数据点 $z$ 的**后验分布**，不是均匀权重。
**误解 5：概率路径等于向量场**
- **纠正**：概率路径规定分布如何变化；向量场规定粒子如何运动。同一条分布路径可能对应不止一个向量场。
**误解 6：网络直接接收数据点 $z$**
- **纠正**：无条件 Flow Matching 的网络输入是 $(x,t)$；$z$ 只在训练时用于构造带噪输入和监督速度。
**误解 7：训练目标是条件速度，所以学到的也只是条件速度**
- **纠正**：均方误差的最优预测是条件期望，配合定理 12，学到的正是边缘速度。
**误解 8：simulation-free 意味着生成时也不用 ODE solver**
- **纠正**：simulation-free 只描述训练阶段；生成阶段仍必须数值积分 ODE。
**误解 9：连续性方程是 $\partial_tp_t=-\operatorname{div}(u_t)$**
- **纠正**：散度作用在概率通量 $p_tu_t$ 上，正确形式是 $\partial_tp_t=-\operatorname{div}(p_tu_t)$。
**误解 10：中文译名混用不影响理解**
- **纠正**：教材术语 “marginal” 可能被译作“边缘”或“边际”，本笔记统一使用**边缘**，阅读其他资料时需留意。

## 十一、本章在实际应用中的体现
**图像与视频生成**
- **Stable Diffusion 3**：基于 rectified flow 的多模态扩散 Transformer
- **FLUX.1**：以 flow matching 目标训练的大规模图像生成模型
- **Movie Gen**（Meta）：视频生成同样采用 flow matching 训练目标
**语音与音频**
- **Voicebox / Audiobox**（Meta）：以 flow matching 训练的语音与音频生成模型
**科学与结构生成**
- 蛋白质骨架、分子构象生成中的 SE(3) flow matching 方法（如 FoldFlow、FrameFlow）
**少步采样**
- CondOT / Rectified Flow 的轨迹近似为直线，少步 Euler 采样即可得到可用样本
- Reflow 与蒸馏方法可进一步把采样压缩到 1–4 步

## 十二、资料来源
- 双语讲义第 3 章 “Flow Matching”，PDF 印刷页 14–24：[Notion 页面](https://app.notion.com/p/3c94bd03a04081b5b319fc4c20f9e177)
- 前置内容：[Notion 页面](https://app.notion.com/p/3c94bd03a040818080def1eacb628126)
- 后续内容：[Notion 页面](https://app.notion.com/p/3cd4bd03a04081a28ebac2ac735fdfc4)
- [MIT 6.S184 课程网站](https://diffusion.csail.mit.edu/)

**本笔记完成时间**：2026-08-29（2026-09-01 按第 4 章体例重排并勘误）
**对应课程讲义**：MIT 6.S184 第 3 章 Flow Matching（PDF 印刷页 14–24）
**笔记版本**：v1.1

