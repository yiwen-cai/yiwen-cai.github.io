+++
title = 'MIT 6.S184: Flow Matching and Diffusion Models — 第 4 章：Score 函数与 Score Matching'
date = 2026-09-11
draft = false
summary = '从条件 score 的后验平均出发，推导高斯路径下 score、去噪器与速度场的转换，用 Fokker–Planck 方程解释如何给 ODE 加噪而保持边缘分布，并将 score matching 化为噪声回归。'
tags = ['Flow Matching', 'Diffusion Models', 'Score Matching', 'SDE', '生成模型']
showReadingTime = true
showTableOfContents = true
+++

{{< katex >}}

> **[MIT 6.S184: Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/)** 课程笔记。对应讲义第 4 章 *Score Functions and Score Matching*，讲义印刷页码 25–33。本文沿用课程从噪声到数据的时间方向；代码是公式的教学实现，未作为训练或生成质量的复现实验。

## 一、本章要解决什么问题

[第 3 章：流匹配](/notes/diffusion/flow-matching/)通过学习速度场，把噪声沿 ODE 输运为数据。第四章进一步问：**如果希望粒子在生成途中随机游走，怎样加入噪声，才能仍然经过同一条概率分布路径？**

答案是同时加入一个由 score 决定的漂移修正项。随机噪声使概率质量扩散，score 修正项在分布演化方程中抵消这部分变化。为了实现它，我们需要学习边缘 score，而这又可以转化为有解析监督目标的条件回归。

| 问题 | 本章的方法 | 结果 |
| --- | --- | --- |
| 如何描述未知边缘密度的 score？ | 对条件 score 取后验平均 | 边缘 score 恒等式 |
| Score 与速度场有什么关系？ | 利用高斯路径的解析形式 | 速度场、score、去噪器可以转换 |
| 如何给 ODE 加随机性？ | 加入 score 漂移修正 | SDE 与 ODE 保持相同边缘分布 |
| 边缘 score 不可计算，如何训练？ | 用条件 score 替代回归目标 | 两种损失相差参数无关的常数 |
| 怎样处理小噪声处的大目标值？ | 改用噪声预测并调整时间权重 | 简单噪声均方误差目标 |

## 二、符号与时间方向

| 符号 | 含义与维度 |
| --- | --- |
| $z\sim p_{\mathrm{data}}$ | 干净数据，$z\in\mathbb R^d$ |
| $\epsilon\sim\mathcal N(0,I_d)$ | 与数据独立的高斯噪声 |
| $x=\alpha_tz+\beta_t\epsilon$ | 时刻 $t$ 的带噪样本，$x\in\mathbb R^d$ |
| $p_t(x\mid z)$ | 固定干净数据 $z$ 后的条件密度 |
| $p_t(x)$ | 将 $z$ 边缘化后的密度 |
| $s_t(x)=\nabla_x\log p_t(x)$ | 边缘 score，输出 $d$ 维向量 |
| $u_t(x)$ | 实现目标概率路径的 ODE 速度场 |
| $\sigma_t$ | SDE 扩散系数，与路径噪声标准差 $\beta_t$ 是不同对象 |

采用理想端点条件

$$
\alpha_0=0,\quad\beta_0=1,\qquad\alpha_1=1,\quad\beta_1=0.
$$

因此 $t=0$ 是噪声，$t=1$ 是数据。含除法的公式首先在 $0<t<1$、相应分母非零的范围讨论；端点退化为点质量时，不能直接套用普通密度的 score 公式。

涉及梯度与积分交换、分布演化方程及 SDE 解时，默认满足所需的光滑性、可积性、边界衰减与适定性条件。非零高斯噪声使中间分布平滑，但不代表所有端点表达式也良好定义。

## 三、条件 Score 与边缘 Score

### 3.1 Score 是对数密度的梯度

给定光滑且为正的密度 $p(x)$，定义

$$
s(x)=\nabla_x\log p(x)=\frac{\nabla_xp(x)}{p(x)}.
$$

它指向对数密度局部上升最快的方向。求导变量是样本 $x$，不是网络参数 $\theta$。如果 $p(x)=\tilde p(x)/Z$，其中 $Z$ 不依赖 $x$，那么 $\nabla_x\log p=\nabla_x\log\tilde p$：归一化常数被求导消去了，但未知分布的 score 仍然需要学习。

对各向同性高斯分布，

$$
p(x)=\mathcal N(x;\mu,\tau^2I_d)
\quad\Longrightarrow\quad s(x)=-\frac{x-\mu}{\tau^2}.
$$

此时 score 指向均值，在均值处为零。一般多峰分布的 score 并不总指向全局中心，也不能仅凭其大小判断密度高低。

### 3.2 高斯条件 Score 有解析目标

对于 $p_t(x\mid z)=\mathcal N(x;\alpha_tz,\beta_t^2I_d)$，

$$
\log p_t(x\mid z)=-\frac{\|x-\alpha_tz\|^2}{2\beta_t^2}+C(t,z),
$$

所以

$$
s_t(x\mid z)=-\frac{x-\alpha_tz}{\beta_t^2}=-\frac{\epsilon}{\beta_t}.
$$

训练时我们自己采样 $z$ 和 $\epsilon$，因此条件目标已知。生成时不知道目标数据 $z$，需要使用边缘 score $s_t(x)$。

### 3.3 边缘 Score 是条件 Score 的后验平均

由 $p_t(x)=\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz$，

$$
\begin{aligned}
\nabla_x\log p_t(x)
&=\frac{1}{p_t(x)}\int\nabla_xp_t(x\mid z)p_{\mathrm{data}}(z)\,dz\\
&=\int\nabla_x\log p_t(x\mid z)
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}\,dz\\
&=\mathbb E_{z\sim p_t(z\mid x)}[s_t(x\mid z)].
\end{aligned}
$$

**权重是观察到 $x$ 后的后验 $p_t(z\mid x)$。** 一个干净样本越有可能产生当前的 $x$，它的条件 score 权重就越大。

这是一条精确恒等式，不需要“小噪声”或“数据各维独立”的近似。将后验换成原始数据分布 $p_{\mathrm{data}}(z)$，一般就不成立。

## 四、Score、去噪器与速度场的转换

### 4.1 去噪器预测条件均值

定义最小均方误差意义下的去噪器 $D_t(x)=\mathbb E[z\mid X_t=x]$。将高斯条件 score 做后验平均，得到

$$
s_t(x)=\frac{\alpha_tD_t(x)-x}{\beta_t^2},
\qquad D_t(x)=\frac{x+\beta_t^2s_t(x)}{\alpha_t}.
$$

第二式要求 $\alpha_t\ne0$。这解释了为什么学习 score 也能实现去噪。

去噪器输出未必是某个真实干净样本。当带噪输入对应多个可能的干净样本时，条件均值会把它们平均，甚至落在低密度区域。完整生成仍依靠动力学过程。

### 4.2 高斯路径下的速度场转换

上一章的条件速度场是

$$
u_t(x\mid z)=\left(\dot\alpha_t-\frac{\dot\beta_t}{\beta_t}\alpha_t\right)z
+\frac{\dot\beta_t}{\beta_t}x.
$$

代入 $\alpha_tz=x+\beta_t^2s_t(x\mid z)$，可写为

$$
u_t(x\mid z)=a_ts_t(x\mid z)+b_tx,
\qquad a_t=\beta_t^2\frac{\dot\alpha_t}{\alpha_t}-\dot\beta_t\beta_t,
\quad b_t=\frac{\dot\alpha_t}{\alpha_t}.
$$

再对后验取期望，就得到讲义命题 1：

$$
\boxed{u_t(x)=a_ts_t(x)+b_tx.}
$$

当 $a_t\ne0$ 时，还可以反求 $s_t(x)=(u_t(x)-b_tx)/a_t$。在同一条高斯路径和非退化条件下，不必分别训练两个独立网络才能获得 score 和速度场。

这是目标函数之间的数学转换。不同时间权重、网络参数化、有限训练误差和数值求解器，仍会带来不同的实际表现。

## 五、给 ODE 加噪，如何保持边缘分布

### 5.1 SDE 扩展技巧

设 ODE $dX_t=u_t(X_t)dt$、$X_0\sim p_{\mathrm{init}}$ 实现了目标路径 $p_t$。讲义定理 17 给出对应的 SDE：

$$
\boxed{dX_t=\left[u_t(X_t)+\frac{\sigma_t^2}{2}s_t(X_t)\right]dt
+\sigma_t\,dW_t.}
$$

在适当的正则性条件下、以同一初始分布启动，它仍满足 $X_t\sim p_t$。Score 前的系数是 **$\sigma_t^2/2$**；这里从 ODE 出发，时间沿噪声到数据递增。

### 5.2 Fokker–Planck 方程解释修正项

对于扩散系数仅依赖时间的 SDE $dX_t=v_t(X_t)dt+\sigma_t dW_t$，密度满足

$$
\partial_tp_t=-\nabla\cdot(p_tv_t)+\frac{\sigma_t^2}{2}\Delta p_t,
\qquad\Delta p_t=\sum_{i=1}^d\frac{\partial^2p_t}{\partial x_i^2}.
$$

第一项描述漂移输运，第二项描述噪声引起的扩散。$\sigma_t=0$ 时退化为连续性方程。

令 $v_t=u_t+\frac{\sigma_t^2}{2}s_t$，利用 $p_ts_t=\nabla p_t$：

$$
\begin{aligned}
\partial_tp_t
&=-\nabla\cdot(p_tu_t)-\frac{\sigma_t^2}{2}\nabla\cdot(p_ts_t)
+\frac{\sigma_t^2}{2}\Delta p_t\\
&=-\nabla\cdot(p_tu_t)-\frac{\sigma_t^2}{2}\Delta p_t
+\frac{\sigma_t^2}{2}\Delta p_t\\
&=-\nabla\cdot(p_tu_t).
\end{aligned}
$$

新增漂移与扩散在密度演化方程中恰好抵消，但不会在单条样本轨迹上逐步抵消：粒子仍然随机运动。

### 5.3 相同边缘分布，不等于相同轨迹

| 比较项 | ODE | 对应 SDE |
| --- | --- | --- |
| 给定初始样本后 | 解唯一时轨迹确定 | 还取决于后续布朗噪声 |
| 随机性来源 | 初始噪声 | 初始噪声与沿途噪声 |
| 理想边缘分布 | $p_t$ | 同一个 $p_t$ |
| 数值求解 | Euler、Heun 等 | Euler–Maruyama 等 |
| 实际误差 | 网络误差与离散误差 | 网络误差、离散误差及其与扩散强度的相互作用 |

**确定性 ODE 仍能生成多样的样本，因为初始噪声是随机的。** 理想条件下两者终点分布相同，不能推断 ODE 的多样性一定更低。实践中扩散强度需要与模型误差、步长和预算一起评估。

### 5.4 静态路径与 Langevin 动力学

若路径始终为同一个密度 $p$，可取 $u_t=0$：

$$
dX_t=\frac{\sigma^2}{2}\nabla\log p(X_t)dt+\sigma dW_t.
$$

这是 Langevin 动力学的一种写法。从 $X_0\sim p$ 开始，它保持 $p$ 不变。从其他分布开始是否收敛、收敛多快，还需要遍历性等条件；平稳性不保证任意初始化都会快速收敛。

## 六、用条件回归学习边缘 Score

### 6.1 两种损失只差参数无关的常数

希望训练 $s_\theta(x,t)$ 逼近边缘 score：

$$
L_{\mathrm{SM}}(\theta)=\mathbb E_{t,x}\|s_\theta(x,t)-s_t(x)\|^2.
$$

但目标不可直接计算。实际使用

$$
L_{\mathrm{CSM}}(\theta)=\mathbb E_{t,z,x}\|s_\theta(x,t)-s_t(x\mid z)\|^2,
$$

其中 $t$ 从选定时间分布采样，$z\sim p_{\mathrm{data}}$，$x\sim p_t(\cdot\mid z)$。

令 $Y=s_t(x\mid z)$。由 $\mathbb E[Y\mid x,t]=s_t(x)$，平方误差可分解为

$$
L_{\mathrm{CSM}}(\theta)=L_{\mathrm{SM}}(\theta)+\mathbb E\|Y-s_t(x)\|^2.
$$

交叉项为零，因为固定 $x,t$ 后，$Y-s_t(x)$ 的条件期望为零。最后一项不含网络参数，所以两种损失的参数梯度相同。这就是讲义定理 22。

**条件回归损失不必降到零，模型才算学对。** 条件目标本身存在不可消除的方差。只有在模型容量、优化与数据条件允许时，最优回归函数才可能精确等于真实边缘 score。

### 6.2 噪声预测与时间权重

代入高斯条件目标：

$$
L_{\mathrm{CSM}}=\mathbb E\left\|s_\theta(\alpha_tz+\beta_t\epsilon,t)+\frac{\epsilon}{\beta_t}\right\|^2.
$$

定义 $\epsilon_\theta(x,t)=-\beta_ts_\theta(x,t)$，原损失精确变为

$$
L_{\mathrm{CSM}}=\mathbb E\left[\frac{1}{\beta_t^2}\|\epsilon_\theta(x,t)-\epsilon\|^2\right].
$$

常用的简单噪声损失是

$$
L_{\mathrm{simple}}=\mathbb E\|\epsilon_\theta(x,t)-\epsilon\|^2.
$$

去掉 $1/\beta_t^2$ **改变了时间权重，并不保持原损失的数值或参数梯度**。在逐时刻拥有无限表达能力、权重为正等理想条件下，两者有相同的最优预测函数；共享网络的有限容量训练会受到权重选择影响。

当 $\beta_t\to0$ 时，条件 score 尺度增大，原损失还可能出现可积性问题。实现时常采用端点截断、权重设计或其他参数化。上面的期望分解需要相关期望有限。

### 6.3 对应公式的 PyTorch 训练代码

以下用 Gaussian CondOT 调度 $\alpha_t=t$、$\beta_t=1-t$ 展示单步训练。`z` 为浮点张量，shape 可以是 `[B, d]` 或 `[B, C, H, W]`；`model(x, t)` 返回与 `x` 相同的 shape，时间输入为 `[B]`。

```python
import torch


def noise_prediction_step(model, optimizer, z, time_eps=1e-3):
    # 简单噪声损失；避开退化端点。
    batch_size = z.shape[0]
    t = time_eps + (1 - 2 * time_eps) * torch.rand(
        batch_size, device=z.device, dtype=z.dtype
    )
    broadcast_shape = (batch_size,) + (1,) * (z.ndim - 1)
    alpha = t.reshape(broadcast_shape)
    beta = (1 - t).reshape(broadcast_shape)
    noise = torch.randn_like(z)
    x = alpha * z + beta * noise
    noise_pred = model(x, t)
    loss = (noise_pred - noise).square().flatten(1).mean(1).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()
```

`randn_like` 生成标准高斯噪声，`rand` 仅用于均匀采样时间。广播形状保证每个样本的一个时间值作用于其全部特征。代码按特征取均值，而公式用平方范数求和；固定样本维度下二者只差整体尺度。

训练直接构造带噪样本，无需逐步模拟 SDE。生成阶段才需要数值积分。

## 七、训练之后如何采样

在内部时刻计算 $s_\theta=-\epsilon_\theta/\beta_t$、$u_\theta=a_ts_\theta+b_tx$。给定递增时间网格、$h_k=t_{k+1}-t_k>0$，Euler–Maruyama 更新为

$$
x_{k+1}=x_k+\left[u_\theta(x_k,t_k)+\frac{\sigma_{t_k}^2}{2}s_\theta(x_k,t_k)\right]h_k
+\sigma_{t_k}\sqrt{h_k}\,\xi_k,
\quad\xi_k\sim\mathcal N(0,I_d).
$$

各步 $\xi_k$ 独立。噪声增量尺度是 $\sqrt{h_k}$，不是 $h_k$。令 $\sigma_t=0$ 就得到 Euler ODE 更新。

CondOT 调度下，

$$
a_t=\frac{1-t}{t},\quad b_t=\frac1t,
\qquad u_\theta(x,t)=\frac{x-\epsilon_\theta(x,t)}{t}.
$$

这个表示在 $t=0$ 处不能直接求值。完整采样器必须处理端点，例如使用有稳定端点定义的速度参数化，或明确采用截断近似。若从 $t=\delta>0$ 启动，理论初始分布应为 $p_\delta$，直接用标准高斯替代是额外近似。训练时间截断不等于解决了采样端点问题。

### 与反向 SDE 的符号对照

另设从数据到噪声递增的时间 $r$，前向加噪 SDE 为 $dZ_r=f_r(Z_r)dr+g_r dW_r$，边缘密度为 $q_r$。其 Probability Flow ODE 速度为 $f_r-\frac12g_r^2\nabla\log q_r$。

改用生成时间 $t=1-r$，$p_t=q_{1-t}$，对应 ODE 速度为

$$
u_t=-f_{1-t}+\frac12g_{1-t}^2\nabla\log p_t.
$$

再用本章 SDE 扩展、取 $\sigma_t=g_{1-t}$，漂移变成

$$
u_t+\frac12\sigma_t^2\nabla\log p_t
=-f_{1-t}+g_{1-t}^2\nabla\log p_t.
$$

这解释了 $1/2$ 与 $1$ 的来源：比较公式前必须对齐时间方向，以及起点是 ODE 还是前向 SDE。仅把随机增量变号不能构成正确的时间反演。

## 八、自测题

### 1. 为什么不能对所有数据的条件 Score 等权平均？

因为固定当前状态 $x$ 后需要的是后验 $p_t(z\mid x)$。不同干净数据产生当前带噪状态的可能性不同，等权平均忽略了这种信息。

### 2. 没显式计算后验，网络怎么学到后验平均？

训练从联合分布 $p_{\mathrm{data}}(z)p_t(x\mid z)$ 采样。平方损失在固定输入 $x,t$ 时的最优解是目标的条件期望，后验权重由联合采样与回归隐式确定。

### 3. 只给 ODE 加布朗噪声会怎样？

密度演化方程多出 $\sigma_t^2\Delta p_t/2$，一般不再沿原来的路径。还需加入 $\sigma_t^2s_t/2$ 的漂移修正。

### 4. 条件损失持续波动，是否说明训练失败？

单个 batch 不能判断。数据、时间和噪声随机采样，条件目标也有不可约方差。应结合固定验证设置、损失统计和生成分布评估；最优条件损失也可能大于零。

### 5. 简单噪声 MSE 与原 Score 损失完全一样吗？

不一样。原损失转换后带有 $1/\beta_t^2$ 权重，去掉它重新分配了不同时刻的训练贡献。

### 6. ODE 固定初始状态就固定结果，为什么仍能生成完整分布？

每次可以重新采样初始噪声。确定性映射能把一个随机变量转换成另一个随机变量，沿途有无新增噪声并不决定终点分布是否多样。

## 九、公式速查与阅读位置

| 对象 | 公式或结论 | 讲义位置 |
| --- | --- | --- |
| 边缘 score | $s_t(x)=\mathbb E[s_t(x\mid z)\mid x]$ | §4.1，式 (38)–(39) |
| 高斯条件 score | $s_t(x\mid z)=-(x-\alpha_tz)/\beta_t^2$ | 示例 15 |
| 速度与 score 转换 | $u_t=a_ts_t+b_tx$ | 命题 1，式 (41)–(42) |
| SDE 扩展 | 漂移 $u_t+\sigma_t^2s_t/2$，扩散系数 $\sigma_t$ | 定理 17，式 (44) |
| Fokker–Planck | $\partial_tp_t=-\nabla\cdot(p_tv_t)+\sigma_t^2\Delta p_t/2$ | 定理 19，式 (49) |
| 条件回归等价性 | $L_{\mathrm{CSM}}=L_{\mathrm{SM}}+C$ | 定理 22；此处常数符号与讲义写法相反 |
| 噪声参数化 | $\epsilon_\theta=-\beta_ts_\theta$ | 示例 23，Algorithm 4 |

推导以课程讲义为主，训练代码与时间方向对照是本文补充说明。完整讲义可从[课程主页](https://diffusion.csail.mit.edu/)获取。

前置阅读：[第 2 章：流模型与扩散模型](/notes/diffusion/flow-and-diffusion-models/)、[第 3 章：流匹配](/notes/diffusion/flow-matching/)。第 5 章将讨论如何通过 Guidance 进行条件生成。
