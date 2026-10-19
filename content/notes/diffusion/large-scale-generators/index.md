+++
title = 'MIT 6.S184: Flow Matching and Diffusion Models — 第 6 章：构建大规模图像与视频生成器'
date = 2026-10-18
draft = false
summary = '从「网络到底长什么样」出发，讲清时间、类别与文本如何嵌入成向量，Diffusion Transformer 的 patchify、AdaLN 与交叉注意力，U-Net 的编解码结构；再推导 VAE 的重建项、先验项与重参数化，说明为什么现代生成模型都在潜空间里训练；最后用 Stable Diffusion 3 与 Movie Gen Video 把前五章的方法串起来。'
tags = ['Flow Matching', 'Diffusion Models', 'DiT', 'VAE', 'Latent Diffusion', '生成模型']
showReadingTime = true
showTableOfContents = true
+++

{{< katex >}}

> **[MIT 6.S184: Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/)** 课程笔记。对应讲义第 6 章 *Building Large-Scale Image or Video Generators*，讲义印刷页码 41–53。本文沿用课程从噪声到数据的时间方向（$t=0$ 是噪声，$t=1$ 是数据）。公式与编号已对照讲义核对；代码是公式的教学实现，未作为训练或生成质量的复现实验。案例研究中的模型规模、采样步数等数字均转述自讲义及其引用的原始论文，不是本文复现的结果。

## 一、本章要解决什么问题

前五章把数学讲完了：[第 3 章：流匹配](/notes/diffusion/flow-matching/)给出训练目标，[第 4 章：Score 函数与 Score Matching](/notes/diffusion/score-based-diffusion/)给出 score 与速度场的转换，[第 5 章：Guidance 与无分类器引导](/notes/diffusion/guidance/)让模型服从提示词。整套方法里只剩一个黑盒：**引导向量场 $u_t^\theta(x\mid y)$ 这个网络本身**。

玩具分布上，把 $x$、$y$、$t$ 拼起来喂给一个 MLP 就够了。但图像和视频不行：

- 网络有三个性质完全不同的输入：高维的 $x\in\mathbb R^d$、离散标签或一段文本 $y$、一个标量 $t\in[0,1]$。
- 输出 $u_t^\theta(x\mid y)\in\mathbb R^d$ **和输入一样大**。图像分类可以一路压缩到几个 logits，流模型不行。
- 一张 $1024\times1024$ 的 RGB 图像就有 $d=3\times1024^2\approx3.1\times10^6$ 维，视频还要再乘上帧数 $T$。

讲义把这一章分成三步，本文顺序相同：

| 部分 | 回答的问题 | 讲义位置 |
| --- | --- | --- |
| 条件嵌入 | $t$、类别、文本怎样变成网络能消化的向量 | §6.1.1 |
| 网络架构 | 用什么网络参数化 $u_t^\theta(x\mid y)$：DiT 或 U-Net | §6.1.2–6.1.3 |
| 潜空间 | 维度太高怎么办：先用 VAE 压缩，再在潜空间里训练流模型 | §6.2 |
| 案例研究 | 工业级模型是怎么把这些拼起来的 | §6.3 |

```text
文本 / 类别 y_raw ──► 冻结的文本编码器或查找表 ──► 条件嵌入 ỹ ─┐
时间 t ───────────► Fourier 特征 TimeEmb ───────► 时间嵌入 t̃ ─┤
                                                              ▼
图像 x ──► VAE 编码器 ──► 潜变量 z ──► DiT / U-Net：u_t^θ(z | y) ──► 速度场
                                                              │
采样：噪声 z_0 ──► ODE 积分到 z_1 ──► VAE 解码器（取均值）──► 生成图像
```

## 二、符号约定

| 符号 | 含义 |
| --- | --- |
| $x\in\mathbb R^{C\times H\times W}$ | 图像；$C$ 是通道数，RGB 图像 $C=3$ |
| $y_{\mathrm{raw}}$，$y$ | 原始条件（类别号或文本）与其嵌入 |
| $\tilde t\in\mathbb R^d$，$\tilde y\in\mathbb R^{S\times d}$ | 送进 DiT 的时间嵌入与提示嵌入，$S$ 是提示序列长度 |
| $P$，$N$ | patch 边长与 token 数，$N=(H/P)(W/P)$ |
| $d$，$L$，$h$ | DiT 的隐藏维度、层数、注意力头数，$d_h=d/h$ |
| $\mu_\phi,\sigma_\phi$ | VAE 编码器输出的均值与标准差 |
| $\mu_\theta,\sigma_\theta$ | VAE 解码器输出的均值与标准差 |
| $z\in\mathbb R^k$ | **本章里 $z$ 指潜变量**，$k\ll d$ |
| $\beta$ | VAE 先验项的权重 |

> 记号提醒：前几章 $z$ 表示数据点（$p_t(x\mid z)$ 里的那个 $z$），本章按讲义 §6.2 把 $z$ 改成 VAE 的潜变量。潜空间训练时，前几章公式里的"数据点"就换成"编码后的潜变量"，其余不变。

## 三、条件变量怎么喂给网络

### 3.1 时间：Fourier 特征

直接把标量 $t$ 拼到输入里，在玩具模型上能用；实践中通常把 $t$ 嵌入到高维空间，让网络更容易捕捉随时间变化很快的成分（讲义式 (68)(69)）：

$$
\begin{aligned}
\mathrm{TimeEmb}(t)=\sqrt{\frac{2}{d}}\,\big[&\cos(2\pi w_1t),\dots,\cos(2\pi w_{d/2}t),\\
&\sin(2\pi w_1t),\dots,\sin(2\pi w_{d/2}t)\big]^\top,
\end{aligned}
$$

$$
w_i=w_{\min}\left(\frac{w_{\max}}{w_{\min}}\right)^{\frac{i-1}{d/2-1}},\qquad i=1,\dots,d/2.
$$

频率在 $w_{\min}$ 到 $w_{\max}$ 之间按几何级数排开，低频分量刻画整体趋势，高频分量刻画细节。前面的系数 $\sqrt{2/d}$ 让嵌入的范数恰好为 1：每对 $\cos^2+\sin^2=1$，一共 $d/2$ 对。讲义强调这只是一个方便的标准选择，具体形式不是必须的。

### 3.2 类别标签：查找表

$y_{\mathrm{raw}}\in\{0,\dots,N\}$ 时，最简单的做法是给 $N+1$ 个取值各学一个嵌入向量，把它当作 $u_t^\theta$ 参数的一部分一起训练。多出来的那个取值正好可以留给第 5 章 label dropping 用的 $\varnothing$。

### 3.3 文本：冻结的预训练编码器

文本要复杂得多，主流做法是借用**冻结的预训练模型**：

- **CLIP**（[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)）用对比损失学一个图文共享的嵌入空间：让图像嵌入靠近它对应的文本，远离其他图像和文本（讲义图 14 右）。取 $y=\mathrm{CLIP}(y_{\mathrm{raw}})\in\mathbb R^{d_{\mathrm{CLIP}}}$，整段提示被压成**一个**向量。
- 有时不希望把整段话压成一个向量，就再用一个预训练 Transformer 得到**一串**嵌入。
- 实践中常把几种编码器的结果组合起来用，各取所长（SD3 和 Movie Gen 都这样做，见第七节）。

后文只需要记住：经过编码器后，提示嵌入的形状是 $\mathrm{PromptEmbed}(y_{\mathrm{raw}})\in\mathbb R^{S\times k}$。

## 四、Diffusion Transformer（DiT）

DiT 建立在 ViT（[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)）之上：把图像切成 patch，每个 patch 当成一个 token，用标准注意力处理，最后再拼回图像形状。讲义讲的是一个通用版本，具体模型的实现细节会有差异（DiT 原始论文见 [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)）。

### 4.1 从图像到 token，再回到图像

**Patchify。** 把 $x\in\mathbb R^{C\times H\times W}$ 重排成 $N$ 个 patch，每个 patch 展平成 $C'=CP^2$ 维向量，再乘一个可学习矩阵 $W\in\mathbb R^{C'\times d}$：

$$
\mathrm{Patchify}(x)\in\mathbb R^{N\times C'},\qquad \mathrm{PatchEmb}(x)=\mathrm{Patchify}(x)\,W\in\mathbb R^{N\times d}.
$$

**三路输入。** 时间、提示、图像都被映射到同一个隐藏维度 $d$：

$$
\begin{aligned}
\tilde t&=\mathrm{TimeEmb}(t)\in\mathbb R^d,\\
\tilde y&=\mathrm{PromptEmb}(y)\in\mathbb R^{S\times d},\\
\tilde x_0&=\mathrm{PatchEmb}(x)\in\mathbb R^{N\times d}.
\end{aligned}
$$

**堆叠 $L$ 层 DiT block**（讲义式 (70)），再 depatchify 回图像形状：

$$
\begin{aligned}
\tilde x_{i+1}&=\mathrm{DiTBlock}(\tilde x_i,\tilde t,\tilde y)\in\mathbb R^{N\times d}\qquad(i=0,\dots,L-1),\\
u&=\mathrm{Depatchify}(\tilde x_L\tilde W)\in\mathbb R^{C\times H\times W},
\end{aligned}
$$

其中 $\tilde W\in\mathbb R^{d\times C'}$。输出 $u$ 就是预测的速度场 $u_t^\theta(x\mid y)$，形状和输入图像一致。

> 本文补充：token 数 $N=(H/P)(W/P)$ 决定了自注意力的开销，后者随 $N^2$ 增长。patch 边长 $P$ 减半，$N$ 变成 4 倍，自注意力开销约变成 16 倍。这也是第五节要先压缩到潜空间的原因之一：例如 $4\times32\times32$ 的潜变量取 $P=2$，只有 $N=256$ 个 token，每个 $C'=16$ 维。

### 4.2 一个 DiT block 里发生了什么

讲义注 29 给出了一个典型 DiT block 的数学描述，它做三件事：patch 之间做自注意力，对提示做交叉注意力，通过自适应归一化（AdaLN）注入时间。

**缩放点积注意力。** $Q\in\mathbb R^{N\times d_h}$，$K,V\in\mathbb R^{M\times d_h}$，softmax 按行做：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V\in\mathbb R^{N\times d_h}.
$$

**多头注意力。** 每个头学三组投影 $W_Q^{(j)},W_K^{(j)},W_V^{(j)}$，查询来自 patch $x$，键和值来自源序列 $s$：

$$
\begin{aligned}
\mathrm{head}_j(x,s)&=\mathrm{Attn}\big(xW_Q^{(j)},\,sW_K^{(j)},\,sW_V^{(j)}\big),\\
\mathrm{MHA}(x,s)&=\mathrm{Concat}\big(\mathrm{head}_1,\dots,\mathrm{head}_h\big)W_O.
\end{aligned}
$$

$s=x$ 是 patch 之间的**自注意力**；$s=\tilde y$ 是对提示的**交叉注意力**，这时 $M=S$。

**用时间做自适应归一化。** 用一个 MLP $g:\mathbb R^d\to\mathbb R^{2d}$ 从时间嵌入生成逐通道的缩放与平移：

$$
(\gamma,\beta)=g(\tilde t),\qquad \mathrm{AdaNorm}_{\tilde t}(x)=(1+\gamma)\odot\mathrm{Norm}(x)+\beta,
$$

$\odot$ 在 token 维上广播，$\mathrm{Norm}$ 通常是 LayerNorm。有的实现给注意力和 MLP 子层各配一对 $(\gamma,\beta)$。注意这里的 $\beta$ 是平移参数，和 VAE 里的权重 $\beta$ 不是一回事。

**组合起来。** 三个残差子层，每个都带一个由时间决定的门控 $g_\cdot(\tilde t)$：

$$
\begin{aligned}
x&\leftarrow x+g_{\mathrm{self}}(\tilde t)\odot\mathrm{MHA}\big(\mathrm{AdaNorm}_{\tilde t}(x),\,\mathrm{AdaNorm}_{\tilde t}(x)\big)\\
x&\leftarrow x+g_{\mathrm{cross}}(\tilde t)\odot\mathrm{MHA}\big(\mathrm{AdaNorm}_{\tilde t}(x),\,\tilde y\big)\\
x&\leftarrow x+g_{\mathrm{MLP}}(\tilde t)\odot\mathrm{MLP}\big(\mathrm{AdaNorm}_{\tilde t}(x)\big)
\end{aligned}
$$

分工很清楚：**时间**通过 AdaNorm 和门控作用在每一层的每个 token 上；**文本提示**通过交叉注意力进入，每个 patch 可以按需查询提示的不同部分。讲义还指出，**类别条件**的 DiT（包括课程实验里实现的那个）通常更简单：去掉交叉注意力层，把时间和类别嵌入一起用 AdaNorm 注入。

```python
# 一个 DiT block 的教学实现（PyTorch 风格，未作为训练实验运行）
class DiTBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Linear(d, 5 * d)      # 由 t̃ 生成 gamma, beta 和三个门控

    def forward(self, x, t_emb, y_emb):     # x: [B,N,d], t_emb: [B,d], y_emb: [B,S,d]
        gamma, beta, g_self, g_cross, g_mlp = self.ada(t_emb)[:, None].chunk(5, dim=-1)
        ada_norm = lambda h: (1 + gamma) * self.norm(h) + beta
        h = ada_norm(x)
        x = x + g_self * self.self_attn(h, h, h)[0]                 # patch 自注意力
        x = x + g_cross * self.cross_attn(ada_norm(x), y_emb, y_emb)[0]  # 对提示交叉注意力
        x = x + g_mlp * self.mlp(ada_norm(x))
        return x
```

## 五、U-Net

U-Net（[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)）是 DiT 之外的另一种选择，本质是卷积网络。它原本为图像分割设计，关键性质是**输入和输出都是图像形状**（通道数可以不同），这正好符合 $x\mapsto u_t^\theta(x\mid y)$ 的要求，所以早期扩散模型大量使用它。

U-Net 由一串编码器 $E_i$、一串对应的解码器 $D_i$ 和夹在中间的处理块组成，讲义把中间块叫 midcoder（讲义自己说明这是非标准叫法）。以 $3\times256\times256$ 的输入为例：

| 阶段 | 运算 | 形状 |
| --- | --- | --- |
| 输入 | $x_t^{\mathrm{input}}$ | $3\times256\times256$ |
| 编码器 | $x_t^{\mathrm{latent}}=E(x_t^{\mathrm{input}})$ | $512\times32\times32$ |
| midcoder | $x_t^{\mathrm{latent}}=M(x_t^{\mathrm{latent}})$ | $512\times32\times32$ |
| 解码器 | $x_t^{\mathrm{output}}=D(x_t^{\mathrm{latent}})$ | $3\times256\times256$ |

编码阶段通道数增加、空间分辨率降低，解码阶段反过来。表里省略了两点：输入通常先过一个预编码块增加通道数；编码器和解码器之间有残差连接，也就是 U 形两侧的跳连（讲义图 15）。讲义的示例是纯卷积的，实际实现常在编解码器里也插入注意力层。

| 对比项 | DiT | U-Net |
| --- | --- | --- |
| 基本单元 | patch token + 注意力 | 卷积 + 上下采样 |
| 时间条件 | AdaNorm + 门控 | 依实现而定 |
| 文本条件 | 交叉注意力 | 依实现而定，常插入交叉注意力 |
| 讲义中的定位 | 当前大模型（SD3、Movie Gen）的骨干 | 早期扩散模型的主流选择 |

## 六、在潜空间里工作：（变分）自编码器

### 6.1 为什么要压缩

回到开头的问题：$d\approx3\times10^6$ 时，在 $\mathbb R^d$ 上直接训练很快就不可行了，视频更是如此。流模型的输出必须和输入一样大，没法像分类网络那样一路收窄。

自然的想法是**压缩**：真实图像大概率分布在高维空间里的一个低维流形附近。

### 6.2 标准自编码器，以及它不够用的地方

取编码器 $\mu_\phi:\mathbb R^d\to\mathbb R^k$ 和解码器 $\mu_\theta:\mathbb R^k\to\mathbb R^d$，$k\ll d$。讲义的例子是 $d=3\times1024\times1024$ 压到 $k=3\times\frac{1024}{16}\times\frac{1024}{16}$，即 256 倍压缩。训练用重建损失：

$$
\mathcal L_{\mathrm{Recon}}(\phi,\theta)=\mathbb E_{x\sim p_{\mathrm{data}}}\left\|\mu_\theta(\mu_\phi(x))-x\right\|^2.
$$

问题在于：我们最终要在潜空间里训练生成模型，目标是潜分布 $p_{\mathrm{latent}}$（$z=\mu_\phi(x)$，$x\sim p_{\mathrm{data}}$）。重建损失**完全不约束** $p_{\mathrm{latent}}$ 的形状。数据被压缩了，却可能被变成一个极难学的分布。所以要换一个能显式正则化潜分布的框架。

### 6.3 变分自编码器：重建项

VAE（[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)）把确定性的编解码器放松成条件分布（讲义式 (71)）：

$$
\begin{aligned}
q_\phi(z\mid x)&=\mathcal N\big(z;\mu_\phi(x),\mathrm{diag}(\sigma_\phi^2(x))\big),\\
p_\theta(x\mid z)&=\mathcal N\big(x;\mu_\theta(z),\sigma_\theta^2(z)I_d\big).
\end{aligned}
$$

编码是采样 $z\sim q_\phi(\cdot\mid x)$，解码是采样 $x\sim p_\theta(\cdot\mid z)$。$\sigma_\phi\equiv0$ 且 $\sigma_\theta\equiv0$ 时退化回标准自编码器。

重建损失换成负对数似然（式 (72)），高斯情形下展开为（式 (73)）：

$$
\begin{aligned}
\mathcal L_{\mathrm{VAE\text{-}Recon}}&=-\mathbb E_{x,\,z\sim q_\phi(\cdot\mid x)}\big[\log p_\theta(x\mid z)\big]\\
&=\mathbb E\left[\frac{1}{2\sigma_\theta^2(z)}\left\|x-\mu_\theta(z)\right\|^2+\frac d2\log\sigma_\theta^2(z)\right]+\mathrm{const}.
\end{aligned}
$$

和标准自编码器的差别只有两点：编码变成了采样；第二项让解码器在"重建精度"和"预测不确定性"之间权衡。许多实现（包括课程实验）把 $\sigma_\theta$ 固定成常数，于是重建项就是带随机编码的 MSE（式 (74)）。

### 6.4 变分自编码器：先验项与总损失

为了让潜分布"好学"，引入先验 $p_{\mathrm{prior}}=\mathcal N(0,I_k)$，要求每个数据点的编码分布都接近它（式 (75)）：

$$
\mathcal L_{\mathrm{VAE\text{-}Prior}}(\phi)=\mathbb E_{x\sim p_{\mathrm{data}}}\Big[D_{\mathrm{KL}}\big(q_\phi(\cdot\mid x)\,\big\|\,p_{\mathrm{prior}}\big)\Big].
$$

KL 散度 $D_{\mathrm{KL}}(q\|p)=\mathbb E_{X\sim q}\big[\log\frac{q(X)}{p(X)}\big]$ 非负，且当且仅当 $q=p$ 时为 0（讲义注 30）。两个对角高斯之间有闭式解（例 31，式 (80)）：

$$
D_{\mathrm{KL}}(q\|p)=\frac12\left[K\!\left(\frac{\sigma_q^2}{\sigma_p^2}\right)+\frac{\left\|\mu_q-\mu_p\right\|^2}{\sigma_p^2}\right],
$$

$$
K(\alpha)=\sum_{i}\big(\alpha_i-\log\alpha_i-1\big).
$$

$K(\alpha)$ 在 $\alpha=1$ 处取唯一最小值 0，所以 KL 在方差相等时最小，并随均值的平方距离增大。代入 $p=\mathcal N(0,I_k)$，再用权重 $\beta\ge0$ 把两项加起来，得到总损失（式 (78)(83)）：

$$
\begin{aligned}
\mathcal L_{\mathrm{VAE}}=\mathbb E\Bigg[&\underbrace{\frac{1}{2\sigma_\theta^2(z)}\left\|x-\mu_\theta(z)\right\|^2}_{\text{重建误差}}+\underbrace{\frac d2\log\sigma_\theta^2(z)}_{\text{解码器置信度}}\\
&+\underbrace{\frac\beta2K\big(\sigma_\phi^2(x)\big)}_{\text{潜变量方差}\to1}+\underbrace{\frac\beta2\left\|\mu_\phi(x)\right\|^2}_{\text{潜变量均值}\to0}\Bigg].
\end{aligned}
$$

四项各管一件事：重建；解码器方差越小越"自信"，但重建误差被罚得越重；后两项把每个编码分布往 $\mathcal N(0,I)$ 拉。对所有 $x$ 都这样做，自然期望整体潜分布也接近高斯。

### 6.5 重参数化与训练流程

期望是对 $q_\phi(z\mid x)$ 取的，而这个分布本身依赖 $\phi$，没法直接反向传播。重参数化把随机性挪到一个与 $\phi$ 无关的噪声上：

$$
\epsilon\sim\mathcal N(0,I_k),\qquad z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon\ \Longrightarrow\ z\sim q_\phi(\cdot\mid x).
$$

于是损失变成对 $(x,\epsilon)$ 的期望，可以用标准的深度学习工具优化。固定 $\sigma_\theta^2=\sigma^2$，就是讲义 Algorithm 6 的 $\beta$-VAE：

```python
# beta-VAE 训练（固定解码器方差 sigma2），对应讲义 Algorithm 6
for x in dataloader:
    mu, logvar = encoder(x)                       # mu_phi(x), log sigma_phi^2(x)
    eps = torch.randn_like(mu)
    z = mu + torch.exp(0.5 * logvar) * eps        # 重参数化
    x_hat = decoder(z)                            # 解码均值 mu_theta(z)
    recon = ((x - x_hat) ** 2).flatten(1).sum(1).mean() / (2 * sigma2)
    kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1).flatten(1).sum(1).mean()
    loss = recon + beta * kl
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

> 易错点：`kl` 里是 $\mu^2+\sigma^2-\log\sigma^2-1$，正是 $K(\sigma^2)+\|\mu\|^2$ 的逐维展开。少写最后的 $-1$ 不影响梯度，但 KL 的数值不再在完美匹配时等于 0，监控曲线会误导人。

### 6.6 讲义列出的四条实践要点

1. **$\beta$ 的选择与 KL warm-up。** $\beta$ 太大，潜变量贴近先验但重建变差，甚至后验坍缩：编码器干脆忽略 $x$，输出 $q_\phi(z\mid x)\approx\mathcal N(0,I_k)$。常见的稳定做法是从 $\beta=0$ 起步，在前几个 epoch 里逐渐升到目标值。讲义特别指出，现代自编码器里的 $\beta$ 都非常小，$\beta\ll1$。
2. **解码器方差。** 学习 $\sigma_\theta^2$ 数值上很敏感，不加正则可能得到退化解，所以多数实现把它固定成常数，重建项就正比于 MSE。
3. **超越像素 MSE。** 像素级高斯似然（MSE）的重建往往过于平滑，实践中会加上感知损失（在预训练网络的特征空间里算距离）来提升清晰度和语义保真度。
4. **对抗与混合目标。** 再加一个判别器作用在解码结果上（VAE-GAN 风格），输出更锐利，代价是额外的优化不稳定性和超参数。

### 6.7 潜空间里的生成模型

讲义注 32 把整套流程说得很直接：**训练流程不变，只是换到潜空间里做**。

- 训练时：$x\sim p_{\mathrm{data}}$，从 $q_\phi(z\mid x)$ **采样**得到潜变量，当作前几章里的"数据点"，照常做条件流匹配；
- 推理时：用潜空间里的流或扩散模型采样 $z$，再用解码器的**均值** $x=\mu_\theta(z)$ 解码，不再额外采样，以免引入噪声伪影。

直观上，训练良好的自编码器过滤掉了高频或语义上无意义的细节，让生成模型"专注"在感知上重要的特征上。写作讲义时，几乎所有最先进的图像与视频生成方法都遵循这种潜扩散范式（代表工作 [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)）。代价是：自编码器要**先**训练好，最终效果也取决于它压缩与还原得有多好。讲义附录 D 有更多 VAE 讨论。

## 七、案例研究：Stable Diffusion 3 与 Movie Gen Video

这两个模型几乎就是前五章加本章的组合，再加上一些为规模和富条件而做的架构增强。

### 7.1 Stable Diffusion 3（图像）

论文：[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)。

| 组成 | 做法 | 对应本系列 |
| --- | --- | --- |
| 训练目标 | 条件流匹配；论文比较了多种流与扩散方案，流匹配表现最好 | 第 3 章，讲义 Algorithm 4 |
| 引导 | CFG 训练，丢弃条件 | 第 5 章 |
| 空间 | 预训练自编码器的潜空间 | 本章第六节 |
| 文本嵌入 | 3 种：CLIP 嵌入，加上 T5-XXL 编码器输出的序列嵌入 | 本章 3.3 |
| 骨干 | MM-DiT：DiT 不仅关注图像 patch，也关注文本嵌入 | 本章第四节 |
| 规模 | 最大 80 亿参数 | — |
| 采样 | Euler 法 50 步（网络评估 50 次），CFG 权重 2.0–5.0 | 第 2 章 Euler，第 5 章 $w$ |

两种文本嵌入的分工：CLIP 给出整段文本粗粒度的整体表示；T5（[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)）给出逐 token 的细粒度上下文，让模型能关注提示里的具体元素。为了容纳这种序列嵌入，SD3 把 DiT 原本基于类别的条件方式扩展成 MM-DiT（讲义图 16）。

讲义脚注 4 提醒：SD3 论文里给噪声做条件的记号约定与本课程不同，但这只是记号，算法是同一个。读原论文时先对齐时间方向，比较 CFG 权重前也要先对齐 $w$ 的定义（见第 5 章第六节）。

### 7.2 Movie Gen Video（视频）

技术报告：[Movie Gen: A Cast of Media Foundation Models](https://arxiv.org/abs/2410.13720)。数据从图像变成视频，$x\in\mathbb R^{T\times C\times H\times W}$，多出一个帧数维 $T$。很多设计都可以看作把图像上的技术推广到这一维。

- **训练目标：** 条件流匹配，直线调度 $\alpha_t=t$，$\beta_t=1-t$（讲义此处记作 $\sigma_t$）。
- **时间自编码器（TAE）：** 同样在冻结的预训练自编码器潜空间里训练。TAE 把原始视频 $x'\in\mathbb R^{T'\times3\times H'\times W'}$ 映射到潜变量 $x\in\mathbb R^{T\times C\times H\times W}$，帧数、高、宽各压缩 8 倍。讲义指出，压缩对视频比对图像更重要，这也是目前多数视频生成器能生成的时长很有限的原因。
- **长视频：** 用时间分块（temporal tiling）把视频切段，各段分别编码，再把潜变量拼起来。
- **骨干：** 类 DiT 结构，$x_t$ 沿时间和空间两个方向做 patchify。patch 之间做自注意力，和语言模型嵌入做交叉注意力，与 SD3 的 MM-DiT 相似。
- **文本嵌入：** 3 种，各司其职：UL2 负责细粒度的文本推理；ByT5 负责字符级细节，比如提示里明确要求画面中出现某段文字；MetaCLIP 提供图文共享嵌入空间。
- **规模：** 最大 300 亿参数。

## 八、常见误解

| 误解 | 纠正 |
| --- | --- |
| DiT 用 AdaLN 同时注入时间和文本 | 讲义描述的通用 DiT block：时间走 AdaNorm 和门控，文本提示走交叉注意力；只有类别条件 DiT 才把类别也并入 AdaNorm |
| VAE 编码是确定性映射 | 训练潜生成模型时从 $q_\phi(z\mid x)$ **采样**；只有推理解码时取解码器均值 |
| 潜扩散换了一套训练目标 | 训练目标不变，仍是条件流匹配，只是"数据点"换成潜变量 |
| $\beta$ 越大潜空间越好 | $\beta$ 太大会后验坍缩；讲义指出现代自编码器 $\beta\ll1$ |
| 自编码器只是预处理，不影响效果 | 最终画质受自编码器压缩与还原质量的上限约束，而且它必须先单独训好 |
| SD3 / Movie Gen 用了与课程不同的生成原理 | 两者都用条件流匹配加 CFG，并在潜空间里训练；差别主要在架构、文本编码器和规模 |
| 文本条件只需要一个 CLIP 向量 | CLIP 只给整体表示；需要关注具体词或字符时，要再加 T5、UL2、ByT5 这类序列嵌入 |

## 九、自测题

### 1. 为什么流模型不能像图像分类网络那样一路压缩特征？

因为输出 $u_t^\theta(x\mid y)\in\mathbb R^d$ 必须和输入一样大。分类网络只需要几个 logits，可以一路收窄；流模型每一步都要输出一个完整的图像形状的速度场。这也是 U-Net 有对称解码器、DiT 最后要 depatchify 的原因。

### 2. Fourier 时间嵌入前面的 $\sqrt{2/d}$ 起什么作用？

让嵌入范数恰好为 1。嵌入里有 $d/2$ 对 $(\cos,\sin)$，每对平方和为 1，总平方和为 $d/2$，乘上 $\sqrt{2/d}$ 后范数为 1。

### 3. 一个 DiT block 里，时间和文本分别从哪里进入？

时间嵌入 $\tilde t$ 经 MLP 生成 AdaNorm 的 $(\gamma,\beta)$ 和三个门控，作用在每个子层；文本嵌入 $\tilde y$ 作为交叉注意力的键和值，让每个 patch 查询提示。类别条件 DiT 可以去掉交叉注意力，把类别并入 AdaNorm。

### 4. 标准自编码器已经能压缩图像，为什么还要 VAE？

重建损失只保证能还原，不约束潜分布 $p_{\mathrm{latent}}$ 的形状，压缩后的分布可能极难学。VAE 用先验项 $D_{\mathrm{KL}}(q_\phi(\cdot\mid x)\|\mathcal N(0,I))$ 把每个编码分布往标准高斯拉，让潜分布对后续生成模型更友好。

### 5. 写出 $q_\phi(z\mid x)$ 与 $\mathcal N(0,I_k)$ 之间的 KL，并说明各项含义。

$\frac12\sum_j\big(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\big)$。$\mu_j^2$ 把均值往 0 拉；$\sigma_j^2-\log\sigma_j^2-1$ 在 $\sigma_j^2=1$ 处取最小值 0，把方差往 1 拉。

### 6. 训练和推理时，潜变量分别怎么得到？

训练时从 $q_\phi(z\mid x)$ 采样，通常用重参数化 $z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$；推理时由潜空间流模型从噪声积分得到 $z$，再取解码器均值 $\mu_\theta(z)$ 作为图像，避免噪声伪影。

### 7. SD3 用了前几章的哪些方法？

条件流匹配目标（第 3 章）、CFG 训练与采样（第 5 章）、Euler 法采样（第 2 章）、预训练自编码器的潜空间（本章），以及在 DiT 基础上扩展出的 MM-DiT 骨干。

## 十、公式速查与阅读位置

| 对象 | 公式或结论 | 讲义位置 |
| --- | --- | --- |
| 时间嵌入 | $\sqrt{2/d}\,[\cos(2\pi w_it),\sin(2\pi w_it)]$，频率几何排开 | 式 (68)(69) |
| 类别 / 文本嵌入 | 查找表；冻结 CLIP 或序列编码器 | §6.1.1 |
| DiT 主干 | $\tilde x_{i+1}=\mathrm{DiTBlock}(\tilde x_i,\tilde t,\tilde y)$，再 depatchify | 式 (70)，图 14 |
| DiT block | 自注意力、交叉注意力、AdaNorm 加门控 | 注 29 |
| U-Net | 编码器、midcoder、解码器，带跳连 | §6.1.3，图 15 |
| VAE 编解码器 | $q_\phi=\mathcal N(\mu_\phi,\mathrm{diag}\,\sigma_\phi^2)$，$p_\theta=\mathcal N(\mu_\theta,\sigma_\theta^2I)$ | 式 (71) |
| 重建项 | 负对数似然，固定方差时即 MSE | 式 (72)–(74) |
| 先验项 | $\mathbb E_x\,D_{\mathrm{KL}}(q_\phi(\cdot\mid x)\,\Vert\,\mathcal N(0,I))$ | 式 (75)，注 30 |
| 高斯 KL | $\frac12[K(\sigma_q^2/\sigma_p^2)+\lVert\mu_q-\mu_p\rVert^2/\sigma_p^2]$ | 例 31，式 (80) |
| VAE 总损失 | 重建 + 解码器置信度 + $\beta$ 倍先验 | 式 (78)–(83) |
| VAE 训练 | 重参数化，$\beta$-VAE | Algorithm 6 |
| 潜空间生成 | 训练时采样编码，推理时取解码均值 | 注 32 |
| 案例研究 | SD3、Movie Gen Video | §6.3，图 16 |

推导以课程讲义为主。数值上我核对了几处：Fourier 时间嵌入在不同 $t$ 下范数均为 1；patchify 与 depatchify 互为逆运算；对角高斯 KL 的闭式解与 Monte Carlo 估计一致；Algorithm 6 的逐维 KL 写法与式 (82) 相同。token 数与注意力开销的估算、两段代码属于本文补充，没有做训练或生成实验。完整讲义可从[课程主页](https://diffusion.csail.mit.edu/)获取。

前置阅读：[第 3 章：流匹配](/notes/diffusion/flow-matching/)、[第 5 章：Guidance 与无分类器引导](/notes/diffusion/guidance/)。下一章讲义转向离散扩散模型：如何用连续时间马尔可夫链在离散 token 上构建语言模型。
