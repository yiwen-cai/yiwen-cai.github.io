#import "@preview/basic-resume:0.2.3": *
// Put your personal information here, replacing mine
#let name = "蔡逸文"
#let location = "北京, 中国"
#let email = "caiyiwen.cs@foxmail.com"
#let github = "github.com/yiwen-cai"
#let phone = "(+86) 189-2876-9793"
#let personal-site = "yiwen-cai.github.io"



#show: resume.with(
  author: name,
  // All the lines below are optional.
  // For example, if you want to to hide your phone number:
  // feel free to comment those lines out and they will not show.
  location: location,
  email: email,
  github: github,
  phone: phone,
  personal-site: personal-site,
  accent-color: "#26428b",
  // 按 glyph 逐个回退：西文走 Libertinus Serif（typst 内置，CI 与本机都有），
  // 中文走 Noto Serif CJK SC（CI 由 fonts-noto-cjk 提供），Songti SC 兜住本机 macOS 预览。
  // 列表里必须有 CJK 字体，否则会全部回退到 Libertinus 并渲染成错误字形。
  font: ("Libertinus Serif", "Noto Serif CJK SC", "Songti SC"),
  paper: "us-letter",
  author-position: center,
  personal-info-position: center,
)

/*
 * Lines that start with == are formatted into section headings
 * You can use the specific formatting functions if needed
 * The following formatting functions are listed below
 * #edu(dates: "", degree: "", gpa: "", institution: "", location: "")
 * #work(company: "", dates: "", location: "", title: "")
 * #project(dates: "", name: "", role: "", url: "")
 * certificates(name: "", issuer: "", url: "", date: "")
 * #extracurriculars(activity: "", dates: "")
 * There are also the following generic functions that don't apply any formatting
 * #generic-two-by-two(top-left: "", top-right: "", bottom-left: "", bottom-right: "")
 * #generic-one-by-two(left: "", right: "")
 */
#align(center)[知乎：#link("https://www.zhihu.com/people/cywww-14/")[zhihu.com/people/cywww-14]]

== 教育经历

#edu(
  institution: "北京邮电大学(BUPT)",
  location: "北京, 中国",
  dates: dates-helper(start-date: "Sep 2026", end-date: "Jun 2029"),
  degree: "硕士学位（推免）, 计算机技术（ParCIS 实验室，导师：李士刚 教授）",
)

#edu(
  institution: "北京邮电大学(BUPT)",
  location: "北京, 中国",
  dates: dates-helper(start-date: "Sep 2022", end-date: "Jun 2026"),
  degree: "学士学位, 计算机科学与技术",
)
- GPA: 3.76 / 4.0 · Grade: 89.37 / 100.0 · Rank: 59 / 389 (15.16%) · English: CET6 552 / 710

== 实习经历

#work(
  title: "AI Infra 实习生（Algorithm–Infra Co-design & 视频生成推理优化）",
  location: "北京, 中国",
  company: "北京踊跃智能（Loopit）",
  dates: dates-helper(start-date: "Oct 2026", end-date: "Present"),
)
- *Zing H3 全 FP8 三档稀疏注意力与常驻压缩 KV 池（Algorithm–Kernel–System Co-design）*：
  - *算法设计（Algorithm）*：提出 *Refine（1/1 精确）/ Skim（1/16 象限低秩粗估）/ Skip（1/64 剪枝）* 三档稀疏注意力算法与全方差/CosGate 动态路由（典型分布 `16.05% / 31.83% / 52.12%`，相对全精细计算余弦相似度达 *0.9964*）；针对视频帧二维空间局部性，打破传统 1D 光栅行主序切块（跨行混合导致块内余弦一致性仅 *0.4741*），设计*单帧对齐 2D 空间正方形 Tile 切块置换算法*，将 Skim 象限块内余弦一致性提升至 *0.9036*；结合 3D RoPE $"SO"(2)$ 群同态与正交伴随性质，设计*滑窗驱逐下 $O(1)$ 零历史 KV 重算的 Query 逆时间旋转对齐算法*，15,921 帧（11 分钟）超长序列全时程 Mean PSNR 达 *23.14 dB*（反超 Dense FP8 基线 *+1.67 dB*）。
  - *算子与系统协同（Kernel & Infra）*：设计去均值 FP8 残差 + 块均值 + 象限均值的常驻压缩 KV 池（`ZingCompressedKVStore`），单块每头显存由 `49,152 B` 降至 `18,248 B`（*KV 显存节省 62.85%*：480p SP1 由 `41.8 GiB` 降至 `15.5 GiB`，736p SP2 由 `40.5 GiB` 降至 `15.0 GiB/rank`）；在 SM90 Triton 内核外利用 Tensor Core 预计算 `qk_dc_table` 消除 FP8 WGMMA 寄存器布局冲突，产出 18 条 `QGMMA.E4M3` 指令且 *0 寄存器溢出*，单算子耗时由 `1.842 ms` 降至 *`0.646 ms`（提速 2.85x，逻辑 MBU 74.47%）*；H200 短序列稳态吞吐达 *23.82 FPS（较 BF16 +26.2%）*，11 分钟超长序列达 *20.40 FPS（端到端提速 26.7% / 节省 210.9s）*。
- *SM90 FP8 GEMM 路由、算子融合与 FA4 调优*：实现针对 Zing H3 的 SM90 原生 FP8 GEMM 调优与动态路由（峰值达 *71.7% MFU*）；设计 Fused Pre-Linear RMSNorm + Indexed Scale/Shift + Online Per-Token FP8 Quant 及 Gated Residual / SwiGLU 融合算子；接入 FlashAttention-4 SplitKV 调优与按 forward 元数据复用，在 L20D 1100W 功耗墙持续负载下实现 SP2/3 调用 *+14.03%* 提升，单卡 denoise 中位延迟降低 *5.85%–7.62%*。
- *端到端分离池化部署、独立 Lynn Decoder 拆分、离线 AdaLN 与整 Chunk CUDA Graph*：
  - *Encoder / Denoiser / Remote Lynn Decoder 独立 GPU 解耦（D1/D2 拆分）*：将 VAE 解码、Fused preparation、RGB `uint8` 量化、D2H 拷贝与 `/dev/shm` 共享内存发布剥离至独立 Decoder GPU 组，并支持 Lynn tile 按 SP rank 切分；在 `1280×736 (E1+SP2+D1)` 下稳态吞吐由 `15.14` 提升至 *`16.24 FPS`（+7.29%）*，DiT 稳态耗时由 `525.66` 降至 *`487.13 ms/chunk`（节省 38.53 ms/chunk，-7.33%，合 243.57 ms/latent）*。
  - *离线 `weights_free` AdaLN 预计算表机制*：将 96,768 维动态 TimeMLP 调制离线化为常驻预计算表并加入软硬件契约校验，消除运行期权重驻留与投影开销。
  - *`after_step0` 延迟 KV 淘汰与 SP>1 整 Chunk CUDA Graph*：通过 `after_step0` 跨 chunk 延迟 KV 淘汰固定图输入 shape，实现 SP>1 去噪前向整 chunk CUDA Graph 默认重放；定位并修复 `cuda_graph_max_bs` 图捕获显存回归，采用 `max_graphs=16` 世代整池排空策略与捕获期禁用 GC，彻底根治显存碎片泄漏。

#work(
  title: "LLM 高性能推理框架调度及高性能算子优化研究",
  location: "北京, 中国",
  company: "腾讯混元 AI Infra 部",
  dates: dates-helper(start-date: "Jul 2026", end-date: "Sep 2026"),
)
- 负责 LLM 推理服务的推理框架调度优化与高性能算子优化，开发 Blackwell 架构 BF16 paged KV attention prefill、`nvfp4_blockwise_gemm`、`per_group` MXFP8 量化、`fused_silu_per_token_quant` 融合算子等
- 主要开源代码库：#link("https://github.com/Tencent/hpc-ops")[Tencent/hpc-ops]，部分贡献代码已并入 vLLM main 分支

== 开源贡献与项目经验

#project(
  name: link("https://github.com/sgl-project/sglang-omni")[SGLang-Omni 开源社区贡献（Issue #2941 P3.6 / P7.4、#3583 & PR #3782）],
  role: text[Algorithm–Infra Co-design],
  dates: "Nov 2026",
)
- *MiniCPM-o 4.5 双流异步投机解码重叠与请求级 Hidden FIFO（Issue #2941 P7.4 / PR #3782）*：针对 `Audio Encoder -> Thinker -> Talker -> Code2Wav` 多阶段流水线中 DSpark 投机草稿器在 `event_loop_spec_overlap` 双流异步重叠（$W_max = 2$）下的跨步覆写（WAW）竞态，将单槽缓存升级为请求级双深双端队列（`dict[str, deque[torch.Tensor]]`），过滤 Chunked Prefill 非末块冗余入队，并在 `OmniScheduler` 注入流同步钩子与 Decode 撤回清理（`on_request_retracted`），移除投机解码强制回退同步模式的限制；在单卡 H100 PCIe 上通过 P0.6 ABBA 全并发（`c1..c64`）评测与 33 次强制撤回压力验证（100% 产出有效 WAV 音频），消除 `MMSU Text Out` 在 `c16` 下的同步投机 `0.90x` 吞吐倒退（恢复至 *`1.00x`*，p95 延迟降低 *`9.6%`*），并在 `MMSU Speech Out`（`c2..c32` 达 *`1.25x–1.40x`*，p95 降低 *`16.7%–39.0%`*）与 `Seed-TTS EN Speech Out`（`c1..c32` 达 *`1.26x–1.46x`*，全并发无退化）上实现端到端无损加速。
- *Qwen3-TTS 双层自回归语音模型投机解码算法设计与 H100 双维度无损证伪（Issue #2941 P3.6 & #3583，获 Maintainer 采纳定论）*：独立负责官方 Roadmap *P3.6*，在 H100 PCIe 上完成 `Qwen3-TTS-12Hz-1.7B` 端到端瓶颈拆解（双层 AR 占比 *>85%*：28 层 `Main Talker` `2.45 ms/frame` 占 48%，15 步 `CodePredictor` `2.60 ms/frame` 占 52%），训练内层分块草稿器（Route A）与外层 1.1M 参数 `Gated-ResMLP` 跨帧草稿器（Route B，CUDA Graph 耗时 `0.082 ms`，对语义主码 $c_(t+1,0)$ 达 *85.44%* Top-50 Shaped 无损接受率）；证明内层 Route A 受深层声学残差码高熵（$g >= 5, H_g > 5.2 "bits"$）制约致 15 层全过率仅 `~1.2%`、无损平均仅接受 *2.46 步*（*0.330x 负加速*），揭示外层 Route B 受跨帧双向因果锁制约致严格无损联合推进退化至 *`~1.01` 帧/轮* 且近似残差引发 *21.9%–38.5% TV 漂移*，从信息论与系统执行流双维度确立双层 RVQ 自回归结构无损投机的 *No-Go 架构定论*。

#work(
  title: "第十届华为 ICT 大赛中国总决赛挑战赛",
  location: "华中科技大学, 武汉",
  company: "华为",
  dates: dates-helper(start-date: "Apr 2026", end-date: "Apr 2026"),
)
- 在 64 支高校队伍中排名第 8 进入决赛，决赛 16 支队伍中总排名第 3 获*全国一等奖*，作为唯一开发者获*最高算子迁移优化奖*（全场比赛最佳优化）
- 代表项目：#link("https://github.com/yiwen-cai/fused_add_rmsnorm_kernel")[*fused_add_rmsnorm_kernel*]（Triton 融合 RMSNorm + Add 算子，多行 program 映射与 `hidden_size` 分档调度，相比 PyTorch 提速 *19.28x*）；#link("https://github.com/yiwen-cai/xllm-ictfinal")[*xllm-ictfinal*]（基于 xllm 引擎完成 Qwen3.5 推理适配，实现 GDN 线性注意力与 MTP 投机采样，解决 KVCache 占用异常并打通 32k 长序列推理）

#project(
  name: link("https://github.com/yiwen-cai/cuda-gemm")[cuda-gemm],
  dates: "Jun 2025",
)
- 从零手写 8 个 CUDA GEMM kernel（覆盖 FP32 CUDA Core 与 FP16 Tensor Core WMMA），渐进引入 Shared Memory Tiling、Bank Conflict Avoidance、Double Buffering、`cp.async` 异步拷贝；SGEMM v3 达 cuBLAS 的 *95.1%*（4096²: 37,898 GFLOPS; 8192²: 36,799 GFLOPS），HGEMM v3 Tensor Core 达 *213 TFLOPS* (4096²)，含 Nsight Compute/Systems 性能剖析

#project(name: link("https://github.com/yiwen-cai/fastGNN")[FastGNN], dates: "May 2026")
- 基于 H100 Tensor Core 的全图 GNN 训练系统，自研稀疏算子后端（Voltrix block-sparse SpMM、Fused3S sparse attention、Rabbit 节点重排），端到端相对 DGL/PyG 提速 *1.8x–3.3x*

#work(
  title: "新加坡国立大学夏令营 NUS Summer Workshop",
  location: "新加坡",
  company: "National University of Singapore",
  dates: dates-helper(start-date: "Jul 2024", end-date: "Aug 2024"),
)
- 云计算与大数据课程成绩 *A+*（排名*第 1 名*）；完成云原生多 Agent 协同与 Kubernetes 调度平台 #link("https://github.com/uplion")[*UPLION*]

== 奖项
#project(
  name: "第十届华为 ICT 大赛全国总决赛挑战赛",
  role: text[一等奖，最高算子迁移优化奖],
  dates: "Apr 2026",
)

#project(
  name: "NUS Summer Workshop",
  role: text[A+（第 1 名）],
  dates: "Jul 2024",
)

#project(
  name: "第十五届蓝桥杯全国软件和信息技术专业人才大赛北京赛区",
  role: text[二等奖],
  dates: "Apr 2024",
)

#project(name: "北京邮电大学校级奖学金（大一、大二、大三学年）", dates: "2023 – 2025")

== 技能
- *编程技能*: Python, C/C++, CUDA, Triton, Rust, Go
- *研究方向*: GPU 高性能算子、大模型训推系统与 Algorithm–Infra Co-design（算子–算法–推理系统协同设计）、分布式与高性能计算
