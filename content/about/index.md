+++
title = '关于'
date = 2026-07-10
draft = false
showDate = false
showReadingTime = false
showWordCount = false
showAuthor = false
showPagination = false
showRelatedContent = false
showTaxonomies = false
+++

我是蔡逸文（Yiwen Cai），北京邮电大学计算机技术专业硕士在读。我的研究兴趣集中在**算法与硬件之间的那一层**——GPU 高性能算子、大模型训推系统与 Algorithm–Infra Co-design。

## 教育经历

- **北京邮电大学 · 硕士**（2026.09 – 2029.06）计算机技术（推免）
  - 实验室：ParCIS 并行计算与智能系统实验室
  - 导师：李士刚 教授
- **北京邮电大学 · 本科**（2022.09 – 2026.06）计算机科学与技术
  - GPA 3.76 / 4.0 · Rank 59/389 · CET6 552

## 实习经历

- **北京踊跃智能（Loopit） · AI Infra 实习生**（2026.10 – 至今）—— 视频生成大模型推理系统与 Algorithm–Infra Co-design
  - 负责 Zing H3 实时视频生成模型的全 FP8 三档稀疏注意力与常驻压缩 KV 池、SM90 FP8 GEMM / FA4 算子融合调优，以及 Encoder–DiT–Decoder 分离池化部署与整 Chunk CUDA Graph 优化。
- **腾讯混元 AI Infra**（2026.07 – 2026.09，实习已结束）—— LLM 推理框架调度与高性能算子优化
  - 开发 Blackwell 架构 attention / MXFP8 & NVFP4 量化 / 融合类算子，主要开源仓库 [Tencent/hpc-ops](https://github.com/Tencent/hpc-ops)，部分贡献已合入 vLLM main 分支。

## 研究方向

- GPU 高性能算子（CUDA / Triton）
- 大模型训推系统与 Algorithm–Infra Co-design
- 分布式计算与云计算、高性能计算

## 代表性工作

- **[SGLang-Omni](https://github.com/sgl-project/sglang-omni) 开源贡献**——实现 MiniCPM-o 4.5 双流异步投机解码重叠与请求级 Hidden FIFO（[PR #3782](https://github.com/sgl-project/sglang-omni/pull/3782)，Speech Out 达 1.25x–1.46x 无损加速）；完成 Qwen3-TTS 双层 RVQ 自回归投机解码算法设计与 H100 双维度无损证伪（[Issue #3583](https://github.com/sgl-project/sglang-omni/issues/3583)，获社区 Maintainer 采纳）。
- **华为 ICT 大赛全国总决赛一等奖**（最高算子迁移优化奖）——64 进 8、决赛第 3；作为队伍唯一开发者获算子迁移优化赛题所有队伍最佳。
- **[cuda-gemm](https://github.com/yiwen-cai/cuda-gemm)**——从零手写 CUDA GEMM；SGEMM 达 cuBLAS 95.1%，HGEMM Tensor Core 达 213 TFLOPS。
- **[FastGNN](https://github.com/yiwen-cai/fastGNN)**——基于 H100 Tensor Core 的全图 GNN 训练系统；相对 DGL/PyG 加速 1.8x–3.3x。
- **NUS Summer Workshop**——新加坡国立大学夏令营，云计算与大数据课程 A+（排名第 1 名）。
- **第十五届蓝桥杯**北京赛区二等奖。
- 详见 [项目](../projects/) 页与下方简历。

## 简历

- [中文简历（PDF）](/resume/chinese.pdf)
- [English Resume (PDF)](/resume/main.pdf)

## 联系

- GitHub：[@yiwen-cai](https://github.com/yiwen-cai)
- 知乎：[我的知乎主页](https://www.zhihu.com/people/cywww-14/)
- 邮箱：caiyiwen.cs@foxmail.com
