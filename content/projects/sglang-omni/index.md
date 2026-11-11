+++
title = 'SGLang-Omni：多模态语音大模型投机解码与异步重叠调度'
date = 2026-11-10
draft = false
summary = '面向 sgl-project/sglang-omni 的 Algorithm–Infra Co-design 开源贡献：（1）实现 MiniCPM-o 4.5 双流异步投机解码重叠与请求级双深 Hidden FIFO（PR #3782），消除 WAW 竞态与 c16 同步投机倒退，在 H100 上实现 Speech Out 1.25×–1.46× 无损加速；（2）完成 Qwen3-TTS 双层 RVQ 自回归模型内/外层投机草稿器训练与信息论/系统流双维度无损证伪（Issue #2941 P3.6 & #3583）。'
tags = ['SGLang', 'LLM 推理', '投机采样', 'Algorithm–Infra Co-design']
externalUrl = 'https://github.com/sgl-project/sglang-omni/pull/3782'
showReadingTime = false

[build]
  render = false
  list = "local"
+++
