+++
title = 'Fused Add + RMSNorm Triton Kernel'
date = 2026-04-01
draft = false
summary = '用 Triton 融合 Add 与 RMSNorm，多行 program 映射 + hidden_size 分档调度，相比 PyTorch 基线 19.28× 加速。'
tags = ['Triton', '算子优化']
externalUrl = 'https://github.com/yiwen-cai/fused_add_rmsnorm_kernel'
showReadingTime = false

[build]
  render = false
  list = "local"
+++
