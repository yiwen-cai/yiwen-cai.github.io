+++
title = 'MoE KV Cache 优化'
date = 2026-07-08
draft = false
summary = '面向 MoE 模型推理的 KV Cache 内存与吞吐优化实验。'
tags = ['LLM', '推理优化', 'MoE']
+++

## 问题背景

MoE（Mixture of Experts）模型在推理时，专家路由的稀疏性使 KV Cache 的访问模式与稠密模型不同，存在显存浪费与带宽利用不充分的问题。

## 技术栈

- PyTorch、Triton
- vLLM（参考实现）

## 个人工作

- 分析了 MoE 路由下的 KV 访问热点
- 设计并实现了一种按专家分组的缓存策略

## 实验结果

<!-- TODO: 填入真实实验数据 -->

| 指标 | 基线 | 优化后 |
|---|---|---|
| 显存占用 | 100% | TBD |
| 吞吐 | 1.0× | TBD |

## 代码

详见 [GitHub 仓库](https://github.com/drink-less-milktea)（链接待替换）。
