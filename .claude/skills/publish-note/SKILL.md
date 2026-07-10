---
name: publish-note
description: 将 Markdown 笔记发布到本项目的 Hugo + Blowfish 网站（yiwen-cai.github.io）。负责完整的发布流程：清理 Obsidian/PPT 残留、转 Hugo front matter、选目录、加 KaTeX/Mermaid 触发、构建验证、git commit + push 部署。触发场景：用户说"发布这篇笔记""把这篇 md 发到网站""把 wiki 里的 XX 笔记上线""加一篇新笔记"等。
---

# 发布笔记到网站

本项目是 Hugo + Blowfish 主题的静态站，部署到 GitHub Pages（`yiwen-cai.github.io`）。笔记源文件通常是用户 wiki 里的 Obsidian `.md` 或其他 Markdown，发布前必须清理并转成 Hugo 格式。

## 目录与路径

- 笔记放 `content/notes/<分类>/<slug>/index.md`（每篇一个目录，`index.md` 是叶子页）
- 分类目录（已存在）：`cuda` / `triton` / `systems` / `papers` / `llm-inference` / `research`
- slug 用短 kebab-case 英文（如 `flashattention`、`memory-coalescing`），是 URL 的一段
- 选分类：CUDA/算子→`cuda`，Triton→`triton`，分布式/并行→`systems`，单篇论文解读→`papers`，推理/综述→`llm-inference`

## front matter（TOML，`+++` 包裹）

每篇笔记必须有：

```toml
+++
title = '标题（中文即可，论文名/方法名保留英文）'
date = 2026-XX-XX
draft = false
summary = '一两句话摘要（手写，必须——因 hugo.toml 设 summaryLength=0，否则会灌入整段正文）'
tags = ['tag1', 'tag2']
showReadingTime = true
showTableOfContents = true
+++
```

- `date`：用笔记原日期或今天
- `tags`：中英文混用均可（如 `['KV Cache', 'LLM 推理']`），会自动生成 `/tags/` 聚合页
- 长文（含多个 H2/H3）开 `showTableOfContents = true`，目录自动从 H2-H4 生成

## 清理 Obsidian / PPT 残留（关键步骤）

源文件常见三种污染，发布前必须处理：

1. **YAML frontmatter → 转 TOML**：wiki 笔记用 `---` YAML（含 `created/updated/type/confidence/sources`），发布时换成上面的 `+++` TOML。`sources`（本地路径）**删除**。
2. **Obsidian 双链 `[[xxx]]`**：转成纯文本（论文标题）或直接删除。决不能让 `[[ ]]` 出现在发布稿里。
3. **本地路径 / 私人标注**：`raw/papers/xxx.pdf`、`/Users/...`、`沙盒复现见 analysis/xxx.py`、`调研人：蔡逸文`、草稿标注 `⚠️ 待确认`——一律删除或改成中性描述。
4. **PPT 残留**（若源是 PPT 指令文档）：`Slide N`、`配色 #xxx`、`字体 28pt`、`布局/设计/Takeaway/演讲者备注`、`.pptx` 输出路径、`答辩口径`——全部删除，把「演讲者备注」里的技术补充融进正文段落。

## 公式与图表渲染

- **KaTeX 公式**：含 `$...$` 或 `$$...$$` 的笔记，在正文**开头**（front matter 之后、第一个内容之前）加一行 `{{< katex >}}`。这是触发 Blowfish 加载 KaTeX 的开关（按需加载，不加就不渲染）。markup.toml 已配 `$...$` 行内定界符，assets/js/katex-render.js 已覆盖好。
- **代码块和表格**：原生 Markdown，自动受益于正文加宽（已配 900px），无需额外处理。
- **图示**：**不要用 Mermaid**（本站 Mermaid 渲染效果差，已弃用）。流程图/对比图用：代码块文字示意图（` ```text ` 包裹）、Markdown 表格、有序列表。矩阵型对比一律用表格。

## 发布流程

1. **读源文件**，判断：分类、slug、是否含公式、需要清理什么。
2. **建文件** `content/notes/<分类>/<slug>/index.md`，写 front matter + 清理后的正文。正文以 `> 引用块` 标注来源（如「基于 Stanford CS336 Lecture 5 整理」或「原论文：xxx，arXiv:xxx」）开头。
3. **构建验证**：`rm -rf public resources && hugo --quiet`，确认退出码 0、页面生成。
4. **确认无残留**：`grep -cE '\[\[|Slide [0-9]|配色|\.pptx' <文件>` 应为 0。
5. **commit + push**：
   ```
   git add -A && git commit -m "feat: 新增笔记「<标题>」

   <一两句话说明来源和清理>
   "
   git push
   ```
6. push 后 GitHub Actions 自动构建部署（约 1-2 分钟），线上即更新。

## 验证部署

push 后用 `gh run list --repo yiwen-cai/yiwen-cai.github.io --limit 1` 看 Actions 状态。已知非阻塞警告（不用管）：`Module blowfish not compatible with Hugo 0.164`、`Node.js 20 deprecated`。

如需本地预览，`hugo server` 后用 Claude Browser MCP（preview_start + preview_eval 查 `getComputedStyle` 或 screenshot）实测渲染——尤其首次验证 KaTeX/Mermaid 是否真渲染。

## 论文引用约定

- 正文方法名首次出现用 Markdown 链接指向 arXiv：`[SnapKV](https://arxiv.org/abs/2404.14469)`，不写完整引用串（`Li et al., arXiv:..., 2024` 是 PPT 页脚残留，删）。
- 末尾可选 `## 参考` 分类列表。**无 arXiv 号的论文只写「方法名 — 年份」，绝不编造链接**。
- 实验/报告数据若非自己复现，必须声明（如「资料口径，非复现实验」），作独立 `>` 引用块。

## 约束

- **不碰 `themes/blowfish/` 源码**——所有定制走 `assets/`（css/js）、`config/`、`layouts/`（项目根覆盖）。
- slug 用英文，正文用中文（避免中文 URL 编码问题）。
- 每篇笔记正文坚持标准 Markdown，专属 shortcode（`{{< katex >}}`）仅用于触发渲染。
