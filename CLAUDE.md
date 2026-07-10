# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

中文技术博客，Hugo + Blowfish 主题，部署到 GitHub Pages（`yiwen-cai.github.io`，用户主页仓库，根路径部署，无 `base`）。

## 构建与本地预览

- 本机 Hugo 版本必须与 CI 对齐：**0.164.0 extended**（Blowfish 编译 SCSS 需 extended 版）。
- 本地预览：`hugo server`（`baseURL` 不写死在 config，本地默认 `http://example.org/` 不影响开发）。
- 干净构建验证（发布笔记前的必跑检查）：`rm -rf public resources && hugo --quiet`，退出码 0 即通过。
- 生产构建参数（CI 用，本地一般不需要）：`hugo --gc --minify --baseURL <url>`。
- 主题是 git submodule（`themes/blowfish`）。克隆/拉取后要 `git submodule update --init --recursive`。

部署全自动：push 到 `main` 触发 `.github/workflows/hugo.yml`（官方 `deploy-pages` 方案）。CI 会 `checkout --submodules --fetch-depth 0`（`enableGitInfo` 需要完整 git 历史取文章最后修改时间）。

## 配置结构

配置拆分在 `config/_default/`，Hugo 自动合并（中文单语言站，无多语言配置）：

- `hugo.toml` — 站点级。`hasCJKLanguage = true`（中文字数统计/摘要）、`enableGitInfo = true`、`mainSections` 在 params 里定为 `["notes"]`（首页「最近文章」只聚合笔记，项目走导航）。
- `params.toml` — 主题行为。首页 `layout = "profile"`，`defaultAppearance = "light"` + `autoSwitchAppearance = true`。
- `markup.toml` — KaTeX 关键：`passthrough` 已配 `$...$` / `$$...$$` / `\(...\)` / `\[...\]` 行界符，`goldmark.renderer.unsafe = true`。
- `languages.zh-cn.toml` — 中文 locale、日期格式、作者信息（Profile 首页头像/标语/简介/社交链接）。
- `menus.zh-cn.toml` — 顶部导航：首页 / 项目 / 笔记 / 关于 / 标签。

## 内容组织

两类内容，front matter 都是 **TOML（`+++` 包裹）**：

- **笔记** `content/notes/<分类>/<slug>/index.md`（每篇一个目录，`index.md` 是叶子页）。分类目录（已建好）：`cuda` / `triton` / `systems` / `papers` / `llm-inference` / `research`。
- **项目** `content/projects/<slug>/index.md`，用 `externalUrl = "https://github.com/..."` 指向 GitHub，并加 `[build] render = false, list = "local"`（项目卡片外链，不渲染站内页）。

### front matter 必填项（笔记）

```toml
+++
title = '标题'
date = 2026-XX-XX
draft = false
summary = '手写摘要（必须！summaryLength=0，不写会灌入整段正文）'
tags = ['tag1', 'tag2']
showReadingTime = true
showTableOfContents = true   # 长文开，目录自动从 H2-H4 生成
+++
```

## 关键渲染规则（容易踩坑）

- **KaTeX**：含公式的笔记，必须在正文**开头**（front matter 后、第一段前）加一行 `{{< katex >}}` 作为加载开关。不加则公式不渲染。
- **Mermaid 已弃用**：本站 Mermaid 渲染效果差，不要用。流程图/对比图用 ` ```text ` 文字示意图、Markdown 表格或有序列表；矩阵型对比一律表格。
- **正文宽度**：已在 assets 配为 900px，代码块/表格原生 Markdown 即可，无需额外处理。
- 自定义样式在 `assets/css/custom.css`，KaTeX 渲染逻辑在 `assets/js/katex-render.js`。无 `layouts/` 覆盖，全部继承 Blowfish。

## 发布笔记

有专门的 skill：**`publish-note`**（`.claude/skills/publish-note/SKILL.md`）。用户说「发布这篇笔记」「把 XX 笔记上线」等时走它。核心是：清理 Obsidian 残留（YAML→TOML、删 `[[ ]]` 双链和本地路径）、选分类、加 `{{< katex >}}`、构建验证（`grep -cE '\[\[|Slide [0-9]|配色|\.pptx'` 应为 0）、commit + push 部署。

## 项目背景资料

- `memory-bank/` — 需求、决策、进度等背景文档（active-context / decision-log / product-context / progress / system-patterns / user-profile）。
- `personal-website-requirements.md`、`personal-website-template-research.md` — 站点需求与主题选型记录。
