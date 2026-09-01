# AGENTS.md

面向所有 AI 编码助手的项目说明。本文件是唯一事实来源，`CLAUDE.md` 只做指引。

中文技术博客，Hugo + Blowfish 主题，部署到 GitHub Pages（`yiwen-cai.github.io` 是用户主页仓库，根路径部署，无子路径 `base`）。内容以 GPU 算子优化、LLM 训推优化方向的技术笔记和项目卡片为主。

## 环境准备

- Hugo **0.164.0 extended**，必须与 CI 对齐。Blowfish 要编译 SCSS，非 extended 版会构建失败。
- macOS 上用 `brew install hugo` 安装（当前 brew stable 恰好是 0.164.0，装出来是 `v0.164.0+extended+withdeploy`）。**不要去 GitHub Release 找 darwin 包**——Hugo 早已不再发布 macOS 二进制，那里只有 linux/windows/bsd。要装指定版本得用 `brew extract` 或从源码编译。
- 主题是 git submodule（`themes/blowfish` → nunocoracao/blowfish，当前锁在 v2.104.0）。**克隆或拉取后必跑** `git submodule update --init --recursive`，否则 `themes/blowfish` 是空目录、构建直接报错。
- 不需要单独装 dart-sass，Hugo extended 自带的够用。也无 Node 依赖（没有 `package.json`），CI 里的 `npm ci` 步骤是条件跳过的。

## 常用命令

| 目的 | 命令 |
| --- | --- |
| 本地预览 | `hugo server`（`baseURL` 不写死在 config，本地走默认 `http://example.org/`，不影响开发） |
| 干净构建验证（发布前必跑） | `rm -rf public resources && hugo --quiet`，退出码 0 即通过 |
| 复现 CI 构建 | `hugo --gc --minify --baseURL https://yiwen-cai.github.io/` |
| 查部署状态 | `gh run list --repo yiwen-cai/yiwen-cai.github.io --limit 1` |

全量构建约 1–2 秒，产出 77 个页面、3.4 MB。数量明显对不上就是哪里出问题了。

部署全自动：push 到 `main` 触发 `.github/workflows/hugo.yaml`（官方 `upload-pages-artifact` + `deploy-pages` 方案）。CI 用 `submodules: recursive` + `fetch-depth: 0`（`enableGitInfo` 需要完整历史取文章最后修改时间），时区固定 `Asia/Shanghai`。

**唯一的已知警告，每次构建都会出现，忽略即可**：

```
WARN  Module "blowfish" is not compatible with this Hugo version: 0.158.0/0.163.3 extended
```

主题声明的兼容上限低于当前 Hugo 版本而已，不影响构建（退出码仍是 0）。CI 里另有 `Node.js 20 deprecated`，同样无害。

## 配置结构

配置拆在 `config/_default/`，Hugo 自动合并。单语言中文站，不要引入多语言配置。

- `hugo.toml` — 站点级。`hasCJKLanguage = true`（中文字数统计/摘要）、`enableGitInfo = true`、`summaryLength = 0`（**所以每篇必须手写 `summary`**）、`pagerSize = 100`。
- `params.toml` — 主题行为。`mainSections = ["notes"]`（首页「最近内容」只聚合笔记，项目走导航）、首页 `layout = "profile"`、`defaultAppearance = "light"` + `autoSwitchAppearance = true`、全站 `showComments = true`、全站 `showTableOfContents = false`（长文按篇在 front matter 里打开）。
- `markup.toml` — KaTeX 的关键：`goldmark.extensions.passthrough` 已配 `$...$` / `$$...$$` / `\(...\)` / `\[...\]` 四种定界符，`goldmark.renderer.unsafe = true`，TOC 取 H2–H4。
- `languages.zh-cn.toml` — locale、日期格式 `2006年1月2日`、作者信息（Profile 首页的头像/标语/简介/社交链接都在这里）。
- `menus.zh-cn.toml` — 顶部导航：首页 / 项目 / 笔记 / 关于 / 标签。

## 内容组织

两类内容，front matter 一律 **TOML（`+++` 包裹）**，不要用 YAML。

**笔记** `content/notes/<分类>/<slug>/index.md`，每篇一个目录，`index.md` 是叶子页。现有分类：`cuda` / `triton` / `systems` / `papers` / `llm-inference`。新增分类要同时建 `content/notes/<分类>/_index.md`（`title` + `date` + `draft` + 一句话说明）。slug 用短 kebab-case 英文，正文用中文。

```toml
+++
title = '标题'
date = 2026-XX-XX
draft = false
summary = '手写摘要，必填——summaryLength = 0，不写会灌入整段正文'
tags = ['tag1', 'tag2']
showReadingTime = true
showTableOfContents = true   # 长文开，目录自动从 H2–H4 生成
+++
```

**项目** `content/projects/<slug>/index.md`，用 `externalUrl` 直链 GitHub，并加 `[build] render = false, list = "local"`，只出卡片、不生成站内详情页。列表页的展示开关走 `content/projects/_index.md` 的 `[cascade]`。

```toml
+++
title = 'CUDA GEMM Kernels'
date = 2025-06-01
draft = false
summary = '一句话讲清做了什么、指标多少。'
tags = ['CUDA', 'Tensor Core']
externalUrl = 'https://github.com/yiwen-cai/cuda-gemm'
showReadingTime = false

[build]
  render = false
  list = "local"
+++
```

## 渲染规则（容易踩坑）

- **KaTeX 开关**：含公式的笔记，必须在正文最开头（front matter 之后、第一段之前）单独一行写 `{{< katex >}}`。Blowfish 按需加载 KaTeX，不写这行公式不渲染。
- **不要用 Mermaid**：本站 Mermaid 渲染效果差，已弃用。流程图用 ` ```text ` 文字示意图，对比/矩阵型内容一律用 Markdown 表格或有序列表。
- **正文与目录宽度**：`assets/css/custom.css` 已优化全站容器与笔记详情页排版：全站容器放宽至 1440px、正文保持黄金阅读宽度（~54rem/864px，解开 Blowfish 原版两层 `65ch` 限制）、TOC 目录放宽到 18.5rem–23rem（296px–368px），并优化了目录缩进、断词与间距，完整展示长标题。代码块和表格写原生 Markdown 即可，不需要额外处理。
- **单 `$` 行内公式**：`assets/js/katex-render.js` 覆盖了主题同名文件，追加了 `$...$` 定界符。定界符顺序里 `$` 必须放最后，否则 `$$` 会被拆成两个 `$`。验证覆盖是否生效：构建后 `rg -o 'left:"[^"]{1,3}"' public/js/main.bundle.min.*.js` 应出现 4 个定界符且 `$` 在末位（主题原版只有 3 个）。
- **评论区**：`layouts/partials/comments.html` 覆盖了主题 partial，接的是 Giscus，并有一段 JS 跟随站点亮/暗模式切换 iframe 主题。这是目前**唯一**的 `layouts/` 覆盖。

## 硬约束

- **绝不修改 `themes/blowfish/`**。所有定制走 `assets/`（css/js）、`config/`、项目根 `layouts/` 覆盖三条路。
- slug 用英文（避免中文 URL 编码问题），正文用中文。
- 正文坚持标准 Markdown，shortcode 只用 `{{< katex >}}` 这一个触发器。
- 引用论文时，方法名首次出现用 Markdown 链接指向 arXiv；**没有 arXiv 号就只写「方法名 — 年份」，不要编造链接**。非自己复现的实验数据必须声明口径。

## 发布笔记

有专门的流程文档：`.claude/skills/publish-note/SKILL.md`（Claude Code 里注册为 `publish-note` skill，其他工具直接当 checklist 读）。用户说「发布这篇笔记」「把 XX 笔记上线」时走它。核心步骤：源文件常是 Obsidian 笔记，要清理残留（YAML→TOML、删 `[[双链]]` 和本地路径、删 PPT 痕迹）→ 选分类和 slug → 补 `{{< katex >}}` → 干净构建验证 → `grep -cE '\[\[|Slide [0-9]|配色|\.pptx' <文件>` 应为 0 → commit + push。

## Git 约定

- 直接在 `main` 上提交并 push，没有 PR 流程，push 即部署。
- commit message 用 conventional commits 前缀 + 中文描述，例如 `feat: 新增笔记「FlashAttention」`、`style: 笔记详情页正文加宽到 ~900px`、`chore: 新增 CLAUDE.md 与本地预览 launch.json`。
- `/public/`、`/resources/_gen/` 已在 `.gitignore` 中，不要提交构建产物。

## 仓库里的其他文件

- `personal-website-requirements.md`、`personal-website-template-research.md` — 站点需求与主题选型的调研记录，`custom.css` 的设计依据出自后者。改视觉风格前值得先读。
- `memory-bank/` — 早期 Roo Code / Cline 工作流留下的目录，除 `user-profile.md` 外基本还是模板占位内容，不要当作可信上下文。
- `.clinerules-*` — 同样是 Roo Code 模式配置遗留，与当前工作流无关，除非用户明确提起，否则忽略。
