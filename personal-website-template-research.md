# 中文个人网站模板与技术栈调研

> 调研日期：2026-07-10  
> 目标场景：中文个人网站，包含个人信息、项目展示和技术笔记；视觉倾向简约、精致；关注社区规模、维护稳定性和长期可升级性。  
> 本文只呈现调研结果与取舍，不给出最终选型结论。

## 1. 需求边界

本次调研针对的不是单纯博客，也不是以论文和学术履历为中心的学术主页，而是由三类内容共同组成的个人网站：

1. **个人信息**：个人定位、教育经历、研究与工程方向、技能、社交链接和联系方式。
2. **项目展示**：项目概览、技术栈、个人贡献、实验结果、代码仓库和详情页。
3. **技术笔记**：CUDA、Triton、LLM 推理优化、系统实验和论文阅读等 Markdown 内容。

附加要求：

- 网站主要使用简体中文。
- 中文正文、中英文混排、代码块和公式应具有良好的阅读体验。
- 视觉风格应克制、留白充分，避免过强的博客侧栏感或营销落地页风格。
- 能以较低成本部署到 GitHub Pages。
- 模板应有一定的社区采用规模，且上游仍在维护。
- 站点内容应尽量保持可迁移，避免过度绑定专属组件或短代码。

## 2. 调研方法

调研分为 Hugo、Astro 和独立风险审计三条路线，统一检查以下维度：

| 维度 | 检查内容 |
|---|---|
| 需求原生度 | 是否原生支持个人首页、项目、笔记和关于页面 |
| 中文适配 | 是否内置简中界面、中文文档、CJK 排版或 i18n |
| 社区规模 | Stars、Forks、公开案例和衍生项目 |
| 维护活性 | 最近有效提交、Release、Issue/PR 状态 |
| 发布纪律 | 是否有版本号、Release notes 和升级说明 |
| 升级边界 | 主题依赖还是复制源码；是否支持本地覆盖而不修改上游 |
| 内容迁移 | Markdown、front matter、shortcode、MDX 和 schema 的绑定程度 |
| 部署 | GitHub Pages 文档、静态构建和子路径支持 |
| 许可证 | MIT、GPL 等许可证对修改和复用的影响 |

Stars 和 Forks 只能作为采用规模的近似指标，不能等同于真实活跃用户数。因此，本文同时参考发布节奏、提交历史、文档完整性和升级方式。

## 3. 候选项目快照

以下数据来自 2026-07-10 的官方 GitHub 仓库或 GitHub API 快照。

| 模板 | 技术栈 | Stars | Forks | 最近推送 | 最新正式版本 | 许可证 |
|---|---|---:|---:|---|---|---|
| [Blowfish](https://github.com/nunocoracao/blowfish) | Hugo + Tailwind CSS | 2,835 | 734 | 2026-07-07 | v2.104.0，2026-07-02 | MIT |
| [Congo](https://github.com/jpanther/congo) | Hugo + Tailwind CSS | 1,641 | 416 | 2026-06-25 | v2.14.0，2026-05-23 | MIT |
| [PaperMod](https://github.com/adityatelange/hugo-PaperMod) | Hugo | 13,744 | 3,412 | 2026-05-10 | v8.0，2024-11-03 | MIT |
| [Hugo Profile](https://github.com/gurusabarish/hugo-profile) | Hugo + Bootstrap | 1,082 | 578 | 2026-02-11 | v4.06，2026-02-08 | MIT |
| [AstroPaper](https://github.com/satnaing/astro-paper) | Astro + TypeScript + Tailwind + MDX | 4,811 | 1,016 | 2026-07-07 | v6.1.0，2026-06-06 | MIT |
| [Dante](https://github.com/JustGoodUI/dante-astro-theme) | Astro + Tailwind + MDX | 501 | 323 | 2026-04-08 | 1.0.0，2025-11-11 | GPL-3.0 |
| [Retypeset](https://github.com/radishzzz/astro-theme-retypeset) | Astro + UnoCSS + MDX | 677 | 195 | 2026-04-12 | v1.0.0，2025-08-07 | MIT |
| [Fuwari](https://github.com/saicaca/fuwari) | Astro + Tailwind + Svelte | 4,769 | 1,236 | 2026-03-10 | 无正式 Release | MIT |
| [Astro Sphere](https://github.com/markhorn-dev/astro-sphere) | Astro + Tailwind + SolidJS | 678 | 182 | 2025-06-16 | v1.0.1，2024-04-03 | MIT |

## 4. Hugo 路线

### 4.1 Blowfish

Blowfish 官方定位为 **Personal Website & Blog Theme for Hugo**，不是纯博客模板。

核心能力：

- 提供 `Profile`、`Page`、`Hero`、`Background`、`Card` 和 `Custom` 多种首页布局。
- 首页可以加载 Markdown，也可以通过自定义 partial 完全控制内容结构。
- `mainSections` 可以同时包含 `projects`、`notes` 或 `posts` 等多个内容区。
- 支持卡片列表、缩略图、搜索、目录、相关内容、系列文章和多作者。
- 技术内容支持代码高亮、复制按钮、Mermaid、Chart.js、KaTeX、GitHub 卡片、轮播和图片画廊。
- 内置简体中文界面，README 与官方文档均提供简体中文版。
- 提供 GitHub Pages 部署文档。
- 可以通过 Hugo Module、Git submodule 或手动方式安装。

项目展示方式：

Blowfish 没有强约束的 Portfolio 数据模型，但 Hugo 的内容区可以直接建立 `content/projects/`。每个项目使用 Markdown 页面和 front matter 描述，再通过列表页或 Card Gallery 展示。

升级方式：

- 主题可以作为 Hugo Module 或 submodule 与站点内容分离。
- 本地可以通过同路径文件覆盖主题的 layout 和 asset，不必直接修改主题源码。
- 版本发布频繁，适合固定某个 Release 后按周期升级。

需要关注的点：

- 功能和配置项较多，需要主动裁剪，才能保持克制的视觉风格。
- 中文界面已经适配，但正文排版仍需要额外调整字体栈、行高、段落间距和标题字重。
- 笔记中大量使用 Blowfish shortcode 会降低未来迁移到其他主题的便利性。

官方资料：

- [仓库](https://github.com/nunocoracao/blowfish)
- [首页布局](https://blowfish.page/docs/homepage-layout/)
- [简体中文文档](https://blowfish.page/zh-cn/docs/)
- [安装和更新](https://blowfish.page/docs/installation/)
- [部署](https://blowfish.page/docs/hosting-deployment/)

### 4.2 Congo

Congo 是另一套基于 Hugo 和 Tailwind CSS 的个人网站与内容主题，视觉比 Blowfish 更克制。

核心能力：

- 提供 Profile、Page 和 Custom 等首页形态。
- 支持自定义内容区，因此可以分别组织 `projects` 和 `notes`。
- 内置简体中文和繁体中文翻译。
- 官方文档包含 GitHub Pages 部署流程。
- 支持 Hugo Module，主题与站点内容可以分离升级。

项目展示方式：

项目通常作为独立 Hugo section 存放，通过列表模板或自定义首页片段展示。默认组件与卡片样式比 Blowfish 少，首页项目区往往需要更多定制。

需要关注的点：

- 默认视觉更安静，但项目卡片、项目筛选和首页组合能力不如 Blowfish 丰富。
- 如果需要比较精细的作品集效果，可能需要编写少量 Hugo template。

官方资料：

- [仓库](https://github.com/jpanther/congo)
- [文档](https://jpanther.github.io/congo/docs/)
- [部署](https://jpanther.github.io/congo/docs/hosting-deployment/)

### 4.3 PaperMod

PaperMod 是本次候选中社区规模最大的 Hugo 主题，结构成熟、依赖简单、长期使用案例较多。

核心能力：

- 提供 Regular、Home Info 和 Profile 三种首页模式。
- 内置简体中文和繁体中文翻译。
- 支持搜索、标签、归档、目录、代码高亮、封面图、SEO 和深浅色模式。
- 没有 Node.js 构建依赖，构建链路相对简单。
- 支持模板覆盖，不必直接修改主题源文件。

项目展示方式：

Profile Mode 可以承担个人介绍和社交入口，但官方没有完整的 Portfolio 或 Project Card 模型。项目通常需要建立独立 section，再自定义 list template 或首页 partial。

维护特征：

- Star 和 Fork 数量显著高于其他候选。
- 2026 年仍有代码更新，但正式 Release 节奏比 Blowfish 慢，近期变化主要在默认分支。

需要关注的点：

- 默认信息架构更偏技术博客。
- 项目展示和个人首页的视觉层次需要自行补充。
- 默认风格干净但较朴素，达到“精致”效果需要调整字体、间距和项目组件。

官方资料：

- [仓库与功能](https://github.com/adityatelange/hugo-PaperMod)
- [中文翻译列表](https://github.com/adityatelange/hugo-PaperMod/wiki/Translations)
- [安装与升级](https://github.com/adityatelange/hugo-PaperMod/wiki/Installation)

### 4.4 Hugo Profile

Hugo Profile 更接近开箱即用的简历式个人主页。

核心能力：

- 首页包含 About、Experience、Education、Projects、Achievements、Contact、Blog 和 Gallery 等模块。
- 项目、经历和技能结构已经预设，初始搭建速度较快。
- 使用 Bootstrap，传统 Portfolio 组件比较完整。

需要关注的点：

- 默认视觉具有明显 Bootstrap Portfolio 风格，需要较多 CSS 才能达到更克制的中文个人站效果。
- 内置翻译主要覆盖英语、西班牙语和法语，中文需要自行建立翻译文件。
- 社区规模和文档完整性低于 PaperMod、Blowfish 等候选。

官方资料：

- [仓库](https://github.com/gurusabarish/hugo-profile)

## 5. Astro 路线

### 5.1 AstroPaper

AstroPaper 是社区规模和维护活性较强的 Astro 内容模板。

核心能力：

- Astro、TypeScript、Tailwind CSS 和 MDX。
- 类型安全的 Markdown 内容。
- Pagefind 静态搜索、目录、分页、RSS、SEO 和动态 OG 图片。
- 深浅色模式和可访问性支持。
- 官方标注为 `i18n ready`。
- 2026 年仍有持续 Release 和依赖升级。

项目展示方式：

默认内容模型主要是 `pages` 和 `posts`，没有独立 Projects。需要新增：

- `projects` Content Collection schema；
- 项目列表和详情路由；
- 首页个人 Hero 与精选项目区；
- 项目搜索或标签逻辑。

中文适配：

`i18n ready` 表示代码结构支持国际化，不等于简体中文开箱即用。仍需补充中文界面词条、日期格式、站点语言配置和 CJK 样式。

升级方式：

AstroPaper 是源码模板。项目创建后，模板代码成为站点代码的一部分；深度修改组件和 schema 后，同步上游版本通常需要手工合并冲突。

官方资料：

- [仓库](https://github.com/satnaing/astro-paper)
- [v6.1.0 Release](https://github.com/satnaing/astro-paper/releases/tag/v6.1.0)
- [配置文档](https://astro-paper.pages.dev/posts/how-to-configure-astropaper-theme/)

### 5.2 Dante

Dante 的默认信息架构与个人网站需求比较接近。

核心能力：

- 首页 Hero 和个人简介。
- Portfolio Collection 和项目列表。
- Blog、项目详情、标签、Markdown/MDX。
- 深浅色模式、SEO、RSS 和响应式布局。
- Astro + Tailwind CSS，修改组件比较直接。

中文适配：

没有内置中文 i18n 或专门的 CJK 排版，需要自行处理界面文案、日期、字体、换行和中英文间距。

维护与许可证：

- 社区规模明显小于 AstroPaper、Fuwari、Blowfish 和 PaperMod。
- 许可证为 GPL-3.0，而不是多数候选使用的 MIT。
- 它是源码 starter，缺少类似 Hugo Module 的独立主题更新边界。
- 官方部署入口偏向 Netlify；GitHub Pages 需要补 Astro 官方 Action。

官方资料：

- [仓库](https://github.com/JustGoodUI/dante-astro-theme)

### 5.3 Retypeset

Retypeset 强调字体、留白和阅读体验，是本次候选中对中文排版关注较多的 Astro 主题。

核心能力：

- 原生简体中文、繁体中文和多语言支持。
- 优化过的正文排版。
- 支持 MDX、LaTeX、Mermaid、目录、评论和深浅色模式。
- 使用 Astro 和 UnoCSS。
- 提供主题更新脚本。

项目展示方式：

官方定位仍是 Static Blog Theme，没有独立 Portfolio 或 Projects 模型。需要自行新增项目集合、列表页和个人首页结构。

需要关注的点：

- 项目较年轻，正式 Release 历史较短。
- UnoCSS 与专门的排版系统提高了接手和二次开发成本。
- 更新脚本本质仍需合并源码；深度修改首页和内容 schema 后可能产生冲突。

官方资料：

- [仓库](https://github.com/radishzzz/astro-theme-retypeset)

### 5.4 Fuwari

Fuwari 在 Astro 中文博客社区中有较高采用规模，功能也比较完整。

核心能力：

- 中文 README 和多语言说明。
- Pagefind 搜索、目录、RSS、深浅色模式。
- Expressive Code、KaTeX、GitHub 卡片和扩展 Markdown。
- Astro + Tailwind CSS + Svelte。

需要关注的点：

- 官方定位是静态博客模板，首页结构和视觉中心都是文章，而不是个人项目。
- 默认包含 Banner、侧栏、圆角卡片和动画，视觉表达比“克制的工程师主页”更活跃。
- 依赖面包含 Astro、Tailwind、Svelte、Swup、Pagefind 等，升级面相对较宽。
- 社区 Star/Fork 较高，但没有清晰的正式 Release 节奏，Issue 和 PR 积压相对明显。
- 专属 Markdown 扩展使用越多，迁移到其他模板的成本越高。

官方资料：

- [仓库](https://github.com/saicaca/fuwari)
- [Pull Requests](https://github.com/saicaca/fuwari/pulls)
- [Discussions](https://github.com/saicaca/fuwari/discussions)

### 5.5 Astro Sphere

Astro Sphere 的站点形态包括个人介绍、Posts、Projects 和 Work，结构上接近个人作品集。

核心能力：

- Astro、Tailwind CSS 和少量 SolidJS。
- 支持 Posts、Projects、Work、Markdown/MDX 和搜索。
- 默认视觉简约，包含深浅色模式和轻量动画。

需要关注的点：

- 最新 Release 停留在 2024 年，最近代码推送为 2025 年。
- 仍使用 Astro 4、Tailwind 3，并引入少量 SolidJS。
- 文章目录仍在官方 roadmap 中。
- 无内置中文 i18n。

官方资料：

- [仓库](https://github.com/markhorn-dev/astro-sphere)

## 6. 中文适配分析

中文适配可以拆成三个层次，不能只看模板是否有中文 README。

### 6.1 界面翻译

需要覆盖：导航、搜索、上一篇/下一篇、阅读时间、目录、归档、标签、分页和 404 页面。

- Blowfish、Congo、PaperMod：内置中文界面。
- Retypeset：原生简繁中文与多语言。
- AstroPaper：具备 i18n 结构，但需要补中文内容。
- Dante、Astro Sphere：需要自行中文化。
- Fuwari：有中文文档和文章语言配置，但仍需核查全部界面词条。

### 6.2 CJK 排版

无论选择哪一个模板，都建议检查：

- 正文行高约 `1.75–1.9`；
- 正文宽度约 `720–780px`；
- 中文段落之间有明确留白；
- 中文标题不要套用过大的英文字距；
- 标点悬挂、长英文链接和行内代码不会破坏换行；
- 中文、英文、数字和等宽代码字体之间的视觉高度协调；
- 粗体字重在 Windows、macOS、iOS 和 Android 上均可正常显示。

建议优先使用系统 CJK 字体栈，减少在中国大陆加载外部字体时的不确定性：

```css
font-family:
  Inter,
  -apple-system,
  BlinkMacSystemFont,
  "Segoe UI",
  "PingFang SC",
  "Hiragino Sans GB",
  "Microsoft YaHei",
  sans-serif;
```

代码字体可以使用：

```css
font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
```

### 6.3 中文内容结构

建议页面 URL 使用稳定的英文 slug，页面标题与正文使用中文：

```text
/about/
/projects/
/notes/
/notes/cuda/memory-coalescing/
```

这样可以避免中文 URL 编码带来的分享和工具兼容问题，也便于未来增加英文内容。

## 7. 内容与项目组织方式

### 7.1 Hugo 内容结构示例

```text
content/
├── _index.md
├── about/
│   └── index.md
├── projects/
│   ├── _index.md
│   ├── sparse-spec/
│   │   └── index.md
│   └── moe-kv-cache/
│       └── index.md
└── notes/
    ├── _index.md
    ├── cuda/
    ├── triton/
    └── llm-inference/
```

### 7.2 Astro 内容结构示例

```text
src/content/
├── projects/
│   ├── sparse-spec.md
│   └── moe-kv-cache.md
└── notes/
    ├── cuda/
    ├── triton/
    └── llm-inference/
```

Astro 需要额外定义 Content Collection schema，例如项目日期、技术栈、仓库、角色、状态和封面图。Hugo 对 front matter 更自由，但类型约束更弱。

## 8. 升级与内容锁定风险

### 8.1 Hugo 主题

Blowfish、Congo 和 PaperMod 可以作为 Hugo Module 或 submodule 安装：

- 内容、配置和主题源代码分离；
- 本地 layout 和 CSS 可以覆盖上游文件；
- 更换主题时，标准 Markdown 内容通常可以保留；
- 主要迁移成本集中在 front matter 和专属 shortcode。

控制风险的方式：

- 固定主题版本，不在每次构建时自动追踪最新分支；
- 不直接修改 `themes/` 内的源码；
- 核心笔记尽量使用标准 Markdown；
- 专属 shortcode 主要用于首页和项目展示。

### 8.2 Astro 源码模板

AstroPaper、Dante、Retypeset、Fuwari 和 Astro Sphere 通常作为完整源码复制到项目中：

- 组件层定制更自由；
- 项目代码与模板代码没有天然边界；
- 深度修改后，同步上游版本容易产生冲突；
- 内容可能绑定 Content Collection schema、MDX 组件、图片导入方式和特定路由结构。

控制风险的方式：

- 将内容 schema 保持简单；
- 笔记优先使用 `.md`，只在确有需要时使用 `.mdx`；
- 将模板上游设置为只读 remote，按版本手工评估更新；
- 避免把业务内容写进组件源码。

## 9. 部署比较

所有候选都能生成静态文件并部署到 GitHub Pages。

### Hugo

- 使用 Hugo Extended 构建。
- Blowfish、Congo 有较完整的 GitHub Pages 文档。
- PaperMod 可以使用标准 Hugo GitHub Actions 流程。
- Hugo Module 需要在 CI 中安装 Go 和对应版本的 Hugo Extended。

### Astro

- 使用 Node.js 和包管理器安装依赖后执行 `build`。
- 可以使用 Astro 官方 GitHub Pages Action。
- 用户主页仓库 `drink-less-milktea.github.io` 通常部署在根路径，不需要额外 `base`。
- 如果未来部署到普通项目仓库，则需要配置 `/repo-name` 子路径。

## 10. 视觉风格适配

“简约精致”更依赖二次设计约束，而不是模板名称。无论采用哪一个候选，都可以统一采用以下原则：

- 白色或接近白色的背景；
- 深灰正文，避免纯黑造成过强对比；
- 只保留一种低饱和强调色；
- 首页不使用大面积渐变、粒子背景和自动轮播；
- 精选项目控制在 3–4 个；
- 首页只展示最近 3–5 篇笔记；
- 使用线性图标，减少彩色技术徽章；
- 项目卡片优先展示“问题、贡献、结果”，而不是堆技术标签；
- 动画只用于状态变化和页面切换，持续时间控制在 150–250ms；
- 暗色模式作为可选能力，不让暗色视觉主导整体设计。

## 11. 需要用户决定的取舍

后续选型主要取决于以下问题，本文不代替用户回答：

1. 更看重主题与内容分离、升级稳定，还是更看重 TypeScript 组件层自由度？
2. 项目展示需要标准卡片和详情页，还是需要复杂交互与动态可视化？
3. 中文排版是否高于项目首页结构的优先级？
4. 是否愿意维护 Astro 依赖和处理大版本升级？
5. 是否接受 GPL-3.0 模板，还是只考虑 MIT 等宽松许可证？
6. 首页更偏个人名片、项目作品集，还是最新内容入口？
7. 是否计划将现有 Obsidian/Markdown 笔记批量同步到网站？

## 12. 候选集合汇总

按照不同技术与产品倾向，可以保留如下候选集合，以下不表示排名：

- **Hugo、个人站与内容并重**：Blowfish、Congo。
- **Hugo、社区规模和技术博客生态**：PaperMod。
- **Hugo、简历式 Portfolio 快速搭建**：Hugo Profile。
- **Astro、社区活跃且适合长期二次开发**：AstroPaper。
- **Astro、原生 Portfolio + Blog 结构**：Dante。
- **Astro、中文长文与排版优先**：Retypeset。
- **Astro、中文博客社区和丰富组件**：Fuwari。
- **Astro、项目结构直接但维护较弱**：Astro Sphere。

