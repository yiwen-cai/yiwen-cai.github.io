---
name: publish-note
description: 清理、检查并发布 yiwen-cai.github.io 的 Hugo 技术笔记，覆盖新笔记导入和已有笔记审核通过后的正式上线。用户说“发布这篇笔记”“审核没问题，可以发布”“把这篇 md 上线”或要求提交、推送并确认 GitHub Pages 部署时使用；负责限定 Git 变更范围、检查 Markdown/KaTeX/资源文件、执行本地与 CI 同口径构建、精确暂存、提交推送、等待 Actions 完成并验证线上页面。
---

# 发布 Hugo 技术笔记

先完整阅读仓库根目录的 `AGENTS.md`，以其中的当前环境、内容格式和发布约束为准。本技能补充从“内容已准备好”到“线上已验证”的发布闭环。

## 发布原则

- 将“审核没问题，可以发布”等明确表述视为提交并推送已审核内容的授权。
- 只发布目标笔记及其实际引用的资源。保留工作区里的无关修改、未跟踪文件和本地辅助 symlink。
- 禁止使用 `git add -A`、`git add .` 或强制推送；始终显式列出暂存路径。
- 用户只要求本地预览或内容审核时，不得提前提交或推送。
- 不修改 `themes/blowfish/`，不提交 `public/` 或 `resources/_gen/`。

## 1. 确定发布范围

1. 运行 `git status --short --branch` 和 `git diff --name-status`。
2. 确认目标文件位于 `content/notes/<分类>/<slug>/index.md`。
3. 搜索正文中的本地图片、SVG 等引用，只把确实属于该笔记的资源纳入范围。
4. 检查目标文件的 diff；若同一文件混入无法安全拆分的无关改动，先向用户说明，不要擅自整文件发布。
5. 记录所有不纳入发布的工作区文件，完成后保持它们原样。

新建笔记时，从现有 `content/notes/` 子目录中选择分类；新增分类必须同时创建 `_index.md`。slug 使用短 kebab-case 英文。

## 2. 执行内容门禁

检查 front matter 使用 `+++` 包裹的 TOML，并至少包含 `title`、`date`、`draft = false`、手写 `summary`、`tags`、`showReadingTime` 和 `showTableOfContents`。含公式的文章必须在正文开头放置 `{{< katex >}}`。

确认以下要求：

- 来源说明、论文链接和非复现实验数据声明符合 `AGENTS.md`。
- 不含 Obsidian 双链、私人路径、PPT 指令、Notion 元数据和 HTML 空格实体等残留。
- 展示公式中没有单独占一行的 `=` 或 `-`，以免被 Markdown 误解析为标题。
- Markdown 图片引用的本地资源均存在；SVG 使用 `xmllint --noout <svg>` 验证。
- 不使用 Mermaid；优先采用 SVG、Markdown 表格或文字图。

运行残留扫描，并逐项判断所有命中：

```bash
grep -cE '\[\[|Slide [0-9]|配色|\.pptx|notion_id|notion_url' <note>
rg -n '/Users/|raw/papers/|沙盒复现|调研人：|⚠️ 待确认|演讲者备注|答辩口径|&#x20;|\\\*\\\*|^[[:space:]]*(=|-)[[:space:]]*$' <note>
git diff --check
```

第一条应输出 `0`；后两条不得留下未解释的问题。

## 3. 完成本地预览审核

内容尚未获用户确认时，运行 `hugo server` 并打开对应本地 URL，检查：

- 行内与块级 KaTeX 公式；
- 标题层级、TOC、正文换行；
- 图片和 SVG；
- 正文到附录等自定义锚点；
- 桌面端和必要的窄屏布局。

发现问题后修改并重新检查。只有用户明确确认可以发布，才进入下一步；若本轮开始前用户已经确认，则无需重复询问。

## 4. 执行发布前构建

先确认 Hugo 为 `v0.164.0+extended`，再依次运行：

```bash
hugo version
rm -rf public resources && hugo --quiet
hugo --gc --minify --baseURL https://yiwen-cai.github.io/
```

两次构建都必须以退出码 `0` 完成。`Module "blowfish" is not compatible...` 是已知非阻塞警告；其他新警告需要调查。

检查生成页面存在，并在 `public/notes/<分类>/<slug>/index.html` 中搜索关键标题、资源路径和自定义锚点。构建失败时停止发布，不得通过跳过检查来继续。

## 5. 精确提交并推送

使用明确路径暂存本次文章和资源：

```bash
git add content/notes/<分类>/<slug>/index.md content/notes/<分类>/<slug>/<resource>
git status --short
git diff --cached --stat
git diff --cached --check
```

核对暂存区只包含本次范围后，使用 conventional commit 中文消息提交。新增文章用 `feat:`，完善已有文章通常用 `docs:`。然后推送到 `main`：

```bash
git commit -m "docs: 完善笔记「<标题>」"
git push origin main
```

若普通 push 被远端更新拒绝，先检查远端状态并向用户报告；不得擅自 force push、reset 或改写用户提交。

## 6. 等待部署并验证线上页面

推送成功不等于发布完成。取得与刚才 commit SHA 对应的 Actions 运行，并等待最终结果：

```bash
gh run list --repo yiwen-cai/yiwen-cai.github.io --limit 1 \
  --json databaseId,name,status,conclusion,headSha,url,createdAt
gh run watch <run-id> --repo yiwen-cai/yiwen-cai.github.io --exit-status --interval 5
```

确认 `headSha` 与本次提交一致，且 build、deploy 均成功。Node.js 20 deprecated 是已知非阻塞警告。

最后访问 `https://yiwen-cai.github.io/notes/<分类>/<slug>/`，至少验证：

- 页面返回成功；
- 一个本次新增或修改的独特文本存在；
- 本地资源 URL 和关键锚点存在。

若 CDN 尚未刷新，短暂重试；若 Actions 失败，读取失败日志并修复后重新走构建与发布流程。

## 7. 交付结果

最终向用户报告：

- commit 短 SHA 与消息；
- GitHub Pages/Actions 最终状态；
- 线上文章和 Actions 链接；
- 本地构建、残留扫描及线上关键内容验证结果；
- 有意保留、未纳入提交的无关工作区文件。
