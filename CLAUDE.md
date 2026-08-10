# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**项目说明统一维护在 [`AGENTS.md`](./AGENTS.md)，开工前先完整读一遍。** 那里有环境准备（Hugo 0.164.0 extended + 主题 submodule）、构建命令、配置结构、内容组织、渲染踩坑点和硬约束，本文件不重复。

以下是 Claude Code 特有的部分。

## Skill

`publish-note`（`.claude/skills/publish-note/SKILL.md`）—— 把 Markdown 笔记发布到本站的完整流程。用户说「发布这篇笔记」「把这篇 md 发到网站」「把 wiki 里的 XX 笔记上线」「加一篇新笔记」时走它，不要手搓流程。

## 本地预览

`.claude/launch.json` 里配了 `hugo-server`（`hugo server --bind 0.0.0.0 --port 1313`）。首次验证 KaTeX 是否真渲染时，起服务后用 Browser MCP 实测（`preview_start` + `preview_eval` 查 `getComputedStyle`，或直接截图），不要只看构建退出码。

## 提交前自查

1. `rm -rf public resources && hugo --quiet`，退出码必须为 0。
2. 新增笔记跑 `grep -cE '\[\[|Slide [0-9]|配色|\.pptx' <文件>`，必须为 0。
3. 含公式的笔记确认正文开头有 `{{< katex >}}`。
4. commit message 用 conventional commits 前缀 + 中文描述，直接 push 到 `main` 即触发部署。
