// 本地覆盖 themes/blowfish/assets/js/katex-render.js
// 让 KaTeX auto-render 识别单 $ 行内公式（默认只认 \( \)、\[ \)、$$）
// 注意：$ 定界符必须放最后，否则 $$ 会被匹配成两个 $
document.getElementById("katex-render") &&
  document.getElementById("katex-render").addEventListener("load", () => {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  });
