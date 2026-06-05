# 期末报告 LaTeX 源文件

## 格式说明

- **单栏排版**，正文 **五号字**（`10.5pt`）
- 页边距 2.54 cm，行距 1.25 倍（与课程 Word 模板接近）
- 含：图、算法、定理/命题及证明、示例、实验、参考文献、两人分工

## 编译

需安装 TeX Live（含 `ctex`、`xelatex`、`bibtex`）：

```bash
cd report_latex
make          # 或手动 xelatex → bibtex → xelatex × 2
```

## 图片

图片路径为 `../figures/`。若尚未生成，请先运行：

```bash
cd ..
python -m experiments.run_all
python -m experiments.extended_experiments
```

未生成图片时，PDF 中会显示占位框，不影响文字与公式编译。

## 文件结构

```
report_latex/
├── report.tex          # 主文件
├── preamble.tex        # 宏包与版式
├── references.bib      # 参考文献
├── sections/           # 正文 §1–§8
├── appendices/         # 附录 A–C（精简版）
└── Makefile
```
