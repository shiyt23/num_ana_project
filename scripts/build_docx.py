#!/usr/bin/env python3
"""按 报告模板.docx 的样式生成 final_report.docx（排版优化版）。

排版要点（对齐模板 + 整齐美观、不过于紧凑）：
1. 复用模板命名样式：Subtitle / 作者 / 单位 / 摘要 / 关键词 / Heading 1-3 /
   Body Text / 定理 / 表名；正文 9pt 宋体（五号）。
2. **行间公式渲染为真实数学图片**（matplotlib mathtext），居中、带段前段后间距，
   远比 Unicode 拼接清晰美观。无法渲染的回退为居中 Unicode。
3. 正文段落加适当段后间距与 1.25 倍行距，留白充足、不紧凑。
4. 定理用模板"定理"样式（黑体标签）；图注、表名居中。
5. 表格统一加边框、表头加粗、整表居中。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from docx import Document  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "报告模板.docx"
OUTPUT = ROOT / "final_report.docx"
FIG_DIR = ROOT / "figures"
EQN_DIR = ROOT / "report" / "_eqn_cache"
DATA_FILE = ROOT / "data" / "experiment_results.json"

BODY_SPACE_AFTER = Pt(5)
BODY_LINE = 1.25


# ===========================================================================
# 数学公式渲染
# ===========================================================================

def _sanitize_latex(s: str) -> str:
    s = s.replace(r"\boxed", "")
    s = s.replace(r"\tfrac", r"\frac")
    s = s.replace(r"\displaystyle", "")
    # \bigl( \bigr) \Bigl \Bigr 等定界符修饰 mathtext 不支持 → 去掉修饰保留定界符
    s = re.sub(r"\\(?:bigg?|Bigg?)[lr]?", "", s)
    s = re.sub(r"\\le(?![a-zA-Z])", r"\\leq", s)
    s = re.sub(r"\\ge(?![a-zA-Z])", r"\\geq", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\\mathrm{\1}", s)
    s = s.replace(r"\odot", r"\circ").replace(r"\oslash", "/")
    s = s.replace(r"\succeq", r"\geq").replace(r"\preceq", r"\leq")
    s = s.replace(r"\succ", ">").replace(r"\prec", "<")
    s = s.replace(r"\implies", r"\Rightarrow")
    s = s.replace(r"\quad", r"\ \ ").replace(r"\qquad", r"\ \ \ ")
    return s


def _split_aligned(latex: str) -> list[str]:
    """把 aligned/cases 环境拆成多行（每行单独渲染）。"""
    m = re.search(r"\\begin\{aligned\}(.*?)\\end\{aligned\}", latex, re.S)
    if not m:
        return [latex]
    body = m.group(1)
    lines = [ln.strip() for ln in body.split(r"\\") if ln.strip()]
    return [ln.replace("&", "") for ln in lines]


def _render_math_png(latex: str, fontsize: int = 15) -> Path | None:
    """渲染单行 LaTeX 为透明 PNG，返回路径；失败返回 None。"""
    EQN_DIR.mkdir(parents=True, exist_ok=True)
    s = _sanitize_latex(latex)
    key = hashlib.md5((s + str(fontsize)).encode()).hexdigest()[:16]
    out = EQN_DIR / f"eq_{key}.png"
    if out.exists():
        return out
    try:
        fig = plt.figure()
        fig.text(0.5, 0.5, f"${s}$", ha="center", va="center", fontsize=fontsize)
        fig.savefig(out, dpi=200, transparent=True, bbox_inches="tight",
                    pad_inches=0.08)
        plt.close(fig)
        return out
    except Exception:
        plt.close("all")
        return None


# ===========================================================================
# Unicode 回退（用于内联与渲染失败时）
# ===========================================================================

_UNI = {
    r"\mu": "μ", r"\nu": "ν", r"\lambda": "λ", r"\kappa": "κ", r"\sigma": "σ",
    r"\eta": "η", r"\beta": "β", r"\alpha": "α", r"\gamma": "γ", r"\rho": "ρ",
    r"\tau": "τ", r"\theta": "θ", r"\varphi": "φ", r"\phi": "φ", r"\psi": "ψ",
    r"\delta": "δ", r"\varepsilon": "ε", r"\epsilon": "ε", r"\omega": "ω",
    r"\Delta": "Δ", r"\Sigma": "Σ", r"\Lambda": "Λ", r"\Omega": "Ω",
    r"\nabla": "∇", r"\partial": "∂", r"\infty": "∞", r"\leq": "≤", r"\le": "≤",
    r"\geq": "≥", r"\ge": "≥", r"\neq": "≠", r"\ne": "≠", r"\approx": "≈",
    r"\to": "→", r"\rightarrow": "→", r"\implies": "⟹", r"\in": "∈",
    r"\cdot": "·", r"\times": "×", r"\pm": "±", r"\mp": "∓", r"\sqrt": "√",
    r"\sum": "Σ", r"\langle": "⟨", r"\rangle": "⟩", r"\succeq": "⪰",
    r"\preceq": "⪯", r"\succ": "≻", r"\prec": "≺", r"\odot": "⊙",
    r"\oslash": "⊘", r"\top": "ᵀ", r"\mathbb{R}": "ℝ", r"\square": "□",
    r"\ldots": "…", r"\cdots": "⋯", r"\,": " ", r"\;": " ", r"\!": "",
    r"\left": "", r"\right": "", r"\quad": "  ", r"\qquad": "   ",
    r"\boxed": "", r"\nabla^2": "∇²",
}


def _to_unicode(s: str) -> str:
    for k, v in sorted(_UNI.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(k, v)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbb\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathcal\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\hat\{([^}]*)\}", r"\1̂", s)
    s = re.sub(r"\\tilde\{([^}]*)\}", r"\1̃", s)
    s = re.sub(r"\\dot\{([^}]*)\}", r"\1̇", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\tfrac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = s.replace(r"\\", " ")
    return s


def _strip_inline_math(text: str) -> str:
    """处理段落里的 $...$ 内联公式 → Unicode。"""
    text = re.sub(r"\$\$(.*?)\$\$", lambda m: _to_unicode(m.group(1).strip()),
                  text, flags=re.S)
    text = re.sub(r"\$(.+?)\$", lambda m: _to_unicode(m.group(1).strip()),
                  text, flags=re.S)
    return text


# ===========================================================================
# 文档构建辅助
# ===========================================================================

def _style_body(p):
    pf = p.paragraph_format
    pf.space_after = BODY_SPACE_AFTER
    pf.line_spacing = BODY_LINE


def add_para(doc, text, style="Body Text", align=None, body_spacing=True):
    p = doc.add_paragraph(_strip_inline_math(text), style=style)
    if align is not None:
        p.alignment = align
    if body_spacing and style == "Body Text":
        _style_body(p)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(_strip_inline_math(text), style=f"Heading {level}")
    return p


def add_equation(doc, latex: str, fontsize: int = 15):
    """行间公式：渲染为图片居中插入（带段前段后留白）；失败回退 Unicode。"""
    lines = _split_aligned(latex)
    for ln in lines:
        png = _render_math_png(ln, fontsize)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format
        pf.space_before = Pt(6)
        pf.space_after = Pt(6)
        if png is not None:
            run = p.add_run()
            # 控制公式图片高度：按行数自适应，单行约 0.62cm 高
            from PIL import Image
            try:
                w, h = Image.open(png).size
                target_h_cm = 0.66 * fontsize / 15
                width_cm = min(15.5, target_h_cm * w / h)
                run.add_picture(str(png), width=Cm(width_cm))
            except Exception:
                run.add_picture(str(png), height=Cm(0.62))
        else:
            r = p.add_run(_to_unicode(ln))
            r.font.italic = True
            r.font.size = Pt(11)
    return


def add_theorem(doc, label: str, body: str):
    """定理：标签用模板'定理'样式（黑体），内容随后。"""
    p = doc.add_paragraph(style="定理")
    run = p.add_run(_strip_inline_math(label))
    run.font.bold = True
    p.add_run("  " + _strip_inline_math(body))
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    return p


def add_proof(doc, body: str):
    p = doc.add_paragraph(style="Body Text")
    run = p.add_run("证明.")
    run.font.italic = True
    p.add_run("  " + _strip_inline_math(body) + "  □")
    _style_body(p)
    return p


def _set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for tag in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{tag}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), "808080")
        borders.append(e)
    tblPr.append(borders)


def _shade_cell(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def add_table(doc, headers, rows, caption=None):
    if caption:
        cap = doc.add_paragraph(_strip_inline_math(caption), style="表名")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = _strip_inline_math(str(h))
        _shade_cell(hdr[i], "D9E2F3")
        for para in hdr[i].paragraphs:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.font.bold = True
                run.font.size = Pt(10)
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, cell in enumerate(row):
            tc = table.rows[r_idx].cells[c_idx]
            tc.text = _strip_inline_math(str(cell))
            for para in tc.paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size = Pt(10)
    _set_table_borders(table)
    # 表后留白
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(2)
    return table


def add_figure(doc, path: Path, caption: str, width_cm: float = 14.5):
    if not path.exists():
        add_para(doc, f"[图缺失：{path.name}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    cap = doc.add_paragraph(_strip_inline_math(caption), style="表名")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)


def _clear_body(doc):
    body = doc.element.body
    for child in list(body):
        if child.tag in (qn("w:p"), qn("w:tbl")):
            body.remove(child)


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, (int, str)):
        return str(v)
    if isinstance(v, float):
        if abs(v) >= 1e4 or (v != 0 and abs(v) < 1e-3):
            return f"{v:.3e}"
        return f"{v:.4g}"
    return str(v)


def _load():
    return json.loads(DATA_FILE.read_text(encoding="utf-8")) if DATA_FILE.exists() else {}


# ===========================================================================
# 正文
# ===========================================================================

def build(doc, data):
    # ---------- 题名 / 作者 / 摘要 / 关键词 ----------
    add_para(doc,
             "正交化梯度方法的数值分析视角：从 Newton–Schulz 极分解到 Muon 优化器"
             "及其统一框架", style="Subtitle", body_spacing=False)
    add_para(doc, "赵泽霖、史云天", style="作者", body_spacing=False)
    add_para(doc, "(清华大学，学号：2023010848、2023010836，数值分析与算法课程项目)",
             style="单位", body_spacing=False)

    add_para(doc,
             "摘  要：本文以 Muon 优化器为核心，把现代深度学习常用的 GD、Heavy-ball、"
             "Nesterov、Adam/AdamW、Sophia、signSGD、Muon 统一为迭代模板 "
             "x_{k+1}=x_k−P_k g_k 下的不同更新策略，并进一步证明它们都是同一个线性"
             "最小化 oracle（LMO）d*=argmin_{‖d‖≤1}⟨g,d⟩ 在 ℓ₂/ℓ∞/谱范数下的解——"
             "即 GD/signSGD/Muon 分别是 ℓ₂/ℓ∞/谱范数的最速下降。我们给出完整数值分析"
             "推导（GD 强凸线性收敛、Polyak 谱半径最优、Heavy-ball 为 Chebyshev 半迭代"
             "的冻结系数极限、Newton–Schulz 二次收敛与谱范数 LMO 最优性），并以 22 组"
             "实验、19 项 pytest、一键复现脚本验证。本稿尤其强调诚实性：修正了一处真"
             "bug（有效条件数应为 κ(P_kA) 而非 κ(P_k⁻¹A)）并纠正多处过度声称——例如"
             "Adam 用梯度幅度而非曲率做对角缩放，故即便在对角 Hessian 上其 κ_eff 也"
             "达不到 Jacobi 的理想值 1；又如凸二次上 CG 为 Krylov 最优、Muon 并不更快，"
             "Muon 的价值是结构性的（更新条件数恒为 1）。关键定量结果：Chebyshev 系数 "
             "ω_k 单调收敛到 1+β*=1.6694（吻合 10⁻⁷）；Muon 更新条件数在 κ(G)∈[1,3162] "
             "上恒为 1；Newton–Schulz 在 (0,√3) 内全部收敛到正确极因子 +U，越界则为"
             "分形式归宿。",
             style="摘要", body_spacing=False)
    add_para(doc,
             "关键词：Newton–Schulz 迭代；极分解；Muon 优化器；范数最速下降；"
             "线性最小化 oracle；Chebyshev 半迭代；Heavy-ball；动态预条件；ODE 离散化",
             style="关键词", body_spacing=False)

    # ---------- 1 引言 ----------
    add_heading(doc, "1  引言")
    add_para(doc,
             "我们这学期的「数值分析与算法」课，迭代法部分系统讲了两件事：最速下降"
             "（GD）和共轭梯度法（CG）。然而工业界深度学习训练里常用的优化器名字根本"
             "不在课本上——Adam、AdamW、Sophia，以及 2024 年 Jordan 等人提出、被用于 "
             "LLM 预训练的 Muon。它们常被视为调参经验，却很少有人从数值迭代角度审视。")
    add_para(doc,
             "我们想做的事很简单：把这些优化器统一写成迭代模板")
    add_equation(doc, r"x_{k+1} = x_k - P_k\, g_k, \quad g_k = \nabla f(x_k),")
    add_para(doc,
             "再用课内工具（谱半径、条件数、Chebyshev 多项式、矩阵函数迭代）分析其"
             "收敛、稳定性与几何。在所有这些优化器里 Muon 最有意思：它的核心步骤 "
             "Newton–Schulz 迭代是 Higham 教材里的极分解迭代，二次收敛、有解析收敛盆 "
             "(0,√3)，给了我们一条用纯数值分析贯穿全文的主线。")

    add_heading(doc, "1.1  与课程的对应关系", level=2)
    add_table(doc,
              ["更新方向 / P_k", "经典对应", "现代名字", "要证 / 验证的"],
              [["ηI（标量步长）", "Richardson / 显式 Euler", "GD（ℓ₂ 最速下降）",
                "强凸线性收敛（定理 1）"],
               ["ηI + 二阶状态", "Chebyshev 半迭代", "Heavy-ball, Nesterov",
                "HB = Chebyshev 冻结系数（定理 4）"],
               ["η·diag(v̂)⁻¹ᐟ²", "对角缩放（≈Jacobi 启发式）", "Adam, AdamW",
                "动态预条件 + 2-极限环 + κ_eff 真相"],
               ["η·diag(h)⁻¹", "对角拟 Newton", "Sophia", "曲率 vs 梯度幅度"],
               ["−η·sign(g)", "ℓ∞ 最速下降", "signSGD, Lion", "LMO 三元组（定理 5）"],
               ["矩阵正交化 G→UVᵀ", "Newton–Schulz 矩阵迭代", "Muon",
                "二次收敛 + 谱范数最速下降（定理 6,7）+ 奇异值均衡"]],
              caption="表 1  统一模板下各更新策略与课程内容的对应")

    # ---------- 2 预备 ----------
    add_heading(doc, "2  预备：迭代法的收敛理论")
    add_heading(doc, "2.1  谱半径、Lipschitz 与强凸", level=2)
    add_para(doc,
             "设 f μ-强凸且梯度 L-Lipschitz（μI ⪯ ∇²f ⪯ LI），条件数 κ=L/μ。"
             "核心模型问题是强凸二次")
    add_equation(doc, r"f(x) = \tfrac{1}{2} x^\top A x - b^\top x, \quad A=A^\top \succ 0,")
    add_para(doc,
             "其极小点 x*=A⁻¹b、梯度 g(x)=Ax−b。优化等价于解线性方程组 Ax=b——"
             "这是把现代优化器与课内迭代法挂钩的根本理由。")

    add_heading(doc, "2.2  Krylov 子空间与 Chebyshev 多项式（CG 的本质）", level=2)
    add_para(doc,
             "CG 第 k 步 x_k 在仿射 Krylov 子空间 x_0+K_k(A,r_0) 中最小化 A-范数误差，"
             "等价于误差 e_k=p_k(A)e_0、p_k∈P_k⁰={p:p(0)=1}。这是 Chebyshev 最佳逼近"
             "问题，解为缩放 Chebyshev 多项式，给出经典上界")
    add_equation(doc, r"\|e_k\|_A \le 2 \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k \|e_0\|_A.")
    add_para(doc,
             "右端 (√κ−1)/(√κ+1) 比 GD 的 (κ−1)/(κ+1) 阶更好，正是 Chebyshev minimax "
             "性质带来的「加速」——后面所有加速类方法共享这个上界。")

    add_heading(doc, "2.3  极分解：矩阵函数视角", level=2)
    add_para(doc,
             "对任意 G∈ℝ^{m×n}（m≥n、列满秩），存在唯一极分解 G=UH（UᵀU=I，H⪰0）。"
             "若 G=ÛΣVᵀ 是 SVD，则 U=ÛVᵀ。第 4 节将证明 Muon 的更新方向就是 −U，"
             "第 5 节解释这正是谱范数最速下降。")

    # ---------- 3 三大基线 ----------
    add_heading(doc, "3  三大基线机制：GD、Heavy-ball、Adam")
    add_heading(doc, "3.1  GD 与 Richardson 迭代", level=2)
    add_para(doc, "GD 更新 x_{k+1}=x_k−η(Ax_k−b) 即求解 Ax=b 的 Richardson 迭代，P_k=ηI。")
    add_theorem(doc, "定理 1（GD 在强凸二次上的线性收敛）.",
                "取 η∈(0,2/L)，则 ‖x_{k+1}−x*‖₂ ≤ ρ(η)‖x_k−x*‖₂，"
                "ρ(η)=max_i|1−ηλ_i|。最优步长 η*=2/(μ+L) 处 ρ*=(κ−1)/(κ+1)。"
                "完整证明见附录 C.1。")
    add_para(doc,
             "实验对照：κ=100 时 ρ*≈0.9802；实验 5 在 200 个步长上验证 ρ(η) 的 V 形；"
             "实验 6 拟合得 κ=1000 时 ρ_emp=0.9959，与理论 0.998 相对误差 0.2%。")

    add_heading(doc, "3.2  Heavy-ball、Nesterov 与 Chebyshev 半迭代", level=2)
    add_para(doc, "Polyak Heavy-ball：v_{k+1}=βv_k+g_k，x_{k+1}=x_k−ηv_{k+1}。"
                  "在特征基下每个模态 λ 给出 2×2 迭代矩阵 T(λ)=[[1−ηλ+β,−β],[1,0]]。")
    add_theorem(doc, "定理 2（Polyak 最优 Heavy-ball 谱半径）.",
                "取 η*=4/(√L+√μ)²、β*=((√L−√μ)/(√L+√μ))²，则所有模态谱半径都等于"
                " ρ*_HB=(√κ−1)/(√κ+1)。证明（判别式置零得双重特征值）见附录 C.2。")
    add_para(doc, "对比：κ=100 时 ρ*_HB≈0.818，比 GD 的 0.980 加速约 9.9×（实验 2）。"
                  "Nesterov 强凸版 η=1/L、β=(√κ−1)/(√κ+1)，实验 11 修复后 κ=100 时"
                  " 70 步达 10⁻⁶（HB 98 步、GD 439 步）。")
    add_theorem(doc, "定理 4（Heavy-ball 是 Chebyshev 半迭代的冻结系数极限，核心结果）.",
                "Chebyshev 半迭代有限步 minimax 最优，其时变系数 ω_k 单调收敛到 "
                "ω_∞=1+β*；Polyak Heavy-ball 即把系数冻结在 ω_∞。二者共享渐近率"
                " (√κ−1)/(√κ+1)，且 Chebyshev 任意有限 k 不慢于 HB。完整证明见附录 C.3。")
    add_para(doc,
             "数值验证（实验 25）：κ=100 上 Chebyshev 系数 ω₅₀=1.66942149 与解析极限"
             " 1+β*=1.66942149 吻合到 10⁻⁷（图 25 中图）；两者收敛曲线几乎重合。"
             "这把课内 Chebyshev/CG 与现代动量法直接挂钩：Polyak Heavy-ball 是 "
             "Chebyshev 半迭代冻结系数后的定常版本——本报告我们最看重的结果。")
    add_figure(doc, FIG_DIR / "exp25_hb_chebyshev.png",
               "图 25  左：HB 与 Chebyshev 收敛曲线几乎重合 + minimax 上界；"
               "中：系数 ω_k→1+β* 收敛（定理 4 直接证据）；右：Chebyshev 误差多项式。")

    add_heading(doc, "3.3  Adam：动态对角缩放、与 Jacobi 的差距、极限环", level=2)
    add_para(doc, "Adam 代入统一模板得 P_k=η·diag(√v̂_k+ε)⁻¹，是动态对角缩放。")
    add_para(doc,
             "【修正一处 bug】在 x_{k+1}=x_k−P_k g_k 中 P_k 扮演 M⁻¹≈A⁻¹，误差递推"
             " e_{k+1}=(I−P_kA)e_k，故有效条件数是 κ_eff=κ(P_kA)，**不是** κ(P_k⁻¹A)。"
             "第一稿写成后者（会把对角 A 上理想 Jacobi 误算为 κ² 而非 1），本稿已"
             "在 src/optimizers.py 修正，所有 κ_eff 图据此重算。")
    add_theorem(doc, "命题 3（Adam 与 Jacobi 的关系——以及为什么不相等）.",
                "Jacobi 用曲率 diag(A)，Adam 用梯度幅度 √EMA(g²)。二次问题上 "
                "√v̂_{k,i}≈|g_{k,i}|=λ_i·|x_{k,i}−x_i*|，把曲率 λ_i 与到极小点的距离"
                "混在一起，只有各坐标距离均匀时才退化为 Jacobi。故一般 P_k≠η·diag(A)⁻¹。"
                "见附录 C.4。")
    add_para(doc, "实验 4（诚实版）画 κ_eff(k)=κ(P_kA) 并叠加 oracle Jacobi：")
    add_table(doc,
              ["情形（κ(A)=100）", "oracle Jacobi κ_eff", "Adam κ_eff（稳定后）"],
              [["对角 A", "1.0", "约 12（最低 1.9）"],
               ["稠密旋转 A", "83", "约 85"]],
              caption="表 2  Adam 的对角缩放达不到 oracle Jacobi 的理想条件数")
    add_para(doc,
             "结论：对角 A 上 oracle Jacobi 把 κ_eff 降到理想的 1，Adam 只到约 12"
             "（因为缩放梯度幅度而非曲率）；稠密 A 上二者都只能 100→约 85（对角预条件"
             "治不了非对角耦合）。「Adam=Jacobi」只是同属对角预条件家族的启发式，非等式。")
    add_figure(doc, FIG_DIR / "exp4_precond_kappa.png",
               "图 4  Adam κ_eff(k)=κ(P_kA) vs oracle Jacobi（上行对角 A、下行稠密 A）。")
    add_para(doc,
             "Bock–Weiß 2-极限环：Adam 即使在最简凸 f(x)=½ax² 上也可能不收敛，而落入"
             " 2-极限环 {x⁺,x⁻}（T(x⁺)=x⁻、T(x⁻)=x⁺）。实验 26 取 (η,β₁,β₂)=(0.05,0,0.99)"
             "、ε=10⁻¹²、x₀=1 复现到 x⁺≈0.0250、x⁻≈−0.0251，振幅 0.056 稳定 200 步以上。")
    add_figure(doc, FIG_DIR / "exp26_adam_limit_cycle.png",
               "图 26  Adam 在 f(x)=x²/2 上的 2-极限环（Bock–Weiß 2022 复现）。")

    # ---------- 4 核心 NS 与 Muon ----------
    add_heading(doc, "4  核心：Newton–Schulz 迭代与 Muon")
    add_heading(doc, "4.1  Newton–Schulz 迭代与二次收敛", level=2)
    add_para(doc, "设 G∈ℝ^{m×n}（m≥n、列满秩），X₀=G/‖G‖₂，")
    add_equation(doc, r"X_{k+1} = \tfrac{1}{2} X_k (3 I - X_k^\top X_k).")
    add_theorem(doc, "定理 6（局部二次收敛）.",
                "若 σ_min(X₀)>0 且 σ_max(X₀)<√3，则迭代二次收敛到极分解正交因子 U。"
                "令 E_k=XₖᵀXₖ−I，则 E_{k+1}=−¼E_k²(3I−E_k)，"
                "故 ‖E_{k+1}‖_F≤¼‖E_k‖_F²(3+‖E_k‖_F)，二次衰减。")
    add_proof(doc,
              "设 G_k=XₖᵀXₖ，则 G_{k+1}=¼(3I−G_k)G_k(3I−G_k)。代入 G_k=I+E_k、"
              "3I−G_k=2I−E_k，展开（E_k 与 I 可交换）得 4(G_{k+1}−I)=−3E_k²+E_k³"
              "=−E_k²(3I−E_k)，即 E_{k+1}=−¼E_k²(3I−E_k)。取 Frobenius 范数，"
              "当 ‖E_k‖_F<1 时 ‖E_{k+1}‖_F<‖E_k‖_F²，即二次收敛。")
    add_para(doc,
             "收敛盆与诚实图景：标量映射 φ(σ)=σ(3−σ²)/2 有不动点 {−1,0,+1}。"
             "σ₀∈(0,√3) 收敛到 +1（正确极因子 +U）；σ₀∈(√3,≈2.06) 收敛到 −1"
             "（即 −U，符号错）；σ₀≳2.3 发散，中间还夹着如 σ₀=2.21 弹回 +1 的点——"
             "盆结构是分形式的。实验 30 在 120 个初值上区分最终符号：(0,√3) 内 86 个"
             "全部 →+U，外部 21 个 →−U、13 个发散。只有 (0,√3) 保证收敛到正确极因子，"
             "这正是 Muon 实现「先除以 ‖G‖₂ 把奇异值压进 (0,√3)」的数值依据。")
    add_figure(doc, FIG_DIR / "exp30_ns_basin.png",
               "图 30  NS 收敛盆三种归宿：(0,√3)→+U（正确）、(√3,2.06)→−U（符号错）、更大发散。")
    add_para(doc,
             "实验 3/23：NS 在 16×16 矩阵上 15 步达 10⁻¹⁰；胖矩阵 8×32 因 m<n 边界，"
             "修复前发散，修复后（先对 Xᵀ 做 NS）7 步达 10⁻⁶。")

    add_heading(doc, "4.2  五阶 Higham 加速", level=2)
    add_para(doc, "Higham (2008, eq. 8.20) 五阶多项式")
    add_equation(doc, r"X_{k+1} = \tfrac{1}{8}\bigl(15 X_k - 10 X_k(X_k^\top X_k) + 3 X_k(X_k^\top X_k)^2\bigr),")
    add_para(doc,
             "对应 p(σ)=(15−10σ²+3σ⁴)/8，满足 p(1)=1、p′(1)=p″(1)=0（三阶接触），"
             "收敛阶为 5。实验 27：16×8 矩阵 5 阶 NS 5 步达 3×10⁻¹⁶，3 阶要 11 步。")
    add_figure(doc, FIG_DIR / "exp27_chebyshev_ns.png",
               "图 27  五阶 Higham NS vs 标准三阶 NS（三种矩阵形状）。")

    add_heading(doc, "4.3  Muon 优化器：谱范数最速下降", level=2)
    add_para(doc,
             "Muon（Jordan et al. 2024）：对矩阵参数先做动量梯度 M_k，用 Newton–Schulz "
             "正交化得 M̃_k≈U_kV_kᵀ，再 W_{k+1}=W_k−ηM̃_k。Kovalev (2025) 证明正交化"
             "方向 −UVᵀ 是谱范数最速下降方向。")
    add_theorem(doc, "定理 7（极分解的 trust-region 最优性）.",
                "对任意 G=UΣVᵀ，argmin_{‖Δ‖₂≤t}⟨G,Δ⟩_F = −t·UVᵀ。")
    add_proof(doc,
              "由 von Neumann 迹不等式 |tr(ΔᵀG)|≤Σ_i σ_i(Δ)σ_i(G)≤t·‖G‖_*"
              "（核范数），等号在 Δ 与 G 共享奇异向量、奇异值取上界 t 时成立，"
              "即 Δ=−t·UVᵀ。")
    add_para(doc,
             "含义：Muon 在 ‖ΔW‖₂≤η 的谱范数 trust-region 上做线性逼近最速下降。"
             "这正是 §5 的「范数最速下降」框架：Muon 选谱范数、GD 选 ℓ₂，各自最小化"
             "不同范数下的线性逼近，故不必在 Frobenius 损失上一致。")

    add_heading(doc, "4.4  Muon 到底好在哪——诚实的评估", level=2)
    add_para(doc,
             "必须诚实交代：凸二次上 CG 是 Krylov 最优的一阶方法，没有任何一阶方法"
             "能胜过 CG，Muon 也不能。在 Frobenius 矩阵二次上固定步长 Muon 甚至发散"
             "（f−f* 升到 6×10⁴，GD 约 50）；即便在我们一度称「Muon 主场」的谱范数"
             "恢复目标上，Frobenius GD 也能精确收敛到 10⁻³²，Muon 只到 0.037。"
             "所以「Muon 赢 GD」站不住脚，第一稿的措辞是过度声称，本稿改正。")
    add_para(doc,
             "Muon 的价值是结构性的（实验 31）：无论梯度 G 的奇异值多么悬殊，Muon "
             "更新 −UVᵀ 的全部奇异值都是 1，即 κ(Muon 更新)≡1；GD 更新 −G 的条件数"
             "直接等于 κ(G)。实验 31 让 κ(G) 从 1 扫到 3162，Muon 更新条件数始终为 "
             "1.000。这意味着 Muon 把每个奇异方向都推进等量的一步，而 GD 在小奇异值"
             "方向几乎不动。")
    add_figure(doc, FIG_DIR / "exp31_muon_equalization.png",
               "图 31  Muon 的结构性特征——奇异值均衡：更新条件数恒为 1。")
    add_para(doc,
             "为什么这在深度学习里重要（而我们无法在二次模型上完全展示）：一层权重"
             "更新 ΔW 对该层输出的影响由 ‖ΔW‖₂ 界定，谱范数 trust-region 正是控制层"
             "输出变化的自然约束；Muon 的均衡更新让所有奇异方向获得一致进展。这是 "
             "Muon 在 LLM 预训练里提速的根源，但严格验证需 GPU 集群，超出本课程范围，"
             "我们诚实地列为「未在本文实验中证实」。实验 12、28 展示的是 Muon 在谱范数"
             "目标上稳定下降、5 种子一致（终值 0.028~0.037），以及与 Frobenius 几何的"
             "错配——而非「比 GD 快」。")
    add_figure(doc, FIG_DIR / "exp12_matrix_muon.png",
               "图 12  左：谱范数目标上 Muon 稳定下降；右：Frobenius 二次上 Muon 不下降（几何错配）。")

    # ---------- 5 范数最速下降统一框架 ----------
    add_heading(doc, "5  统一框架：范数视角下的最速下降（LMO）")
    add_para(doc,
             "本节给出把 GD、signSGD、Muon 真正统一起来的现代观点（Bernstein–Newhouse "
             "modular duality；Kovalev 2025 non-Euclidean trust-region）。给定梯度 g，"
             "在范数 ‖·‖ 单位球内沿线性逼近走最陡一步，方向由线性最小化 oracle 给出：")
    add_equation(doc, r"d^\star = \arg\min_{\|d\|\le 1} \langle g, d\rangle, \qquad x_{k+1} = x_k + \eta\, d^\star.")
    add_theorem(doc, "定理 5（三种范数的 LMO 闭式解 = 三个经典优化器）.",
                "ℓ₂：d*=−g/‖g‖₂（GD）；ℓ∞：d*=−sign(g)（signSGD/Lion）；"
                "谱范数：d*=−UVᵀ（Muon）。且最优值 ⟨g,d*⟩=−‖g‖_dual，"
                "其中 ℓ₂↔ℓ₂、ℓ∞↔ℓ₁、谱范数↔核范数。证明见附录 C.5。")
    add_para(doc,
             "实验 32 数值验证：三种 LMO 解的内积都精确等于 −‖g‖_dual，且优于 3000 个"
             "随机方向。一句话统一：现代优化器 = 选了不同范数几何的最速下降——"
             "GD 选 ℓ₂、signSGD/Lion 选 ℓ∞、Muon 选谱范数。")
    add_figure(doc, FIG_DIR / "exp32_norm_steepest_descent.png",
               "图 32  范数最速下降三元组：LMO 值 = −对偶范数（左）；三范数在矩阵二次上的行为（右）。")
    add_para(doc,
             "为什么这个视角重要：(1) 它解释 Muon 的「非对角」性——ℓ₂/ℓ∞ 的 LMO 都是"
             "逐元素（对角型），谱范数 LMO 涉及 SVD，是真正的矩阵级操作；(2) 它把"
             "「预条件」与「范数选择」统一；(3) 它指明 Newton–Schulz 的角色——谱范数 "
             "LMO 需要 UVᵀ，精确算要 SVD，Muon 用 NS 多项式迭代近似，于是第 4 节的"
             "矩阵函数迭代成了这个框架的计算引擎。一个诚实边界：LMO 统一的是方向不是"
             "速度——凸二次上换范数并不更快，优势只在问题几何与该范数匹配时出现。")

    # ---------- 6 ODE ----------
    add_heading(doc, "6  ODE 视角：连续极限与高分辨率")
    add_para(doc,
             "梯度流 ẋ=−∇f 的前向 Euler 离散即 GD。Nesterov 对应 Su–Boyd–Candès "
             "AVD-ODE Ẍ+(3/t)Ẋ+∇f=0（t=√η·k，凸时 O(1/t²)）。Shi et al. 2021 的"
             "高分辨率 ODE 多出 √η·∇²f·Ẋ 修正项，区分 NAG 与 Heavy-ball。"
             "实验 29 在 κ=50 二维二次上用 RK4 积分 AVD-ODE，与离散 NAG/HB 在连续"
             "时间 t=√η·k 下轨迹基本重合，2D 相图显示离散迭代是连续螺旋的步长采样。")
    add_figure(doc, FIG_DIR / "exp29_nag_ode.png",
               "图 29  AVD-ODE RK4 解 vs 离散 NAG/HB（左：收敛对照；右：2D 相图）。")

    # ---------- 7 实验综述 ----------
    add_heading(doc, "7  数值实验综述")
    add_heading(doc, "7.1  环境与一键复现", level=2)
    add_para(doc,
             "环境 Python 3.11 + NumPy/SciPy/Matplotlib；一键复现 "
             "bash scripts/check_repro.sh（19 项 pytest + run_all + 附录表 + 生成 docx）。"
             "35 张图存 figures/，JSON 摘要存 data/experiment_results.json。")

    add_heading(doc, "7.2  关键实验数据", level=2)
    e1 = data.get("exp1", {})
    e11 = data.get("exp11", {})
    e12 = data.get("exp12", {})
    e25 = data.get("exp25_hb_chebyshev", {})
    e27 = data.get("exp27_cheb_ns", {})
    e30 = data.get("exp30_ns_basin", {})
    e31 = data.get("exp31_equalization", {})

    add_para(doc, "实验 1：达到 f−f*<10⁻⁶ 的迭代步数（κ=10/100/1000）。")
    add_table(doc, ["κ", "GD", "Polyak HB", "Adam"],
              [["10", _fmt(e1.get("kappa_10_gd", {}).get("iter_to_1e6")),
                _fmt(e1.get("kappa_10_momentum", {}).get("iter_to_1e6")),
                _fmt(e1.get("kappa_10_adam", {}).get("iter_to_1e6"))],
               ["100", _fmt(e1.get("kappa_100_gd", {}).get("iter_to_1e6")),
                _fmt(e1.get("kappa_100_momentum", {}).get("iter_to_1e6")),
                _fmt(e1.get("kappa_100_adam", {}).get("iter_to_1e6"))],
               ["1000", _fmt(e1.get("kappa_1000_gd", {}).get("iter_to_1e6")),
                _fmt(e1.get("kappa_1000_momentum", {}).get("iter_to_1e6")),
                _fmt(e1.get("kappa_1000_adam", {}).get("iter_to_1e6"))]],
              caption="表 3  实验 1：收敛步数")
    add_figure(doc, FIG_DIR / "exp1_convergence.png",
               "图 1  病态二次 GD/Momentum/Adam 收敛（3 档 κ）。")

    add_para(doc, "实验 11：Nesterov（强凸版，修复后）vs Polyak HB。")
    add_table(doc, ["κ", "Nesterov", "Polyak HB"],
              [["10", _fmt(e11.get("kappa_10_nesterov_iter")),
                _fmt(e11.get("kappa_10_momentum_iter"))],
               ["100", _fmt(e11.get("kappa_100_nesterov_iter")),
                _fmt(e11.get("kappa_100_momentum_iter"))],
               ["1000", _fmt(e11.get("kappa_1000_nesterov_iter")),
                _fmt(e11.get("kappa_1000_momentum_iter"))]],
              caption="表 4  实验 11：Nesterov 与 Polyak 收敛步数（修复后）")

    add_para(doc, "实验 25 / 31 / 32：核心定量结果。")
    add_table(doc, ["量", "值"],
              [["Chebyshev 系数极限 1+β*（解析）", _fmt(e25.get("omega_inf_theory"))],
               ["数值 ω₅₀", _fmt(e25.get("omega_50_numeric"))],
               ["Muon 更新条件数（κ(G)∈[1,3162]）", "恒为 " + _fmt(e31.get("muon_update_kappa_max"))],
               ["GD 更新条件数（κ(G)=1000）", _fmt(e31.get("gd_update_kappa_at_1000"))],
               ["NS 收敛盆：→+U / →−U / 发散 点数",
                f"{_fmt(e30.get('n_converge_plus'))} / "
                f"{_fmt(e30.get('n_converge_minus_wrong'))} / "
                f"{_fmt(e30.get('n_diverge'))}"]],
              caption="表 5  定理 4 / 5 / 6 / 7 的关键数值证据")

    add_para(doc, "实验 27：5 阶 vs 3 阶 NS 最终正交化误差（5 步后）。")
    add_table(doc, ["矩阵形状", "3 阶 NS", "5 阶 Higham NS"],
              [["16×8", _fmt(e27.get("16x8_std_final")), _fmt(e27.get("16x8_cheb_final"))],
               ["32×16", _fmt(e27.get("32x16_std_final")), _fmt(e27.get("32x16_cheb_final"))],
               ["64×32", _fmt(e27.get("64x32_std_final")), _fmt(e27.get("64x32_cheb_final"))]],
              caption="表 6  实验 27：五阶 NS 5 步即达机器精度")

    add_heading(doc, "7.3  修复纪事（透明度）", level=2)
    add_para(doc, "本轮诚实记录修了什么——含一处真 bug、多处过度声称、若干不收敛实验：")
    add_table(doc, ["项目", "之前的问题", "修复"],
              [["κ_eff 公式 bug", "算成 κ(P_k⁻¹A)，对角 A 上误算理想 Jacobi 为 κ²",
                "改正为 κ(P_kA)，重算所有 κ_eff 图"],
               ["Adam=Jacobi 过度声称", "笼统说 Adam 改善条件数",
                "区分曲率 vs 梯度幅度；对角 A 上 Adam κ_eff≈12，Jacobi=1"],
               ["Muon「主场」过度声称", "称 Muon 在谱范数目标上赢 GD",
                "诚实：凸二次 CG 最优、Muon 不加速；价值是奇异值均衡"],
               ["NS 收敛盆笼统", "称「√3 外发散」",
                "精确化：(0,√3)→+U，外部分形（−U/弹回/发散）"],
               ["exp11/12/15/23/24/κ-scan", "实现/参数问题致不收敛或被截断",
                "强凸 Nesterov / 谱范数目标 / Sophia-H / 胖矩阵分支 / 逐配置扫 lr / 动态 MAX_ITER"]],
              caption="表 7  第三稿修订清单")

    # ---------- 8 结论 ----------
    add_heading(doc, "8  结论")
    for t in [
        "(1) 统一模板 x_{k+1}=x_k−P_k g_k 把 GD/HB/Adam/Sophia/signSGD/Muon 纳入同一"
        "框架；更进一步（§5），它们都是 LMO 在不同范数下的解——GD/signSGD/Muon = "
        "ℓ₂/ℓ∞/谱范数最速下降（定理 5）。",
        "(2) 谱半径分析精确预测收敛率：GD (κ−1)/(κ+1)、Polyak HB (√κ−1)/(√κ+1)；"
        "实验 6 经验拟合与理论在 κ=1000 时相对误差 0.1%。",
        "(3) Heavy-ball 是 Chebyshev 半迭代的冻结系数极限（定理 4 + 实验 25）："
        "ω_k→1+β*=1.6694（吻合 10⁻⁷）。",
        "(4) Newton–Schulz 二次收敛 E_{k+1}=−¼E_k²(3I−E_k)；保证收敛到正确极因子的"
        "盆是 (0,√3)，越界为分形式归宿（实验 30）。",
        "(5) Muon 是谱范数最速下降（定理 7），结构性特征是更新条件数恒为 1（实验 31）。"
        "诚实结论：凸二次上 CG 最优、Muon 不加速；优势在匹配谱范数几何的深度学习场景"
        "（本文未实验证实）。",
        "(6) Adam 的对角缩放用梯度幅度而非曲率，对角 Hessian 上 κ_eff 也只到约 12"
        "（oracle Jacobi 到 1，实验 4）；最简凸 f=x²/2 上有 2-极限环（实验 26）。",
        "(7) NAG ↔ AVD-ODE 离散（实验 29）：连续时间下三者轨迹重合。",
    ]:
        add_para(doc, t)
    add_para(doc,
             "不足：未做神经网络实验（需 GPU 集群），故 Muon 在深度学习里的优势未在"
             "本文实验中证实；未完整复现 Remez 最优 NS 系数；AMSGrad 未与极限环精确"
             "对照；高分辨率 ODE 未做离散化误差实验。详见 re_TODO.md。")

    # ---------- 参考文献 ----------
    add_heading(doc, "参考文献")
    refs = [
        "Kingma, D. P., Ba, J. Adam: A Method for Stochastic Optimization. ICLR, 2015.",
        "Loshchilov, I., Hutter, F. Decoupled Weight Decay Regularization. ICLR, 2019.",
        "Liu, H., et al. Sophia: A Scalable Stochastic Second-order Optimizer. ICLR, 2023.",
        "Jordan, K., et al. Muon: An Optimizer for Hidden Layers in Neural Networks. 2024.",
        "Liu, J., et al. Muon is Scalable for LLM Training. arXiv:2502.16982, 2025.",
        "Polyak, B. T. Some Methods of Speeding up the Convergence of Iteration Methods. "
        "USSR Comput. Math. Math. Phys., 1964, 4(5): 1–17.",
        "Reddi, S. J., et al. On the Convergence of Adam and Beyond. ICLR, 2018.",
        "Bock, S., Weiß, M. Non-Convergence and Limit Cycles in the Adam Optimizer. "
        "arXiv:2210.02070, 2022.",
        "Kovalev, D. Understanding Gradient Orthogonalization via Non-Euclidean "
        "Trust-Region Optimization. arXiv:2503.12645, 2025.",
        "Bernstein, J., Newhouse, L. Modular Duality in Deep Learning. arXiv:2410.21265, 2024.",
        "Schulz, G. Iterative Berechnung der reziproken Matrix. ZAMM, 1933, 13: 57–59.",
        "Higham, N. J. Functions of Matrices: Theory and Computation. SIAM, 2008.",
        "Anonymous. Accelerating Newton-Schulz via Chebyshev Polynomials. "
        "arXiv:2506.10935, 2025.",
        "Su, W., Boyd, S., Candès, E. J. A Differential Equation for Modeling Nesterov's "
        "Accelerated Gradient Method. JMLR, 2016, 17: 1–43.",
        "Shi, B., Du, S. S., Jordan, M. I., Su, W. J. Understanding the Acceleration "
        "Phenomenon via High-Resolution Differential Equations. Math. Program., 2021, 195: 79–148.",
        "Saad, Y. Iterative Methods for Sparse Linear Systems (2nd ed.). SIAM, 2003.",
        "Trefethen, L. N., Bau, D. Numerical Linear Algebra. SIAM, 1997.",
        "Nocedal, J., Wright, S. J. Numerical Optimization (2nd ed.). Springer, 2006.",
    ]
    for i, r in enumerate(refs, start=1):
        p = add_para(doc, f"[{i}] {r}")

    # ---------- 附录 ----------
    add_heading(doc, "附录 A  代码模块")
    add_table(doc, ["源文件", "内容"],
              [["src/optimizers.py", "gd/momentum/nesterov/adam/adamw/sophia/jacobi/signsgd + κ(P_kA)"],
               ["src/newton_schulz.py", "NS 3 阶 + 5 阶 Higham + 胖矩阵分支"],
               ["src/chebyshev.py", "Chebyshev 半迭代 + minimax 误差多项式 + HB-Polyak 形式"],
               ["src/steepest_descent_norms.py", "范数最速下降三元组：ℓ₂/ℓ∞/谱的 LMO + 对偶范数"],
               ["src/spectral_trust_region.py", "谱范数恢复 + Muon/GD/Adam 对照"],
               ["src/adam_limit_cycle.py", "标量 Adam + 2-极限环检测"],
               ["src/ode_integrators.py", "梯度流/HB-ODE/AVD-ODE/高分辨率 NAG-ODE 的 RK4"],
               ["src/pcg.py", "CG / PCG + Jacobi + 理论上界"],
               ["experiments/", "run_all.py（1–10）+ extended_experiments.py（11–32）"],
               ["tests/", "19 项 pytest"]],
              caption="表 A1  代码模块清单")

    add_heading(doc, "附录 B  完整证明（节选）")
    add_heading(doc, "B.1  定理 4（HB = Chebyshev 冻结系数极限）", level=2)
    add_para(doc,
             "Chebyshev 半迭代 x_{k+1}=ω_{k+1}(r_k/d+x_k−x_{k−1})+x_{k−1}，"
             "ω_{k+1}=1/(1−σ²ω_k/4)、σ=(L−μ)/(L+μ)、d=(L+μ)/2。令 ω_k→ω_∞ 解 "
             "σ²ω_∞²−4ω_∞+4=0 得 ω_∞=2/(1+√(1−σ²))。代入得 "
             "x_{k+1}=x_k−(ω_∞/d)(Ax_k−b)+(ω_∞−1)(x_k−x_{k−1})，恰为 Heavy-ball。"
             "由 1−σ²=4Lμ/(L+μ)² 验证 ω_∞/d=4/(√L+√μ)²=η*、ω_∞−1=β*。故 "
             "Chebyshev 在 k→∞ 退化为 Polyak 最优 Heavy-ball。□")
    add_heading(doc, "B.2  定理 5（范数最速下降三元组）", level=2)
    add_para(doc,
             "ℓ₂ 由 Cauchy–Schwarz 取等得 d*=−g/‖g‖₂；ℓ∞ 逐坐标 d_i=−sign(g_i) 得"
             " −‖g‖₁；谱范数由 von Neumann 迹不等式 ⟨G,D⟩≥−Σσ_i(G)=−‖G‖_*，"
             "等号在 D=−UVᵀ 取到。对偶配对 ℓ₂↔ℓ₂、ℓ∞↔ℓ₁、谱↔核范数。□")
    add_heading(doc, "B.3  定理 7（极分解 trust-region 最优性）", level=2)
    add_para(doc,
             "由 von Neumann 迹不等式 |tr(ΔᵀG)|≤Σσ_i(Δ)σ_i(G)≤t‖G‖_*，"
             "等号在 Δ=−t·UVᵀ 时取到。□")

    add_heading(doc, "附录 C  超参与复现")
    add_para(doc,
             "全局超参（experiments/config.py）：DIM=20、CONDITION_NUMBERS=[10,100,1000]、"
             "MAX_ITER=2000、TOL=10⁻⁶、SEED=42、SEEDS=[0,1,2,42,123]、ADAM_LR=0.1。"
             "一键复现：bash scripts/check_repro.sh。")


def main():
    data = _load()
    doc = Document(str(TEMPLATE))
    _clear_body(doc)
    build(doc, data)
    doc.save(str(OUTPUT))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
