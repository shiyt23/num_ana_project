#!/usr/bin/env python3
"""从 experiment_results.json 生成附录 Markdown（完整 + 嵌入 report 的精简表）。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "experiment_results.json"
OUT_FULL = ROOT / "report" / "appendix_auto.md"
OUT_EMBED = ROOT / "report" / "appendix_b_embed.md"


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:
            return "NaN"
        if abs(v) < 1e-3 or abs(v) > 1e4:
            return f"{v:.4e}"
        return f"{v:.4g}"
    return str(v)


def _table(rows: list[tuple[str, str]]) -> list[str]:
    lines = ["| 项 | 值 |", "|----|-----|"]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    return lines


def build_embed(data: dict) -> str:
    lines = [
        "<!-- 本文件由 scripts/generate_appendix_tables.py 自动生成，勿手工编辑 -->",
        "",
        "### 实验 1：达到 $10^{-6}$ 的迭代步数",
        "",
    ]
    e1 = data.get("exp1", {})
    for kappa in [10, 100, 1000]:
        lines.extend(
            _table(
                [
                    (f"κ={kappa} GD", _fmt(e1.get(f"kappa_{kappa}_gd", {}).get("iter_to_1e6"))),
                    (f"κ={kappa} Momentum", _fmt(e1.get(f"kappa_{kappa}_momentum", {}).get("iter_to_1e6"))),
                    (f"κ={kappa} Adam", _fmt(e1.get(f"kappa_{kappa}_adam", {}).get("iter_to_1e6"))),
                ]
            )
        )
        lines.append("")

    e7 = data.get("exp7", {})
    lines += [
        "### 实验 7 / E22（κ=100）",
        "",
        *_table(
            [
                ("Polyak β*", _fmt(e7.get("beta_opt"))),
                ("β=0.9 达 tol 步数", _fmt(e7.get("iter_1e6_beta_0.9"))),
                ("Polyak β* 达 tol 步数", _fmt(e7.get("iter_1e6_beta_0.6694214876033059"))),
            ]
        ),
        "",
    ]
    e22 = data.get("exp22", {})
    if e22:
        lines += [
            "",
            *_table(
                [
                    ("E22 ρ* (Polyak)", _fmt(e22.get("rho_polyak"))),
                    ("E22 ρ at β=0.9", _fmt(e22.get("rho_at_beta09"))),
                ]
            ),
            "",
        ]

    e13 = data.get("exp13", {})
    lines += [
        "### 实验 13：Adam 学习率（κ=100）",
        "",
        *_table(
            [
                ("best_lr", _fmt(e13.get("best_lr"))),
                ("best_iter", _fmt(e13.get("best_iter"))),
            ]
        ),
        "",
    ]

    e2 = data.get("exp2", {})
    lines += [
        "### 实验 2（κ=100 谱半径）",
        "",
        *_table(
            [
                ("ρ Polyak", _fmt(e2.get("kappa_100_rho_polyak"))),
                ("ρ GD", _fmt(e2.get("kappa_100_rho_gd"))),
            ]
        ),
        "",
        "完整键值见 `report/appendix_auto.md` 与 `data/experiment_results.json`。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    lines = ["# 附录：实验数值自动摘要", "", f"来源：`{DATA.relative_to(ROOT)}`", ""]
    for key in sorted(data.keys()):
        lines.append(f"## {key}")
        lines.append("")
        if isinstance(data[key], dict):
            lines.append("| 键 | 值 |")
            lines.append("|----|-----|")
            for k, v in data[key].items():
                lines.append(f"| `{k}` | {v} |")
        lines.append("")
    OUT_FULL.write_text("\n".join(lines), encoding="utf-8")
    OUT_EMBED.write_text(build_embed(data), encoding="utf-8")
    print(f"Wrote {OUT_FULL}")
    print(f"Wrote {OUT_EMBED}")


if __name__ == "__main__":
    main()
