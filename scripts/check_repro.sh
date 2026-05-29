#!/usr/bin/env bash
# 提交前一键复现：pytest + 实验 + 附录表 + docx 生成
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== pytest =="
python -m pytest tests/ -q

echo "== experiments =="
python experiments/run_all.py

echo "== appendix tables =="
python scripts/generate_appendix_tables.py

echo "== build final_report.docx =="
python scripts/build_docx.py

echo "== figure count =="
ls -1 figures/*.png | wc -l

echo "OK: figures/, data/experiment_results.json, report/appendix_*.md, final_report.docx 已更新。"
