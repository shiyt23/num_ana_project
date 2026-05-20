#!/usr/bin/env bash
# 提交前一键复现检查（对应 TODO §2.2）
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== pytest =="
python -m pytest tests/ -q

echo "== experiments =="
python experiments/run_all.py

echo "== appendix tables =="
python scripts/generate_appendix_tables.py

echo "== figure count =="
ls -1 figures/*.png | wc -l

echo "OK: figures/ and data/experiment_results.json updated."
