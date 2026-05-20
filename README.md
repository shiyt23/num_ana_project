# 数值分析课程项目：经典迭代法与现代优化器

选题 2 的完整实现，含理论推导（见 `report/report.md`）、实验代码与可复现图表。

**成员**：赵泽霖（2023010848）、史云天（2023010836）

**分工（参考）**：赵泽霖 — 理论推导与报告 §3；史云天 — 实验代码与图表；共同 — 实验设计与结论。

**答辩 1 页素材**：`report/defense_one_page.md`（三图 + 话术，可转 PPT）

## 快速开始

```bash
cd num_ana_project
conda create -n num_ana_opt python=3.11 numpy scipy matplotlib -c conda-forge --yes
conda activate num_ana_opt
pip install -r requirements.txt   # 含 pytest

python -m pytest tests/ -q
python experiments/run_all.py
python scripts/generate_appendix_tables.py
bash scripts/check_repro.sh          # 提交前一键检查
```

## 理论—代码—图表索引

| 报告章节 | 内容 | 实验 | 主要图表 |
|----------|------|------|----------|
| §3.1 | GD / Richardson | 1, 5, 6 | `exp1_convergence.png`, `exp5_step_sensitivity.png`, `exp6_*.png` |
| §3.2 | Momentum / Polyak | 2, 7 | `exp2_*.png`, `exp7_*.png` |
| §3.3 | Adam / Jacobi | 4, 8, 13, 15 | `exp4_precond_kappa.png`, `exp8_*.png`, `exp13_*.png`, `exp15_sophia.png` |
| §3.4 | Newton–Schulz / Muon | 3, 9, 12, 23 | `exp3_*.png`, `exp9_ns_domain.png`, `exp12_matrix_muon.png`, `exp23_*.png` |
| §4 扩展 | Nesterov, PCG, 噪声, 模态/消融 | 11, 14, 17, 21, 24, 1b | `exp11_*.png`, `exp21_*.png`, `exp24_*.png`, `exp1b_*.png` |
| κ 扫描 | 迭代数 vs 理论 ρ | — | `exp_kappa_scan.png` |
| 可视化 | 2D 轨迹 | 10 | `exp10_trajectories_2d.png` |

完整 TODO 完成情况见 **`re_TODO.md`**。

## 目录结构

```
src/optimizers.py          # gd, momentum, nesterov, adam, adamw, sophia, jacobi
src/matrix_quadratic.py    # 矩阵二次 + Muon-NS
src/pcg.py                 # PCG 基线
experiments/config.py      # 统一超参
experiments/run_all.py     # 入口
experiments/extended_experiments.py
figures/                   # 约 26 张图
report/appendix_auto.md    # JSON 自动表格
report/appendix C–D      # 定理证明、答辩备忘（见 report.md）
data/experiment_results.json
report/report.md
report/appendix_auto.md    # 脚本生成
```

## 配置说明（`experiments/config.py`）

| 变量 | 值 | 用途 |
|------|-----|------|
| `ADAM_LR` | 0.1 | 实验 1、6、8、11、15、17 等主收敛曲线 |
| `ADAM_LR_PRECOND_STUDY` | 0.5 | 实验 4：观察 $\kappa(P_k^{-1}A)$ 演化 |
| `ADAM_LR_2D` | 0.3 | 实验 10：二维轨迹 |

分场景学习率是为避免「单一 lr 无法同时适合预条件研究与轨迹可视化」；见 `re_TODO.md` 第八节说明。
