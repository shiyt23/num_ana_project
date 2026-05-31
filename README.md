# 正交化梯度方法的数值分析视角

**课程**：数值分析与算法 · **小组成员**：赵泽霖（2023010848）、史云天（2023010836）

将 GD、Heavy-ball、Adam、Muon 等现代优化器统一为迭代模板 \(x_{k+1} = x_k - P_k g_k\)，从 Newton–Schulz 极分解、Chebyshev 半迭代与谱范数 trust-region 等数值分析视角加以分析，并配套 20 组数值实验与可复现代码。

> **完整报告**（理论推导、实验解读、证明与数值表）见 [`report/report.md`](report/report.md)。

## 快速开始

```bash
conda create -n num_ana_opt python=3.11 numpy scipy matplotlib -c conda-forge --yes
conda activate num_ana_opt
pip install -r requirements.txt
pytest tests/ -q
python experiments/run_all.py
python experiments/extended_experiments.py
```

## 项目结构

```
num_ana_project/
├── src/                        # 核心算法
├── experiments/                # 实验脚本
├── tests/                      # 回归测试（14 项）
├── figures/                    # 实验图（运行后生成）
├── data/                       # 实验 JSON 摘要
└── report/                     # 报告与附录
```

### `src/` — 算法实现

| 文件 | 功能 |
|------|------|
| `quadratic.py` | 病态二次问题构造、目标函数与梯度 |
| `optimizers.py` | GD / Heavy-ball / Nesterov / Adam / AdamW / Sophia / Jacobi |
| `momentum_spectrum.py` | Polyak Heavy-ball 谱半径计算 |
| `chebyshev.py` | Chebyshev 半迭代、minimax 误差多项式 |
| `newton_schulz.py` | 标准 Newton–Schulz 与 Higham 五次多项式极分解迭代 |
| `spectral_trust_region.py` | 谱范数恢复问题、Muon 与基线对照 |
| `matrix_quadratic.py` | Frobenius 矩阵二次问题、Muon-NS 方向 |
| `adam_limit_cycle.py` | 标量 Adam 迭代与 2-极限环检测 |
| `ode_integrators.py` | 梯度流 / Heavy-ball / AVD-ODE / 高分辨率 NAG-ODE（RK4） |
| `pcg.py` | CG / PCG（Jacobi 预条件）及理论上界 |

### `experiments/` — 实验入口

| 文件 | 功能 |
|------|------|
| `run_all.py` | 实验 1–10 |
| `extended_experiments.py` | 实验 11–30 |
| `config.py` | 全局超参（`MAX_ITER`、`SEED`、`ADAM_LR` 等） |

### `tests/` — 单元测试

覆盖优化器、谱半径、Newton–Schulz、CG/Chebyshev、ODE 与 Muon 等核心模块。

### `report/` — 文档

| 文件 | 功能 |
|------|------|
| `report.md` | 主报告（正文、结论、完整证明） |
| `appendix_auto.md` | 实验数值自动摘要 |
