# 数值分析课程项目：以 Muon 为核心的现代优化器统一框架

**题目**：正交化梯度方法的数值分析视角：从 Newton–Schulz 极分解到 Muon 优化器及其统一框架（选题 2）
**成员**：赵泽霖（2023010848）、史云天（2023010836）

> 第二稿（C 路线）。把核心从"五机制并列"调成"Muon-centric"，并补完整 Chebyshev/Krylov 链条、Adam 极限环、NAG-ODE 视角与 Muon trust-region 理论。修复了第一稿中 7 个不收敛的实验（详见 `report/report.md` §6.4 修复纪事）。

## 一键复现

```bash
conda create -n num_ana_opt python=3.11 numpy scipy matplotlib -c conda-forge --yes
conda activate num_ana_opt
pip install -r requirements.txt   # 含 pytest, python-docx

bash scripts/check_repro.sh        # = pytest + run_all + appendix tables + figure count
```

输出：
- `figures/` 共 33 张 PNG
- `data/experiment_results.json`（自动）
- `report/appendix_auto.md`、`report/appendix_b_embed.md`（自动）
- 14 项 pytest 应全部通过（约 0.1 秒）

## 报告

- **`report/report.md`**：主报告（约 920 行），9 节正文 + 附录 A–D
- **`report/defense_one_page.md`**：答辩 1 页素材（3 图 + 话术）
- **`final_report.docx`**：按 `报告模板.docx` 样式填好的最终交付件（由 `scripts/build_docx.py` 生成）

### 章节—代码—图—数据索引

| 报告章节 | 主题 | 代码模块 | 关键图 |
|----------|------|----------|--------|
| §2.2 | Krylov + Chebyshev minimax | `src/chebyshev.py`, `src/pcg.py` | exp17 (CG 理论上界), exp25 (Cheb 多项式) |
| §3.1 | GD = Richardson | `src/optimizers.py:_run_gd` | exp1, exp5, exp6 |
| §3.2 | HB ≡ Chebyshev 半迭代 | `src/momentum_spectrum.py`, `src/chebyshev.py` | exp2, exp7, exp22, **exp25** |
| §3.3 | Adam 动态 Jacobi + 极限环 | `src/optimizers.py:_run_adam`, `src/adam_limit_cycle.py` | exp4, exp8, exp24, **exp26** |
| §4.1 | Newton–Schulz 二次收敛 | `src/newton_schulz.py` | exp3, exp9, exp23, **exp30** |
| §4.2 | 5 阶 Higham NS | `src/newton_schulz.py:chebyshev_ns_iterate` | **exp27** |
| §4.3 | Muon = 谱范数 trust-region | `src/spectral_trust_region.py` | **exp12**, **exp28** |
| §5 | NAG-ODE / 高分辨率 ODE | `src/ode_integrators.py` | **exp29** |
| §6 | 综合实验列表 | `experiments/{run_all,extended_experiments}.py` | 全部 33 张 |

**粗体**为本稿修复或新增的实验。

## 项目结构

```
num_ana_project/
├── src/
│   ├── quadratic.py              # 病态二次问题
│   ├── optimizers.py             # GD/HB/Nesterov/Adam/AdamW/Sophia/Jacobi
│   ├── momentum_spectrum.py      # Polyak 谱半径计算
│   ├── newton_schulz.py          # 3 阶 + 5 阶 NS（含胖矩阵分支）
│   ├── chebyshev.py              # Chebyshev 半迭代 + minimax 多项式
│   ├── spectral_trust_region.py  # 谱范数恢复目标 + Muon/GD/Adam 对比
│   ├── adam_limit_cycle.py       # Bock-Weiß 2-极限环数值复现
│   ├── ode_integrators.py        # 梯度流/HB-ODE/AVD-ODE/高分辨率 NAG-ODE
│   ├── pcg.py                    # CG + PCG + 理论上界
│   └── matrix_quadratic.py       # 矩阵二次 + Muon 接口
├── experiments/
│   ├── config.py                 # MAX_ITER=2000, SEED=42, ADAM_LR=0.1, ...
│   ├── run_all.py                # exp1–10 + 入口
│   └── extended_experiments.py   # exp11–30
├── tests/                        # 14 项 pytest
├── scripts/
│   ├── check_repro.sh            # 一键复现脚本
│   ├── generate_appendix_tables.py  # 从 JSON 自动生成附录
│   └── build_docx.py             # 按模板生成 final_report.docx
├── figures/                      # 33 张 PNG
├── data/experiment_results.json  # 实验数值摘要
├── report/
│   ├── report.md                 # 主报告
│   ├── appendix_auto.md          # 完整自动附录
│   ├── appendix_b_embed.md       # 精选附录（嵌入正文）
│   └── defense_one_page.md       # 答辩素材
├── 报告模板.docx                  # 课程提供的模板
└── final_report.docx             # 最终交付件（按模板）
```

## 关键超参（`experiments/config.py`）

| 变量 | 值 | 备注 |
|------|-----|------|
| `DIM` | 20 | 二次问题维度 |
| `MAX_ITER` | 2000 | 提到 2000 以便 GD 在 κ=1000 上不撞顶 |
| `CONDITION_NUMBERS` | [10, 100, 1000] | 主扫 κ |
| `SEED` | 42 | 主实验 |
| `SEEDS` | [0,1,2,42,123] | 多种子带（exp1_multiseed, exp28） |
| `ADAM_LR` | 0.1 | exp1/6/8/11/17 等 |
| `TOL` | 1e-6 | 一致达 tol 阈值 |

## 这一稿的修订纪事（与第一稿对比）

| 类别 | 修订 |
|------|------|
| 方向 | E 综合 → **C Muon-centric** |
| 不收敛实验 | exp11/12/15/23/24/exp_kappa_scan/exp4 全部修复 |
| 新增前沿实验 | E25 HB≡Cheb, E26 Adam 极限环, E27 5 阶 NS, E28 谱范数 trust-region, E29 NAG-ODE, E30 NS 收敛盆 |
| 新增模块 | `chebyshev.py`, `ode_integrators.py`, `spectral_trust_region.py`, `adam_limit_cycle.py` |
| 测试 | 7 → **14 项 pytest** |
| 图表 | 27 → **33 张** |
| 报告 | 570 行 → **923 行**，含 7 个完整定理证明 |
| 输出 | docx 模板填充版 `final_report.docx` |
