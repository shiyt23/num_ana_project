# 数值分析课程项目：以 Muon 为核心的现代优化器统一框架

**题目**：正交化梯度方法的数值分析视角：从 Newton–Schulz 极分解到 Muon 优化器及其统一框架（选题 2）
**成员**：赵泽霖（2023010848）、史云天（2023010836）

> 第三稿（C 路线 + 范数最速下降统一框架）。在 Muon-centric 主线上新增"GD/signSGD/Muon = ℓ₂/ℓ∞/谱范数最速下降"的 LMO 统一框架（§5）。**本稿尤其强调诚实性**：修正一处真 bug（有效条件数应为 κ(P_kA) 而非 κ(P_k⁻¹A)）、纠正多处过度声称（Adam≠Jacobi、Muon 在凸二次上不加速、NS 收敛盆三种归宿）。完整修订见 `report/report.md` §7.4 修复纪事。

## 一键复现

```bash
conda create -n num_ana_opt python=3.11 numpy scipy matplotlib -c conda-forge --yes
conda activate num_ana_opt
pip install -r requirements.txt   # 含 pytest, python-docx

bash scripts/check_repro.sh        # = pytest + run_all + appendix tables + figure count
```

输出：
- `figures/` 共 **35 张 PNG**
- `data/experiment_results.json`（自动）
- `report/appendix_auto.md`、`report/appendix_b_embed.md`（自动）
- `final_report.docx`（按模板自动生成，含图片渲染的行间公式）
- **19 项 pytest** 应全部通过（约 0.1 秒）

## 报告

- **`report/report.md`**：主报告（约 990 行），8 节正文 + 附录 A–D
- **`report/defense_one_page.md`**：答辩 1 页素材（3 图 + 话术）
- **`final_report.docx`**：按 `报告模板.docx` 样式填好的最终交付件（由 `scripts/build_docx.py` 生成；行间公式用 matplotlib 渲染为真实数学图片）

### 章节—代码—图—数据索引

| 报告章节 | 主题 | 代码模块 | 关键图 |
|----------|------|----------|--------|
| §2.2 | Krylov + Chebyshev minimax | `src/chebyshev.py`, `src/pcg.py` | exp17, exp25 |
| §3.1 | GD = Richardson | `src/optimizers.py:_run_gd` | exp1, exp5, exp6 |
| §3.2 | HB = Chebyshev 冻结系数极限 | `src/chebyshev.py` | exp2, exp7, exp22, **exp25** |
| §3.3 | Adam 对角缩放 + κ_eff 真相 + 极限环 | `src/optimizers.py`, `src/adam_limit_cycle.py` | **exp4**, exp8, exp24, **exp26** |
| §4.1 | Newton–Schulz 二次收敛 + 收敛盆 | `src/newton_schulz.py` | exp3, exp9, exp23, **exp30** |
| §4.2 | 5 阶 Higham NS | `src/newton_schulz.py:chebyshev_ns_iterate` | **exp27** |
| §4.3–4.4 | Muon = 谱范数最速下降 + 奇异值均衡 | `src/spectral_trust_region.py` | **exp12**, **exp28**, **exp31** |
| **§5** | **范数最速下降统一框架（LMO）** | `src/steepest_descent_norms.py` | **exp32** |
| §6 | NAG-ODE / 高分辨率 ODE | `src/ode_integrators.py` | **exp29** |
| §7 | 综合实验列表 | `experiments/{run_all,extended_experiments}.py` | 全部 35 张 |

**粗体**为本稿修复或新增的实验。

## 项目结构

```
num_ana_project/
├── src/
│   ├── quadratic.py              # 病态二次问题
│   ├── optimizers.py             # GD/HB/Nesterov/Adam/AdamW/Sophia/Jacobi/signSGD + κ(P_kA)
│   ├── momentum_spectrum.py      # Polyak 谱半径计算
│   ├── newton_schulz.py          # 3 阶 + 5 阶 NS（含胖矩阵分支）
│   ├── chebyshev.py              # Chebyshev 半迭代 + minimax 多项式
│   ├── steepest_descent_norms.py # 范数最速下降三元组：ℓ₂/ℓ∞/谱的 LMO + 对偶范数
│   ├── spectral_trust_region.py  # 谱范数恢复目标 + Muon/GD/Adam 对比
│   ├── adam_limit_cycle.py       # Bock-Weiß 2-极限环数值复现
│   ├── ode_integrators.py        # 梯度流/HB-ODE/AVD-ODE/高分辨率 NAG-ODE
│   ├── pcg.py                    # CG + PCG + 理论上界
│   └── matrix_quadratic.py       # 矩阵二次 + Muon 接口
├── experiments/
│   ├── config.py                 # MAX_ITER=2000, SEED=42, ADAM_LR=0.1, ...
│   ├── run_all.py                # exp1–10 + 入口
│   └── extended_experiments.py   # exp11–32
├── tests/                        # 19 项 pytest
├── scripts/
│   ├── check_repro.sh            # 一键复现脚本
│   ├── generate_appendix_tables.py  # 从 JSON 自动生成附录
│   └── build_docx.py             # 按模板生成 final_report.docx（公式渲染为图片）
├── figures/                      # 35 张 PNG
├── data/experiment_results.json  # 实验数值摘要
├── report/
│   ├── report.md                 # 主报告
│   ├── _eqn_cache/               # docx 行间公式的渲染缓存（自动）
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

## 修订纪事

### 第三稿（本稿）：诚实性修复 + 范数统一框架

| 类别 | 修订 |
|------|------|
| **真 bug** | 有效条件数 `effective_preconditioned_condition_number` 由 κ(P_k⁻¹A) 改正为 κ(P_kA)（误差递推 e_{k+1}=(I−P_kA)e_k）|
| **过度声称纠正** | Adam≠Jacobi（曲率 vs 梯度幅度）；Muon 在凸二次上不加速；NS 收敛盆三种归宿 |
| 新增统一框架 | §5 范数最速下降（GD/signSGD/Muon = ℓ₂/ℓ∞/谱的 LMO），定理 5 |
| 新增模块 | `steepest_descent_norms.py` + `optimizers.py` 加 `signsgd` |
| 新增实验 | E31 Muon 奇异值均衡、E32 范数 LMO 三元组；改进 E25（ω_k 收敛）、E30（三种归宿）|
| docx 排版 | 行间公式渲染为真实数学图片、定理用模板「定理」样式、表头着色、整体留白 |
| 测试 / 图 | 14 → **19 项 pytest**；33 → **35 张图** |

### 第二稿：E 综合 → C Muon-centric

补完整 Chebyshev/Krylov 链条、Adam 极限环、NAG-ODE、Muon trust-region；修复 7 个不收敛实验（exp11/12/15/23/24/exp_kappa_scan/exp4）。
