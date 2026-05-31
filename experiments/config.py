"""实验全局配置（统一超参）。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"

# 问题规模
DIM = 20
CONDITION_NUMBERS = [10, 100, 1000]

# 迭代
MAX_ITER = 2000  # 取较大上限以便 GD 在 κ=1000 时仍可到 tol
TOL = 1e-6

# 随机性：主实验固定种子；多种子实验使用 SEEDS
SEED = 42
SEEDS = [0, 1, 2, 42, 123]

# Adam 族：二次问题上梯度范数较大，lr 略大于典型 DL 设置
# exp1/6/8/11/17 使用 ADAM_LR；exp4 可略大以观察预条件演化
ADAM_LR = 0.1
ADAM_LR_PRECOND_STUDY = 0.5  # 实验 4：强调预条件矩阵变化
ADAM_LR_2D = 0.3  # 实验 10：二维轨迹可视化

# 其他
NESTEROV_USE_STRONG_CONVEX_PARAMS = True  # 与 Polyak 同阶加速率参数便于对照
SGD_NOISE_SIGMA = 0.01  # 实验 14
