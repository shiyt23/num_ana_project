"""真实数据上的全连接层训练：在统一模板 x_{k+1}=x_k-P_k g_k 下比较
SGD / Adam / Muon。

数据集为 scikit-learn 自带的手写数字 digits（8x8 灰度，1797 张，10 类），
网络是一个单隐层 MLP：

    x (64) -> W1 (64xH) -> ReLU -> W2 (Hx10) -> softmax

其中 W1, W2 都是矩阵参数层，正是 Muon 设计的作用对象：Muon 对每个矩阵
梯度做 Newton-Schulz 正交化，再沿正交化方向更新；偏置为向量参数，统一用
带动量的一阶规则更新。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .newton_schulz import newton_schulz_iterate


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_digits_split(
    test_ratio: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """载入 digits，标准化特征并按比例划分训练/测试集。"""
    from sklearn.datasets import load_digits

    data = load_digits()
    x = data.data.astype(np.float64)
    y = data.target.astype(np.int64)

    # 标准化（用全体统计量；避免常数列除零）
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x = (x - mean) / std

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    n_test = int(round(test_ratio * len(x)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


# ---------------------------------------------------------------------------
# 网络前向 / 反向
# ---------------------------------------------------------------------------
@dataclass
class MLPParams:
    w1: np.ndarray  # (d_in, hidden)
    b1: np.ndarray  # (hidden,)
    w2: np.ndarray  # (hidden, n_class)
    b2: np.ndarray  # (n_class,)


def init_params(d_in: int, hidden: int, n_class: int, seed: int) -> MLPParams:
    rng = np.random.default_rng(seed)
    # He 初始化（隐层 ReLU）+ 较小的输出层尺度
    w1 = rng.standard_normal((d_in, hidden)) * np.sqrt(2.0 / d_in)
    w2 = rng.standard_normal((hidden, n_class)) * np.sqrt(1.0 / hidden)
    return MLPParams(w1, np.zeros(hidden), w2, np.zeros(n_class))


def _softmax_cross_entropy(
    logits: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    """返回平均交叉熵损失与 softmax 概率。"""
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
    return float(loss), probs


def forward_backward(
    params: MLPParams, x: np.ndarray, y: np.ndarray
) -> tuple[float, dict[str, np.ndarray]]:
    """一次完整前向 + 反向，返回 batch 平均损失与各参数梯度。"""
    n = len(x)
    h_pre = x @ params.w1 + params.b1            # (n, hidden)
    h = np.maximum(h_pre, 0.0)                   # ReLU
    logits = h @ params.w2 + params.b2           # (n, n_class)
    loss, probs = _softmax_cross_entropy(logits, y)

    d_logits = probs.copy()
    d_logits[np.arange(n), y] -= 1.0
    d_logits /= n

    g_w2 = h.T @ d_logits
    g_b2 = d_logits.sum(axis=0)
    d_h = d_logits @ params.w2.T
    d_h_pre = d_h * (h_pre > 0.0)
    g_w1 = x.T @ d_h_pre
    g_b1 = d_h_pre.sum(axis=0)

    grads = {"w1": g_w1, "b1": g_b1, "w2": g_w2, "b2": g_b2}
    return loss, grads


def accuracy(params: MLPParams, x: np.ndarray, y: np.ndarray) -> float:
    h = np.maximum(x @ params.w1 + params.b1, 0.0)
    logits = h @ params.w2 + params.b2
    return float(np.mean(np.argmax(logits, axis=1) == y))


# ---------------------------------------------------------------------------
# 优化器状态与更新
# ---------------------------------------------------------------------------
def _muon_direction(g: np.ndarray, ns_steps: int) -> np.ndarray:
    """对矩阵梯度做 Newton-Schulz 正交化，得到极分解正交因子近似。"""
    xs, _ = newton_schulz_iterate(g, max_iter=ns_steps, scale_factor=1.0)
    return xs[-1]


def train_mlp(
    optimizer: str,
    hidden: int = 32,
    epochs: int = 60,
    lr: float | None = None,
    seed: int = 0,
    *,
    momentum: float = 0.9,
    ns_steps: int = 5,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    test_ratio: float = 0.3,
) -> dict:
    """在 digits 上整批（full-batch）训练单隐层 MLP。

    optimizer ∈ {"sgd", "adam", "muon"}。三者共享同一前向/反向，仅更新规则不同：

    - sgd  : 带 (Nesterov 风格) 动量的梯度下降，所有参数同规则。
    - adam : 矩阵权重展平后做对角自适应；偏置同样用 Adam。
    - muon : 矩阵权重 W1,W2 用带动量梯度的 Newton-Schulz 正交化方向更新；
             偏置（向量）退化为带动量的 SGD。
    """
    x_tr, y_tr, x_te, y_te = load_digits_split(test_ratio=test_ratio, seed=seed)
    d_in, n_class = x_tr.shape[1], int(y_tr.max() + 1)
    params = init_params(d_in, hidden, n_class, seed=seed)

    default_lr = {"sgd": 0.2, "adam": 0.02, "muon": 0.5}
    if lr is None:
        lr = default_lr[optimizer]

    keys = ["w1", "b1", "w2", "b2"]
    matrix_keys = {"w1", "w2"}
    mom = {k: np.zeros_like(getattr(params, k)) for k in keys}
    v_state = {k: np.zeros_like(getattr(params, k)) for k in keys}
    beta1, beta2 = adam_betas

    train_losses: list[float] = []
    test_accs: list[float] = []

    for t in range(1, epochs + 1):
        loss, grads = forward_backward(params, x_tr, y_tr)
        train_losses.append(loss)
        test_accs.append(accuracy(params, x_te, y_te))

        for k in keys:
            g = grads[k]
            p = getattr(params, k)
            if optimizer == "sgd":
                mom[k] = momentum * mom[k] + g
                p = p - lr * mom[k]
            elif optimizer == "adam":
                mom[k] = beta1 * mom[k] + (1 - beta1) * g
                v_state[k] = beta2 * v_state[k] + (1 - beta2) * (g * g)
                m_hat = mom[k] / (1 - beta1 ** t)
                v_hat = v_state[k] / (1 - beta2 ** t)
                p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
            elif optimizer == "muon":
                mom[k] = momentum * mom[k] + g
                if k in matrix_keys:
                    # Nesterov 风格：用 g + momentum*buf 作正交化输入
                    update_dir = g + momentum * mom[k]
                    direction = _muon_direction(update_dir, ns_steps)
                    # 按 fan-out 缩放，使更新幅度与矩阵形状无关（Muon 常用做法）
                    scale = np.sqrt(max(p.shape[0], p.shape[1]))
                    p = p - lr * scale * direction / max(p.shape[0], p.shape[1])
                else:
                    p = p - lr * mom[k]
            else:
                raise ValueError(f"未知优化器: {optimizer}")
            setattr(params, k, p)

    # 末次评估
    final_loss, _ = forward_backward(params, x_tr, y_tr)
    train_losses.append(final_loss)
    test_accs.append(accuracy(params, x_te, y_te))

    return {
        "train_losses": train_losses,
        "test_accs": test_accs,
        "final_train_loss": float(train_losses[-1]),
        "final_test_acc": float(test_accs[-1]),
        "best_test_acc": float(max(test_accs)),
        "n_train": int(len(x_tr)),
        "n_test": int(len(x_te)),
        "d_in": int(d_in),
        "hidden": int(hidden),
        "n_class": int(n_class),
    }
