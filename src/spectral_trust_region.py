"""谱范数 trust-region 问题与 Muon 优化器的对照实验设计。

核心动机（Kovalev 2025）：Muon 等价于
    Δ W = argmin_{‖U‖_σ ≤ 1} ⟨G, U⟩_F · t
即沿 ⟨G, ·⟩_F 的谱范数 trust-region 线性子问题最速下降。其闭式解恰为极分解 G = U Σ V^T 中
的正交因子 U V^T 乘以 trust-region 半径 t。

为观察 Muon-NS 在匹配谱范数几何的任务上的下降行为，我们设计两类适配目标：

(1) 谱范数加权矩阵恢复：
    f(W) = (1/2) ‖U Σ V^T - W‖_σ^2  (谱范数损失)
   用于展示 Muon 正交化方向与谱范数 trust-region 线性化子问题的匹配。

(2) 算子拟合（Frobenius 内积线性目标）：
    f(W) = -⟨G_*, W⟩_F + (1/2) ‖W‖_σ^2
   trust-region 解析解为 W* = G_* / ‖G_*‖_σ 的正交因子方向。

这两个目标避开 Frobenius 损失下 Muon 不下降的几何错配。
"""

from __future__ import annotations

import numpy as np

from .newton_schulz import newton_schulz_iterate, polar_factor_via_svd


def make_spectral_recovery_problem(
    m: int,
    n: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成目标矩阵 W* 与其极分解参考。
    使用 W* = U Σ V^T，其中 Σ 接近半正交（奇异值 in [0.5, 1.5]）。
    """
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.standard_normal((m, min(m, n))))
    v, _ = np.linalg.qr(rng.standard_normal((n, min(m, n))))
    sigma = 0.5 + rng.uniform(size=min(m, n))
    w_star = u @ np.diag(sigma) @ v.T
    return w_star, polar_factor_via_svd(w_star)


def spectral_recovery_loss(w: np.ndarray, w_star: np.ndarray) -> float:
    """谱范数下的距离平方（用顶奇异值近似）。"""
    return 0.5 * float(np.linalg.svd(w - w_star, compute_uv=False)[0] ** 2)


def trust_region_objective(
    w: np.ndarray,
    g_target: np.ndarray,
    radius: float,
) -> float:
    """
    f(W) = -⟨G_*, W⟩_F + (1/2 radius^2) ‖W‖_σ^2 · radius^2 / 2
    （一个凸的谱范数惩罚目标；其最小化对应极分解方向。）
    """
    sigma_max = float(np.linalg.svd(w, compute_uv=False)[0])
    return -float((g_target * w).sum()) + 0.5 * sigma_max ** 2


def run_muon_on_spectral_recovery(
    w_star: np.ndarray,
    w0: np.ndarray,
    max_iter: int,
    lr: float,
    ns_steps: int = 8,
) -> tuple[list[float], list[np.ndarray]]:
    """
    Muon 在谱范数恢复目标上：
        ∇_W (1/2) ‖W - W*‖_F^2 = W - W*  (用 Frobenius 梯度作下降方向，
        但 Muon 把它正交化后作用于 W)
    谱范数损失一般非光滑；这里检验的是正交化方向与谱范数 trust-region
    线性化几何相匹配时的下降行为。
    """
    w = w0.copy()
    losses = [spectral_recovery_loss(w, w_star)]
    ws = [w.copy()]
    for _ in range(max_iter):
        g = w - w_star  # Frobenius 梯度
        xs, _ = newton_schulz_iterate(g, max_iter=ns_steps)
        direction = xs[-1]  # 正交因子近似
        w = w - lr * direction
        losses.append(spectral_recovery_loss(w, w_star))
        ws.append(w.copy())
    return losses, ws


def run_gd_on_spectral_recovery(
    w_star: np.ndarray,
    w0: np.ndarray,
    max_iter: int,
    lr: float,
) -> tuple[list[float], list[np.ndarray]]:
    """对照：Frobenius 梯度下降（朝 W* 直线推进）。"""
    w = w0.copy()
    losses = [spectral_recovery_loss(w, w_star)]
    ws = [w.copy()]
    for _ in range(max_iter):
        g = w - w_star
        w = w - lr * g
        losses.append(spectral_recovery_loss(w, w_star))
        ws.append(w.copy())
    return losses, ws


def run_adam_on_spectral_recovery(
    w_star: np.ndarray,
    w0: np.ndarray,
    max_iter: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[list[float], list[np.ndarray]]:
    """对照：将 W 展平后使用 Adam，对谱范数损失。"""
    shape = w0.shape
    x = w0.copy().reshape(-1)
    m_state = np.zeros_like(x)
    v_state = np.zeros_like(x)
    losses = [spectral_recovery_loss(w0, w_star)]
    for t in range(1, max_iter + 1):
        w = x.reshape(shape)
        g = (w - w_star).reshape(-1)
        m_state = beta1 * m_state + (1 - beta1) * g
        v_state = beta2 * v_state + (1 - beta2) * (g * g)
        m_hat = m_state / (1 - beta1 ** t)
        v_hat = v_state / (1 - beta2 ** t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        losses.append(spectral_recovery_loss(x.reshape(shape), w_star))
    return losses, [w0]
