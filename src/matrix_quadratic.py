"""矩阵变量二次损失：min_W 0.5 ||A W - B||_F^2。"""

from __future__ import annotations

import numpy as np

from .newton_schulz import newton_schulz_iterate


def make_matrix_problem(
    m: int,
    n: int,
    condition_number: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """返回 A, B, W_star, mu, L（A 为 SPD）。"""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((m, m)))
    eigvals = np.geomspace(1.0, condition_number, m)
    a = q @ np.diag(eigvals) @ q.T
    w_star = rng.standard_normal((m, n))
    b = a @ w_star
    return a, b, w_star, float(eigvals.min()), float(eigvals.max())


def objective(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    r = a @ w - b
    return 0.5 * float(np.sum(r * r))


def gradient(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    return a.T @ (a @ w - b)


def muon_direction(g: np.ndarray, ns_steps: int = 10) -> np.ndarray:
    """Newton-Schulz 正交化梯度，作为 Muon 更新方向。"""
    xs, _ = newton_schulz_iterate(g, max_iter=ns_steps, scale_factor=1.0)
    return xs[-1]


def run_matrix_gd(
    a: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    max_iter: int,
    lr: float,
) -> tuple[list[float], list[np.ndarray]]:
    w = w0.copy()
    fs, ws = [objective(a, b, w)], [w.copy()]
    for _ in range(max_iter):
        w = w - lr * gradient(a, b, w)
        fs.append(objective(a, b, w))
        ws.append(w.copy())
    return fs, ws


def run_matrix_muon(
    a: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    max_iter: int,
    lr: float,
    ns_steps: int = 10,
) -> tuple[list[float], list[np.ndarray]]:
    w = w0.copy()
    fs, ws = [objective(a, b, w)], [w.copy()]
    for _ in range(max_iter):
        g = gradient(a, b, w)
        direction = muon_direction(g, ns_steps=ns_steps)
        w = w - lr * direction
        fs.append(objective(a, b, w))
        ws.append(w.copy())
    return fs, ws


def run_matrix_adam_flat(
    a: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    max_iter: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[list[float], list[np.ndarray]]:
    """将 W 展平后使用 Adam。"""
    w = w0.copy()
    m_shape = w.shape
    x = w.reshape(-1)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    fs = [objective(a, b, w)]

    for t in range(1, max_iter + 1):
        w = x.reshape(m_shape)
        g = gradient(a, b, w).reshape(-1)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        w = x.reshape(m_shape)
        fs.append(objective(a, b, w))
    return fs, [w0]  # 不存全部 ws 以省内存
