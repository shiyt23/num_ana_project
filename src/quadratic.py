"""病态二次函数构造与梯度/目标值计算。"""

from __future__ import annotations

import numpy as np


def make_quadratic_problem(
    dim: int,
    condition_number: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    构造 f(x) = 0.5 x^T A x - b^T x，其中 A 对称正定，κ(A)=condition_number。

    Returns
    -------
    A, b, mu, L : 矩阵、右端项、最小/最大特征值（强凸参数）
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    eigvals = np.geomspace(1.0, condition_number, dim)
    a = q @ np.diag(eigvals) @ q.T
    x_star = rng.standard_normal(dim)
    b = a @ x_star
    mu, L = float(eigvals.min()), float(eigvals.max())
    return a, b, mu, L


def objective(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return 0.5 * float(x @ (a @ x)) - float(b @ x)


def gradient(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return a @ x - b


def optimal_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(a, b)
