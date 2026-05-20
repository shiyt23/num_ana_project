"""预条件共轭梯度（PCG）基线。"""

from __future__ import annotations

import numpy as np

from .quadratic import objective


def run_pcg(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    tol: float = 1e-12,
    use_jacobi: bool = True,
) -> tuple[list[float], int]:
    """PCG 求解 Ax=b，返回目标值序列与迭代次数。"""
    n = len(x0)
    diag = np.diag(a).copy()
    diag = np.where(np.abs(diag) < 1e-12, 1.0, diag)
    m_inv = 1.0 / diag if use_jacobi else np.ones(n)
    return _pcg_iterations(a, b, x0, max_iter, tol, m_inv)


def _pcg_iterations(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    tol: float,
    m_inv: np.ndarray,
) -> tuple[list[float], int]:
    """手写 PCG，记录每步目标值。"""
    x = x0.copy()
    fs = [objective(a, b, x)]
    r = b - a @ x
    z = m_inv * r
    p = z.copy()
    rz_old = float(r @ z)

    for k in range(max_iter):
        ap = a @ p
        alpha = rz_old / float(p @ ap)
        x = x + alpha * p
        r = r - alpha * ap
        fs.append(objective(a, b, x))
        if np.linalg.norm(r) < tol:
            return fs, k + 1
        z = m_inv * r
        rz_new = float(r @ z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return fs, max_iter
