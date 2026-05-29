"""共轭梯度法 (CG) 与预条件 CG (PCG)：课内核心算法。

数值分析事实：
- CG 在 k 步后误差落在 K_k(A, r_0) = span(r_0, A r_0, ..., A^{k-1} r_0) 中
- 误差界：||x_k - x*||_A ≤ 2 ((√κ - 1)/(√κ + 1))^k · ||x_0 - x*||_A
- PCG 把 κ 换成 κ(M^{-1} A)
- Jacobi 预条件 M = diag(A)
"""

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
    if use_jacobi:
        diag = np.diag(a).copy()
        diag = np.where(np.abs(diag) < 1e-12, 1.0, diag)
        m_inv = 1.0 / diag
    else:
        m_inv = np.ones(n)
    return _pcg_iterations(a, b, x0, max_iter, tol, m_inv)


def run_cg(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    tol: float = 1e-12,
) -> tuple[list[float], int, list[np.ndarray]]:
    """无预条件 CG，额外返回每步的解 (供分析用)。"""
    x = x0.copy()
    xs = [x.copy()]
    fs = [objective(a, b, x)]
    r = b - a @ x
    p = r.copy()
    rr_old = float(r @ r)

    for k in range(max_iter):
        ap = a @ p
        alpha = rr_old / float(p @ ap)
        x = x + alpha * p
        r = r - alpha * ap
        xs.append(x.copy())
        fs.append(objective(a, b, x))
        if np.sqrt(rr_old) < tol:
            return fs, k + 1, xs
        rr_new = float(r @ r)
        beta = rr_new / rr_old
        p = r + beta * p
        rr_old = rr_new
    return fs, max_iter, xs


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


def cg_theoretical_bound(k: int, kappa: float, e0_anorm: float = 1.0) -> float:
    """CG 在 A-范数下的标准 Chebyshev 上界。"""
    rho = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    return 2.0 * e0_anorm * rho ** k
