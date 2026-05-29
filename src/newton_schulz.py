"""Newton-Schulz 迭代：矩阵极分解 / 正交化（Muon 核心数值步骤）。

支持任意形状 m×n。当 m<n（胖矩阵）时，对 X^T 迭代再转置回来；
这样统一以 min(m,n) 阶单位阵作目标，避免 X^T X - I 在大 n 下退化。
"""

from __future__ import annotations

import numpy as np


def newton_schulz_step(x: np.ndarray) -> np.ndarray:
    """X_{k+1} = 0.5 X_k (3I - X_k^T X_k)。仅当 X 为列正交目标时直接用。"""
    return 0.5 * x @ (3.0 * np.eye(x.shape[1]) - x.T @ x)


def _newton_schulz_step_thin(x: np.ndarray) -> np.ndarray:
    """瘦/方矩阵 (m>=n) 的 NS 一步。"""
    return 0.5 * x @ (3.0 * np.eye(x.shape[1]) - x.T @ x)


def newton_schulz_iterate(
    g: np.ndarray,
    max_iter: int,
    normalize: bool = True,
    scale_factor: float = 1.0,
) -> tuple[list[np.ndarray], list[float]]:
    """
    对矩阵 G 运行 Newton-Schulz，返回迭代序列与正交化误差。

    误差度量：用 min(m,n) 边的 Gram 矩阵偏离 I 的 Frobenius 距离，
    这样瘦矩阵 (m>=n) 用 X^T X，胖矩阵 (m<n) 用 X X^T，
    保证 NS 在两种形状下都收敛到极分解的部分等距。
    """
    x = g.copy().astype(float)
    m, n = x.shape
    transposed = False
    if m < n:
        x = x.T  # 让算法始终面对瘦矩阵
        transposed = True

    if normalize:
        fro = np.linalg.norm(x, "fro")
        if fro > 0:
            x = x / fro
        sigma_max = np.linalg.svd(x, compute_uv=False)[0]
        if sigma_max > 0:
            x = x / (scale_factor * sigma_max)

    xs = [x.T.copy() if transposed else x.copy()]
    errors = [_orthogonality_error_min_side(xs[-1])]
    for _ in range(max_iter):
        x = _newton_schulz_step_thin(x)
        xk = x.T.copy() if transposed else x.copy()
        xs.append(xk)
        errors.append(_orthogonality_error_min_side(xk))
    return xs, errors


def _orthogonality_error_min_side(x: np.ndarray) -> float:
    """min(m,n) 边的 Gram 偏离 I。"""
    m, n = x.shape
    if m >= n:
        g = x.T @ x
        return float(np.linalg.norm(g - np.eye(n), "fro"))
    else:
        g = x @ x.T
        return float(np.linalg.norm(g - np.eye(m), "fro"))


def orthogonality_error(x: np.ndarray) -> float:
    """兼容旧 API：返回 min-side 度量。"""
    return _orthogonality_error_min_side(x)


def polar_factor_via_svd(g: np.ndarray) -> np.ndarray:
    """参考解：G = U Sigma V^T，正交因子 U V^T（即极分解 G=QH 中的 Q）。"""
    u, _, vt = np.linalg.svd(g, full_matrices=False)
    return u @ vt


def chebyshev_ns_step(x: np.ndarray, coeffs: tuple[float, float, float]) -> np.ndarray:
    """
    Chebyshev-加速 NS 一步：X <- a X + b X (X^T X) + c X (X^T X)^2
    系数 (a,b,c) 由 Chebyshev/Remez 优化得到（arXiv 2506.10935）。
    标准 NS 对应 (a,b,c) = (1.5, -0.5, 0)。
    """
    a, b, c = coeffs
    g = x.T @ x
    return a * x + b * (x @ g) + c * (x @ g @ g)


def chebyshev_ns_iterate(
    g: np.ndarray,
    max_iter: int,
    coeffs: tuple[float, float, float] = (15 / 8, -10 / 8, 3 / 8),
    normalize: bool = True,
) -> tuple[list[np.ndarray], list[float]]:
    """
    五阶 Newton-Schulz 极分解迭代（Higham, *Functions of Matrices*, eq. 8.20）：

        X_{k+1} = (15 X_k - 10 X_k(X_k^T X_k) + 3 X_k(X_k^T X_k)^2) / 8

    对应多项式 p(σ) = (15 - 10σ² + 3σ⁴)/8，满足 p(1)=1，p'(1)=p''(1)=0，
    即三阶接触；故 |1 - σ p(σ)| = O((1-σ²)³)，收敛阶为 5。
    收敛域为 σ ∈ (0, √3)，与三阶 NS 相同（同 Higham §8.6）。
    """
    x = g.copy().astype(float)
    m, n = x.shape
    transposed = False
    if m < n:
        x = x.T
        transposed = True

    if normalize:
        fro = np.linalg.norm(x, "fro")
        if fro > 0:
            x = x / fro
        sigma_max = np.linalg.svd(x, compute_uv=False)[0]
        if sigma_max > 0:
            x = x / sigma_max

    xs = [x.T.copy() if transposed else x.copy()]
    errors = [_orthogonality_error_min_side(xs[-1])]
    for _ in range(max_iter):
        x = chebyshev_ns_step(x, coeffs)
        xk = x.T.copy() if transposed else x.copy()
        xs.append(xk)
        errors.append(_orthogonality_error_min_side(xk))
    return xs, errors
