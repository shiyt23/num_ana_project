"""Newton-Schulz 迭代：矩阵极分解 / 正交化（Muon 核心数值步骤）。"""

from __future__ import annotations

import numpy as np


def newton_schulz_step(x: np.ndarray) -> np.ndarray:
    """X_{k+1} = 0.5 X_k (3I - X_k^T X_k)。"""
    return 0.5 * x @ (3.0 * np.eye(x.shape[1]) - x.T @ x)


def newton_schulz_iterate(
    g: np.ndarray,
    max_iter: int,
    normalize: bool = True,
    scale_factor: float = 1.0,
) -> tuple[list[np.ndarray], list[float]]:
    """
    对矩阵 G 运行 Newton-Schulz，返回迭代序列与 ||X^T X - I||_F。
    """
    x = g.copy().astype(float)
    if normalize:
        fro = np.linalg.norm(x, "fro")
        if fro > 0:
            x = x / fro
        # 谱范数初值：使奇异值落入收敛域 (0, sqrt(3))
        sigma_max = np.linalg.svd(x, compute_uv=False)[0]
        if sigma_max > 0:
            x = x / (scale_factor * sigma_max)

    xs = [x.copy()]
    errors = [orthogonality_error(x)]
    for _ in range(max_iter):
        x = newton_schulz_step(x)
        xs.append(x.copy())
        errors.append(orthogonality_error(x))
    return xs, errors


def orthogonality_error(x: np.ndarray) -> float:
    r = x.T @ x - np.eye(x.shape[1])
    return float(np.linalg.norm(r, "fro"))


def polar_factor_via_svd(g: np.ndarray) -> np.ndarray:
    """参考解：G = U Sigma V^T，正交因子 U V^T。"""
    u, _, vt = np.linalg.svd(g, full_matrices=False)
    return u @ vt
