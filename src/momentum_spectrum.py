"""Heavy-ball / Momentum 在二次函数上的谱半径分析。"""

from __future__ import annotations

import numpy as np


def momentum_iteration_matrix(lam: float, step: float, beta: float) -> np.ndarray:
    """
    对特征值 λ，Heavy-ball 在标量二次项上的 2×2 迭代矩阵。
    状态 s_k = [x_k; x_{k-1}]，见 Polyak 最优参数分析。
    """
    return np.array(
        [
            [1 - step * lam + beta, -beta],
            [1.0, 0.0],
        ]
    )


def spectral_radius(lam: float, step: float, beta: float) -> float:
    m = momentum_iteration_matrix(lam, step, beta)
    eigs = np.linalg.eigvals(m)
    return float(np.max(np.abs(eigs)))


def optimal_polyak_params(mu: float, L: float) -> tuple[float, float]:
    """Polyak 最优 heavy-ball 参数。"""
    step = 4.0 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    beta = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    return step, beta


def worst_case_spectral_radius(mu: float, L: float, step: float, beta: float) -> float:
    return max(spectral_radius(mu, step, beta), spectral_radius(L, step, beta))
