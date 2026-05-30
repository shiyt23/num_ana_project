"""范数视角下的最速下降统一框架：GD / signSGD / Muon 的 LMO 三元组。

核心思想（Kovalev 2025 non-Euclidean trust-region；Bernstein–Newhouse modular
duality）：给定光滑 f，"在范数 ‖·‖ 的单位球内沿线性逼近最速下降"的方向是

    d* = argmin_{‖d‖ ≤ 1} ⟨g, d⟩       （线性最小化 oracle, LMO）

对不同范数：
    ‖·‖₂ (Euclidean)  →  d* = -g / ‖g‖₂                （GD / 归一化 GD）
    ‖·‖_∞ (entrywise) →  d* = -sign(g)                  （signSGD / Lion）
    ‖·‖_σ (spectral)  →  d* = -U Vᵀ，其中 G = U Σ Vᵀ    （Muon）

这三者只是"度量几何"不同：同一个梯度，在不同范数球上找最速下降方向。
Muon 的更新方向 -UVᵀ 的全部奇异值都是 1（条件数恒为 1），这是它区别于
GD 的结构性特征：无论梯度 G 的奇异值多么悬殊，Muon 都把它"均衡化"。

这把课内的最速下降（ℓ₂）、符号梯度法（ℓ∞）、Muon（谱范数）统一成
同一个 LMO 模板，是本项目的核心统一观点。
"""

from __future__ import annotations

import numpy as np

from .newton_schulz import polar_factor_via_svd


# ---------------------------------------------------------------------------
# 三种范数的线性最小化 oracle (LMO)：argmin_{‖d‖≤1} ⟨g, d⟩
# ---------------------------------------------------------------------------

def lmo_euclidean(g: np.ndarray) -> np.ndarray:
    """ℓ₂ 球的 LMO：-g/‖g‖_F。对应 GD / 归一化最速下降。"""
    norm = np.linalg.norm(g)
    return -g / norm if norm > 0 else np.zeros_like(g)


def lmo_linf(g: np.ndarray) -> np.ndarray:
    """ℓ∞ 球的 LMO：-sign(g)。对应 signSGD / Lion。"""
    return -np.sign(g)


def lmo_spectral(g: np.ndarray) -> np.ndarray:
    """谱范数球的 LMO：-U Vᵀ（极分解正交因子）。对应 Muon。"""
    return -polar_factor_via_svd(g)


def dual_norm_value(g: np.ndarray, norm: str) -> float:
    """LMO 的最优值 -⟨g, d*⟩ 等于 g 的对偶范数。

    ‖·‖₂ ↔ ‖·‖₂；‖·‖_∞ ↔ ‖·‖₁；‖·‖_σ ↔ ‖·‖_*（核范数）。
    """
    if norm == "euclidean":
        return float(np.linalg.norm(g))
    if norm == "linf":
        return float(np.sum(np.abs(g)))  # ℓ₁ 对偶
    if norm == "spectral":
        return float(np.sum(np.linalg.svd(g, compute_uv=False)))  # 核范数对偶
    raise ValueError(f"未知范数: {norm}")


def update_condition_number(g: np.ndarray, norm: str) -> float:
    """各 LMO 给出的更新方向矩阵的条件数。

    Muon (spectral) 的更新恒为条件数 1（全奇异值=1）；
    GD (euclidean) 的更新继承 g 的条件数；
    signSGD (linf) 的更新是 ±1 矩阵，条件数取决于符号模式。
    """
    if norm == "euclidean":
        d = lmo_euclidean(g)
    elif norm == "linf":
        d = lmo_linf(g)
    elif norm == "spectral":
        d = lmo_spectral(g)
    else:
        raise ValueError(f"未知范数: {norm}")
    sv = np.linalg.svd(d, compute_uv=False)
    sv = sv[sv > 1e-12]
    if len(sv) < 2:
        return 1.0
    return float(sv.max() / sv.min())


def verify_lmo_optimality(g: np.ndarray, norm: str, n_random: int = 2000,
                          seed: int = 0) -> dict:
    """蒙特卡洛验证：LMO 解确实是单位球内 ⟨g, d⟩ 的最小值。

    随机采样单位球内的 d，确认 ⟨g, d*⟩ ≤ ⟨g, d_random⟩。
    """
    rng = np.random.default_rng(seed)
    if norm == "euclidean":
        d_star = lmo_euclidean(g)
    elif norm == "linf":
        d_star = lmo_linf(g)
    elif norm == "spectral":
        d_star = lmo_spectral(g)
    else:
        raise ValueError(norm)
    val_star = float((g * d_star).sum())

    worst_random = np.inf
    for _ in range(n_random):
        r = rng.standard_normal(g.shape)
        if norm == "euclidean":
            r = r / np.linalg.norm(r)
        elif norm == "linf":
            r = r / np.max(np.abs(r))
        elif norm == "spectral":
            r = r / np.linalg.svd(r, compute_uv=False)[0]
        worst_random = min(worst_random, float((g * r).sum()))
    return {
        "lmo_value": val_star,
        "best_random": worst_random,
        "lmo_is_optimal": val_star <= worst_random + 1e-9,
        "dual_norm": dual_norm_value(g, norm),
    }
