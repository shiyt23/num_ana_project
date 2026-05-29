"""Chebyshev 半迭代法 (Chebyshev semi-iteration) 与 Heavy-ball 的等价分析。

数值分析事实：对二次问题 f(x) = (1/2) x^T A x - b^T x，求 Ax=b
等价于多项式预条件 e_k = p_k(A) e_0，其中 p_k 在 spec(A) 上 minimax 最小。
Chebyshev minimax 解给出最优多项式，其三项递推恰好等价于
Heavy-ball 迭代在 Polyak 最优 (η*, β*) 下的形式。

参考：
- Saad, Iterative Methods, §5.3, §6.11
- Hageman & Young, Applied Iterative Methods
- Polyak (1964)
"""

from __future__ import annotations

import numpy as np


def chebyshev_semi_iterative(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    mu: float,
    L: float,
) -> tuple[list[np.ndarray], list[float]]:
    """
    标准 Chebyshev 半迭代求 Ax=b（A 对称正定，谱在 [μ, L]）。

    递推（Saad §5.3）：
        d = (L+μ)/2,  c = (L-μ)/2
        x_1 = x_0 + (1/d) r_0,  r_k = b - A x_k
        α_k = ... by Chebyshev coefficients
        x_{k+1} = ω_{k+1} (x_k + α_k r_k - x_{k-1}) + x_{k-1}

    用更紧凑的递推（Hageman-Young）：
        γ = 2/(L+μ),  σ = (L-μ)/(L+μ),  ρ_1 = 1/(1 - σ²/2)
        x_1 = x_0 + γ r_0
        ρ_{k+1} = 1/(1 - σ² ρ_k / 4)
        x_{k+1} = ρ_{k+1} (γ r_k + x_k - x_{k-1}) + x_{k-1}
    """
    d = (L + mu) / 2.0
    c = (L - mu) / 2.0
    sigma = c / d

    x_prev = x0.copy()
    r_prev = b - a @ x_prev
    x = x_prev + r_prev / d  # 第一步用 Richardson
    xs = [x_prev.copy(), x.copy()]
    fs = [0.5 * float(x_prev @ (a @ x_prev)) - float(b @ x_prev),
          0.5 * float(x @ (a @ x)) - float(b @ x)]

    rho = 1.0 / (1.0 - 0.5 * sigma * sigma)
    for _ in range(max_iter - 1):
        r = b - a @ x
        x_next = rho * (r / d + x - x_prev) + x_prev
        x_prev = x
        x = x_next
        rho = 1.0 / (1.0 - 0.25 * sigma * sigma * rho)
        xs.append(x.copy())
        fs.append(0.5 * float(x @ (a @ x)) - float(b @ x))
    return xs, fs


def heavy_ball_polyak_form(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    mu: float,
    L: float,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Polyak Heavy-ball 写成"以 x_{k-1}, x_k 二项递推"的标准形式：
        x_{k+1} = x_k - η g_k + β (x_k - x_{k-1})
    使用最优 η* = 4/(√L + √μ)², β* = ((√L-√μ)/(√L+√μ))².

    与 Chebyshev 半迭代在二次目标上**渐近等价**——
    Chebyshev 用时变 ρ_k 收敛到 β*，故 HB 是 Chebyshev 的"定常版本"。
    """
    eta = 4.0 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    beta = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2

    x_prev = x0.copy()
    g = a @ x_prev - b
    x = x_prev - eta * g
    xs = [x_prev.copy(), x.copy()]
    fs = [0.5 * float(x_prev @ (a @ x_prev)) - float(b @ x_prev),
          0.5 * float(x @ (a @ x)) - float(b @ x)]

    for _ in range(max_iter - 1):
        g = a @ x - b
        x_next = x - eta * g + beta * (x - x_prev)
        x_prev = x
        x = x_next
        xs.append(x.copy())
        fs.append(0.5 * float(x @ (a @ x)) - float(b @ x))
    return xs, fs


def chebyshev_polynomial_value(k: int, t: float) -> float:
    """T_k(t) (复数域上 cosh/cos 实现)。用于绘制 minimax 多项式。"""
    if abs(t) <= 1:
        return float(np.cos(k * np.arccos(t)))
    sign = 1.0 if t > 0 else (-1.0) ** k
    return sign * float(np.cosh(k * np.arccosh(abs(t))))


def chebyshev_error_polynomial(k: int, lam: float, mu: float, L: float) -> float:
    """
    Chebyshev 误差多项式 e_k(λ) = T_k((L+μ-2λ)/(L-μ)) / T_k((L+μ)/(L-μ))。
    在 spec(A) ⊆ [μ, L] 上达到 minimax 最小：
        max |e_k(λ)| = 1 / T_k((L+μ)/(L-μ)) ≈ 2 ((√κ-1)/(√κ+1))^k.
    """
    t_num = (L + mu - 2.0 * lam) / (L - mu)
    t_den = (L + mu) / (L - mu)
    return chebyshev_polynomial_value(k, t_num) / chebyshev_polynomial_value(k, t_den)


def chebyshev_minimax_bound(k: int, kappa: float) -> float:
    """1/T_k((κ+1)/(κ-1))，理论 minimax 上界。"""
    t = (kappa + 1.0) / (kappa - 1.0)
    return 1.0 / chebyshev_polynomial_value(k, t)
