"""优化算法对应 ODE 的数值积分器，用于"算法 vs 连续流"对比。

ODE 视角概览：
- GD 对应梯度流 ẋ = -∇f(x)，前向 Euler 离散即 GD。
- Heavy-ball 对应 ẍ + γ ẋ + ∇f(x) = 0；Polyak 临界阻尼 γ = 2√μ。
- Nesterov-AVD: Ẍ + (3/t) Ẋ + ∇f(X) = 0 (Su, Boyd, Candès 2016)。
- 高分辨率 ODE (Shi et al. 2021)：捕捉 √η 阶修正，区分 NAG vs HB。

我们用四阶 RK4 数值积分作"参考解"，便于与小步长离散迭代对比误差。
"""

from __future__ import annotations

import numpy as np


def gradient_flow_rk4(
    grad_fn,
    x0: np.ndarray,
    t_end: float,
    n_steps: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """积分 ẋ = -∇f(x) 用 RK4。返回时间网格与轨迹。"""
    dt = t_end / n_steps
    t = np.linspace(0, t_end, n_steps + 1)
    xs = [x0.copy()]
    x = x0.copy()
    for _ in range(n_steps):
        k1 = -grad_fn(x)
        k2 = -grad_fn(x + 0.5 * dt * k1)
        k3 = -grad_fn(x + 0.5 * dt * k2)
        k4 = -grad_fn(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x.copy())
    return t, xs


def heavy_ball_ode_rk4(
    grad_fn,
    x0: np.ndarray,
    v0: np.ndarray,
    gamma: float,
    t_end: float,
    n_steps: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    积分 ẍ + γ ẋ + ∇f = 0，写成一阶系统 (x, v=ẋ)：
        ẋ = v
        v̇ = -γ v - ∇f(x)
    """
    dt = t_end / n_steps
    t = np.linspace(0, t_end, n_steps + 1)
    xs = [x0.copy()]
    x, v = x0.copy(), v0.copy()
    for _ in range(n_steps):
        def f(x_, v_):
            return v_, -gamma * v_ - grad_fn(x_)

        k1x, k1v = f(x, v)
        k2x, k2v = f(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
        k3x, k3v = f(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
        k4x, k4v = f(x + dt * k3x, v + dt * k3v)
        x = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        xs.append(x.copy())
    return t, xs


def nesterov_avd_ode_rk4(
    grad_fn,
    x0: np.ndarray,
    t_start: float,
    t_end: float,
    n_steps: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    积分 Ẍ + (3/t) Ẋ + ∇f(X) = 0  (Su-Boyd-Candès AVD ODE)。
    起点 t_start > 0 以避免奇点；典型取 t_start = √η 量级。
    """
    dt = (t_end - t_start) / n_steps
    t = np.linspace(t_start, t_end, n_steps + 1)
    xs = [x0.copy()]
    x = x0.copy()
    v = np.zeros_like(x0)  # Ẋ(t_start) = 0 近似
    tt = t_start
    for _ in range(n_steps):
        def f(x_, v_, tau):
            return v_, -(3.0 / max(tau, 1e-6)) * v_ - grad_fn(x_)

        k1x, k1v = f(x, v, tt)
        k2x, k2v = f(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, tt + 0.5 * dt)
        k3x, k3v = f(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, tt + 0.5 * dt)
        k4x, k4v = f(x + dt * k3x, v + dt * k3v, tt + dt)
        x = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        tt += dt
        xs.append(x.copy())
    return t, xs


def high_resolution_nag_ode_rk4(
    grad_fn,
    hess_grad_fn,
    x0: np.ndarray,
    eta: float,
    gamma: float,
    t_end: float,
    n_steps: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    高分辨率 Nesterov ODE (Shi et al. 2021)：
        Ẍ + γ Ẋ + (1 + √η · γ) ∇f(X) + √η · ∇²f(X) Ẋ = 0
    捕捉 √η 阶修正项，区分 NAG vs 经典 Heavy-ball。
    hess_grad_fn(x, v) 返回 ∇²f(x) v（避免显式构造 Hessian）。
    """
    sqrt_eta = np.sqrt(eta)
    dt = t_end / n_steps
    t = np.linspace(0, t_end, n_steps + 1)
    xs = [x0.copy()]
    x = x0.copy()
    v = np.zeros_like(x0)
    for _ in range(n_steps):
        def f(x_, v_):
            return v_, -gamma * v_ - (1.0 + sqrt_eta * gamma) * grad_fn(x_) - sqrt_eta * hess_grad_fn(x_, v_)

        k1x, k1v = f(x, v)
        k2x, k2v = f(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
        k3x, k3v = f(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
        k4x, k4v = f(x + dt * k3x, v + dt * k3v)
        x = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        xs.append(x.copy())
    return t, xs
