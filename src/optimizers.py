"""统一迭代模板 x_{k+1} = x_k - P_k g_k 下的经典与现代优化器。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .quadratic import gradient, objective


@dataclass
class RunResult:
    xs: list[np.ndarray]
    fs: list[float]
    gs: list[np.ndarray]
    preconditioners: list[np.ndarray] = field(default_factory=list)


def _resolve_spectrum(
    a: np.ndarray,
    mu: float | None,
    L: float | None,
) -> tuple[float, float]:
    if mu is not None and L is not None:
        return float(mu), float(L)
    eigs = np.linalg.eigvalsh(a)
    return float(eigs.min()), float(eigs.max())


def run_optimizer(
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    lr: float | None = None,
    mu: float | None = None,
    L: float | None = None,
    **kwargs,
) -> RunResult:
    """在二次问题上运行指定优化器。建议传入 mu, L 避免重复特征值分解。"""
    runners = {
        "gd": _run_gd,
        "momentum": _run_momentum,
        "nesterov": _run_nesterov,
        "adam": _run_adam,
        "adamw": _run_adamw,
        "sophia": _run_sophia,
        "jacobi": _run_jacobi_gd,
    }
    if name not in runners:
        raise ValueError(f"未知优化器: {name}")
    return runners[name](a, b, x0, max_iter, lr, mu=mu, L=L, **kwargs)


def _run_gd(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    lr: float | None,
    mu: float | None = None,
    L: float | None = None,
    **_,
) -> RunResult:
    mu, L = _resolve_spectrum(a, mu, L)
    step = lr if lr is not None else 2.0 / (mu + L)

    x = x0.copy()
    xs, fs, gs, ps = [x.copy()], [objective(a, b, x)], [gradient(a, b, x)], [np.eye(len(x)) * step]
    for _ in range(max_iter):
        g = gradient(a, b, x)
        p_k = np.eye(len(x)) * step
        x = x - p_k @ g
        xs.append(x.copy())
        fs.append(objective(a, b, x))
        gs.append(gradient(a, b, x))
        ps.append(p_k.copy())
    return RunResult(xs, fs, gs, ps)


def _run_momentum(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    lr: float | None,
    beta: float = 0.9,
    mu: float | None = None,
    L: float | None = None,
    **_,
) -> RunResult:
    mu, L = _resolve_spectrum(a, mu, L)
    kappa = L / mu
    step = lr if lr is not None else 4.0 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    if beta < 0:
        beta = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2

    x, v = x0.copy(), np.zeros_like(x0)
    xs, fs, gs, ps = [x.copy()], [objective(a, b, x)], [gradient(a, b, x)], [np.eye(len(x)) * step]
    for _ in range(max_iter):
        g = gradient(a, b, x)
        v = beta * v + g
        p_k = np.eye(len(x)) * step
        x = x - p_k @ v
        xs.append(x.copy())
        fs.append(objective(a, b, x))
        gs.append(gradient(a, b, x))
        ps.append(p_k.copy())
    return RunResult(xs, fs, gs, ps)


def _run_nesterov(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    lr: float | None,
    mu: float | None = None,
    L: float | None = None,
    **_,
) -> RunResult:
    """
    强凸 Nesterov 加速梯度（Nesterov 2004, §2.2.1）：
        y_k = x_k + β (x_k - x_{k-1})
        x_{k+1} = y_k - (1/L) ∇f(y_k)
    使用 β = (√κ - 1)/(√κ + 1)。注意梯度在 y_k 处取，
    曲线 f(x_k) 不一定单调（典型 Nesterov 行为，正确实现亦如此）。
    """
    mu, L = _resolve_spectrum(a, mu, L)
    kappa = L / mu
    step = lr if lr is not None else 1.0 / L
    beta = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)

    x = x0.copy()
    x_prev = x0.copy()
    xs, fs, gs, ps = [x.copy()], [objective(a, b, x)], [gradient(a, b, x)], [np.eye(len(x)) * step]
    for _ in range(max_iter):
        y = x + beta * (x - x_prev)
        g_y = gradient(a, b, y)
        x_new = y - step * g_y
        x_prev = x
        x = x_new
        p_k = np.eye(len(x)) * step
        xs.append(x.copy())
        fs.append(objective(a, b, x))
        gs.append(gradient(a, b, x))
        ps.append(p_k.copy())
    return RunResult(xs, fs, gs, ps)


def _run_adam(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    lr: float | None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    mu: float | None = None,
    L: float | None = None,
    **_,
) -> RunResult:
    _ = _resolve_spectrum(a, mu, L)  # 仅校验/统一接口
    step = lr if lr is not None else 0.1
    dim = len(x0)
    x = x0.copy()
    m = np.zeros(dim)
    v = np.zeros(dim)
    xs, fs, gs, ps = [x.copy()], [objective(a, b, x)], [gradient(a, b, x)], [np.eye(dim)]

    for t in range(1, max_iter + 1):
        g = gradient(a, b, x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        precond_diag = 1.0 / (np.sqrt(v_hat) + eps)
        p_k = np.diag(precond_diag * step)
        x = x - p_k @ m_hat
        xs.append(x.copy())
        fs.append(objective(a, b, x))
        gs.append(gradient(a, b, x))
        ps.append(p_k.copy())
    return RunResult(xs, fs, gs, ps)


def _run_adamw(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    lr: float | None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    mu: float | None = None,
    L: float | None = None,
    **_,
) -> RunResult:
    _ = _resolve_spectrum(a, mu, L)
    step = lr if lr is not None else 0.1
    dim = len(x0)
    x = x0.copy()
    m = np.zeros(dim)
    v = np.zeros(dim)
    xs, fs, gs, ps = [x.copy()], [objective(a, b, x)], [gradient(a, b, x)], [np.eye(dim)]

    for t in range(1, max_iter + 1):
        g = gradient(a, b, x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        precond_diag = step / (np.sqrt(v_hat) + eps)
        p_k = np.diag(precond_diag)
        # 解耦权重衰减：自适应步 + decoupled WD
        x = x - p_k @ m_hat - step * weight_decay * x
        xs.append(x.copy())
        fs.append(objective(a, b, x))
        gs.append(gradient(a, b, x))
        ps.append(p_k.copy())
    return RunResult(xs, fs, gs, ps)


def _run_sophia(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    lr: float | None,
    beta1: float = 0.9,
    beta2: float = 0.99,
    eps: float = 1e-12,
    rho: float = 0.04,
    hess_interval: int = 10,
    mu: float | None = None,
    L: float | None = None,
    **_,
) -> RunResult:
    """
    Sophia-H 风格（Liu et al. 2023, Algorithm 1，二次问题特化）：
        h_k+1 = β2 h_k + (1-β2) diag(A)           (二次问题 Hessian 对角)
        m_k+1 = β1 m_k + (1-β1) g_k
        x_{k+1} = x_k - η · clip( m_hat / max(h_hat, eps),  ρ )
    取 ρ 较小以维持稳定；hess_interval 为 Hessian 重采样间隔。
    """
    _ = _resolve_spectrum(a, mu, L)
    step = lr if lr is not None else 0.5
    dim = len(x0)
    x = x0.copy()
    m = np.zeros(dim)
    h = np.zeros(dim)
    diag_h = np.diag(a).copy()
    xs, fs, gs, ps = [x.copy()], [objective(a, b, x)], [gradient(a, b, x)], [np.eye(dim)]

    for t in range(1, max_iter + 1):
        g = gradient(a, b, x)
        m = beta1 * m + (1 - beta1) * g
        if (t - 1) % hess_interval == 0:
            h = beta2 * h + (1 - beta2) * diag_h
        h_clamped = np.maximum(h, eps)
        update = np.clip(m / h_clamped, -rho, rho)
        p_k = np.diag(step / h_clamped)
        x = x - step * update
        xs.append(x.copy())
        fs.append(objective(a, b, x))
        gs.append(gradient(a, b, x))
        ps.append(p_k.copy())
    return RunResult(xs, fs, gs, ps)


def _run_jacobi_gd(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    lr: float | None,
    mu: float | None = None,
    L: float | None = None,
    **_,
) -> RunResult:
    _ = _resolve_spectrum(a, mu, L)
    diag_a = np.diag(a).copy()
    diag_a = np.where(np.abs(diag_a) < 1e-12, 1e-12, diag_a)
    jacobi_scale = 1.0 / diag_a
    step = lr if lr is not None else 1.0

    x = x0.copy()
    dim = len(x0)
    xs, fs, gs, ps = [x.copy()], [objective(a, b, x)], [gradient(a, b, x)], [np.diag(jacobi_scale * step)]
    for _ in range(max_iter):
        g = gradient(a, b, x)
        p_k = np.diag(jacobi_scale * step)
        x = x - p_k @ g
        xs.append(x.copy())
        fs.append(objective(a, b, x))
        gs.append(gradient(a, b, x))
        ps.append(p_k.copy())
    return RunResult(xs, fs, gs, ps)


def effective_preconditioned_condition_number(
    a: np.ndarray, p_k: np.ndarray
) -> float:
    """κ(P_k^{-1} A) 用于分析动态预条件。"""
    try:
        m = np.linalg.solve(p_k, a)
    except np.linalg.LinAlgError:
        m = np.linalg.pinv(p_k) @ a
    sym = 0.5 * (m + m.T)
    eigs = np.linalg.eigvalsh(sym)
    eigs = eigs[eigs > 1e-12]
    if len(eigs) < 2:
        return 1.0
    return float(eigs.max() / eigs.min())
