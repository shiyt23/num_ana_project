"""复现 Bock & Weiß (2022) 中 Adam 在二次目标上的 2-极限环现象。

定理（Bock-Weiß 2022）：对一维二次 f(x) = (a/2) x^2 (a>0)，
全 batch 带 bias-correction 的 Adam 不存在收敛到 x*=0 的吸引子；
对所有超参数 (η, β₁, β₂, ε) 选择，迭代会落入一个**2-极限环**：
存在两点 x⁺, x⁻ 满足 T(x⁺)=x⁻、T(x⁻)=x⁺。

这是 Adam **在最简凸问题上**不收敛的最强反例。
本模块给出标量版本的 Adam 迭代算子，便于做相图与极限环可视化。
"""

from __future__ import annotations

import numpy as np


def adam_step_scalar(
    x: float,
    m: float,
    v: float,
    t: int,
    a: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> tuple[float, float, float]:
    """对 f(x) = a x²/2，Adam 一步：返回 (x_next, m_next, v_next)。"""
    g = a * x
    m_next = beta1 * m + (1 - beta1) * g
    v_next = beta2 * v + (1 - beta2) * g * g
    m_hat = m_next / (1 - beta1 ** t)
    v_hat = v_next / (1 - beta2 ** t)
    x_next = x - lr * m_hat / (np.sqrt(v_hat) + eps)
    return x_next, m_next, v_next


def adam_trajectory_scalar(
    x0: float,
    a: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    max_iter: int,
) -> tuple[list[float], list[float], list[float]]:
    """在 f(x) = a x²/2 上跑 Adam max_iter 步，返回 (x_seq, m_seq, v_seq)。"""
    xs = [x0]
    ms = [0.0]
    vs = [0.0]
    x, m, v = x0, 0.0, 0.0
    for t in range(1, max_iter + 1):
        x, m, v = adam_step_scalar(x, m, v, t, a, lr, beta1, beta2, eps)
        xs.append(x)
        ms.append(m)
        vs.append(v)
    return xs, ms, vs


def detect_limit_cycle(
    xs: list[float], window: int = 100, tol: float = 1e-3
) -> tuple[bool, float | None, float | None]:
    """
    简单 2-极限环检测：观察末段 |x_k - x_{k-2}| 是否收敛到 0
    但 |x_k - x_{k-1}| 不收敛。返回 (is_2cycle, x_plus, x_minus)。
    """
    if len(xs) < window + 2:
        return False, None, None
    tail = xs[-window:]
    # 偶/奇位置分开
    even = tail[::2]
    odd = tail[1::2]
    if len(even) < 5 or len(odd) < 5:
        return False, None, None
    even_std = float(np.std(even[-min(10, len(even)):]))
    odd_std = float(np.std(odd[-min(10, len(odd)):]))
    even_mean = float(np.mean(even[-min(10, len(even)):]))
    odd_mean = float(np.mean(odd[-min(10, len(odd)):]))
    if even_std < tol and odd_std < tol and abs(even_mean - odd_mean) > 3 * tol:
        return True, even_mean, odd_mean
    return False, None, None
