"""优化器与 PCG 的补充测试。"""

import numpy as np

from src.optimizers import run_optimizer
from src.pcg import run_pcg
from src.quadratic import make_quadratic_problem, objective, optimal_point


def test_nesterov_reduces_quadratic():
    a, b, mu, L = make_quadratic_problem(8, 10.0, seed=0)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)
    run = run_optimizer("nesterov", a, b, np.zeros(8), 50, mu=mu, L=L)
    assert run.fs[-1] - f_star < run.fs[0] - f_star


def test_pcg_reaches_tight_tol():
    a, b, mu, L = make_quadratic_problem(12, 50.0, seed=1)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)
    fs, iters = run_pcg(a, b, np.zeros(12), 200, tol=1e-10)
    assert fs[-1] - f_star < 1e-8
    assert iters < 200


def test_adamw_differs_from_adam_on_quadratic():
    a, b, mu, L = make_quadratic_problem(10, 20.0, seed=2)
    x0 = np.ones(10)
    run_a = run_optimizer("adam", a, b, x0.copy(), 30, lr=0.1, mu=mu, L=L)
    run_w = run_optimizer("adamw", a, b, x0.copy(), 30, lr=0.1, mu=mu, L=L, weight_decay=0.1)
    assert not np.allclose(run_a.xs[-1], run_w.xs[-1])
