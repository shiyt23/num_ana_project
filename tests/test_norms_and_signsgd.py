"""范数最速下降三元组 + signSGD + NS 收敛盆 的回归测试。"""

import numpy as np

from src.optimizers import run_optimizer
from src.quadratic import make_quadratic_problem, objective, optimal_point
from src.steepest_descent_norms import (
    dual_norm_value,
    lmo_euclidean,
    lmo_linf,
    lmo_spectral,
    update_condition_number,
    verify_lmo_optimality,
)


def test_lmo_value_equals_negative_dual_norm():
    """LMO 最优值 ⟨g, d*⟩ = -‖g‖_dual（三种范数）。"""
    rng = np.random.default_rng(0)
    g = rng.standard_normal((6, 4))
    for norm in ["euclidean", "linf", "spectral"]:
        v = verify_lmo_optimality(g, norm, n_random=500, seed=1)
        assert v["lmo_is_optimal"]
        assert abs(v["lmo_value"] + v["dual_norm"]) < 1e-9


def test_muon_update_condition_number_is_one():
    """Muon (谱范数 LMO) 的更新方向条件数恒为 1（奇异值全为 1）。"""
    rng = np.random.default_rng(1)
    for kappa_g in [10.0, 100.0, 1000.0]:
        u, _ = np.linalg.qr(rng.standard_normal((8, 8)))
        v, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        s = np.geomspace(1.0, kappa_g, 5)
        g = u[:, :5] @ np.diag(s) @ v.T
        assert update_condition_number(g, "spectral") < 1.0001
        # GD 更新继承 G 的条件数
        assert update_condition_number(g, "euclidean") > 0.5 * kappa_g


def test_spectral_lmo_inner_product_is_nuclear_norm():
    """⟨G, -UVᵀ⟩ = -‖G‖_*（核范数）。"""
    rng = np.random.default_rng(2)
    g = rng.standard_normal((7, 4))
    d = lmo_spectral(g)
    inner = float((g * d).sum())
    nuclear = float(np.sum(np.linalg.svd(g, compute_uv=False)))
    assert abs(inner + nuclear) < 1e-9


def test_signsgd_descends_on_quadratic():
    """signSGD (衰减步长) 在二次问题上下降。"""
    a, b, mu, L = make_quadratic_problem(15, 30.0, seed=3)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)
    run = run_optimizer("signsgd", a, b, np.zeros(15), 300, mu=mu, L=L)
    assert run.fs[-1] - f_star < 0.1 * (run.fs[0] - f_star)


def test_dual_norm_pairs():
    """对偶范数配对：ℓ₂↔ℓ₂, ℓ∞↔ℓ₁, 谱↔核。"""
    rng = np.random.default_rng(4)
    g = rng.standard_normal((5, 5))
    assert abs(dual_norm_value(g, "euclidean") - np.linalg.norm(g)) < 1e-9
    assert abs(dual_norm_value(g, "linf") - np.sum(np.abs(g))) < 1e-9
    assert abs(dual_norm_value(g, "spectral")
               - np.sum(np.linalg.svd(g, compute_uv=False))) < 1e-9
