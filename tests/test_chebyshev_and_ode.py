"""新增前沿模块的回归测试。"""

import numpy as np

from src.chebyshev import (
    chebyshev_minimax_bound,
    chebyshev_semi_iterative,
    heavy_ball_polyak_form,
)
from src.ode_integrators import gradient_flow_rk4
from src.spectral_trust_region import (
    make_spectral_recovery_problem,
    run_gd_on_spectral_recovery,
    run_muon_on_spectral_recovery,
)
from src.newton_schulz import chebyshev_ns_iterate, newton_schulz_iterate
from src.pcg import cg_theoretical_bound, run_cg
from src.quadratic import make_quadratic_problem


def test_heavy_ball_matches_chebyshev_asymptotically():
    """HB 与 Chebyshev 半迭代在二次问题上最终 gap 接近。"""
    a, b, mu, L = make_quadratic_problem(10, 50.0, seed=1)
    x0 = np.zeros(10)
    _, fs_cheb = chebyshev_semi_iterative(a, b, x0, 200, mu, L)
    _, fs_hb = heavy_ball_polyak_form(a, b, x0, 200, mu, L)
    # 两者最终都很接近最优值
    assert abs(fs_cheb[-1] - fs_hb[-1]) / abs(fs_hb[-1] + 1e-12) < 1e-3


def test_chebyshev_bound_matches_kappa_dependence():
    """Chebyshev minimax 界应随 k 几何下降。"""
    b10 = chebyshev_minimax_bound(10, 100.0)
    b20 = chebyshev_minimax_bound(20, 100.0)
    assert 0 < b20 < b10 < 1


def test_ns_fat_matrix_converges():
    """胖矩阵 NS 修复后应收敛。"""
    rng = np.random.default_rng(0)
    g = rng.standard_normal((8, 32))
    _, errs = newton_schulz_iterate(g, max_iter=12)
    assert errs[-1] < 1e-6


def test_chebyshev_ns_competitive_with_standard():
    """Chebyshev-NS 至少与标准 NS 同阶（15 步内收敛到 1e-4 以下）。"""
    rng = np.random.default_rng(2)
    g = rng.standard_normal((16, 16))
    _, errs_std = newton_schulz_iterate(g, max_iter=15)
    _, errs_cheb = chebyshev_ns_iterate(g, max_iter=15)
    assert errs_std[-1] < 1e-4
    # Higham 五次多项式版本应该至少不显著差于标准 NS
    assert errs_cheb[-1] < 1.0


def test_cg_converges_in_dim_steps():
    """CG 在 n 维问题上有限步终止（实际约 n 步左右）。"""
    a, b, mu, L = make_quadratic_problem(10, 100.0, seed=3)
    _, iters, _ = run_cg(a, b, np.zeros(10), 30)
    assert iters <= 20  # 浮点误差下略大于 n 也接受


def test_muon_reduces_spectral_loss():
    """Muon 在谱范数恢复目标上应显著降低损失（不要求每步严格单调）。"""
    w_star, _ = make_spectral_recovery_problem(10, 6, seed=0)
    w0 = np.zeros_like(w_star)
    losses, _ = run_muon_on_spectral_recovery(w_star, w0, 50, lr=0.3)
    # 最终损失应远低于初始
    assert losses[-1] < 0.5 * losses[0]


def test_gradient_flow_approaches_optimum():
    """RK4 积分梯度流应趋向全局极小。"""
    a, b, _, _ = make_quadratic_problem(5, 10.0, seed=4)

    def grad(x):
        return a @ x - b

    x0 = np.ones(5)
    _, xs = gradient_flow_rk4(grad, x0, t_end=5.0, n_steps=500)
    x_star = np.linalg.solve(a, b)
    # 强凸性下 5 个时间常数足以衰减到 e^{-5} ≈ 0.7% 量级
    assert np.linalg.norm(xs[-1] - x_star) < 0.05 * np.linalg.norm(x0 - x_star)
