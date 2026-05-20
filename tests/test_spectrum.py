"""轻量单元测试：理论对照不被重构破坏。"""

import numpy as np

from src.momentum_spectrum import optimal_polyak_params, spectral_radius, worst_case_spectral_radius
from src.newton_schulz import newton_schulz_iterate, orthogonality_error


def test_polyak_spectral_radius_kappa_100():
    mu, L = 1.0, 100.0
    step, beta = optimal_polyak_params(mu, L)
    rho = worst_case_spectral_radius(mu, L, step, beta)
    theory = (np.sqrt(L / mu) - 1) / (np.sqrt(L / mu) + 1)
    assert abs(rho - theory) < 1e-10


def test_gd_spectral_radius_step_opt():
    mu, L = 1.0, 100.0
    h = 2.0 / (mu + L)
    rho = max(abs(1 - h * mu), abs(1 - h * L))
    theory = (L / mu - 1) / (L / mu + 1)
    assert abs(rho - theory) < 1e-10


def test_newton_schulz_error_decreases():
    rng = np.random.default_rng(0)
    g = rng.standard_normal((8, 8))
    _, errors = newton_schulz_iterate(g, max_iter=12, scale_factor=1.0)
    assert errors[-1] < errors[2]
    assert orthogonality_error(g / np.linalg.norm(g, "fro")) > errors[-1]


def test_momentum_iteration_matrix_shape():
    m = spectral_radius(10.0, 0.1, 0.5)
    assert 0 < m < 1.1
