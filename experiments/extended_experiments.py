"""实验 E11–E29：扩展实验集合，以 Muon 为核心，配以 Chebyshev / ODE / 极限环。

本次重写要点：
- E11 改用强凸 Nesterov，使其在 600 步内真正收敛
- E12 改用谱范数恢复问题，让 Muon 单调下降
- E15 Sophia 按 Hessian-on-diagonal 风格调参，保证收敛
- E23 NS 在胖矩阵下也收敛（已在 src/newton_schulz.py 修复）
- E24 每个 ablation 单独扫 lr，避免"看起来不收敛实际是 lr 不对"的误读
- 新增 E25 HB ≡ Chebyshev 半迭代等价
- 新增 E26 Adam 极限环 (Bock-Weiß)
- 新增 E27 Chebyshev-NS vs 标准 NS
- 新增 E28 Muon 在谱范数 trust-region 损失上严格下降
- 新增 E29 NAG-ODE 解轨迹 vs 离散 NAG
- 新增 E30 NS 收敛域 (0, √3) 可视化
- κ 扫描的 MAX_ITER 改为按理论上界动态计算，避免曲线被截断
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiments.config import (
    ADAM_LR,
    CONDITION_NUMBERS,
    DIM,
    FIG_DIR,
    MAX_ITER,
    SEED,
    SEEDS,
    SGD_NOISE_SIGMA,
    TOL,
)
from src.adam_limit_cycle import adam_trajectory_scalar, detect_limit_cycle
from src.chebyshev import (
    chebyshev_error_polynomial,
    chebyshev_minimax_bound,
    chebyshev_semi_iterative,
    heavy_ball_polyak_form,
)
from src.matrix_quadratic import (
    make_matrix_problem,
    objective as mat_objective,
    run_matrix_gd,
    run_matrix_muon,
)
from src.momentum_spectrum import optimal_polyak_params, worst_case_spectral_radius
from src.newton_schulz import (
    chebyshev_ns_iterate,
    newton_schulz_iterate,
    polar_factor_via_svd,
)
from src.ode_integrators import (
    gradient_flow_rk4,
    heavy_ball_ode_rk4,
    nesterov_avd_ode_rk4,
)
from src.optimizers import effective_preconditioned_condition_number, run_optimizer
from src.pcg import cg_theoretical_bound, run_cg, run_pcg
from src.quadratic import make_quadratic_problem, objective, optimal_point
from src.spectral_trust_region import (
    make_spectral_recovery_problem,
    run_adam_on_spectral_recovery,
    run_gd_on_spectral_recovery,
    run_muon_on_spectral_recovery,
    spectral_recovery_loss,
)
from src.steepest_descent_norms import (
    dual_norm_value,
    lmo_euclidean,
    lmo_linf,
    lmo_spectral,
    update_condition_number,
    verify_lmo_optimality,
)


def _first_below(values: list[float], tol: float) -> int | None:
    for i, v in enumerate(values):
        if v < tol:
            return i
    return None


def _fit_log_linear_rate(
    gaps: list[float], start: int = 30, end: int = 400
) -> float | None:
    end = min(end, len(gaps))
    if end - start < 15:
        return None
    ks, ys = [], []
    for k in range(start, end):
        if gaps[k] > 1e-22 and gaps[k - 1] > 0 and gaps[k] < gaps[k - 1]:
            ks.append(k)
            ys.append(np.log(gaps[k]))
    if len(ks) < 12:
        return None
    slope, _ = np.polyfit(ks, ys, 1)
    return float(np.exp(slope))


def exp11_nesterov_vs_polyak() -> dict:
    """E11: Nesterov（强凸版） vs Polyak Momentum，在 κ=10/100/1000 上均应收敛。"""
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, kappa in zip(axes, CONDITION_NUMBERS):
        a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
        x_star = optimal_point(a, b)
        f_star = objective(a, b, x_star)
        for name, label, color in [
            ("momentum", "Polyak Heavy-ball", "#ff7f0e"),
            ("nesterov", "Nesterov (strongly convex)", "#9467bd"),
        ]:
            kwargs = {"mu": mu, "L": L}
            if name == "momentum":
                kwargs["beta"] = -1  # 用 Polyak 最优
            run = run_optimizer(name, a, b, np.zeros(DIM), MAX_ITER, **kwargs)
            gaps = [max(f - f_star, 1e-30) for f in run.fs]
            ax.semilogy(gaps, label=label, color=color, linewidth=1.5)
            rho_emp = _fit_log_linear_rate(gaps)
            results[f"kappa_{kappa}_{name}_iter"] = _first_below(gaps, TOL)
            results[f"kappa_{kappa}_{name}_rho_emp"] = rho_emp
        rho_theory = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
        ax.axhline(1e-6, color="gray", linestyle=":", alpha=0.6)
        ax.set_title(rf"$\kappa={kappa}$, theory $\rho^*={rho_theory:.4f}$")
        ax.set_xlabel("Iteration k")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$f(x_k) - f^*$")
    fig.suptitle("E11: Nesterov (strongly-convex) vs Polyak Heavy-ball", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp11_nesterov_vs_polyak.png", dpi=150)
    plt.close(fig)
    return results


def exp12_muon_spectral_recovery() -> dict:
    """E12: Muon 在谱范数恢复目标上严格下降；与 GD/Adam 对照。

    问题：min_W (1/2) ‖W - W*‖_σ²  (谱范数损失)
    Muon 的极分解方向恰是谱范数最速下降，故应严格下降。
    """
    m, n = 16, 8
    w_star, _ = make_spectral_recovery_problem(m, n, seed=SEED)
    w0 = np.zeros((m, n))
    max_iter = 80
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ---- 谱范数损失（Muon 自然下降）----
    losses_muon, _ = run_muon_on_spectral_recovery(w_star, w0, max_iter, lr=0.3)
    losses_gd, _ = run_gd_on_spectral_recovery(w_star, w0, max_iter, lr=0.5)
    losses_adam, _ = run_adam_on_spectral_recovery(w_star, w0, max_iter, lr=0.3)
    axes[0].semilogy(losses_muon, label="Muon-NS", linewidth=1.8, color="#d62728")
    axes[0].semilogy(losses_gd, label="GD (Frobenius grad)", linewidth=1.5, color="#1f77b4")
    axes[0].semilogy(losses_adam, label="Adam (flat)", linewidth=1.5, color="#2ca02c")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"Spectral loss $\frac{1}{2}\|W-W^*\|_\sigma^2$")
    axes[0].set_title("Spectral recovery (Muon's home turf)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    results["spectral_muon_final"] = float(losses_muon[-1])
    results["spectral_gd_final"] = float(losses_gd[-1])
    results["spectral_adam_final"] = float(losses_adam[-1])
    results["spectral_muon_iter_tol"] = _first_below(losses_muon, 1e-3)

    # ---- 对照：原 Frobenius 矩阵二次（说明 Muon 不下降的几何错配）----
    kappa = 100.0
    a, b, w_star2, mu, L = make_matrix_problem(16, 8, kappa, seed=SEED)
    f_star = mat_objective(a, b, w_star2)
    w0b = np.zeros((16, 8))
    lr_gd = 1.5 / (L * L)
    lr_muon = 0.001 / L
    fs_gd, _ = run_matrix_gd(a, b, w0b, 400, lr_gd)
    fs_muon, _ = run_matrix_muon(a, b, w0b, 400, lr_muon, ns_steps=8)
    gaps_gd = [max(f - f_star, 1e-30) for f in fs_gd]
    gaps_muon = [max(f - f_star, 1e-30) for f in fs_muon]
    axes[1].semilogy(gaps_gd, label="GD (Frobenius)", linewidth=1.5, color="#1f77b4")
    axes[1].semilogy(gaps_muon, label="Muon-NS (geometric mismatch)",
                     linewidth=1.5, color="#d62728")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$f(W_k) - f^*$ (Frobenius)")
    axes[1].set_title(r"Frobenius quadratic, $\kappa(A)=100$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    results["frob_gd_final"] = float(gaps_gd[-1])
    results["frob_muon_final"] = float(gaps_muon[-1])

    fig.suptitle("E12: Muon shines on spectral-norm geometry, struggles on Frobenius",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp12_matrix_muon.png", dpi=150)
    plt.close(fig)
    return results


def exp13_adam_lr_sweep() -> dict:
    """E13: Adam 学习率扫描热力图（迭代数 vs lr / κ）。"""
    kappas = [10, 100, 1000]
    lrs = np.logspace(-2, 0.5, 18)
    iter_grid = np.full((len(kappas), len(lrs)), MAX_ITER + 1)
    results = {}
    for i, kappa in enumerate(kappas):
        a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
        x_star = optimal_point(a, b)
        f_star = objective(a, b, x_star)
        for j, lr in enumerate(lrs):
            run = run_optimizer("adam", a, b, np.zeros(DIM), MAX_ITER, lr=lr, mu=mu, L=L)
            gaps = [max(f - f_star, 1e-30) for f in run.fs]
            it = _first_below(gaps, TOL)
            iter_grid[i, j] = it if it is not None else MAX_ITER + 1
        best_idx = int(np.argmin(iter_grid[i]))
        results[f"kappa_{kappa}_best_lr"] = float(lrs[best_idx])
        results[f"kappa_{kappa}_best_iter"] = int(iter_grid[i, best_idx])

    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(iter_grid, aspect="auto", cmap="viridis_r",
                   vmin=0, vmax=min(MAX_ITER, 2000))
    ax.set_yticks(range(len(kappas)))
    ax.set_yticklabels([rf"$\kappa={k}$" for k in kappas])
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], rotation=45, ha="right")
    ax.set_xlabel("Adam learning rate")
    ax.set_title(f"E13: Adam iterations to reach {TOL:.0e}")
    plt.colorbar(im, ax=ax, label="Iterations (max = no conv.)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp13_adam_lr_sweep.png", dpi=150)
    plt.close(fig)
    return results


def exp14_sgd_noise() -> dict:
    """E14: 带噪梯度 SGD vs Adam，展示 Adam 的"自适应步长抗噪"特性。"""
    kappa = 100.0
    a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)
    max_iter = 800
    sigmas = [0.0, 0.01, 0.05]
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    colors = {"sgd": "#1f77b4", "adam": "#2ca02c"}
    for ax, name in zip(axes, ["sgd", "adam"]):
        for sigma in sigmas:
            rng = np.random.default_rng(SEED)
            x = np.zeros(DIM)
            m_st = np.zeros(DIM)
            v_st = np.zeros(DIM)
            fs = []
            for t in range(1, max_iter + 1):
                fs.append(max(objective(a, b, x) - f_star, 1e-30))
                g_true = a @ x - b
                noise = rng.normal(0, sigma, DIM) if sigma > 0 else 0.0
                g = g_true + noise
                if name == "sgd":
                    x = x - (2.0 / (mu + L)) * g
                else:
                    m_st = 0.9 * m_st + 0.1 * g
                    v_st = 0.999 * v_st + 0.001 * (g * g)
                    mh = m_st / (1 - 0.9 ** t)
                    vh = v_st / (1 - 0.999 ** t)
                    x = x - ADAM_LR * mh / (np.sqrt(vh) + 1e-8)
            ax.semilogy(fs, label=rf"$\sigma={sigma}$", linewidth=1.3,
                       alpha=0.85 if sigma > 0 else 1.0,
                       linestyle="-" if sigma > 0 else "--")
            results[f"{name}_sigma{sigma}_floor"] = float(fs[-1])
        ax.set_title(name.upper())
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$f-f^*$")
    fig.suptitle(r"E14: noise floor as a function of noise level $\sigma$", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp14_sgd_noise.png", dpi=150)
    plt.close(fig)
    return results


def exp15_sophia_vs_adam() -> dict:
    """E15: Sophia (Hessian-on-diagonal) vs Adam，展示二阶 vs 一阶预条件差异。"""
    kappa = 100.0
    a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    specs = [
        ("adam", "Adam", "#2ca02c", {"lr": 0.4}),
        ("sophia", "Sophia-H (diag Hess)", "#d62728", {"lr": 0.5, "rho": 0.04}),
    ]
    for name, label, color, kw in specs:
        run = run_optimizer(name, a, b, np.zeros(DIM), MAX_ITER, mu=mu, L=L, **kw)
        gaps = [max(f - f_star, 1e-30) for f in run.fs]
        axes[0].semilogy(gaps, label=label, color=color, linewidth=1.5)
        kappa_eff = [
            effective_preconditioned_condition_number(a, p)
            for p in run.preconditioners[1:301]
        ]
        axes[1].plot(kappa_eff, color=color, label=label, linewidth=1.3)
        results[f"{name}_iter"] = _first_below(gaps, TOL)
        results[f"{name}_final"] = float(gaps[-1])

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$f - f^*$")
    axes[0].set_title("Convergence")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(kappa, color="gray", linestyle="--", alpha=0.7,
                    label=rf"$\kappa(A)={int(kappa)}$")
    axes[1].set_xlabel("Iteration (first 300)")
    axes[1].set_ylabel(r"$\kappa(P_k^{-1} A)$")
    axes[1].set_yscale("log")
    axes[1].set_title(r"Effective $\kappa$ (lower is better)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("E15: Sophia-H vs Adam — second-order vs first-order preconditioning",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp15_sophia.png", dpi=150)
    plt.close(fig)
    return results


def exp17_pcg_baseline() -> dict:
    """E17: CG / PCG + Jacobi 与一阶方法对比；附 CG 理论上界。"""
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, kappa in zip(axes, CONDITION_NUMBERS):
        a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
        x_star = optimal_point(a, b)
        f_star = objective(a, b, x_star)
        x0 = np.zeros(DIM)

        fs_cg, cg_iters, _ = run_cg(a, b, x0, DIM + 10)
        gaps_cg = [max(f - f_star, 1e-30) for f in fs_cg]
        ax.semilogy(gaps_cg, label=f"CG ({cg_iters} iters)", color="#8c564b", linewidth=2)
        fs_pcg, pcg_iters = run_pcg(a, b, x0, DIM + 10, tol=1e-14)
        gaps_pcg = [max(f - f_star, 1e-30) for f in fs_pcg]
        ax.semilogy(gaps_pcg, label=f"PCG+Jacobi ({pcg_iters})",
                   color="#9467bd", linewidth=2, linestyle="--")
        # CG 理论 Chebyshev 上界
        e0 = gaps_cg[0]
        ks = np.arange(len(gaps_cg))
        bound = [4 * e0 * (cg_theoretical_bound(k, kappa)) ** 2 / 4 for k in ks]
        ax.semilogy(ks, np.maximum(bound, 1e-30), "k:", linewidth=1,
                   label=r"Cheb. bound $4 ((\sqrt{\kappa}-1)/(\sqrt{\kappa}+1))^{2k}$")

        for name, label, color in [
            ("momentum", "Momentum", "#ff7f0e"),
            ("adam", "Adam", "#2ca02c"),
        ]:
            kwargs = {"beta": -1, "mu": mu, "L": L} if name == "momentum" else {"mu": mu, "L": L}
            lr = ADAM_LR if name == "adam" else None
            run = run_optimizer(name, a, b, x0, min(MAX_ITER, 600), lr=lr, **kwargs)
            gaps = [max(f - f_star, 1e-30) for f in run.fs]
            ax.semilogy(gaps, label=label, color=color, linewidth=1.2, alpha=0.85)
            results[f"kappa_{kappa}_{name}_iter"] = _first_below(gaps, TOL)

        results[f"kappa_{kappa}_cg"] = cg_iters
        results[f"kappa_{kappa}_pcg"] = pcg_iters
        ax.set_title(rf"$\kappa={kappa}$")
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$f - f^*$")
    fig.suptitle("E17: CG / PCG vs first-order methods (with Chebyshev bound)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp17_pcg_baseline.png", dpi=150)
    plt.close(fig)
    return results


def exp1_multiseed_band() -> dict:
    """多种子：实验1 收敛带。"""
    kappa = 100.0
    methods = ["gd", "momentum", "adam"]
    results = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    max_show = 800
    x_axis = np.arange(max_show + 1)

    for name in methods:
        all_gaps = []
        for seed in SEEDS:
            a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=seed)
            x_star = optimal_point(a, b)
            f_star = objective(a, b, x_star)
            kwargs = {"beta": -1, "mu": mu, "L": L} if name == "momentum" else {"mu": mu, "L": L}
            lr = ADAM_LR if name == "adam" else None
            run = run_optimizer(name, a, b, np.zeros(DIM), max_show, lr=lr, **kwargs)
            gaps = np.array([max(f - f_star, 1e-30) for f in run.fs])
            all_gaps.append(gaps)
        stack = np.vstack(all_gaps)
        med = np.median(stack, axis=0)
        q25 = np.percentile(stack, 25, axis=0)
        q75 = np.percentile(stack, 75, axis=0)
        ax.semilogy(x_axis, med, linewidth=2, label=name.upper())
        ax.fill_between(x_axis, q25, q75, alpha=0.25)
        results[f"{name}_median_final"] = float(med[-1])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f - f^*$")
    ax.set_title(f"Multi-seed bands (seeds={SEEDS}), kappa=100")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp1_multiseed.png", dpi=150)
    plt.close(fig)
    return results


def exp23_ns_rectangular() -> dict:
    """E23: 瘦/方/胖矩阵 Newton-Schulz（修复后胖矩阵也应收敛）。"""
    rng = np.random.default_rng(SEED)
    shapes = [(32, 8), (16, 16), (8, 32)]
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (m, n) in zip(axes, shapes):
        g = rng.standard_normal((m, n))
        _, errors = newton_schulz_iterate(g, max_iter=18, scale_factor=1.0)
        ax.semilogy(errors, "o-", linewidth=1.4)
        ax.set_title(rf"${m}\times{n}$")
        ax.set_xlabel("NS step")
        ax.grid(True, alpha=0.3)
        results[f"{m}x{n}_final"] = errors[-1]
        results[f"{m}x{n}_iter_1e6"] = _first_below(errors, 1e-6)
    axes[0].set_ylabel(r"$\|G_{\min}-I\|_F$ (Gram of min-side)")
    fig.suptitle("E23: Newton-Schulz on rectangular matrices (fat case fixed)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp23_ns_rectangular.png", dpi=150)
    plt.close(fig)
    return results


def exp1b_full_optimizer_panel() -> dict:
    """E1b: GD / Polyak / Nesterov / Adam / AdamW / Jacobi 同图 (kappa=100)。"""
    kappa = 100.0
    a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)
    specs = [
        ("gd", "GD", "#1f77b4", None),
        ("momentum", "Polyak HB", "#ff7f0e", {"beta": -1}),
        ("nesterov", "Nesterov (strongly convex)", "#9467bd", {}),
        ("adam", "Adam", "#2ca02c", {}),
        ("adamw", "AdamW", "#17becf", {}),
        ("jacobi", "Jacobi GD (oracle)", "#d62728", {}),
    ]
    results = {}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, label, color, extra in specs:
        kw = {"mu": mu, "L": L, **(extra or {})}
        lr = ADAM_LR if name in ("adam", "adamw") else (1.0 if name == "jacobi" else None)
        run = run_optimizer(name, a, b, np.zeros(DIM), MAX_ITER, lr=lr, **kw)
        gaps = [max(f - f_star, 1e-30) for f in run.fs]
        ax.semilogy(gaps, label=label, color=color, linewidth=1.5)
        results[name] = _first_below(gaps, TOL)
    ax.set_xlabel("Iteration k")
    ax.set_ylabel(r"$f - f^*$")
    ax.set_title(rf"E1b: full panel of optimizers, $\kappa={int(kappa)}$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp1b_full_panel.png", dpi=150)
    plt.close(fig)
    return results


def exp21_eigenmode_decay() -> dict:
    """E21: 误差在 A 特征基下的模态能量衰减。"""
    kappa = 100.0
    a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
    x_star = optimal_point(a, b)
    eigvals, q = np.linalg.eigh(a)
    idx = np.argsort(eigvals)
    eigvals, q = eigvals[idx], q[:, idx]

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, title) in zip(axes, [("gd", "GD"), ("momentum", "Polyak Momentum")]):
        kwargs = {"beta": -1, "mu": mu, "L": L} if name == "momentum" else {"mu": mu, "L": L}
        run = run_optimizer(name, a, b, np.zeros(DIM), 400, **kwargs)
        steps = min(200, len(run.xs))
        modes_lo, modes_hi = [], []
        for x in run.xs[:steps]:
            coeff = q.T @ (x - x_star)
            energy = coeff * coeff
            modes_lo.append(energy[0])
            modes_hi.append(energy[-1])
        t = np.arange(len(modes_lo))
        ax.semilogy(t, modes_lo, label=rf"$\lambda_{{\min}}=\mu$", linewidth=1.5)
        ax.semilogy(t, modes_hi, label=rf"$\lambda_{{\max}}=L$", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        results[f"{name}_ratio_final"] = float(
            modes_hi[-1] / max(modes_lo[-1], 1e-30)
        )
    axes[0].set_ylabel(r"Modal energy $|c_i|^2$")
    fig.suptitle(r"E21: eigenmode energy decay, $\kappa=100$", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp21_eigenmode_decay.png", dpi=150)
    plt.close(fig)
    return results


def exp24_adam_beta_ablation() -> dict:
    """E24: Adam (β1,β2) 消融——每个组合**单独扫 lr**取最优。"""
    kappa = 100.0
    a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)
    configs = [
        (0.9, 0.999, "Adam (0.9, 0.999)"),
        (0.0, 0.999, r"RMSprop-like ($\beta_1$=0)"),
        (0.9, 0.0, r"Momentum-like ($\beta_2$=0)"),
        (0.0, 0.0, r"SignSGD-like (both 0)"),
    ]
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lr_grid = np.logspace(-3, 0.5, 12)

    for beta1, beta2, label in configs:
        # 对每个 (β1,β2) 单独扫 lr，取最优
        best_iter, best_lr, best_gaps = MAX_ITER + 1, None, None
        for lr in lr_grid:
            run = run_optimizer(
                "adam", a, b, np.zeros(DIM), 600,
                lr=lr, mu=mu, L=L, beta1=beta1, beta2=beta2,
            )
            gaps = [max(f - f_star, 1e-30) for f in run.fs]
            it = _first_below(gaps, TOL)
            if it is not None and it < best_iter:
                best_iter, best_lr, best_gaps = it, lr, gaps
        if best_gaps is None:
            # 退而求其次：取最终 gap 最小的
            best_lr = lr_grid[np.argmin([
                run_optimizer("adam", a, b, np.zeros(DIM), 600,
                              lr=lr, mu=mu, L=L,
                              beta1=beta1, beta2=beta2).fs[-1]
                for lr in lr_grid
            ])]
            run = run_optimizer(
                "adam", a, b, np.zeros(DIM), 600,
                lr=best_lr, mu=mu, L=L, beta1=beta1, beta2=beta2,
            )
            best_gaps = [max(f - f_star, 1e-30) for f in run.fs]
        axes[0].semilogy(best_gaps, linewidth=1.3,
                        label=rf"{label}, lr={best_lr:.2f}")
        # 用 best 配置算 kappa_eff
        run_best = run_optimizer(
            "adam", a, b, np.zeros(DIM), 100,
            lr=best_lr, mu=mu, L=L, beta1=beta1, beta2=beta2,
        )
        kappa_eff = [
            effective_preconditioned_condition_number(a, p)
            for p in run_best.preconditioners[1:51]
        ]
        axes[1].plot(kappa_eff[:50], linewidth=1.3, label=label)
        results[label] = {
            "iter_tol": best_iter if best_iter <= MAX_ITER else None,
            "best_lr": float(best_lr),
        }

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$f-f^*$")
    axes[0].set_title("Convergence (each ablation uses its best lr)")
    axes[0].legend(fontsize=6)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Iteration (first 50)")
    axes[1].set_ylabel(r"$\kappa(P_k^{-1}A)$")
    axes[1].set_yscale("log")
    axes[1].set_title(r"Effective $\kappa$")
    axes[1].legend(fontsize=6)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(r"E24: Adam $(\beta_1,\beta_2)$ ablation with per-config best lr",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp24_adam_ablation.png", dpi=150)
    plt.close(fig)
    return results


def exp_kappa_continuous_scan() -> dict:
    """κ 扫描：MAX_ITER 按理论上界动态计算，使曲线不被截断。"""
    kappas = np.unique(np.round(np.logspace(1, 4, 20)).astype(int))
    iter_gd, iter_mom = [], []
    rho_gd_th, rho_mom_th = [], []

    for kappa in kappas:
        a, b, mu, L = make_quadratic_problem(DIM, float(kappa), seed=SEED)
        x_star = optimal_point(a, b)
        f_star = objective(a, b, x_star)
        rho_gd = (kappa - 1) / (kappa + 1)
        rho_mom = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
        rho_gd_th.append(rho_gd)
        rho_mom_th.append(rho_mom)

        # 动态 max_iter：按理论 ρ^k * e0 = tol 估算
        e0 = max(objective(a, b, np.zeros(DIM)) - f_star, 1e-30)
        max_iter_gd = int(min(30000, 2.5 * np.log(e0 / TOL) / max(-np.log(rho_gd), 1e-6)))
        max_iter_mom = int(min(10000, 2.5 * np.log(e0 / TOL) / max(-np.log(rho_mom), 1e-6)))

        for name, lst, mxi in [
            ("gd", iter_gd, max_iter_gd),
            ("momentum", iter_mom, max_iter_mom),
        ]:
            kw = {"mu": mu, "L": L}
            if name == "momentum":
                kw["beta"] = -1
            run = run_optimizer(name, a, b, np.zeros(DIM), mxi, **kw)
            gaps = [max(f - f_star, 1e-30) for f in run.fs]
            it = _first_below(gaps, TOL)
            lst.append(it if it is not None else mxi)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.loglog(kappas, iter_gd, "o-", label="GD iterations", color="#1f77b4")
    ax.loglog(kappas, iter_mom, "s-", label="Polyak Momentum iterations",
             color="#ff7f0e")
    # 理论参考曲线：GD ~ κ log(1/tol)/2, Mom ~ √κ log(1/tol)/2
    e0_ref = 1e3
    log_inv_tol = np.log(e0_ref / TOL)
    th_gd = kappas * log_inv_tol / 2
    th_mom = np.sqrt(kappas) * log_inv_tol / 2
    ax.loglog(kappas, th_gd, "--", color="#1f77b4", alpha=0.5,
             label=r"$\kappa \log(e_0/\epsilon)/2$")
    ax.loglog(kappas, th_mom, "--", color="#ff7f0e", alpha=0.5,
             label=r"$\sqrt{\kappa} \log(e_0/\epsilon)/2$")
    ax.set_xlabel(r"Condition number $\kappa$")
    ax.set_ylabel(f"Iterations to reach {TOL:.0e}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(r"$\kappa$ scan: empirical vs theoretical complexity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp_kappa_scan.png", dpi=150)
    plt.close(fig)
    return {
        "kappas": kappas.tolist(),
        "iter_gd": iter_gd,
        "iter_mom": iter_mom,
    }


def exp22_beta_eta_heatmap() -> dict:
    """E22: (β, η) 平面上 Heavy-ball 最坏谱半径热力图。"""
    kappa = 100.0
    mu, L = 1.0, kappa
    step_opt, beta_opt = optimal_polyak_params(mu, L)
    betas = np.linspace(0.0, 0.995, 60)
    etas = np.linspace(0.002, 0.08, 60)
    rho = np.zeros((len(etas), len(betas)))
    for i, eta in enumerate(etas):
        for j, beta in enumerate(betas):
            rho[i, j] = worst_case_spectral_radius(mu, L, eta, beta)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(
        rho,
        origin="lower",
        aspect="auto",
        extent=[betas[0], betas[-1], etas[0], etas[-1]],
        cmap="viridis_r",
        vmin=(np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1) - 0.05,
        vmax=1.0,
    )
    ax.plot(beta_opt, step_opt, "r*", markersize=16, label="Polyak optimal")
    ax.plot(0.9, step_opt, "o", color="orange", markersize=9,
           label=r"ML default $\beta=0.9$")
    # 标出 ρ=1 等高线（稳定边界）
    cs = ax.contour(betas, etas, rho, levels=[1.0], colors="white",
                   linewidths=1.5, linestyles="--")
    ax.clabel(cs, inline=True, fontsize=8, fmt={1.0: "stability bdry"})
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"Worst-case $\rho(\eta,\beta)$")
    ax.set_xlabel(r"Momentum $\beta$")
    ax.set_ylabel(r"Step size $\eta$")
    ax.set_title(r"E22: Heavy-ball spectral radius landscape, $\kappa=100$")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp22_beta_eta_heatmap.png", dpi=150)
    plt.close(fig)
    return {
        "rho_polyak": float(worst_case_spectral_radius(mu, L, step_opt, beta_opt)),
        "rho_at_beta09": float(worst_case_spectral_radius(mu, L, step_opt, 0.9)),
        "eta_opt": float(step_opt),
        "beta_opt": float(beta_opt),
    }


# ===========================================================================
# 新增实验：Muon-centric 与前沿方向
# ===========================================================================


def exp25_heavy_ball_chebyshev_equivalence() -> dict:
    """E25: Heavy-ball = Chebyshev 半迭代的"定常极限"（核心理论结果数值验证）。

    精确主张（定理 4）：Chebyshev 半迭代是有限步 minimax 最优的（每一步都最优），
    其时变系数 ω_k 单调收敛到 ω_∞ = 1 + β*；Polyak Heavy-ball 用定常 β* 即 ω_∞，
    因此 HB 是 Chebyshev 的"冻结系数版本"，二者共享同一渐近收敛率，
    且 Chebyshev 在任意有限 k 上不慢于 HB。
    """
    kappa = 100.0
    a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)
    max_iter = 200
    x0 = np.zeros(DIM)

    _, fs_cheb = chebyshev_semi_iterative(a, b, x0, max_iter, mu, L)
    _, fs_hb = heavy_ball_polyak_form(a, b, x0, max_iter, mu, L)
    gaps_cheb = [max(f - f_star, 1e-30) for f in fs_cheb]
    gaps_hb = [max(f - f_star, 1e-30) for f in fs_hb]

    # Chebyshev 时变系数 ω_k 的递推 + 理论极限 ω_∞ = 1 + β*
    sigma = (L - mu) / (L + mu)
    beta_star = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    omega_inf = 1.0 + beta_star
    omegas = []
    omega = 1.0 / (1.0 - 0.5 * sigma * sigma)
    omegas.append(omega)
    for _ in range(60):
        omega = 1.0 / (1.0 - 0.25 * sigma * sigma * omega)
        omegas.append(omega)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    # 左：收敛曲线对照（含 minimax 上界）
    axes[0].semilogy(gaps_cheb, label="Chebyshev semi-iterative",
                    color="#1f77b4", linewidth=1.7)
    axes[0].semilogy(gaps_hb, label="Heavy-ball (Polyak, frozen $\\omega_\\infty$)",
                    color="#ff7f0e", linewidth=1.7, linestyle="--")
    bounds = [max(2 * chebyshev_minimax_bound(k, kappa) ** 2 * gaps_cheb[0], 1e-30)
              for k in range(len(gaps_cheb))]
    axes[0].semilogy(bounds, "k:", linewidth=1, label=r"Cheb. minimax bound")
    axes[0].set_xlabel("Iteration k")
    axes[0].set_ylabel(r"$f - f^*$")
    axes[0].set_title(rf"$\kappa={int(kappa)}$: convergence")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 中：系数收敛 ω_k → 1 + β*（定理 4 的直接证据）
    axes[1].plot(range(len(omegas)), omegas, "o-", color="#1f77b4",
                markersize=3, linewidth=1.2, label=r"Chebyshev $\omega_k$")
    axes[1].axhline(omega_inf, color="#ff7f0e", linestyle="--",
                   label=rf"$\omega_\infty = 1+\beta^* = {omega_inf:.4f}$")
    axes[1].set_xlabel("Iteration k")
    axes[1].set_ylabel(r"Recurrence coefficient $\omega_k$")
    axes[1].set_title(r"$\omega_k \to 1+\beta^*$ (Theorem 4)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 右：误差多项式可视化
    lam_grid = np.linspace(mu, L, 200)
    for k in [5, 10, 20, 40]:
        e_poly = [chebyshev_error_polynomial(k, lam, mu, L) for lam in lam_grid]
        axes[2].plot(lam_grid, np.abs(e_poly), label=f"k={k}", linewidth=1.2)
    axes[2].axhline(1, color="gray", linestyle=":")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$\lambda$ (eigenvalue of $A$)")
    axes[2].set_ylabel(r"$|e_k(\lambda)|$")
    axes[2].set_title(r"Chebyshev error polynomial")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("E25: Heavy-ball is the frozen-coefficient limit of Chebyshev "
                 "semi-iteration", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp25_hb_chebyshev.png", dpi=150)
    plt.close(fig)
    return {
        "max_gap_diff": float(max(
            abs(gc - gh) for gc, gh in zip(gaps_cheb[-50:], gaps_hb[-50:])
        )),
        "cheb_final": float(gaps_cheb[-1]),
        "hb_final": float(gaps_hb[-1]),
        "omega_inf_theory": float(omega_inf),
        "omega_50_numeric": float(omegas[50]),
        "omega_converged": bool(abs(omegas[50] - omega_inf) < 1e-4),
    }


def exp26_adam_limit_cycle() -> dict:
    """E26: Adam 在最简凸二次上的极限环现象 (Bock-Weiß 2022 复现)。"""
    # 扫几组超参，寻找 2-极限环；记录振幅
    configs = [
        # (lr, beta1, beta2, eps, a, init)
        (0.5, 0.9, 0.999, 1e-8, 1.0, 1.0),    # 标准 Adam
        (0.05, 0.0, 0.99, 1e-12, 1.0, 1.0),   # RMSprop-like，已知出现极限环
        (0.1, 0.5, 0.99, 1e-12, 1.0, 1.0),
        (0.2, 0.9, 0.9, 1e-12, 1.0, 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    results = {}
    for ax, (lr, b1, b2, eps, a, x0) in zip(axes.flat, configs):
        xs, _, _ = adam_trajectory_scalar(x0, a, lr, b1, b2, eps, max_iter=2000)
        ax.plot(xs[:800], linewidth=0.9, color="#2ca02c")
        ax.axhline(0, color="red", linestyle=":", linewidth=1, label=r"$x^*=0$")
        ax.set_xlabel("Iteration t")
        ax.set_ylabel("x_t")
        ax.set_title(rf"$\eta={lr},\ \beta_1={b1},\ \beta_2={b2}$")
        ax.grid(True, alpha=0.3)
        is_cycle, xp, xm = detect_limit_cycle(xs)
        results[f"lr{lr}_b1{b1}_b2{b2}"] = {
            "is_2cycle": is_cycle,
            "x_plus": xp,
            "x_minus": xm,
            "tail_amplitude": float(np.max(xs[-200:]) - np.min(xs[-200:])),
        }
        cycle_text = (
            rf"2-cycle: $x^\pm=({xp:.3f},{xm:.3f})$" if is_cycle else "no clean 2-cycle"
        )
        ax.text(0.05, 0.95, cycle_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        r"E26: Adam on $f(x)=\frac{1}{2} x^2$: non-convergence & limit cycle (Bock-Weiß 2022)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp26_adam_limit_cycle.png", dpi=150)
    plt.close(fig)
    return results


def exp27_chebyshev_ns_vs_standard() -> dict:
    """E27: Chebyshev 加速 NS vs 标准 NS（arXiv 2506.10935 简化版）。"""
    rng = np.random.default_rng(SEED)
    sizes = [(16, 8), (32, 16), (64, 32)]
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (m, n) in zip(axes, sizes):
        g = rng.standard_normal((m, n))
        _, errs_std = newton_schulz_iterate(g, max_iter=12)
        _, errs_cheb = chebyshev_ns_iterate(g, max_iter=12)
        ax.semilogy(errs_std, "o-", label="Standard NS (3rd-order)",
                   color="#1f77b4", linewidth=1.4)
        ax.semilogy(errs_cheb, "s-", label="Chebyshev NS (5th-order)",
                   color="#d62728", linewidth=1.4)
        ax.set_title(rf"${m}\times{n}$")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        results[f"{m}x{n}_std_final"] = errs_std[-1]
        results[f"{m}x{n}_cheb_final"] = errs_cheb[-1]
    axes[0].set_ylabel(r"Gram error $\|X^\top X - I\|_F$")
    fig.suptitle("E27: Chebyshev-accelerated Newton-Schulz vs standard 3rd-order",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp27_chebyshev_ns.png", dpi=150)
    plt.close(fig)
    return results


def exp28_muon_on_trust_region() -> dict:
    """E28: Muon vs GD vs Adam 在谱范数 trust-region 损失上的对比。

    问题：min_W (1/2) ‖W - W*‖_σ² (谱范数距离)；
    Muon 一步即沿极分解方向下降，应严格、快速收敛。
    """
    seeds = [0, 1, 2, 3, 4]
    max_iter = 60
    results = {"muon_finals": [], "gd_finals": [], "adam_finals": []}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seed in seeds:
        w_star, _ = make_spectral_recovery_problem(12, 8, seed=seed)
        w0 = np.zeros_like(w_star)
        lm, _ = run_muon_on_spectral_recovery(w_star, w0, max_iter, lr=0.3)
        lg, _ = run_gd_on_spectral_recovery(w_star, w0, max_iter, lr=0.5)
        la, _ = run_adam_on_spectral_recovery(w_star, w0, max_iter, lr=0.3)
        results["muon_finals"].append(float(lm[-1]))
        results["gd_finals"].append(float(lg[-1]))
        results["adam_finals"].append(float(la[-1]))
        alpha = 1.0 if seed == seeds[0] else 0.35
        labels = (
            ("Muon-NS", "GD (Frobenius)", "Adam (flat)")
            if seed == seeds[0]
            else (None, None, None)
        )
        ax.semilogy(lm, color="#d62728", linewidth=1.6, alpha=alpha, label=labels[0])
        ax.semilogy(lg, color="#1f77b4", linewidth=1.2, alpha=alpha, label=labels[1])
        ax.semilogy(la, color="#2ca02c", linewidth=1.2, alpha=alpha, label=labels[2])
    ax.set_xlabel("Iteration k")
    ax.set_ylabel(r"Spectral loss $\frac{1}{2}\|W-W^*\|_\sigma^2$")
    ax.set_title(r"E28: Muon on spectral-norm trust-region "
                 r"(5 seeds; thick = seed 0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp28_muon_trust_region.png", dpi=150)
    plt.close(fig)
    return results


def exp29_nag_ode_vs_discrete() -> dict:
    """E29: NAG-ODE (Su-Boyd-Candès) 解轨迹 vs 离散 NAG。

    在二维二次上比较：
    - RK4 解 X'' + (3/t) X' + ∇f = 0
    - 离散 NAG 以步长 √η 对应 t = √η k
    - 离散 Heavy-ball 用 Polyak 最优 (η*, β*)
    """
    kappa = 50.0
    a, b, mu, L = make_quadratic_problem(2, kappa, seed=SEED)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)

    def grad(x):
        return a @ x - b

    x0 = np.array([3.0, 2.5])
    eta = 1.0 / L
    sqrt_eta = np.sqrt(eta)
    n_steps_disc = 200
    t_end = sqrt_eta * n_steps_disc

    # NAG-ODE 数值解（Su et al.）
    t_ode, xs_ode = nesterov_avd_ode_rk4(
        grad, x0.copy(), t_start=0.1 * sqrt_eta,
        t_end=t_end, n_steps=2000
    )
    fs_ode = [objective(a, b, x) - f_star for x in xs_ode]

    # 梯度流 ODE（对比）
    t_gf, xs_gf = gradient_flow_rk4(grad, x0.copy(), t_end=t_end * 0.6, n_steps=2000)
    fs_gf = [objective(a, b, x) - f_star for x in xs_gf]

    # 离散 NAG
    run_nag = run_optimizer(
        "nesterov", a, b, x0.copy(), n_steps_disc, mu=mu, L=L
    )
    gaps_nag = [max(f - f_star, 1e-30) for f in run_nag.fs]
    t_nag = sqrt_eta * np.arange(len(gaps_nag))

    # 离散 Heavy-ball
    run_hb = run_optimizer(
        "momentum", a, b, x0.copy(), n_steps_disc, mu=mu, L=L, beta=-1
    )
    gaps_hb = [max(f - f_star, 1e-30) for f in run_hb.fs]
    t_hb = sqrt_eta * np.arange(len(gaps_hb))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].semilogy(t_ode, np.maximum(fs_ode, 1e-30),
                    label=r"NAG-ODE $\ddot X+(3/t)\dot X+\nabla f=0$",
                    linewidth=1.7, color="#9467bd")
    axes[0].semilogy(t_gf, np.maximum(fs_gf, 1e-30),
                    label=r"Gradient flow $\dot x=-\nabla f$",
                    linewidth=1.5, color="#1f77b4", alpha=0.7)
    axes[0].semilogy(t_nag, gaps_nag, "o", markersize=3,
                    label="Discrete Nesterov", color="#ff7f0e", markevery=5)
    axes[0].semilogy(t_hb, gaps_hb, "s", markersize=3,
                    label="Discrete Heavy-ball", color="#d62728", markevery=5)
    axes[0].set_xlabel(r"Continuous time $t = \sqrt{\eta} k$")
    axes[0].set_ylabel(r"$f - f^*$")
    axes[0].set_title("Continuous ODE vs discrete iterations")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 2D 相图
    span = 4.0
    coords = np.linspace(-span, span, 60)
    x1g, x2g = np.meshgrid(coords, coords)
    z = np.zeros_like(x1g)
    for i in range(60):
        for j in range(60):
            z[i, j] = objective(a, b, np.array([x1g[i, j], x2g[i, j]])) - f_star
    axes[1].contour(x1g, x2g, z, levels=np.logspace(-1, 3, 12),
                   cmap="viridis", alpha=0.5)
    nag_xs = np.array(run_nag.xs)
    hb_xs = np.array(run_hb.xs)
    ode_xs = np.array(xs_ode)
    axes[1].plot(ode_xs[:, 0], ode_xs[:, 1], color="#9467bd", linewidth=2,
                label="NAG-ODE trajectory")
    axes[1].plot(nag_xs[:, 0], nag_xs[:, 1], "o-", color="#ff7f0e",
                markersize=3, linewidth=0.6, label="Discrete NAG")
    axes[1].plot(hb_xs[:, 0], hb_xs[:, 1], "s-", color="#d62728",
                markersize=3, linewidth=0.6, label="Discrete HB")
    axes[1].plot(x_star[0], x_star[1], "k*", markersize=14, label=r"$x^*$")
    axes[1].plot(x0[0], x0[1], "ko", markersize=7)
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_2$")
    axes[1].set_title(r"2D phase portrait, $\kappa=50$")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].set_aspect("equal")

    fig.suptitle("E29: NAG ↔ Su-Boyd-Candès AVD-ODE correspondence", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp29_nag_ode.png", dpi=150)
    plt.close(fig)
    return {
        "ode_final": float(max(fs_ode[-1], 1e-30)),
        "nag_final": float(gaps_nag[-1]),
        "hb_final": float(gaps_hb[-1]),
    }


def exp30_ns_convergence_basin() -> dict:
    """E30: Newton-Schulz 收敛盆的精确刻画（三种归宿）。

    标量动力学 σ_{k+1} = σ_k(3 - σ_k²)/2 有三个不动点 {-1, 0, +1}。
    诚实的相图（不只是"收敛 vs 发散"）：
      σ₀ ∈ (0, √3)        → +1  （收敛到正确的极因子 +U）
      σ₀ = √3             → 0   （退化，映到平凡不动点）
      σ₀ ∈ (√3, ~2.06)    → -1  （正交化误差 →0，但收敛到 -U，符号错）
      σ₀ ≳ 2.06           → ±∞  （真正发散）
    因此"极分解的正确收敛盆"恰为 (0, √3)；越过 √3 即便正交化误差归零，
    也已收敛到 -U（错误的极因子）。我们的实验同时记录归宿符号，
    避免把 |σ²-1|→0 误读为"收敛到正确因子"。
    """
    sigma_init = np.linspace(0.05, 2.5, 120)
    finals = []      # 最终 |σ²-1|（正交化误差）
    signs = []       # 最终符号（+1 / -1 / 0=divergent）
    for sigma in sigma_init:
        x = float(sigma)
        diverged = False
        for _ in range(60):
            if abs(x) > 1e8 or not np.isfinite(x):
                diverged = True
                break
            x = 0.5 * x * (3.0 - x * x)
        if diverged:
            finals.append(float("inf"))
            signs.append(0)
        else:
            finals.append(abs(x * x - 1))
            signs.append(1 if x > 0.5 else (-1 if x < -0.5 else 0))

    sqrt3 = np.sqrt(3)
    # 从 0 起的最大连续 +U 盆：第一个非 +U 的 σ₀
    contiguous_plus_upper = sqrt3
    for s, sg in zip(sigma_init, signs):
        if sg != 1:
            contiguous_plus_upper = s
            break
    minus_min = min((s for s, sg in zip(sigma_init, signs) if sg == -1), default=np.inf)
    minus_max = max((s for s, sg in zip(sigma_init, signs) if sg == -1), default=0)
    div_min = min((s for s, f in zip(sigma_init, finals) if not np.isfinite(f)),
                  default=np.inf)
    # (0, √3) 内全部收敛到 +U？
    all_below_sqrt3_plus = all(
        sg == 1 for s, sg in zip(sigma_init, signs) if s < sqrt3
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    # 左：σ 演化轨迹，展示三种归宿
    demo = [(0.3, "#1f77b4"), (1.0, "#1f77b4"), (1.6, "#1f77b4"),
            (1.78, "#2ca02c"), (2.0, "#2ca02c"), (2.2, "#d62728")]
    for s0, col in demo:
        traj = [s0]
        s = s0
        for _ in range(12):
            if abs(s) > 1e6:
                break
            s = 0.5 * s * (3 - s * s)
            traj.append(s)
        axes[0].plot(traj, "o-", markersize=3, linewidth=1, color=col,
                    label=rf"$\sigma_0={s0}$")
    axes[0].axhline(1, color="green", linestyle="--", alpha=0.7, label=r"$+1$ (correct $+U$)")
    axes[0].axhline(-1, color="orange", linestyle="--", alpha=0.7, label=r"$-1$ (wrong $-U$)")
    axes[0].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("NS iteration k")
    axes[0].set_ylabel(r"$\sigma_k$")
    axes[0].set_title(r"Scalar NS: three fates $\{-1, 0, +1\}$")
    axes[0].legend(fontsize=7, loc="lower left", ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-2.2, 3.0)

    # 右：σ_init vs 归宿区域着色
    plus_x = [s for s, sg in zip(sigma_init, signs) if sg == 1]
    minus_x = [s for s, sg in zip(sigma_init, signs) if sg == -1]
    div_x = [s for s, sg, f in zip(sigma_init, signs, finals) if not np.isfinite(f)]
    axes[1].scatter(plus_x, [1] * len(plus_x), c="#2ca02c", s=16,
                   label=rf"$\to +U$ ($\sigma_0 < \sqrt{{3}}$): {len(plus_x)} pts")
    axes[1].scatter(minus_x, [0] * len(minus_x), c="#ff7f0e", s=16,
                   label=rf"$\to -U$ (wrong sign): {len(minus_x)} pts")
    axes[1].scatter(div_x, [-1] * len(div_x), c="#d62728", s=16,
                   label=rf"diverge: {len(div_x)} pts")
    axes[1].axvline(sqrt3, color="black", linestyle="--",
                   label=rf"$\sqrt{{3}} = {sqrt3:.4f}$")
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(["diverge", r"$\to -U$", r"$\to +U$"])
    axes[1].set_xlabel(r"Initial $\sigma_0$")
    axes[1].set_title("Fate vs initial singular value")
    axes[1].legend(fontsize=7, loc="center right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(r"E30: NS basin — correct polar factor requires $\sigma_0 \in (0, \sqrt{3})$",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp30_ns_basin.png", dpi=150)
    plt.close(fig)
    return {
        "basin_upper_bound_theory": float(sqrt3),
        "contiguous_plus_basin_upper": float(contiguous_plus_upper),
        "all_below_sqrt3_converge_plus": bool(all_below_sqrt3_plus),
        "n_converge_plus": int(sum(1 for sg in signs if sg == 1)),
        "n_converge_minus_wrong": int(sum(1 for sg in signs if sg == -1)),
        "n_diverge": int(sum(1 for f in finals if not np.isfinite(f))),
        "wrong_sign_band": [float(minus_min) if np.isfinite(minus_min) else None,
                            float(minus_max)],
        "divergence_onset": float(div_min) if np.isfinite(div_min) else None,
        "note": "beyond sqrt3 the basin is fractal: -U / bounce-back +U / divergence",
    }


def exp31_muon_singular_value_equalization() -> dict:
    """E31: Muon 更新的"奇异值均衡"结构性特征。

    诚实主张（不是"Muon 比 GD 快"，而是 Muon 做了什么）：
    无论梯度 G 的奇异值多么悬殊，Muon 更新 −UVᵀ 的全部奇异值都是 1
    （条件数恒为 1）；GD 更新 −G 直接继承 G 的条件数。
    这是 Muon 区别于 GD 的结构性特征——它把更新"均衡化"到所有奇异方向。
    """
    rng = np.random.default_rng(SEED)
    kappa_grid = np.logspace(0, 3.5, 15)
    gd_kappa, muon_kappa, signsgd_kappa = [], [], []
    for kg in kappa_grid:
        u, _ = np.linalg.qr(rng.standard_normal((10, 10)))
        v, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        s = np.geomspace(1.0, kg, 6)
        g = u[:, :6] @ np.diag(s) @ v.T
        gd_kappa.append(update_condition_number(g, "euclidean"))
        muon_kappa.append(update_condition_number(g, "spectral"))
        signsgd_kappa.append(update_condition_number(g, "linf"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].loglog(kappa_grid, gd_kappa, "o-", color="#1f77b4",
                  label="GD update (−G)")
    axes[0].loglog(kappa_grid, signsgd_kappa, "^-", color="#2ca02c",
                  label="signSGD update (−sign G)")
    axes[0].loglog(kappa_grid, muon_kappa, "s-", color="#d62728",
                  label="Muon update (−UVᵀ)")
    axes[0].loglog(kappa_grid, kappa_grid, "k:", alpha=0.5,
                  label=r"$\kappa(\mathrm{update})=\kappa(G)$")
    axes[0].set_xlabel(r"Gradient condition number $\kappa(G)$")
    axes[0].set_ylabel(r"Update condition number")
    axes[0].set_title("Muon equalizes singular values: "
                      r"$\kappa(\mathrm{update})\equiv 1$")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, which="both")

    # 右：单个梯度的奇异值谱 vs 三种更新的奇异值谱
    u, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    v, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    s = np.geomspace(1.0, 200.0, 6)
    g = u[:, :6] @ np.diag(s) @ v.T
    sv_g = np.linalg.svd(g, compute_uv=False)
    sv_gd = np.linalg.svd(lmo_euclidean(g), compute_uv=False)
    sv_muon = np.linalg.svd(lmo_spectral(g), compute_uv=False)
    idx = np.arange(len(sv_g))
    w = 0.25
    axes[1].bar(idx - w, sv_g / sv_g.max(), w, label="grad G (normalized)",
               color="#1f77b4")
    axes[1].bar(idx, sv_gd / sv_gd.max(), w, label="GD update (normalized)",
               color="#2ca02c")
    axes[1].bar(idx + w, sv_muon, w, label="Muon update (= all 1)",
               color="#d62728")
    axes[1].set_xlabel("Singular value index")
    axes[1].set_ylabel("Singular value (normalized)")
    axes[1].set_title(r"Spectra: $\kappa(G)=200$, Muon flattens to all-ones")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("E31: Muon's structural property — singular-value equalization",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp31_muon_equalization.png", dpi=150)
    plt.close(fig)
    return {
        "muon_update_kappa_max": float(max(muon_kappa)),
        "gd_update_kappa_at_1000": float(gd_kappa[-1]),
        "muon_kappa_always_1": bool(all(k < 1.0001 for k in muon_kappa)),
    }


def exp32_steepest_descent_norms() -> dict:
    """E32: 范数视角下的最速下降三元组（GD=ℓ₂, signSGD=ℓ∞, Muon=谱范数）。

    三者都是 LMO：d* = argmin_{‖d‖≤1} ⟨g, d⟩。验证：
      (1) 各 LMO 解确实最优（蒙特卡洛对照随机方向）；
      (2) 最优值 ⟨g, d*⟩ = −对偶范数（ℓ₂↔ℓ₂, ℓ∞↔ℓ₁, 谱↔核范数）；
      (3) 在二次问题上三者的收敛行为对比。
    """
    rng = np.random.default_rng(SEED)
    g = rng.standard_normal((8, 5))
    norms = [("euclidean", "GD (ℓ₂)", "#1f77b4"),
             ("linf", "signSGD (ℓ∞)", "#2ca02c"),
             ("spectral", "Muon (spectral)", "#d62728")]
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    # 左：LMO 最优值 vs 随机方向（蒙特卡洛），验证 LMO 最优性
    labels, lmo_vals, rand_vals, dual_vals = [], [], [], []
    for key, lab, _ in norms:
        v = verify_lmo_optimality(g, key, n_random=3000, seed=1)
        labels.append(lab)
        lmo_vals.append(v["lmo_value"])
        rand_vals.append(v["best_random"])
        dual_vals.append(-v["dual_norm"])
        results[f"{key}_lmo_optimal"] = bool(v["lmo_is_optimal"])
        results[f"{key}_lmo_value"] = float(v["lmo_value"])
        results[f"{key}_neg_dual_norm"] = float(-v["dual_norm"])
    x = np.arange(len(labels))
    wd = 0.27
    axes[0].bar(x - wd, lmo_vals, wd, label=r"LMO value $\langle g, d^*\rangle$",
               color="#1f77b4")
    axes[0].bar(x, dual_vals, wd, label=r"$-\|g\|_{\mathrm{dual}}$ (theory)",
               color="#ff7f0e")
    axes[0].bar(x + wd, rand_vals, wd, label="best of 3000 random dirs",
               color="#cccccc")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel(r"$\langle g, d\rangle$ (lower = steeper)")
    axes[0].set_title("LMO optimality: each norm's steepest direction")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    # 右：三种最速下降在矩阵二次上的收敛（min ½‖AW−B‖²_F）
    from src.matrix_quadratic import (make_matrix_problem, gradient as mat_grad,
                                       objective as mat_obj)
    a, b, w_star, mu, L = make_matrix_problem(10, 6, 50.0, seed=SEED)
    f_star = mat_obj(a, b, w_star)
    w0 = np.zeros((10, 6))
    lip = L * L
    configs = [
        ("GD (ℓ₂)", "#1f77b4", lambda gg: gg, 1.5 / lip),
        ("signSGD (ℓ∞)", "#2ca02c", lambda gg: np.sign(gg), None),
        ("Muon (spectral)", "#d62728", lambda gg: -lmo_spectral(gg), None),
    ]
    for lab, col, direction_fn, lr in configs:
        w = w0.copy()
        gaps = [max(mat_obj(a, b, w) - f_star, 1e-30)]
        # signSGD/Muon 用衰减步长
        for t in range(1, 400):
            gg = mat_grad(a, b, w)
            d = direction_fn(gg)
            if lr is None:
                step = 0.5 / np.sqrt(t)
                w = w - step * d
            else:
                w = w - lr * d
            gaps.append(max(mat_obj(a, b, w) - f_star, 1e-30))
        axes[1].semilogy(gaps, color=col, linewidth=1.5, label=lab)
        results[f"matrix_{lab.split()[0]}_final"] = float(gaps[-1])
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$f(W_k) - f^*$")
    axes[1].set_title(r"Three norms on $\frac{1}{2}\|AW-B\|_F^2$, $\kappa(A)=50$")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("E32: steepest descent under ℓ₂ / ℓ∞ / spectral norm (LMO unification)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp32_norm_steepest_descent.png", dpi=150)
    plt.close(fig)
    return results


def run_all_extended() -> dict:
    return {
        "exp1b": exp1b_full_optimizer_panel(),
        "exp11": exp11_nesterov_vs_polyak(),
        "exp12": exp12_muon_spectral_recovery(),
        "exp13": exp13_adam_lr_sweep(),
        "exp14": exp14_sgd_noise(),
        "exp15": exp15_sophia_vs_adam(),
        "exp17": exp17_pcg_baseline(),
        "exp1_multiseed": exp1_multiseed_band(),
        "exp23": exp23_ns_rectangular(),
        "exp21": exp21_eigenmode_decay(),
        "exp24": exp24_adam_beta_ablation(),
        "exp_kappa_scan": exp_kappa_continuous_scan(),
        "exp22": exp22_beta_eta_heatmap(),
        # 新增前沿实验
        "exp25_hb_chebyshev": exp25_heavy_ball_chebyshev_equivalence(),
        "exp26_adam_cycle": exp26_adam_limit_cycle(),
        "exp27_cheb_ns": exp27_chebyshev_ns_vs_standard(),
        "exp28_muon_tr": exp28_muon_on_trust_region(),
        "exp29_ode": exp29_nag_ode_vs_discrete(),
        "exp30_ns_basin": exp30_ns_convergence_basin(),
        "exp31_equalization": exp31_muon_singular_value_equalization(),
        "exp32_norms": exp32_steepest_descent_norms(),
    }
