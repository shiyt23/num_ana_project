"""实验 E11–E29：扩展实验集合，以 Muon 为核心，配以 Chebyshev / ODE / 极限环。
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
from src.mlp_training import train_mlp
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
    """E12: Muon 在谱范数恢复目标上下降；与 GD/Adam 对照。

    问题：min_W (1/2) ‖W - W*‖_σ²  (谱范数损失)
    用于展示 Muon 正交化方向与谱范数 trust-region 线性化几何的匹配。
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
    """E25: Heavy-ball 是 Chebyshev 半迭代的定常极限（核心理论结果数值验证）。

    主张：在二次目标上，Chebyshev 半迭代的时变系数收敛后，
    递推退化为使用最优 (η*, β*) 的 Polyak Heavy-ball。
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

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].semilogy(gaps_cheb, label="Chebyshev semi-iterative",
                    color="#1f77b4", linewidth=1.7)
    axes[0].semilogy(gaps_hb, label="Heavy-ball (Polyak optimal)",
                    color="#ff7f0e", linewidth=1.7, linestyle="--")
    # 理论上界（Chebyshev minimax）
    bounds = [max(2 * chebyshev_minimax_bound(k, kappa) ** 2 * gaps_cheb[0], 1e-30)
              for k in range(len(gaps_cheb))]
    axes[0].semilogy(bounds, "k:", linewidth=1, label=r"Cheb. minimax bound")
    axes[0].set_xlabel("Iteration k")
    axes[0].set_ylabel(r"$f - f^*$")
    axes[0].set_title(rf"$\kappa={int(kappa)}$: Chebyshev $\to$ Polyak HB")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 误差多项式可视化
    lam_grid = np.linspace(mu, L, 200)
    for k in [5, 10, 20, 40]:
        e_poly = [chebyshev_error_polynomial(k, lam, mu, L) for lam in lam_grid]
        axes[1].plot(lam_grid, np.abs(e_poly), label=f"k={k}", linewidth=1.2)
    axes[1].axhline(1, color="gray", linestyle=":")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\lambda$ (eigenvalue of $A$)")
    axes[1].set_ylabel(r"$|e_k(\lambda)|$")
    axes[1].set_title(r"Chebyshev error polynomial $|e_k(\lambda)|$")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("E25: Heavy-ball as stationary limit of Chebyshev semi-iteration",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp25_hb_chebyshev.png", dpi=150)
    plt.close(fig)
    return {
        "max_gap_diff": float(max(
            abs(gc - gh) for gc, gh in zip(gaps_cheb[-50:], gaps_hb[-50:])
        )),
        "cheb_final": float(gaps_cheb[-1]),
        "hb_final": float(gaps_hb[-1]),
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
    """E27: 五次多项式 Higham NS vs 标准 NS（arXiv 2506.10935 简化版）。"""
    rng = np.random.default_rng(SEED)
    sizes = [(16, 8), (32, 16), (64, 32)]
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (m, n) in zip(axes, sizes):
        g = rng.standard_normal((m, n))
        _, errs_std = newton_schulz_iterate(g, max_iter=12)
        _, errs_cheb = chebyshev_ns_iterate(g, max_iter=12)
        ax.semilogy(errs_std, "o-", label="Standard NS",
                   color="#1f77b4", linewidth=1.4)
        ax.semilogy(errs_cheb, "s-", label="Higham 5th-degree polynomial NS",
                   color="#d62728", linewidth=1.4)
        ax.set_title(rf"${m}\times{n}$")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        results[f"{m}x{n}_std_final"] = errs_std[-1]
        results[f"{m}x{n}_cheb_final"] = errs_cheb[-1]
    axes[0].set_ylabel(r"Gram error $\|X^\top X - I\|_F$")
    fig.suptitle("E27: Higham fifth-degree Newton-Schulz vs standard NS",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp27_chebyshev_ns.png", dpi=150)
    plt.close(fig)
    return results


def exp28_muon_on_trust_region() -> dict:
    """E28: Muon vs GD vs Adam 在谱范数 trust-region 损失上的对比。

    问题：min_W (1/2) ‖W - W*‖_σ² (谱范数距离)；
    用于检验 Muon 正交化方向与谱范数 trust-region 线性化几何相匹配时的下降行为。
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
    """E30: Newton-Schulz 收敛盆 (0, √3) 的数值可视化。

    NS 迭代 X_{k+1} = 0.5 X_k (3I - X_k^T X_k) 在标量情形下化为
    σ_{k+1} = σ_k (3 - σ_k²) / 2，收敛域为 σ ∈ (0, √3)。
    用奇异值最大初值 σ_max 在这个区间扫描，记录是否收敛。
    """
    # σ 从 0.05 到 2.5，跨越收敛盆边界 √3 ≈ 1.732 两侧
    sigma_init = np.linspace(0.05, 2.5, 100)
    iters_to_conv = []
    finals = []
    for sigma in sigma_init:
        x = float(sigma)
        errs = [abs(x * x - 1)]
        diverged = False
        for _ in range(40):
            if abs(x) > 1e8 or not np.isfinite(x):
                diverged = True
                break
            x = 0.5 * x * (3.0 - x * x)
            errs.append(abs(x * x - 1))
        finals.append(float("inf") if diverged else errs[-1])
        below = [i for i, e in enumerate(errs) if e < 1e-6]
        iters_to_conv.append(below[0] if below else None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    # 左：σ 演化轨迹
    sig_demo = [0.2, 0.7, 1.0, 1.5, 1.7, 1.85]
    for s0 in sig_demo:
        traj = [s0]
        s = s0
        for _ in range(15):
            s = 0.5 * s * (3 - s * s)
            traj.append(s)
        axes[0].plot(traj, "o-", markersize=3, linewidth=1,
                    label=rf"$\sigma_0={s0}$")
    axes[0].axhline(1, color="red", linestyle="--", alpha=0.6, label=r"$\sigma^*=1$")
    axes[0].axhline(np.sqrt(3), color="gray", linestyle=":", alpha=0.7,
                   label=r"$\sqrt{3}$ (basin bdry)")
    axes[0].set_xlabel("NS iteration k")
    axes[0].set_ylabel(r"$\sigma_k$")
    axes[0].set_title(r"NS on scalar: $\sigma_{k+1} = \sigma_k(3 - \sigma_k^2)/2$")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.5, 3.0)

    # 右：σ_init vs 收敛步数（含发散段）
    finite = [(s, f) for s, f in zip(sigma_init, finals) if np.isfinite(f)]
    finite_x, finite_y = zip(*finite) if finite else ([], [])
    div = [(s, f) for s, f in zip(sigma_init, finals) if not np.isfinite(f)]
    axes[1].semilogy(finite_x, np.maximum(finite_y, 1e-18), "o-",
                    color="#1f77b4", markersize=2, linewidth=1,
                    label="Final |σ²-1| after 30 NS")
    axes[1].axvline(np.sqrt(3), color="red", linestyle="--",
                   label=r"$\sqrt{3}$ (theoretical basin)")
    axes[1].axvline(1.0, color="green", linestyle=":", label=r"$\sigma=1$ (fixed pt)")
    if div:
        axes[1].axvspan(min(s for s, _ in div), 2.2, alpha=0.15, color="red",
                       label="Divergence region")
    axes[1].set_xlabel(r"Initial $\sigma_0$")
    axes[1].set_ylabel("Final orthogonality error")
    axes[1].set_title("Convergence basin of Newton-Schulz")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(r"E30: NS convergence basin $\sigma \in (0, \sqrt{3})$", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp30_ns_basin.png", dpi=150)
    plt.close(fig)
    return {
        "basin_upper_bound_theory": float(np.sqrt(3)),
        "n_div_above_sqrt3": int(sum(1 for s, f in zip(sigma_init, finals)
                                     if s > np.sqrt(3) and not np.isfinite(f))),
    }


def exp31_real_mlp_digits() -> dict:
    """E31: 真实数据（scikit-learn digits）上的单隐层 MLP 训练。

    网络 x(64) -> W1(64x32) -> ReLU -> W2(32x10) -> softmax，两个矩阵权重
    W1,W2 是真实参数层。比较 SGD / Adam / Muon：Muon 对矩阵梯度做 Newton-Schulz
    正交化后更新，验证矩阵正交化预条件在真实分类任务上的可用性。
    """
    optimizers = [
        ("sgd", "SGD (momentum)", "#1f77b4"),
        ("adam", "Adam", "#2ca02c"),
        ("muon", "Muon (NS orthogonalized)", "#d62728"),
    ]
    epochs = 60
    seeds = SEEDS
    results: dict = {}

    # 收集每个优化器跨种子的曲线
    loss_curves: dict[str, np.ndarray] = {}
    acc_curves: dict[str, np.ndarray] = {}
    meta = {}
    for name, _, _ in optimizers:
        losses, accs = [], []
        for seed in seeds:
            run = train_mlp(name, hidden=32, epochs=epochs, seed=seed)
            losses.append(run["train_losses"])
            accs.append(run["test_accs"])
            meta = run  # 记录维度等元信息
        loss_curves[name] = np.array(losses)
        acc_curves[name] = np.array(accs)
        results[f"{name}_final_test_acc_mean"] = float(
            np.mean(acc_curves[name][:, -1])
        )
        results[f"{name}_final_test_acc_std"] = float(
            np.std(acc_curves[name][:, -1])
        )
        results[f"{name}_best_test_acc_mean"] = float(
            np.mean(np.max(acc_curves[name], axis=1))
        )
        results[f"{name}_final_train_loss_mean"] = float(
            np.mean(loss_curves[name][:, -1])
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    epoch_axis = np.arange(loss_curves[optimizers[0][0]].shape[1])
    for name, label, color in optimizers:
        med = np.median(loss_curves[name], axis=0)
        lo = np.percentile(loss_curves[name], 25, axis=0)
        hi = np.percentile(loss_curves[name], 75, axis=0)
        axes[0].semilogy(epoch_axis, np.maximum(med, 1e-6), color=color,
                         linewidth=1.8, label=label)
        axes[0].fill_between(epoch_axis, np.maximum(lo, 1e-6),
                             np.maximum(hi, 1e-6), color=color, alpha=0.2)
        med_acc = np.median(acc_curves[name], axis=0)
        lo_a = np.percentile(acc_curves[name], 25, axis=0)
        hi_a = np.percentile(acc_curves[name], 75, axis=0)
        axes[1].plot(epoch_axis, med_acc, color=color, linewidth=1.8, label=label)
        axes[1].fill_between(epoch_axis, lo_a, hi_a, color=color, alpha=0.2)

    axes[0].set_xlabel("Epoch (full-batch)")
    axes[0].set_ylabel("Training cross-entropy")
    axes[0].set_title("Training loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch (full-batch)")
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Test accuracy")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(
        r"E31: single fully-connected layer on real digits "
        rf"({meta['d_in']}$\to${meta['hidden']}$\to${meta['n_class']}, "
        rf"{meta['n_train']} train / {meta['n_test']} test, {len(seeds)} seeds)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp31_mlp_digits.png", dpi=150)
    plt.close(fig)

    results["epochs"] = epochs
    results["n_seeds"] = len(seeds)
    results["n_train"] = meta["n_train"]
    results["n_test"] = meta["n_test"]
    results["architecture"] = f"{meta['d_in']}-{meta['hidden']}-{meta['n_class']}"
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
        "exp31_mlp_digits": exp31_real_mlp_digits(),
    }
