"""实验 E11–E17 及 TODO 中的扩展实验。"""

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
from src.matrix_quadratic import (
    make_matrix_problem,
    objective as mat_objective,
    run_matrix_adam_flat,
    run_matrix_gd,
    run_matrix_muon,
)
from src.momentum_spectrum import optimal_polyak_params, worst_case_spectral_radius
from src.newton_schulz import newton_schulz_iterate
from src.optimizers import effective_preconditioned_condition_number, run_optimizer
from src.pcg import run_pcg
from src.quadratic import make_quadratic_problem, objective, optimal_point


def _first_below(values: list[float], tol: float) -> int | None:
    for i, v in enumerate(values):
        if v < tol:
            return i
    return None


def _fit_log_linear_rate(gaps: list[float], start: int = 30, end: int = 200) -> float | None:
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
    """E11: Nesterov vs Polyak Momentum。"""
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, kappa in zip(axes, CONDITION_NUMBERS):
        a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
        x_star = optimal_point(a, b)
        f_star = objective(a, b, x_star)
        for name, label, color in [
            ("momentum", "Polyak HB", "#ff7f0e"),
            ("nesterov", "Nesterov AG", "#9467bd"),
        ]:
            kwargs = {"beta": -1, "mu": mu, "L": L} if name == "momentum" else {"mu": mu, "L": L}
            run = run_optimizer(name, a, b, np.zeros(DIM), MAX_ITER, **kwargs)
            gaps = [max(f - f_star, 1e-30) for f in run.fs]
            ax.semilogy(gaps, label=label, color=color, linewidth=1.5)
            rho_emp = _fit_log_linear_rate(gaps)
            results[f"kappa_{kappa}_{name}_iter"] = _first_below(gaps, TOL)
            results[f"kappa_{kappa}_{name}_rho_emp"] = rho_emp
        ax.set_title(f"kappa={kappa}")
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$f(x_k)-f^*$")
    fig.suptitle("E11: Nesterov vs Polyak momentum", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp11_nesterov_vs_polyak.png", dpi=150)
    plt.close(fig)

    kappa = 100.0
    step_p, beta_p = optimal_polyak_params(1.0, kappa)
    results["kappa_100_rho_polyak"] = worst_case_spectral_radius(1.0, kappa, step_p, beta_p)
    results["kappa_100_note"] = "Nesterov uses eta=1/L, beta=(sqrt(k)-1)/(sqrt(k)+1)"
    return results


def exp12_matrix_muon() -> dict:
    """E12: 矩阵二次 + Muon-NS vs GD vs Adam(flat)。"""
    m, n, kappa = 16, 16, 100.0
    a, b, w_star, mu, L = make_matrix_problem(m, n, kappa, seed=SEED)
    f_star = mat_objective(a, b, w_star)
    w0 = np.zeros((m, n))
    max_iter = 1500
    lip = L * L
    lr_gd = 1.5 / lip
    lr_muon = 0.005 / L  # 正交化方向未必使 Frobenius 目标下降，需小步长
    lr_adam = 0.8 / lip

    runners = [
        ("GD (Frobenius)", run_matrix_gd, lr_gd),
        ("Muon-NS", run_matrix_muon, lr_muon),
        ("Adam (flat)", run_matrix_adam_flat, lr_adam),
    ]
    results = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, fn, lr in runners:
        fs, _ = fn(a, b, w0, max_iter, lr)
        gaps = [max(f - f_star, 1e-30) for f in fs]
        ax.semilogy(gaps, linewidth=1.5, label=label)
        results[label] = {"iter_tol": _first_below(gaps, TOL), "final": gaps[-1]}
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f(W_k)-f^*$")
    ax.set_title(r"E12: matrix quadratic, $\kappa=100$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp12_matrix_muon.png", dpi=150)
    plt.close(fig)
    return results


def exp13_adam_lr_sweep() -> dict:
    """E13: Adam 学习率扫描热力图（迭代数）。"""
    kappa = 100.0
    lrs = np.logspace(-3, 0, 16)
    results = {}
    iter_grid = np.zeros(len(lrs))
    for i, lr in enumerate(lrs):
        a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
        x_star = optimal_point(a, b)
        f_star = objective(a, b, x_star)
        run = run_optimizer("adam", a, b, np.zeros(DIM), MAX_ITER, lr=lr, mu=mu, L=L)
        gaps = [max(f - f_star, 1e-30) for f in run.fs]
        it = _first_below(gaps, TOL)
        iter_grid[i] = it if it is not None else MAX_ITER + 1
        results[f"lr_{lr:.4f}"] = int(iter_grid[i])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(iter_grid.reshape(1, -1), aspect="auto", cmap="viridis_r")
    ax.set_yticks([0])
    ax.set_yticklabels([r"$\kappa=100$"])
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], rotation=45, ha="right")
    ax.set_xlabel("Adam learning rate")
    ax.set_title(f"E13: iterations to reach {TOL:.0e}")
    plt.colorbar(im, ax=ax, label="Iterations")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp13_adam_lr_sweep.png", dpi=150)
    plt.close(fig)
    best_idx = int(np.argmin(iter_grid))
    results["best_lr"] = float(lrs[best_idx])
    results["best_iter"] = int(iter_grid[best_idx])
    return results


def exp14_sgd_noise() -> dict:
    """E14: 带噪梯度 SGD vs Adam。"""
    kappa = 100.0
    a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)
    sigma = SGD_NOISE_SIGMA
    max_iter = 400
    results = {}

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.zeros(DIM)
    fs_gd = []
    rng = np.random.default_rng(SEED)
    for t in range(max_iter + 1):
        fs_gd.append(max(objective(a, b, x) - f_star, 1e-30))
        if t < max_iter:
            g = a @ x - b + rng.normal(0, sigma, DIM)
            x = x - (2.0 / (mu + L)) * g
    m, v = np.zeros(DIM), np.zeros(DIM)
    x = np.zeros(DIM)
    fs_adam = []
    rng2 = np.random.default_rng(SEED)
    for t in range(1, max_iter + 1):
        fs_adam.append(max(objective(a, b, x) - f_star, 1e-30))
        g = a @ x - b + rng2.normal(0, sigma, DIM)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * (g * g)
        mh = m / (1 - 0.9**t)
        vh = v / (1 - 0.999**t)
        x = x - ADAM_LR * mh / (np.sqrt(vh) + 1e-8)
    ax.semilogy(fs_gd, label=f"Noisy GD (sigma={sigma})", color="#1f77b4")
    ax.semilogy(fs_adam, label=f"Noisy Adam (sigma={sigma})", color="#2ca02c")
    run_clean = run_optimizer("adam", a, b, np.zeros(DIM), max_iter, lr=ADAM_LR, mu=mu, L=L)
    gaps_clean = [max(f - f_star, 1e-30) for f in run_clean.fs]
    ax.semilogy(gaps_clean, "--", color="gray", label="Adam (no noise)")
    results["floor_gd"] = fs_gd[-1]
    results["floor_adam"] = fs_adam[-1]
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f(x_k)-f^*$")
    ax.set_title("E14: SGD noise floor")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp14_sgd_noise.png", dpi=150)
    plt.close(fig)
    return results


def exp15_sophia_vs_adam() -> dict:
    """E15: Sophia vs Adam — kappa_eff 与收敛。"""
    kappa = 100.0
    a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for name, color in [("adam", "#2ca02c"), ("sophia", "#d62728")]:
        run = run_optimizer(name, a, b, np.zeros(DIM), MAX_ITER, lr=ADAM_LR if name == "adam" else 0.05, mu=mu, L=L)
        gaps = [max(f - f_star, 1e-30) for f in run.fs]
        axes[0].semilogy(gaps, label=name.upper(), color=color)
        kappa_eff = [effective_preconditioned_condition_number(a, p) for p in run.preconditioners[1:]]
        axes[1].plot(kappa_eff, color=color, label=name.upper())
        results[f"{name}_iter"] = _first_below(gaps, TOL)

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$f-f^*$")
    axes[0].set_title("Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$\kappa(P_k^{-1}A)$")
    axes[1].set_yscale("log")
    axes[1].set_title(r"Effective $\kappa$, $\kappa(A)=100$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("E15: Sophia (simplified) vs Adam", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp15_sophia.png", dpi=150)
    plt.close(fig)
    return results


def exp17_pcg_baseline() -> dict:
    """E17: PCG+Jacobi 与一阶方法对比。"""
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, kappa in zip(axes, CONDITION_NUMBERS):
        a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
        x_star = optimal_point(a, b)
        f_star = objective(a, b, x_star)
        x0 = np.zeros(DIM)

        fs_pcg, pcg_iters = run_pcg(a, b, x0, MAX_ITER, tol=1e-14)
        gaps_pcg = [max(f - f_star, 1e-30) for f in fs_pcg]
        ax.semilogy(gaps_pcg, label=f"PCG+Jacobi ({pcg_iters} iters)", color="#8c564b", linewidth=2)

        for name, label, color in [
            ("momentum", "Momentum", "#ff7f0e"),
            ("adam", "Adam", "#2ca02c"),
        ]:
            kwargs = {"beta": -1, "mu": mu, "L": L} if name == "momentum" else {"mu": mu, "L": L}
            lr = ADAM_LR if name == "adam" else None
            run = run_optimizer(name, a, b, x0, MAX_ITER, lr=lr, **kwargs)
            gaps = [max(f - f_star, 1e-30) for f in run.fs]
            ax.semilogy(gaps, label=label, color=color, linewidth=1.2, alpha=0.85)
            results[f"kappa_{kappa}_{name}"] = _first_below(gaps, TOL)
        results[f"kappa_{kappa}_pcg"] = pcg_iters
        ax.set_title(f"kappa={kappa}")
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$f-f^*$")
    fig.suptitle("E17: PCG baseline vs first-order methods", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp17_pcg_baseline.png", dpi=150)
    plt.close(fig)
    return results


def exp1_multiseed_band() -> dict:
    """多种子：实验1 收敛带（GD / Momentum / Adam）。"""
    kappa = 100.0
    methods = ["gd", "momentum", "adam"]
    results = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    x_axis = np.arange(MAX_ITER + 1)

    for name in methods:
        all_gaps = []
        for seed in SEEDS:
            a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=seed)
            x_star = optimal_point(a, b)
            f_star = objective(a, b, x_star)
            kwargs = {"beta": -1, "mu": mu, "L": L} if name == "momentum" else {"mu": mu, "L": L}
            lr = ADAM_LR if name == "adam" else None
            run = run_optimizer(name, a, b, np.zeros(DIM), MAX_ITER, lr=lr, **kwargs)
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
    ax.set_ylabel(r"$f-f^*$")
    ax.set_title(f"Multi-seed bands (seeds={SEEDS}), kappa=100")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp1_multiseed.png", dpi=150)
    plt.close(fig)
    return results


def exp23_ns_rectangular() -> dict:
    """E23: 瘦/胖矩阵 Newton-Schulz。"""
    rng = np.random.default_rng(SEED)
    shapes = [(32, 8), (16, 16), (8, 32)]
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (m, n) in zip(axes, shapes):
        g = rng.standard_normal((m, n))
        _, errors = newton_schulz_iterate(g, max_iter=18, scale_factor=1.0)
        ax.semilogy(errors, "o-")
        ax.set_title(f"{m}x{n}")
        ax.set_xlabel("NS step")
        ax.grid(True, alpha=0.3)
        results[f"{m}x{n}_final"] = errors[-1]
    axes[0].set_ylabel(r"$\|X^TX-I\|_F$")
    fig.suptitle("E23: rectangular matrices", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp23_ns_rectangular.png", dpi=150)
    plt.close(fig)
    return results


def exp1b_full_optimizer_panel() -> dict:
    """实验1扩展：GD / Momentum / Nesterov / Adam / Jacobi 同图（kappa=100）。"""
    kappa = 100.0
    a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)
    specs = [
        ("gd", "GD", "#1f77b4", None),
        ("momentum", "Polyak HB", "#ff7f0e", {"beta": -1}),
        ("nesterov", "Nesterov", "#9467bd", {}),
        ("adam", "Adam", "#2ca02c", {}),
        ("adamw", "AdamW", "#17becf", {}),
        ("jacobi", "Jacobi GD", "#d62728", {}),
    ]
    results = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, label, color, extra in specs:
        kw = {"mu": mu, "L": L, **(extra or {})}
        lr = ADAM_LR if name in ("adam", "adamw") else (1.0 if name == "jacobi" else None)
        run = run_optimizer(name, a, b, np.zeros(DIM), MAX_ITER, lr=lr, **kw)
        gaps = [max(f - f_star, 1e-30) for f in run.fs]
        ax.semilogy(gaps, label=label, color=color, linewidth=1.5)
        results[name] = _first_below(gaps, TOL)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f-f^*$")
    ax.set_title(r"E1b: full panel, $\kappa=100$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp1b_full_panel.png", dpi=150)
    plt.close(fig)
    return results


def exp21_eigenmode_decay() -> dict:
    """E21: 误差在 A 特征基下的模态能量衰减（Polyak vs GD）。"""
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
        run = run_optimizer(name, a, b, np.zeros(DIM), min(MAX_ITER, 200), **kwargs)
        steps = min(80, len(run.xs))
        modes_lo, modes_hi = [], []
        for x in run.xs[:steps]:
            coeff = q.T @ (x - x_star)
            energy = coeff * coeff
            modes_lo.append(energy[0])
            modes_hi.append(energy[-1])
        t = np.arange(len(modes_lo))
        ax.semilogy(t, modes_lo, label=rf"$\lambda_{{\min}}=\mu$")
        ax.semilogy(t, modes_hi, label=rf"$\lambda_{{\max}}=L$")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        results[f"{name}_ratio_final"] = float(modes_hi[-1] / max(modes_lo[-1], 1e-30))
    axes[0].set_ylabel("Modal energy $|c_i|^2$")
    fig.suptitle(r"E21: eigenmode energy, $\kappa=100$", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp21_eigenmode_decay.png", dpi=150)
    plt.close(fig)
    return results


def exp24_adam_beta_ablation() -> dict:
    """E24: Adam beta1/beta2 消融。"""
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

    for beta1, beta2, label in configs:
        run = run_optimizer(
            "adam", a, b, np.zeros(DIM), MAX_ITER,
            lr=ADAM_LR, mu=mu, L=L, beta1=beta1, beta2=beta2,
        )
        gaps = [max(f - f_star, 1e-30) for f in run.fs]
        axes[0].semilogy(gaps, linewidth=1.2, label=label)
        kappa_eff = [effective_preconditioned_condition_number(a, p) for p in run.preconditioners[1:51]]
        axes[1].plot(kappa_eff[:50], linewidth=1.2, label=label)
        results[label] = _first_below(gaps, TOL)

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$f-f^*$")
    axes[0].set_title("Convergence")
    axes[0].legend(fontsize=6)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Iteration (first 50)")
    axes[1].set_ylabel(r"$\kappa(P_k^{-1}A)$")
    axes[1].set_yscale("log")
    axes[1].set_title(r"Effective $\kappa$")
    axes[1].legend(fontsize=6)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(r"E24: Adam $(\beta_1,\beta_2)$ ablation, $\kappa=100$", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp24_adam_ablation.png", dpi=150)
    plt.close(fig)
    return results


def exp_kappa_continuous_scan() -> dict:
    """kappa 连续扫描：达到 tol 的迭代数 vs 理论 rho。"""
    kappas = np.unique(np.round(np.logspace(1, 4, 25)).astype(int))
    iter_gd, iter_mom, rho_gd_th, rho_mom_th = [], [], [], []

    for kappa in kappas:
        a, b, mu, L = make_quadratic_problem(DIM, float(kappa), seed=SEED)
        x_star = optimal_point(a, b)
        f_star = objective(a, b, x_star)
        rho_gd = (kappa - 1) / (kappa + 1)
        rho_mom = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
        rho_gd_th.append(rho_gd)
        rho_mom_th.append(rho_mom)

        for name, lst in [("gd", iter_gd), ("momentum", iter_mom)]:
            kw = {"mu": mu, "L": L}
            if name == "momentum":
                kw["beta"] = -1
            run = run_optimizer(name, a, b, np.zeros(DIM), MAX_ITER, **kw)
            gaps = [max(f - f_star, 1e-30) for f in run.fs]
            it = _first_below(gaps, TOL)
            lst.append(it if it is not None else MAX_ITER + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(kappas, iter_gd, "o-", label="GD iterations", color="#1f77b4")
    ax.loglog(kappas, iter_mom, "s-", label="Momentum iterations", color="#ff7f0e")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(f"Iterations to {TOL:.0e}")
    ax2 = ax.twinx()
    ax2.loglog(kappas, rho_gd_th, "--", color="#1f77b4", alpha=0.5, label=r"$\rho^*_{GD}$")
    ax2.loglog(kappas, rho_mom_th, "--", color="#ff7f0e", alpha=0.5, label=r"$\rho^*_{Mom}$")
    ax2.set_ylabel("Theory spectral radius")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="center left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("Iterations vs kappa (theory on right axis)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp_kappa_scan.png", dpi=150)
    plt.close(fig)
    return {"kappas": kappas.tolist(), "iter_gd": iter_gd, "iter_mom": iter_mom}


def exp22_beta_eta_heatmap() -> dict:
    """E22: (beta, eta) 平面上 Heavy-ball 最坏谱半径热力图。"""
    kappa = 100.0
    mu, L = 1.0, kappa
    step_opt, beta_opt = optimal_polyak_params(mu, L)
    betas = np.linspace(0.0, 0.995, 50)
    etas = np.linspace(0.002, 0.08, 50)
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
    ax.plot(beta_opt, step_opt, "r*", markersize=14, label="Polyak optimal")
    ax.plot(0.9, step_opt, "o", color="orange", markersize=8, label=r"ML default $\beta=0.9$")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"Worst-case $\rho(\eta,\beta)$")
    ax.set_xlabel(r"Momentum $\beta$")
    ax.set_ylabel(r"Step size $\eta$")
    ax.set_title(r"E22: spectral radius heatmap, $\kappa=100$")
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


def run_all_extended() -> dict:
    return {
        "exp1b": exp1b_full_optimizer_panel(),
        "exp11": exp11_nesterov_vs_polyak(),
        "exp12": exp12_matrix_muon(),
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
    }
