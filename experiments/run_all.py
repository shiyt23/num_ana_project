#!/usr/bin/env python3
"""运行全部数值实验并保存图表与数据。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.momentum_spectrum import (  # noqa: E402
    optimal_polyak_params,
    spectral_radius,
    worst_case_spectral_radius,
)
from src.newton_schulz import (  # noqa: E402
    newton_schulz_iterate,
    orthogonality_error,
    polar_factor_via_svd,
)
from src.optimizers import (  # noqa: E402
    effective_preconditioned_condition_number,
    run_optimizer,
)
from src.quadratic import make_quadratic_problem, objective, optimal_point  # noqa: E402

from experiments.config import (  # noqa: E402
    ADAM_LR,
    ADAM_LR_2D,
    ADAM_LR_PRECOND_STUDY,
    CONDITION_NUMBERS,
    DATA_DIR,
    DIM,
    FIG_DIR,
    MAX_ITER,
    SEED,
    TOL,
)
from experiments.extended_experiments import run_all_extended  # noqa: E402


def setup_dirs() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)


def exp1_convergence() -> dict:
    """实验1：病态二次函数上 GD / Momentum / Adam 收敛对比。"""
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, kappa in zip(axes, CONDITION_NUMBERS):
        a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
        x_star = optimal_point(a, b)
        f_star = objective(a, b, x_star)
        x0 = np.zeros(DIM)

        for name, label, color in [
            ("gd", "GD (Richardson)", "#1f77b4"),
            ("momentum", "Momentum (Polyak β)", "#ff7f0e"),
            ("adam", "Adam (dynamic Jacobi)", "#2ca02c"),
        ]:
            kwargs = {}
            if name == "momentum":
                kwargs["beta"] = -1  # 使用 Polyak 最优
            lr = ADAM_LR if name == "adam" else None
            run = run_optimizer(name, a, b, x0, MAX_ITER, lr=lr, mu=mu, L=L, **kwargs)
            gaps = [max(f - f_star, 1e-30) for f in run.fs]
            ax.semilogy(gaps, label=label, color=color, linewidth=1.5)
            results[f"kappa_{kappa}_{name}"] = {
                "final_gap": gaps[-1],
                "iter_to_1e6": _first_below(gaps, 1e-6),
            }

        ax.set_title(f"κ = {kappa}")
        ax.set_xlabel("Iteration k")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel(r"$f(x_k) - f^*$")
    fig.suptitle("Ill-conditioned quadratic: optimizer convergence", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp1_convergence.png", dpi=150)
    plt.close(fig)
    return results


def exp2_momentum_spectrum() -> dict:
    """实验2：Momentum 谱半径理论与实测对比。"""
    kappas = np.logspace(1, 3, 30)
    theory_rates, gd_rates, bad_rates = [], [], []

    for kappa in kappas:
        mu, L = 1.0, kappa
        step_opt, beta_opt = optimal_polyak_params(mu, L)
        rho_opt = worst_case_spectral_radius(mu, L, step_opt, beta_opt)
        theory_rates.append(rho_opt)

        step_gd = 2.0 / (mu + L)
        rho_gd = max(
            abs(1 - step_gd * mu),
            abs(1 - step_gd * L),
        )
        gd_rates.append(rho_gd)

        step_bad, beta_bad = 0.01, 0.9
        rho_bad = worst_case_spectral_radius(mu, L, step_bad, beta_bad)
        bad_rates.append(rho_bad)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(kappas, theory_rates, label=r"Momentum (Polyak optimal) $\rho^*$")
    ax.loglog(
        kappas,
        [(np.sqrt(k) - 1) / (np.sqrt(k) + 1) for k in kappas],
        "--",
        label=r"Theory $(\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$",
    )
    ax.loglog(kappas, gd_rates, label=r"GD optimal step $\rho_{\mathrm{GD}}$")
    ax.set_xlabel(r"Condition number $\kappa$")
    ax.set_ylabel("Worst-case spectral radius")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("Momentum vs GD: spectral radius vs kappa")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp2_spectral_radius.png", dpi=150)
    plt.close(fig)

    # 单点验证：κ=100 沿特征方向衰减
    kappa = 100.0
    mu, L = 1.0, kappa
    step, beta = optimal_polyak_params(mu, L)
    steps = np.arange(0, 80)
    rho_mu = spectral_radius(mu, step, beta)
    rho_L = spectral_radius(L, step, beta)
    theory_decay_mu = rho_mu**steps
    theory_decay_L = rho_L**steps

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.semilogy(steps, theory_decay_mu, label=rf"Mode $\lambda=\mu$, $\rho={rho_mu:.4f}$")
    ax2.semilogy(steps, theory_decay_L, label=rf"Mode $\lambda=L$, $\rho={rho_L:.4f}$")
    ax2.set_xlabel("Iteration k")
    ax2.set_ylabel(r"Theoretical decay $\rho^k$")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title(r"$\kappa=100$: Momentum modal decay")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "exp2_modal_decay.png", dpi=150)
    plt.close(fig2)

    return {
        "kappa_100_rho_polyak": float(worst_case_spectral_radius(1, 100, *optimal_polyak_params(1, 100))),
        "kappa_100_rho_gd": float(max(abs(1 - 2 / 101), abs(1 - 200 / 101))),
    }


def exp3_newton_schulz() -> dict:
    """实验3：Newton-Schulz 正交化精度。"""
    rng = np.random.default_rng(SEED)
    sizes = [(8, 8), (16, 16), (32, 24)]
    results = {}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (m, n) in zip(axes, sizes):
        g = rng.standard_normal((m, n))
        _, errors = newton_schulz_iterate(g, max_iter=15)
        ax.semilogy(range(len(errors)), errors, "o-", linewidth=1.5, markersize=4)
        ax.set_title(f"Matrix size {m}x{n}")
        ax.set_xlabel("Newton-Schulz iteration")
        ax.grid(True, alpha=0.3)
        results[f"{m}x{n}_final_error"] = errors[-1]
        results[f"{m}x{n}_iter_to_1e6"] = _first_below(errors, 1e-6)

    axes[0].set_ylabel(r"$\|X^T X - I\|_F$")
    fig.suptitle("Newton-Schulz: orthogonality error (quadratic convergence)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp3_newton_schulz.png", dpi=150)
    plt.close(fig)

    # 与 SVD 极分解对比
    g = rng.standard_normal((16, 16))
    xs, _ = newton_schulz_iterate(g, max_iter=12)
    u_ref = polar_factor_via_svd(g)
    dists = [np.linalg.norm(x - u_ref, "fro") for x in xs]

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.semilogy(dists, "s-", color="#d62728")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\|X_k - UV^T\|_F$")
    ax2.set_title("Distance to SVD polar factor")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "exp3_polar_distance.png", dpi=150)
    plt.close(fig2)

    return results


def _diagonal_quadratic(dim: int, kappa: float, seed: int = 0):
    """对角 Hessian A=diag(geomspace(1,κ))，此时 Jacobi/Adam 的对角预条件可对齐特征方向。"""
    rng = np.random.default_rng(seed)
    eigvals = np.geomspace(1.0, kappa, dim)
    a = np.diag(eigvals)
    x_star = rng.standard_normal(dim)
    b = a @ x_star
    return a, b, 1.0, kappa


def exp4_preconditioner_evolution() -> dict:
    """实验4：Adam 的 κ_eff vs oracle Jacobi —— 诚实版。

    关键诚实点（比"Adam=Jacobi"更精确）：
      Jacobi 预条件用**曲率** diag(A)，在对角 A 上给出 κ(P⁻¹A)=1（理想）；
      Adam 预条件用**梯度幅度** √EMA(g²)。在二次问题上 |g_i|=λ_i·|x_i−x*_i|，
      把曲率 λ_i 与到极小点的距离 |x_i−x*_i| 混在一起，因此即便在对角 A 上
      Adam 的 κ_eff 也**不趋于 1**（反而可达 ~κ² 量级）。
    这说明"Adam = 动态 Jacobi"只是启发式类比：二者都做对角缩放，但
    Adam 缩放的是梯度幅度而非曲率，仅在特定统计假设下才与 Jacobi 重合。
      每个子图：Adam κ_eff(k) 曲线 + oracle Jacobi κ_eff（水平虚线）+ κ(A)。
    """
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    for col, kappa in enumerate(CONDITION_NUMBERS):
        for row, (prob_label, make_prob) in enumerate([
            ("diagonal A", _diagonal_quadratic),
            ("dense rotated A", make_quadratic_problem),
        ]):
            a, b, mu, L = make_prob(DIM, kappa, seed=SEED)
            x0 = np.zeros(DIM)
            run = run_optimizer(
                "adam", a, b, x0, MAX_ITER, lr=0.5, mu=mu, L=L,
                beta1=0.0, beta2=0.99,
            )
            kappa_true = float(np.linalg.cond(a))
            kappa_eff = [
                effective_preconditioned_condition_number(a, p)
                for p in run.preconditioners[1:401]
            ]
            # oracle Jacobi: P = diag(A)^{-1}（曲率预条件）
            p_jac = np.diag(1.0 / np.clip(np.diag(a), 1e-12, None))
            kappa_jac = effective_preconditioned_condition_number(a, p_jac)

            ax = axes[row, col]
            ax.plot(kappa_eff, color="#2ca02c", linewidth=1.4,
                    label=r"Adam $\kappa(P_k^{-1}A)$")
            ax.axhline(kappa_jac, color="#1f77b4", linestyle="-.",
                       label=rf"oracle Jacobi $={kappa_jac:.1f}$")
            ax.axhline(kappa_true, color="gray", linestyle="--",
                       label=rf"$\kappa(A)={kappa_true:.0f}$")
            ax.axhline(1.0, color="black", linestyle=":", alpha=0.5)
            ax.set_title(rf"$\kappa={kappa}$, {prob_label}", fontsize=10)
            if row == 1:
                ax.set_xlabel("Iteration k (first 400)")
            ax.set_yscale("log")
            ax.legend(fontsize=6.5)
            ax.grid(True, alpha=0.3)
            tag = "diag" if row == 0 else "dense"
            results[f"kappa{kappa}_{tag}_adam_kappa_eff_final"] = float(kappa_eff[-1])
            results[f"kappa{kappa}_{tag}_adam_kappa_eff_min"] = float(min(kappa_eff))
            results[f"kappa{kappa}_{tag}_jacobi_kappa_eff"] = float(kappa_jac)

    axes[0, 0].set_ylabel(r"Diagonal $A$: $\kappa(P_k^{-1} A)$")
    axes[1, 0].set_ylabel(r"Dense $A$: $\kappa(P_k^{-1} A)$")
    fig.suptitle(
        "Adam preconditions by gradient magnitude, NOT curvature: "
        "even on diagonal $A$, Adam's $\\kappa_{\\mathrm{eff}} \\neq 1$ "
        "while oracle Jacobi $=1$",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp4_precond_kappa.png", dpi=150)
    plt.close(fig)

    a, b, mu_100, L_100 = make_quadratic_problem(DIM, 100, seed=SEED)
    step = 2.0 / (mu_100 + L_100)
    p_gd = np.eye(DIM) * step
    results["kappa_100_gd_eff"] = effective_preconditioned_condition_number(a, p_gd)
    return results


def exp5_step_size_sensitivity() -> dict:
    """实验5：GD 步长与 Richardson 迭代谱半径。"""
    kappa = 100
    mu, L = 1.0, kappa
    steps = np.linspace(0.001, 0.05, 200)
    rhos = [max(abs(1 - h * mu), abs(1 - h * L)) for h in steps]
    h_opt = 2.0 / (mu + L)
    rho_opt = max(abs(1 - h_opt * mu), abs(1 - h_opt * L))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, rhos, label=r"$\rho(h)=\max|1-h\lambda_i|$")
    ax.axvline(h_opt, color="red", linestyle="--", label=rf"Optimal $h^*=2/(\mu+L)$")
    ax.scatter([h_opt], [rho_opt], color="red", zorder=5)
    ax.set_xlabel("Step size h")
    ax.set_ylabel("Spectral radius")
    ax.set_title(r"Richardson iteration: step-size sensitivity ($\kappa=100$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp5_step_sensitivity.png", dpi=150)
    plt.close(fig)
    return {"h_opt": h_opt, "rho_opt": float(rho_opt)}


def _first_below(values: list[float], tol: float) -> int | None:
    for i, v in enumerate(values):
        if v < tol:
            return i
    return None


def _fit_log_linear_rate(
    gaps: list[float],
    start: int | None = None,
    end: int | None = None,
    floor: float = 1e-18,
) -> float | None:
    """拟合 log(gap_k) ≈ k log ρ + c，返回经验收敛率 ρ_emp。

    自适应窗口：自动避开早期非渐近段与末段精度地板。
    """
    n = len(gaps)
    if n < 30:
        return None
    if start is None:
        # 跳过早期 10% 步或 10 步
        start = max(10, int(0.05 * n))
    if end is None:
        # 末段截到 gap < 1e3 * floor 之前
        end = n
        for i in range(n - 1, start, -1):
            if gaps[i] > 1e3 * floor:
                end = i + 1
                break
    if end - start < 15:
        return None
    ks, ys = [], []
    for k in range(start, end):
        if gaps[k] > floor and gaps[k - 1] > 0 and gaps[k] < gaps[k - 1]:
            ks.append(k)
            ys.append(np.log(gaps[k]))
    if len(ks) < 12:
        return None
    slope, _ = np.polyfit(ks, ys, 1)
    return float(np.exp(slope))


def exp6_empirical_vs_theory_rate() -> dict:
    """实验6：经验收敛率 vs 理论谱半径（线性区域斜率拟合）。"""
    kappas = [10, 100, 1000]
    names = ["gd", "momentum"]
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name in zip(axes, names):
        rho_emp_list, rho_th_list = [], []
        for kappa in kappas:
            a, b, mu, L = make_quadratic_problem(DIM, kappa, seed=SEED)
            x_star = optimal_point(a, b)
            f_star = objective(a, b, x_star)
            kwargs = {"beta": -1} if name == "momentum" else {}
            run = run_optimizer(name, a, b, np.zeros(DIM), MAX_ITER, **kwargs)
            gaps = [max(f - f_star, 1e-30) for f in run.fs]
            rho_emp = _fit_log_linear_rate(gaps)
            if name == "gd":
                rho_th = (kappa - 1) / (kappa + 1)
            else:
                rho_th = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
            if rho_emp is not None:
                rho_emp_list.append(rho_emp)
                rho_th_list.append(rho_th)
                results[f"kappa_{kappa}_{name}_rho_emp"] = rho_emp
                results[f"kappa_{kappa}_{name}_rho_th"] = rho_th
                results[f"kappa_{kappa}_{name}_rel_err"] = abs(rho_emp - rho_th) / rho_th

        x_pos = np.arange(len(kappas[: len(rho_emp_list)]))
        width = 0.35
        ax.bar(x_pos - width / 2, rho_th_list, width, label="Theory ρ*", color="#aec7e8")
        ax.bar(x_pos + width / 2, rho_emp_list, width, label="Empirical ρ_emp", color="#ff7f0e")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(k) for k in kappas[: len(rho_emp_list)]])
        ax.set_xlabel("κ")
        ax.set_ylabel("Convergence rate")
        ax.set_title("GD" if name == "gd" else "Momentum (Polyak)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Empirical vs theoretical convergence rate", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp6_empirical_rate.png", dpi=150)
    plt.close(fig)

    # κ=100 对数图叠加理论直线
    kappa = 100
    a, b, _, _ = make_quadratic_problem(DIM, kappa, seed=SEED)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for name, color, rho_th in [
        ("gd", "#1f77b4", (kappa - 1) / (kappa + 1)),
        ("momentum", "#ff7f0e", (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)),
    ]:
        kwargs = {"beta": -1} if name == "momentum" else {}
        run = run_optimizer(name, a, b, np.zeros(DIM), MAX_ITER, **kwargs)
        gaps = [max(f - f_star, 1e-30) for f in run.fs]
        ax2.semilogy(gaps, color=color, linewidth=1.5, label=f"{name.upper()} data")
        k0 = 30
        if gaps[k0] > 0:
            ref = gaps[k0] * (rho_th ** (np.arange(len(gaps)) - k0))
            ax2.semilogy(ref, "--", color=color, alpha=0.7, label=rf"{name.upper()} theory $\rho={rho_th:.4f}$")
    ax2.set_xlabel("Iteration k")
    ax2.set_ylabel(r"$f(x_k)-f^*$")
    ax2.set_title(r"$\kappa=100$: log-linear fit vs theory")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "exp6_log_linear_fit.png", dpi=150)
    plt.close(fig2)
    return results


def exp7_momentum_beta_sensitivity() -> dict:
    """实验7：动量系数 β 与步长 η 的敏感性（谱半径视角）。"""
    kappa = 100.0
    mu, L = 1.0, kappa
    step_opt, beta_opt = optimal_polyak_params(mu, L)
    beta_grid = np.linspace(0.0, 0.995, 80)
    rho_grid = [worst_case_spectral_radius(mu, L, step_opt, b) for b in beta_grid]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(beta_grid, rho_grid, linewidth=2, color="#1f77b4")
    ax.axvline(beta_opt, color="red", linestyle="--", label=rf"Polyak $\beta^*={beta_opt:.4f}$")
    ax.axhline(
        (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1),
        color="green",
        linestyle=":",
        label=r"Min rate $(\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$",
    )
    ax.scatter([0.9], [worst_case_spectral_radius(mu, L, step_opt, 0.9)], color="orange", zorder=5, label=r"Default $\beta=0.9$")
    ax.set_xlabel(r"Momentum $\beta$")
    ax.set_ylabel(r"Worst-case spectral radius $\rho$")
    ax.set_title(r"$\kappa=100$, fixed Polyak $\eta^*$: $\rho(\beta)$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp7_beta_spectrum.png", dpi=150)
    plt.close(fig)

    # 实际收敛：若干 β
    a, b, _, _ = make_quadratic_problem(DIM, kappa, seed=SEED)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)
    beta_cases = [
        (beta_opt, "Polyak optimal"),
        (0.9, "beta=0.9 (ML default)"),
        (0.0, "beta=0 (no momentum)"),
        (0.99, "beta=0.99 (over-damped)"),
    ]
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    results = {"beta_opt": beta_opt, "rho_at_beta_opt": float(worst_case_spectral_radius(mu, L, step_opt, beta_opt))}
    for beta, label in beta_cases:
        run = run_optimizer("momentum", a, b, np.zeros(DIM), MAX_ITER, beta=beta)
        gaps = [max(f - f_star, 1e-30) for f in run.fs]
        ax2.semilogy(gaps, linewidth=1.5, label=label)
        results[f"iter_1e6_beta_{beta}"] = _first_below(gaps, 1e-6)
    ax2.set_xlabel("Iteration k")
    ax2.set_ylabel(r"$f(x_k)-f^*$")
    ax2.set_title(r"$\kappa=100$: convergence under different $\beta$")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "exp7_beta_convergence.png", dpi=150)
    plt.close(fig2)
    return results


def _make_diagonal_quadratic(dim: int, kappa: float, seed: int = 0):
    """对角 Hessian：Jacobi 预条件可理想消去尺度差异。"""
    rng = np.random.default_rng(seed)
    eigvals = np.geomspace(1.0, kappa, dim)
    a = np.diag(eigvals)
    x_star = rng.standard_normal(dim)
    b = a @ x_star
    return a, b, 1.0, kappa


def exp8_jacobi_vs_adam() -> dict:
    """实验8：Jacobi 预条件 GD vs Adam vs GD（旋转稠密 / 对角两种情形）。"""
    kappas = [10, 100, 1000]
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey="row")

    for row, (problem_name, make_prob) in enumerate(
        [
            ("Rotated dense A", make_quadratic_problem),
            ("Diagonal A (Jacobi ideal)", _make_diagonal_quadratic),
        ]
    ):
        for ax, kappa in zip(axes[row], kappas):
            a, b, _, _ = make_prob(DIM, kappa, seed=SEED)
            x_star = optimal_point(a, b)
            f_star = objective(a, b, x_star)
            x0 = np.zeros(DIM)
            for name, label, color in [
                ("gd", "GD", "#1f77b4"),
                ("jacobi", "Jacobi GD", "#d62728"),
                ("adam", "Adam", "#2ca02c"),
            ]:
                lr = 0.5 if name == "adam" else (1.0 if name == "jacobi" else None)
                run = run_optimizer(name, a, b, x0, MAX_ITER, lr=lr)
                gaps = [max(f - f_star, 1e-30) for f in run.fs]
                ax.semilogy(gaps, label=label, color=color, linewidth=1.5)
                key = f"{problem_name}_kappa_{kappa}_{name}_iter_1e6"
                results[key] = _first_below(gaps, 1e-6)

            if row == 0:
                p_jac = np.diag(1.0 / np.clip(np.diag(a), 1e-12, None))
                results[f"dense_kappa_{kappa}_jacobi_kappa_eff"] = effective_preconditioned_condition_number(
                    a, p_jac
                )

            ax.set_title(f"{problem_name}, κ={kappa}")
            ax.set_xlabel("Iteration k")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6)

    axes[0, 0].set_ylabel(r"$f(x_k)-f^*$ (dense)")
    axes[1, 0].set_ylabel(r"$f(x_k)-f^*$ (diag)")
    fig.suptitle("Jacobi-preconditioned GD vs Adam vs GD", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp8_jacobi_comparison.png", dpi=150)
    plt.close(fig)
    return results


def exp9_newton_schulz_domain() -> dict:
    """实验9：Newton-Schulz 初值缩放与收敛域。"""
    rng = np.random.default_rng(SEED)
    g = rng.standard_normal((16, 16))
    scale_factors = np.linspace(0.3, 2.5, 15)
    results = {}
    iter_counts, final_errors = [], []

    for c in scale_factors:
        _, errors = newton_schulz_iterate(g, max_iter=30, scale_factor=c)
        finite_errors = [e for e in errors if np.isfinite(e)]
        if not finite_errors:
            iter_counts.append(30)
            final_errors.append(float("inf"))
            results[f"scale_{c:.2f}_iter"] = None
            results[f"scale_{c:.2f}_final_err"] = float("inf")
            continue
        it = _first_below(finite_errors, 1e-6)
        iter_counts.append(it if it is not None else 30)
        final_errors.append(finite_errors[-1])
        results[f"scale_{c:.2f}_iter"] = it
        results[f"scale_{c:.2f}_final_err"] = finite_errors[-1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(scale_factors, iter_counts, "o-", color="#1f77b4")
    axes[0].axvline(1.0, color="red", linestyle="--", label="Normalized init (σ_max)")
    axes[0].set_xlabel("Initial scale factor c (X0 = G/(c·σ_max))")
    axes[0].set_ylabel("Iterations to reach ||X'X-I||<1e-6")
    axes[0].set_title("Convergence speed vs initialization")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for c in [0.5, 1.0, 1.8]:
        _, errors = newton_schulz_iterate(g, max_iter=20, scale_factor=c)
        axes[1].semilogy(errors, "o-", linewidth=1.2, label=f"c={c}")
    axes[1].set_xlabel("Newton-Schulz iteration")
    axes[1].set_ylabel(r"$\|X^TX-I\|_F$")
    axes[1].set_title("Error curves for selected scalings")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp9_ns_domain.png", dpi=150)
    plt.close(fig)
    return results


def exp10_trajectories_2d() -> dict:
    """实验10：二维病态二次曲面上的优化轨迹（含 Nesterov）。

    避免之前误读：Adam 路径短 ≠ Adam 更快；同时画 80 步到 x* 的剩余距离。
    """
    kappa = 100.0
    a, b, mu, L = make_quadratic_problem(2, kappa, seed=SEED)
    x_star = optimal_point(a, b)
    f_star = objective(a, b, x_star)

    span = 3.0
    n_grid = 80
    coords = np.linspace(-span, span, n_grid)
    x1g, x2g = np.meshgrid(coords, coords)
    z = np.zeros_like(x1g)
    for i in range(n_grid):
        for j in range(n_grid):
            pt = np.array([x1g[i, j], x2g[i, j]])
            z[i, j] = objective(a, b, pt) - f_star

    x0 = np.array([2.5, 2.0])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    levels = np.logspace(-1, 3.5, 20)
    ax.contour(x1g, x2g, z, levels=levels, cmap="viridis", alpha=0.6)
    ax.plot(x_star[0], x_star[1], "r*", markersize=14, label=r"$x^*$")

    colors = {
        "gd": "#1f77b4",
        "momentum": "#ff7f0e",
        "nesterov": "#9467bd",
        "adam": "#2ca02c",
    }
    results = {}
    n_steps = 80
    convergence_data = {}
    for name in ["gd", "momentum", "nesterov", "adam"]:
        kwargs = {"mu": mu, "L": L}
        if name == "momentum":
            kwargs["beta"] = -1
        lr = ADAM_LR_2D if name == "adam" else None
        run = run_optimizer(name, a, b, x0.copy(), n_steps, lr=lr, **kwargs)
        xs = np.array(run.xs)
        ax.plot(
            xs[:, 0], xs[:, 1], "o-",
            color=colors[name], markersize=2.5, linewidth=1.0,
            label=name.upper(),
        )
        results[f"path_length_{name}"] = float(
            np.sum(np.linalg.norm(np.diff(xs, axis=0), axis=1))
        )
        results[f"dist_to_optimum_{name}"] = float(
            np.linalg.norm(xs[-1] - x_star)
        )
        gaps = [max(f - f_star, 1e-30) for f in run.fs]
        convergence_data[name] = gaps

    ax.plot(x0[0], x0[1], "ko", markersize=8, label=r"$x_0$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(r"2D trajectories, $\kappa=100$")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # 右图：同步收敛曲线
    ax2 = axes[1]
    for name, gaps in convergence_data.items():
        ax2.semilogy(gaps, color=colors[name], linewidth=1.5,
                    label=name.upper())
    ax2.set_xlabel("Iteration k")
    ax2.set_ylabel(r"$f - f^*$")
    ax2.set_title("Convergence over same 80 iterations")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp10_trajectories_2d.png", dpi=150)
    plt.close(fig)
    return results


def main() -> None:
    setup_dirs()
    # 极端 κ（如 1000）下 Heavy-ball 标量递推会数值溢出，属预期边界行为，
    # 结果已被 max(·,1e-30) 截断，这里抑制相应 RuntimeWarning 以保持输出整洁。
    import warnings

    np.seterr(over="ignore", invalid="ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    all_results = {
        "exp1": exp1_convergence(),
        "exp2": exp2_momentum_spectrum(),
        "exp3": exp3_newton_schulz(),
        "exp4": exp4_preconditioner_evolution(),
        "exp5": exp5_step_size_sensitivity(),
        "exp6": exp6_empirical_vs_theory_rate(),
        "exp7": exp7_momentum_beta_sensitivity(),
        "exp8": exp8_jacobi_vs_adam(),
        "exp9": exp9_newton_schulz_domain(),
        "exp10": exp10_trajectories_2d(),
    }
    all_results.update(run_all_extended())
    out_path = DATA_DIR / "experiment_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"实验完成。图表保存在 {FIG_DIR}")
    print(f"数值摘要保存在 {out_path}")


if __name__ == "__main__":
    main()
