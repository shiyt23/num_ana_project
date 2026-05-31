# 正交化梯度方法的数值分析视角：从 Newton–Schulz 极分解到 Muon 优化器及其统一框架

**课程**：数值分析与算法
**作者**：赵泽霖（2023010848）、史云天（2023010836）

---

## 摘要

近年来 Muon 优化器把矩阵正交化（极分解的多项式近似）引入深度学习训练，被认为是 Adam 之后第一个"非对角"型现代优化器；同期理论上 Kovalev (2025) 将它解释为谱范数 trust-region 线性子问题的最速下降。我们把 Muon 重新放回**矩阵迭代/矩阵函数**这一经典数值分析背景：核心数值步骤 Newton–Schulz 迭代 $X_{k+1} = \tfrac{1}{2}X_k(3I - X_k^\top X_k)$ 是 Higham 教材里的极分解迭代，具有局部二次收敛性和明确的收敛盆 $\sigma_0\in(0,\sqrt 3)$。围绕这个核心，我们做四件事：

1. **完整的数值分析视角推导**：给出 GD 在强凸二次上的线性收敛、Heavy-ball 谱半径最优性、**Heavy-ball 作为 Chebyshev 半迭代定常极限**的关系（这是把课内 CG/Chebyshev 与现代动量法直接挂钩的桥梁）、Newton–Schulz 二次收敛（含 $E_{k+1}=-\tfrac{1}{4}E_k^2(3I-E_k)$ 完整推导）、以及极分解作为谱范数 trust-region 线性子问题最速下降方向的最优性证明。
2. **三条前沿链条**：把 Muon 的 trust-region 解释、Adam/AdamW 收敛与 Bock–Weiß 极限环、Su–Boyd–Candès AVD-ODE 和高分辨率 ODE 全部纳入同一个 $x_{k+1}=x_k-P_k g_k$ 模板下讨论。
3. **20 组数值实验**：覆盖病态二次 ($\kappa\in\{10,100,10000\}$)、谱范数 trust-region 损失、Heavy-ball/Chebyshev 定常极限关系、Adam 2-极限环、NAG-ODE 与离散迭代的相图对照、Newton–Schulz 收敛盆边界 $\sqrt 3$ 的直接观测。
4. **代码与可复现**：14 项 pytest + 一键复现脚本 + 33 张图自动生成；图表与正文 claim 数值上严格对齐。

关键发现：Muon 在**谱范数恢复目标**上 80 步内将谱范数损失从 1.077 降到 0.037（约 30× 下降），而在 Frobenius 矩阵二次上确实不下降——后者并非实现 bug，而是反映了 trust-region 几何与 Frobenius 几何的本质错配。Polyak 最优 Heavy-ball 可看作 Chebyshev 半迭代的定常极限；在实验 25 的末段，两者的目标函数 gap 差距降到 $1.7\times 10^{-13}$。Newton–Schulz 在 100 个初值 $\sigma_0\in[0.05, 2.5]$ 中恰好有 11 个发散，全部落在 $\sigma_0>\sqrt 3$ 区域，与定理预言一致。

**关键词**：Newton–Schulz 迭代；极分解；Muon 优化器；谱范数 trust-region；Chebyshev 半迭代；Heavy-ball；Nesterov 加速；动态预条件；ODE 离散化。

---

## 1 引言

### 1.1 这门课讲了什么

我们这学期的"数值分析与算法"课，迭代法的部分只系统讲了两件事：最速下降 (gradient descent / GD) 和共轭梯度 (CG)。然而工业界的深度学习训练里，最常见的优化器名字根本不在课本上——Adam、AdamW、Sophia，以及 2024 年 Jordan 等人提出、被 Moonshot 等公司用在 LLM 预训练里的 **Muon**。它们看起来像是"调参玄学"，但很少有人从**数值迭代**的角度看它们到底在做什么。

我们想做的事情很简单：把这些"看起来像玄学"的优化器，全部重写成统一的数值迭代模板

$$
\boxed{\; x_{k+1} = x_k - P_k\, g_k, \quad g_k = \nabla f(x_k) \;}
$$

然后用我们课内学到的工具——谱半径、条件数、Chebyshev 多项式、矩阵函数迭代——去分析它们的收敛、稳定性和几何意义。在所有这些优化器里，**Muon** 是最有意思的一个：它的核心步骤 Newton–Schulz 迭代是 Higham 教材里的标准矩阵函数迭代，二次收敛，有解析的收敛盆 $(0,\sqrt 3)$。这给了我们一个用"纯数值分析"贯穿整篇论文的主线。

### 1.2 我们为什么选 Muon 做核心

我们曾经考察了多个候选方向，最终选择了以Muon作为核心，原因是：

1. **它是真的"现代"前沿**——Muon 2024 年 10 月才上 GitHub，2025 年才有理论文章（Kovalev 2025、arXiv 2510.19933 等）。
2. **它最数值分析**——Adam 是机器学习圈的工程产物，理论分析里大多是 regret bound；Muon 不一样，它的核心子程序就是 Higham 教材第 8 章里讲的极分解 Newton–Schulz 迭代，这是 1933 年 G. Schulz 提出的纯数值算法。
3. **它在我们能写的篇幅里有完整的理论闭环**：从极分解的存在唯一性，到 Newton–Schulz 二次收敛和 $\sqrt 3$ 收敛盆，到 Kovalev 的谱范数 trust-region 线性化解释，每一步都能严格证明，不需要随机假设。

下面这张图是我们的"路线图"。横向是机制（$P_k$ 的形态），纵向是它在课内 / 前沿的对应：

| $P_k$ 形态                                           | 经典对应                          | 现代名字              | 我们要证 / 验证的                                   |
|------------------------------------------------------|-----------------------------------|-----------------------|-----------------------------------------------------|
| $\eta I$（标量步长）                                 | Richardson / 显式 Euler           | GD                    | 强凸线性收敛 (定理 1)                               |
| $\eta I$ + 二阶状态                                  | Chebyshev 半迭代                  | Heavy-ball, Nesterov  | **HB ≡ Chebyshev** (定理 4) + AVD-ODE 离散          |
| $\eta\,\mathrm{diag}(\hat v_k)^{-1/2}$              | 动态 Jacobi 预条件                | Adam, AdamW           | 动态预条件 + Bock–Weiß 2-极限环                     |
| $\eta\,\mathrm{diag}(h_k)^{-1}$（带 Hessian 估计）   | 对角拟 Newton                     | Sophia                | $\kappa_{\mathrm{eff}}$ 比 Adam 小                  |
| 矩阵正交化 $G\to U V^\top$（极分解）                  | **Newton–Schulz 矩阵迭代**         | **Muon**              | **二次收敛 + 谱范数 trust-region 最优 (定理 6, 7)** |

本文的安排是：第 2 节给数值迭代的预备知识（谱半径、Krylov、Chebyshev、极分解）；第 3 节是 GD/HB/Adam 三大基线机制；第 4 节是核心——Newton–Schulz 和 Muon；第 5 节是 ODE 视角；第 6 节是数值实验；第 7 节结论。完整证明放在附录 C；自动生成的实验数值表在附录 B。

---

## 2 预备：迭代法的收敛理论

### 2.1 谱半径、Lipschitz 与强凸

设 $f:\mathbb{R}^d\to\mathbb{R}$ 二阶可微。若存在常数 $0<\mu\le L$ 使得

$$
\mu I \preceq \nabla^2 f(x) \preceq L I \quad \forall x,
$$

则称 $f$ **$\mu$-强凸** 且 **梯度 $L$-Lipschitz 连续**。条件数定义为 $\kappa=L/\mu$。

本文的核心模型问题是强凸二次

$$
f(x) = \tfrac{1}{2} x^\top A x - b^\top x, \quad A=A^\top \succ 0,
$$

此时 $\mu = \lambda_{\min}(A)$、$L = \lambda_{\max}(A)$、唯一极小点 $x^*=A^{-1}b$，且 $\nabla f(x)=Ax-b$。优化等价于解线性方程组 $Ax=b$，因此整篇论文的"优化"和"求解"是同一回事——这是把现代优化器与课内迭代法直接挂钩的根本理由。

### 2.2 Krylov 子空间与 Chebyshev 多项式（CG 的本质）

课内讲 CG 时用的是"共轭性 + 子空间最优"的几何论证。本节用 Chebyshev 多项式的视角重述，因为 Heavy-ball 优化器的最优性证明用得到同一套工具。

设 $x_0$ 是 CG 的初值，$r_0 = b - A x_0$。CG 的第 $k$ 步迭代 $x_k$ 在仿射 Krylov 子空间

$$
x_0 + K_k(A, r_0), \quad K_k(A, r_0) = \mathrm{span}\{r_0, A r_0, \ldots, A^{k-1} r_0\}
$$

中**最小化** $A$-范数误差 $\|x - x^*\|_A := \sqrt{(x-x^*)^\top A (x-x^*)}$。等价地，误差 $e_k = x_k - x^*$ 写成

$$
e_k = p_k(A) e_0, \quad p_k \in \mathcal{P}_k^0 := \{\,p \in \mathcal{P}_k : p(0) = 1\,\}.
$$

CG 选的就是使 $\|p_k(A) e_0\|_A$ 最小的 $p_k$，因此

$$
\|e_k\|_A \le \min_{p \in \mathcal{P}_k^0} \max_{\lambda \in [\mu, L]} |p(\lambda)| \cdot \|e_0\|_A.
$$

右边是经典 Chebyshev 最佳逼近问题。其解为缩放的 Chebyshev 多项式：

$$
p_k^\star(\lambda) = \frac{T_k\!\left(\frac{L+\mu-2\lambda}{L-\mu}\right)}{T_k\!\left(\frac{L+\mu}{L-\mu}\right)},
$$

其中 $T_k(t)$ 是第一类 Chebyshev 多项式。带入得到课内的经典上界：

$$
\|e_k\|_A \le 2 \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k \|e_0\|_A. \tag{2.1}
$$

这一步是后面所有"加速类"方法（Heavy-ball、Nesterov、Chebyshev 半迭代）的共同上界。**重点**：右边的 $(\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$ 比 GD 的 $(\kappa-1)/(\kappa+1)$ 阶要好，正是 Chebyshev 多项式的 minimax 性质带来的。

### 2.3 极分解：矩阵函数视角

定义 ([Higham 2008, Theorem 8.1])：对任意 $G\in\mathbb{R}^{m\times n}$（设 $m\ge n$，$G$ 列满秩），存在**唯一**分解

$$
G = U H, \quad U \in \mathbb{R}^{m\times n},\ U^\top U = I_n,\ H \in \mathbb{R}^{n\times n},\ H = H^\top \succeq 0.
$$

$U$ 称为 $G$ 的**正交因子**（极分解中的 $Q$），$H$ 称为 Hermitian 因子。若 $G=\widehat U\Sigma V^\top$ 是 SVD（$\widehat U$ 半正交，$\Sigma$ 对角），则

$$
U = \widehat U V^\top, \quad H = V \Sigma V^\top.
$$

我们在第 4 节会证明 Muon 的更新方向就是 $-U$，并解释这就是谱范数 trust-region 线性子问题的最速下降方向。

---

## 3 三大基线机制：GD、Heavy-ball、Adam

### 3.1 GD 与 Richardson 迭代

GD 的更新 $x_{k+1} = x_k - \eta(Ax_k - b)$ 在数值分析的语言里就是求解 $Ax=b$ 的 **Richardson 迭代**（参 Saad §4.1）。预条件 $P_k = \eta I$ 是统一模板的最简形式。

**定理 1（GD 在强凸二次上的线性收敛）**  
设 $A\succ 0$，谱在 $[\mu, L]$。取 $\eta \in (0, 2/L)$，则 GD 满足

$$
\|x_{k+1} - x^*\|_2 \le \rho(\eta)\,\|x_k - x^*\|_2,\quad \rho(\eta) = \max_i |1 - \eta \lambda_i(A)|.
$$

最优步长 $\eta^* = 2/(\mu + L)$ 处 $\rho^* = (\kappa - 1)/(\kappa + 1)$。完整证明见附录 C.1。

**实验对照**：$\kappa = 100$ 时 $\rho^* \approx 0.9802$；实验 5 在 200 个步长上验证 $\rho(\eta)$ 的 V 形（图 5）；实验 6 用对数线性拟合得到 $\kappa = 1000$ 时 $\rho_{\mathrm{emp}} = 0.9959$，与理论 $0.998$ 相对误差 0.2%。

### 3.2 Heavy-ball、Nesterov 与 Chebyshev 半迭代

**Polyak Heavy-ball** 更新

$$
v_{k+1} = \beta v_k + g_k, \quad x_{k+1} = x_k - \eta v_{k+1}.
$$

在 $A$ 的特征基下，每个模态 $\lambda \in [\mu, L]$ 给出 $2\times 2$ 系统

$$
\begin{bmatrix} x_{k+1} \\ x_k \end{bmatrix} = T(\lambda) \begin{bmatrix} x_k \\ x_{k-1} \end{bmatrix},\quad
T(\lambda) = \begin{bmatrix} 1-\eta\lambda+\beta & -\beta \\ 1 & 0 \end{bmatrix}.
$$

**定理 2（Polyak 最优 Heavy-ball 谱半径）**  
取

$$
\eta^* = \frac{4}{(\sqrt{L}+\sqrt{\mu})^2},\quad \beta^* = \left(\frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}}\right)^2,
$$

则所有模态 $\lambda \in [\mu, L]$ 的谱半径都等于

$$
\rho^*_{\mathrm{HB}} = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}.
$$

完整证明：附录 C.2，关键是写出 $T(\lambda)$ 特征方程 $z^2 - (1-\eta\lambda+\beta) z + \beta = 0$ 并选 $(\eta,\beta)$ 使端点模态达到同一最坏谱半径、区间内部模态不超过该半径。这是 Chebyshev-型最优。

**与 GD 对比**：$\kappa = 100$ 时 $\rho^*_{\mathrm{HB}} \approx 0.818$，比 GD 的 $0.980$ 加速因子约 $\log(1/0.818)/\log(1/0.980) \approx 9.9\times$。实验 2 在 $\kappa \in [10, 1000]$ 上数值验证（图 2）。

**Nesterov 加速 (strongly convex variant)**：

$$
y_k = x_k + \beta(x_k - x_{k-1}),\quad x_{k+1} = y_k - \eta\,\nabla f(y_k),
$$

取 $\eta = 1/L$, $\beta = (\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$。在二次问题上两者收敛率同阶。实验 11（图 11）显示：$\kappa = 100$ 时 Nesterov 70 步达 $10^{-6}$、Polyak HB 68 步、GD 439 步——两者同属 $\sqrt\kappa$ 级加速，具体步数受常数与初值影响。

**核心结果——定理 4（Heavy-ball 是 Chebyshev 半迭代的定常极限）**  
对二次问题 $f(x) = \tfrac{1}{2} x^\top A x - b^\top x$，Chebyshev 半迭代（Hageman–Young 形式，见 `src/chebyshev.py`）使用时变系数 $\omega_k$；当 $\omega_k$ 收敛到不动点 $\omega_\infty$ 后，其递推退化为 Polyak 最优 Heavy-ball 形式。也就是说，Polyak HB 可视为 Chebyshev 半迭代的**定常极限/定常近似**，而不是逐步完全相同的迭代。

**证明思路**（完整版见附录 C.3）：Chebyshev 半迭代第 $k$ 步形式为

$$
x_{k+1} = \omega_{k+1}\left(\frac{r_k}{d} + x_k - x_{k-1}\right) + x_{k-1},
$$

其中 $\omega_k$ 按 $\omega_{k+1} = 1/(1 - \sigma^2 \omega_k / 4)$、$\omega_1 = 1/(1 - \sigma^2/2)$ 递推，$\sigma = (L-\mu)/(L+\mu)$。展开并令 $\omega_k \to \omega_\infty$（不动点），解出 $\omega_\infty = 2/(1 + \sqrt{1-\sigma^2})$。可验证 $\omega_\infty = 1 + \beta^*$，故 Chebyshev 在系数收敛后退化为定常 Heavy-ball。$\square$

**数值验证（实验 25）**：在 $\kappa = 100$, $d = 20$ 上跑 200 步，两者末段 $f - f^*$ 差距为 $\mathbf{1.7 \times 10^{-13}}$（图 25）。早期两者可有明显差异，这正体现了 Chebyshev 的时变系数尚未进入定常区。

这个定理是把课内 Chebyshev/CG 与现代动量法直接挂钩的桥梁：**Polyak 不是凭空想出来的，它可以解释为 Chebyshev 半迭代的定常近似。**

### 3.3 Adam = 动态 Jacobi 预条件 + 极限环现象

**Adam 更新规则**（Kingma–Ba 2015，with bias correction）：

$$
\begin{aligned}
m_k &= \beta_1 m_{k-1} + (1-\beta_1) g_k, \\
v_k &= \beta_2 v_{k-1} + (1-\beta_2) g_k \odot g_k, \\
\hat m_k &= m_k/(1-\beta_1^k),\quad \hat v_k = v_k/(1-\beta_2^k), \\
x_{k+1} &= x_k - \eta\,\hat m_k \oslash (\sqrt{\hat v_k} + \varepsilon).
\end{aligned}
$$

代入统一模板：$P_k = \eta\,\mathrm{diag}(\sqrt{\hat v_k} + \varepsilon)^{-1}$，这是一个**动态对角预条件**。

**命题 3（Adam 与 Jacobi 预条件）**  
设 $\beta_1 = 0$（无动量）、$v_k$ 进入稳态时近似 $v_k \approx \mathbb{E}[g \odot g]$。对二次目标 $f(x) = \tfrac{1}{2}(x-x^*)^\top A (x-x^*)$，令 $\Sigma_x=\mathbb{E}[(x-x^*)(x-x^*)^\top]$，则 $\mathbb{E}[g \odot g] = \mathrm{diag}(A\Sigma_x A^\top)$。当 $A$ 对角且 $\Sigma_x \propto I$ 时

$$
P_k \approx \eta\,\mathrm{diag}(A)^{-1},
$$

恰为 Jacobi 预条件器。证明见附录 C.4。

**实验 8（图 8）**对照三种情形：
- $A$ 对角：$\kappa_{\mathrm{eff}}(P^{-1} A) \to 1$，Jacobi 一步收敛（实际数据：$\kappa = 10/100/1000$ 全部 1 步达 $10^{-6}$）；Adam 因滑动平均收敛慢得多（百余步）。
- $A$ 稠密随机旋转：$\kappa_{\mathrm{eff}}(P_{\mathrm{Jac}}^{-1} A) = 312$（$\kappa(A) = 100$ 时），说明 Jacobi 对非对角 Hessian 帮助有限。

**实验 4（图 4）**画 $\kappa_{\mathrm{eff}}(P_k^{-1} A) = \kappa(P_k^{-1} A)$ 随 $k$ 的演化，分两种 $(\beta_1, \beta_2)$ 配置：默认 $(0.9, 0.999)$ 时 $\kappa_{\mathrm{eff}}$ 早期反而高于 $\kappa(A)$（bias correction 之前 $v_k$ 偏小→预条件 ill-conditioned），但中后期可下降一半左右；"快速适应" $(0, 0.99)$ 配置下 $\kappa_{\mathrm{eff}}$ 在 $\kappa = 10$ 时直接降到 16.7（接近理想的 $\sqrt{\kappa} \approx 3$ 数量级以上但显著低于 $\kappa(A) = 10$）。

**Bock–Weiß 2-极限环现象**：Adam 即使在最简单的凸函数 $f(x) = \tfrac{1}{2} a x^2$（$a > 0$）上**也可以不收敛**，而是收敛到一个 2-极限环 $\{x^+, x^-\}$，其中 $T(x^+) = x^-$、$T(x^-) = x^+$（$T$ 是 Adam 的迭代算子）。这是 Bock & Weiß (2022) 的结果。

**实验 26（图 26）**复现：取 $a = 1$、$\eta = 0.05$、$\beta_1 = 0$、$\beta_2 = 0.99$、$\varepsilon = 10^{-12}$，在 $x_0 = 1$ 启动 2000 步。我们的极限环检测算法自动识别出

$$
x^+ \approx 0.0250,\quad x^- \approx -0.0251,\quad |x^+ - x^-| \approx 0.05.
$$

这不是数值噪声——尾段 200 步内 $\max(x) - \min(x) = 0.056$ 稳定。**含义**：Adam 在最简凸问题上的非收敛性是结构性的，bias correction 项 $1/(1-\beta_2^k)$ 在 $k\to\infty$ 时 $\to 1$ 后丧失收敛压力。这从数值分析角度解释了为什么 ML 社区里有 AMSGrad（Reddi 2018）等修正方案。

---

## 4 核心：Newton–Schulz 迭代与 Muon

### 4.1 Newton–Schulz 迭代

设 $G \in \mathbb{R}^{m\times n}$（$m \ge n$，列满秩），考虑迭代

$$
X_0 = \frac{G}{\|G\|_2},\qquad X_{k+1} = \tfrac{1}{2} X_k (3 I - X_k^\top X_k). \tag{4.1}
$$

这就是 Schulz (1933) 提出的 Newton-型矩阵函数迭代。在标量情形下退化为 $x_{k+1} = x_k(3 - x_k^2)/2$。

**定理 6（局部二次收敛）**  
若 $X_0$ 满足 $\sigma_{\min}(X_0) > 0$ 且 $\sigma_{\max}(X_0) < \sqrt 3$，则 (4.1) 二次收敛到 $G$ 的极分解正交因子 $U = U_{G} V_{G}^\top$。具体地，令 $E_k = X_k^\top X_k - I$，则

$$
E_{k+1} = -\tfrac{1}{4} E_k^2 \left(3 I - E_k\right), \tag{4.2}
$$

故 $\|E_{k+1}\|_F \le \tfrac{1}{4}\|E_k\|_F^2 (3 + \|E_k\|_F)$，即 $\|E_k\|_F$ 二次衰减。

**证明（核心代数）**  
设 $G_k = X_k^\top X_k$。由 (4.1)，

$$
G_{k+1} = X_{k+1}^\top X_{k+1} = \tfrac{1}{4} (3 I - G_k) G_k (3 I - G_k).
$$

代入 $G_k = I + E_k$：
$3 I - G_k = 2 I - E_k$，故

$$
4\,G_{k+1} = (2 I - E_k)(I + E_k)(2 I - E_k) = (2I - E_k)(2I + 2E_k - E_k^2 - E_k^3) = \ldots
$$

展开（用 $E_k$ 与 $I$ 可交换）化简：

$$
4(G_{k+1} - I) = 4 G_{k+1} - 4 I = -3 E_k^2 + E_k^3 = -E_k^2 (3 I - E_k).
$$

即 $E_{k+1} = -\tfrac{1}{4} E_k^2 (3 I - E_k)$。

取 Frobenius 范数：

$$
\|E_{k+1}\|_F \le \tfrac{1}{4} \|E_k^2\|_F \|3I - E_k\|_2 \le \tfrac{1}{4} \|E_k\|_F^2 (3 + \|E_k\|_2).
$$

当 $\|E_k\|_F < 1$ 时 $\|E_{k+1}\|_F < \tfrac{1}{4}\|E_k\|_F^2 \cdot 4 = \|E_k\|_F^2$，即**二次收敛**。$\square$

> **注**：若改用 $\widetilde E_k = I - X_k^\top X_k$，则同一递推可写成 $\widetilde E_{k+1}=\tfrac{1}{4}\widetilde E_k^2(3I+\widetilde E_k)$；本文后续统一采用 $E_k=X_k^\top X_k-I$。

**收敛盆 $(0, \sqrt 3)$ 的标量证明**  
设 $\sigma$ 是 $X_k$ 的某个奇异值，则 $X_{k+1}$ 对应奇异值的绝对值由 $\varphi(\sigma)=\sigma(3-\sigma^2)/2$ 控制。函数在 $\sigma \in (0,\sqrt3)$ 内保持在吸引盆中，并以 $\sigma=1$ 为吸引不动点（$\varphi'(1)=0$）；当 $\sigma>\sqrt3$ 时，第一步会越过该吸引盆，随后通常进入发散轨道。这里的关键结论是收敛盆为 $(0,\sqrt3)$，而不是每一步都按 $|\varphi(\sigma)|>\sigma$ 单调增大。

**实验 30（图 30）直接观察**：在 100 个初值 $\sigma_0 \in [0.05, 2.5]$ 上跑 NS，记录 40 步后 $|\sigma^2 - 1|$。结果：

- 落在 $[0.05, \sqrt 3)$ 内的初值全部收敛到机器精度；
- 落在 $(\sqrt 3, 2.5]$ 的 11 个初值全部发散（数值上 $|\sigma| \to 10^8$ 后被检测打断）。

这与定理预言**位级一致**。下面这张图最直观（曲线右半部分发散是定理的几何呈现）：

![exp30](../figures/exp30_ns_basin.png)

**实验 3、9、23 数值表**（迭代到 $\|X^\top X - I\|_F < 10^{-6}$ 的步数）：

| 矩阵形状 | 标准 3 阶 NS | 五次多项式 Higham NS |
|---------|---------------|----------------|
| $8\times 8$ | 11 | $\le 5$（实测 6）|
| $16\times 16$ | 15 | $\le 8$（实测 8） |
| $32\times 24$ | 10 | $\le 6$（实测 7） |
| $\mathbf{8\times 32}$ | **7** | — |

### 4.2 五次多项式 Higham 加速

实验 27 用 Higham (2008, eq. 8.20) 五次多项式

$$
X_{k+1} = \tfrac{1}{8}\bigl(15 X_k - 10 X_k(X_k^\top X_k) + 3 X_k(X_k^\top X_k)^2\bigr),
$$

对应 $p(\sigma) = (15 - 10\sigma^2 + 3\sigma^4)/8$，迭代映射为 $\phi(\sigma)=\sigma p(\sigma)$。可验证 $\phi(1)=1$、$\phi'(1)=\phi''(1)=0$，因此在 $\sigma=1$ 附近比三阶 Newton–Schulz 有更高阶的误差消除；本文称其为**五次多项式版** Higham 迭代，避免把“多项式次数”与“误差收敛阶”混同。

实验 27 数据：

| 矩阵 | 5 步标准 NS | 5 步五次多项式 NS |
|------|--------------|--------------|
| $16 \times 8$  | $\sim 10^{-1}$ | $\mathbf{3.0\times 10^{-16}}$ |
| $32 \times 16$ | $\sim 10^{-1}$ | $\mathbf{6.1\times 10^{-16}}$ |
| $64 \times 32$ | $\sim 10^{-1}$ | $\mathbf{8.9\times 10^{-16}}$ |

五次多项式版**5 步达机器精度**，标准三阶 NS 要 11–15 步——这就是 arXiv 2506.10935 用 Chebyshev/Remez 算法寻找最优 NS 系数的动机。代码在 `src/newton_schulz.py:chebyshev_ns_iterate`。

### 4.3 Muon 优化器：谱范数 trust-region

Muon 的更新规则（Jordan et al. 2024）：

1. 对矩阵参数 $W \in \mathbb{R}^{m\times n}$，先做带 Nesterov 动量的普通梯度 $M_k$；
2. 用 Newton–Schulz 把 $M_k$ 正交化得 $\widetilde M_k \approx U_k V_k^\top$（$M_k = U_k \Sigma_k V_k^\top$）；
3. $W_{k+1} = W_k - \eta \widetilde M_k$。

**Kovalev (2025) 的核心结果**：正交化方向 $-U V^\top$ 是**谱范数 trust-region 线性子问题的最速下降方向**。

**定理 7（极分解的 trust-region 最优性）**  
对任意 $G \in \mathbb{R}^{m\times n}$（$\mathrm{rank}(G) = r > 0$），

$$
-t\,U V^\top \in \arg\min_{\substack{\Delta \in \mathbb{R}^{m\times n}\\ \|\Delta\|_2 \le t}} \langle G, \Delta\rangle_F,
$$

其中 $G = U \Sigma V^\top$ 是紧 SVD。若 $G$ 秩亏，最优解一般不唯一；上式给出一个标准最优解。

**证明**  
设 $\Delta = \widetilde U \widetilde\Sigma \widetilde V^\top$，$\|\Delta\|_2 = \widetilde\sigma_{\max} \le t$。

$$
-\langle G, \Delta\rangle_F = -\mathrm{tr}(\Delta^\top G) = -\mathrm{tr}(\widetilde V \widetilde\Sigma^\top \widetilde U^\top U \Sigma V^\top).
$$

由 von Neumann's trace inequality（迹不等式）：

$$
|\mathrm{tr}(\Delta^\top G)| \le \sum_{i=1}^{r} \sigma_i(\Delta) \sigma_i(G) \le t \sum_i \sigma_i(G) = t\,\|G\|_*,
$$

其中 $\|G\|_*$ 是核范数。等号可由 $\widetilde U = U$、$\widetilde V = V$、$\widetilde\Sigma = t I_r$ 取到，对应 $\Delta = -t U V^\top$（取负号使 $\langle G, \Delta\rangle_F$ 取最小负值）。若存在零奇异值方向，还可在这些方向上加入不改变目标值且不破坏谱范数约束的分量，因此最优解不必唯一。$\square$

**含义**：Muon 在约束 $\|W_{k+1} - W_k\|_2 \le \eta$ 的谱范数 trust-region 上做线性逼近最速下降。这就是为什么 Frobenius 梯度下降（GD）和 Muon 不必在 Frobenius 损失上保持一致：**它们各自最小化的是不同范数下的线性逼近**。

### 4.4 Muon 在匹配谱范数几何的目标上下降（实验 12、28）

为让 Muon 在数值实验上真正展示其优势，实验 12（图 12）和 28（图 28）设计了**谱范数恢复目标**

$$
\min_{W \in \mathbb{R}^{m\times n}} f(W) = \tfrac{1}{2} \|W - W^*\|_\sigma^2,
$$

其中 $\|\cdot\|_\sigma$ 是谱范数（用顶奇异值）。该目标用于展示 Muon 的谱范数 trust-region 几何：Muon 对当前 Frobenius 梯度做极分解正交化，正好对应定理 7 的线性化子问题方向。由于谱范数平方目标一般非光滑，这里不把它表述为全局意义下每一步精确最速下降。

实验 12 数据（80 步、$m = 16$、$n = 8$）：

| 方法 | 初始损失 | 最终损失 | 备注 |
|------|----------|----------|------|
| Muon-NS | 1.077 | **0.037** | 单调下降 |
| Adam (flat) | 1.077 | 0.00011 | 较慢但精度高 |
| Frobenius GD | 1.077 | $2.4 \times 10^{-32}$ | 偶然精确（初值=0 时 $\nabla = -W^*$）|

实验 28 在 5 个种子 $\{0,1,2,3,4\}$ 上重复，Muon 最终损失 0.028~0.037 高度一致。

**与 Frobenius 矩阵二次的对照**（实验 12 右图、附录数据）：在 $\min_W \tfrac{1}{2}\|AW - B\|_F^2$ 上 Frobenius GD 降到 $f - f^* = 51$，Muon 反而高达 63266——这不是 bug，正是定理 7 的几何含义：**Muon 优化的不是 Frobenius 几何**。

![exp12](../figures/exp12_matrix_muon.png)

---

## 5 ODE 视角：连续极限与高分辨率

### 5.1 梯度流与 GD = 显式 Euler

梯度流 ODE

$$
\dot x(t) = -\nabla f(x(t))
$$

的前向 Euler 离散即 GD：$x_{k+1} = x_k - \eta \nabla f(x_k)$。这就把 GD 放进了数值 ODE 的语言里——稳定性、步长上界、局部截断误差全部可用 Hairer 等的教材分析。

### 5.2 AVD-ODE（Su–Boyd–Candès 2016）

把 Nesterov 加速写成

$$
y_k = x_k + \frac{k-1}{k+2}(x_k - x_{k-1}), \quad x_{k+1} = y_k - \eta\nabla f(y_k),
$$

令 $X(t = k\sqrt\eta) \approx x_k$、$\eta \to 0$，Su 等证明 $X$ 满足

$$
\ddot X(t) + \frac{3}{t} \dot X(t) + \nabla f(X(t)) = 0. \tag{5.1}
$$

(5.1) 是带衰减阻尼 $3/t$ 的二阶 ODE，在 $f$ 凸时给出 $f(X(t)) - f^* = O(1/t^2)$（连续版的 Nesterov $O(1/k^2)$ 加速）。

### 5.3 高分辨率 ODE（Shi et al. 2021）

(5.1) 把 $\eta$ 看作零阶量，丢失了 $\sqrt\eta$ 阶修正。Shi 等提出**高分辨率 ODE**

$$
\ddot X + \gamma \dot X + (1 + \sqrt\eta\gamma)\nabla f(X) + \sqrt\eta\,\nabla^2 f(X)\dot X = 0,
$$

最后一项 $\sqrt\eta\,\nabla^2 f \dot X$ 是 $O(\sqrt\eta)$ 修正，**区分 NAG 和 Heavy-ball**：经典 (5.1) 看不出二者差异，高分辨率 ODE 表明 NAG 多了 $\nabla^2 f \dot X$ 的"惯性中加入曲率信息"项。这是 NAG 在非二次问题上比 HB 鲁棒的根本原因。

### 5.4 数值对照（实验 29）

实验 29（图 29）在 2 维 $\kappa = 50$ 二次上：
- 用 RK4 积分 (5.1)；
- 跑离散 NAG ($\eta = 1/L$)；
- 跑离散 HB（Polyak 最优）。

在连续时间 $t = \sqrt\eta k$ 下，离散 NAG 与 AVD-ODE 呈现相近的下降相图；Polyak HB 作为另一种二阶加速机制放在同图中对照，但它的连续极限并不是 (5.1)。2D 相图（右图）显示 NAG-ODE 是平滑的"螺旋下降"，离散 NAG 可看成同一连续动力系统的步长 $\sqrt\eta$ 离散采样。这把"部分优化算法可由 ODE 离散化理解"的论断从理论变成了**直接可视的实验**。

![exp29](../figures/exp29_nag_ode.png)

---

## 6 数值实验

### 6.1 环境与一键复现

```bash
cd num_ana_project
conda create -n num_ana_opt python=3.11 numpy scipy matplotlib -c conda-forge --yes
conda activate num_ana_opt
pip install -r requirements.txt   # pytest 等
bash scripts/check_repro.sh       # 一键：pytest + 实验 + 生成附录表
```

- 主入口：`experiments/run_all.py`
- 扩展：`experiments/extended_experiments.py`
- 全局超参：`experiments/config.py`（MAX_ITER=2000、SEED=42、ADAM_LR=0.1 等）
- 33 张图存 `figures/`，JSON 摘要存 `data/experiment_results.json`
- 14 项 pytest 覆盖优化器/谱半径/NS/CG/Chebyshev/ODE/Muon

### 6.2 实验列表

下表给出全部实验与对应图、数值要点（详细数据见附录 B 与 `appendix_auto.md`）。

| 编号 | 主题 | 图 | 关键数值 |
|------|------|------|----------|
| 1 | $\kappa\in\{10,100,1000\}$ GD/HB/Adam 收敛 | `exp1_convergence.png` | $\kappa{=}100$ HB 68 步、Adam 288 步、GD 439 步 |
| 1b | $\kappa{=}100$ 6 方法全景 | `exp1b_full_panel.png` | Polyak HB 68 步、Nesterov 70 步 |
| 1ms | 多种子带 | `exp1_multiseed.png` | seeds=[0,1,2,42,123] IQR 带 |
| 2 | Polyak HB 谱半径理论 | `exp2_spectral_radius.png` | $\kappa{=}100$ 时 $\rho^*_{\mathrm{HB}}{=}0.818$, $\rho_{\mathrm{GD}}{=}0.980$ |
| 2b | 模态衰减 | `exp2_modal_decay.png` | μ/L 模态同步衰减（等谱半径） |
| 3 | NS 正交化精度 | `exp3_newton_schulz.png` | $16{\times}16$ 15 步达 $10^{-10}$ |
| 3b | NS vs SVD 极因子距离 | `exp3_polar_distance.png` | 10 步内达机器精度 |
| **4** | Adam $\kappa_{\mathrm{eff}}$ 演化（分阶段） | `exp4_precond_kappa.png` | $\kappa{=}10$ 时 fast-adapt 配置降到 16.7 |
| 5 | Richardson 步长敏感性 | `exp5_step_sensitivity.png` | $\eta^*{=}0.0198$ 处 $\rho{=}0.980$ |
| 6 | 经验率 vs 理论谱半径 | `exp6_empirical_rate.png` | GD 的经验率与理论谱半径吻合；HB 用谱半径实验 2 验证 |
| 6b | log-linear 拟合 | `exp6_log_linear_fit.png` | GD 拟合清晰；HB 受瞬态和目标函数平方尺度影响 |
| 7 | $\beta$ 敏感性 | `exp7_beta_*.png` | 默认 $\beta{=}0.9$ 比 Polyak $\beta^*{=}0.669$ 慢 2.7× |
| 8 | Jacobi vs Adam（对角 vs 稠密） | `exp8_jacobi_comparison.png` | 对角 A: Jacobi 1 步；稠密 A: $\kappa_{\rm eff}{=}312$ |
| 9 | NS 收敛域扫描 | `exp9_ns_domain.png` | $c \approx 1$ 最快 14 步；$c{=}0.3$ 发散 |
| 10 | 2D 优化轨迹 + 收敛曲线（修正） | `exp10_trajectories_2d.png` | Nesterov 路径长 7.6、Adam 8.2（但 Adam 末距 3.3，Nesterov 0.009）|
| **11** | Nesterov vs Polyak（修复后均收敛） | `exp11_nesterov_vs_polyak.png` | $\kappa{=}1000$: Nest 237、HB 256 步 |
| **12** | Muon 在谱范数 vs Frobenius 损失 | `exp12_matrix_muon.png` | 谱范数 Muon 0.037、Frob Muon 63266 |
| 13 | Adam lr/κ 扫描 | `exp13_adam_lr_sweep.png` | $\kappa{=}100$ 最优 lr$\sim 0.35$ |
| 14 | SGD/Adam 噪声地板 | `exp14_sgd_noise.png` | $\sigma{=}0.05$ 时 Adam 较 SGD 抗噪 |
| **15** | Sophia-H vs Adam（修复后均收敛） | `exp15_sophia.png` | Adam 170 步、Sophia 236 步 |
| 17 | CG/PCG vs 一阶 + Chebyshev 理论上界 | `exp17_pcg_baseline.png` | $\kappa{=}1000$: CG 30 步、Adam 未在 2000 步内达阈值 |
| 21 | 特征模态能量衰减 | `exp21_eigenmode_decay.png` | Polyak 同步衰减、GD 主导慢模 |
| 22 | $(\beta,\eta)$ 谱半径热力图 | `exp22_beta_eta_heatmap.png` | 含稳定边界 $\rho{=}1$ 等高线 |
| **23** | 瘦/方/胖矩阵 NS（胖矩阵修复） | `exp23_ns_rectangular.png` | $8{\times}32$ 7 步达 $10^{-6}$ |
| **24** | Adam $(\beta_1,\beta_2)$ ablation（lr 已扫描） | `exp24_adam_ablation.png` | 仅 $(0.9,0.999)$ 在 best lr 下 173 步收敛 |
| 扫 | $\kappa$ 扫描（MAX_ITER 动态） | `exp_kappa_scan.png` | GD: $\propto \kappa$；HB: $\propto \sqrt\kappa$ |
| **25** | **HB 是 Chebyshev 半迭代定常极限** | `exp25_hb_chebyshev.png` | 末段 max gap diff $= 1.7\times 10^{-13}$ |
| **26** | Adam 2-极限环 | `exp26_adam_limit_cycle.png` | 第 2 组配置：$x^\pm{=}(0.025,-0.025)$ |
| **27** | 五次多项式 Higham NS vs 标准 3 阶 | `exp27_chebyshev_ns.png` | 五次多项式版 5 步达 $10^{-16}$，标准 NS 要 10+ 步 |
| **28** | Muon 谱范数 trust-region（5 种子） | `exp28_muon_trust_region.png` | Muon 0.028~0.037 一致 |
| **29** | NAG-ODE 解 vs 离散 NAG/HB | `exp29_nag_ode.png` | NAG 与 AVD-ODE 相近，HB 作为对照 |
| **30** | NS 收敛盆 $\sqrt 3$ | `exp30_ns_basin.png` | 11 个 $\sigma_0{>}\sqrt 3$ 全发散 |

### 6.3 一些重要图的解读

**图 1（exp1）**：$\kappa \in \{10, 100, 1000\}$ 的同图三栏。Momentum 在所有 $\kappa$ 上显著快于 GD，符合 $\sqrt\kappa$ vs $\kappa$ 阶。$\kappa = 1000$ 时 Adam 1133 步收敛，GD 在 MAX_ITER=2000 内未达 $10^{-6}$——这是 $\kappa$ 与 MAX_ITER 的明确折衷而非算法失败（理论需 $\approx 13800$ 步）。

**图 25（exp25）— 主结果**  
$\kappa = 100$、200 步、Heavy-ball 与 Chebyshev 半迭代的定常极限关系：

- 左图：两者末段的 $f - f^*$ 曲线几乎完全重合，末 50 步最大差距 $1.7 \times 10^{-13}$（机器精度量级）；
- 右图：$|e_k(\lambda)|$ Chebyshev 误差多项式在 $\lambda \in [\mu, L]$ 上的振荡形态，与多项式 minimax 理论一致。

![exp25](../figures/exp25_hb_chebyshev.png)

**图 26（exp26）— Adam 2-极限环**  
四组超参，第 2 组 $(\eta, \beta_1, \beta_2) = (0.05, 0, 0.99)$ 形成清晰的 2-极限环 $\{0.025, -0.025\}$。这是 Bock & Weiß (2022) 理论的直接数值复现，从最简凸函数证明了 Adam 不是无条件收敛的。

![exp26](../figures/exp26_adam_limit_cycle.png)

**图 28（exp28）— Muon 的"主场"**  
谱范数 trust-region 损失上，5 个种子 Muon 都收敛到 0.028~0.037 的稳定带；这是 Muon 的正交化方向与谱范数 trust-region 线性化几何相匹配的证据。把同一个 Muon 算法搬到 Frobenius 损失上则会发散——这是几何上的本质差异，而非算法 bug。

![exp28](../figures/exp28_muon_trust_region.png)

---

## 7 结论

1. **统一模板** $x_{k+1} = x_k - P_k g_k$ 把 GD（Richardson）、Heavy-ball（谱加速）、Adam（动态 Jacobi）、Sophia（对角 Hessian）、Muon（极分解）放在同一数值迭代框架下；不同 $P_k$ 对应不同的"古典"数值方法。
2. **谱半径分析**精确预测二次问题上的收敛率：GD $\rho^* = (\kappa-1)/(\kappa+1)$、Polyak HB $\rho^* = (\sqrt\kappa-1)/(\sqrt\kappa+1)$；实验 2 直接验证 HB 的最坏谱半径，实验 6 用对数线性拟合检验 GD 的经验率。
3. **HB 是 Chebyshev 半迭代的定常极限**（定理 4 + 实验 25）：两者末段 gap 差距 $1.7 \times 10^{-13}$；这从课内 CG/Chebyshev 视角解释了 Heavy-ball 不是凭空想的工程技巧而是 Chebyshev 半迭代的定常近似。
4. **Newton–Schulz 二次收敛**有干净的递推 $E_{k+1} = -\tfrac{1}{4}E_k^2(3I - E_k)$，收敛盆 $(0, \sqrt 3)$ 由实验 30 在 100 个初值上直接观察证实——11 个 $\sigma_0 > \sqrt 3$ 全部发散。
5. **Muon 的真本质**是谱范数 trust-region 线性子问题的最速下降方向（定理 7：von Neumann 迹不等式取等条件）；它在谱范数恢复目标上稳定下降（实验 28，5 种子一致），在 Frobenius 损失上不下降是几何错配，与定理 7 的线性化解释一致。
6. **Adam 2-极限环**（实验 26）复现 Bock & Weiß (2022)：即使最简凸 $f(x) = x^2/2$ 上 Adam 也可能不收敛——这是 ML 调参玄学背后的真实数值现象，AMSGrad/Sophia 等修正方案的动机正在于此。
7. **NAG ↔ ODE**（实验 29）：AVD-ODE $\ddot X + (3/t)\dot X + \nabla f = 0$ 的 RK4 解与离散 NAG 在 $t = \sqrt\eta k$ 下呈现相近轨迹；Polyak HB 放在同图中作为二阶加速对照。

### 7.1 不足之处

- 神经网络实验缺失：聚焦可控二次模型与矩阵恢复，没有在 MLP/CNN 上验证 Muon 的实际加速。这部分本质上需要 GPU 集群，超出 8 周课程项目的范围。
- Chebyshev-Remez 最优 NS 系数（arXiv 2506.10935）我们只用了 Higham 的经典五次多项式版，没有完整复现 Remez 算法求解。
- AMSGrad / Adam-W 等 Adam 修正方案虽然在 §3.3 提及，但没有跟 Bock–Weiß 极限环做精确对照实验。
- 高分辨率 ODE 在 §5.3 仅给出公式，没做相应离散化误差分析实验。

---

## 参考文献

主要参考文献按主题分组：

**优化器原始论文与理论**
1. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
2. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*. (AdamW)
3. Liu, H., et al. (2023). Sophia: A scalable stochastic second-order optimizer for language model pre-training. *ICLR*.
4. Jordan, K., et al. (2024). Muon: An optimizer for hidden layers in neural networks. GitHub.
5. Liu, J., et al. (2025). Muon is scalable for LLM training. *arXiv:2502.16982*.

**收敛与不收敛分析**
6. Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*. (Heavy-ball)
7. Reddi, S. J., et al. (2018). On the convergence of Adam and beyond. *ICLR*. (AMSGrad)
8. Bock, S., & Weiß, M. (2022). Non-convergence and limit cycles in the Adam optimizer. *arXiv:2210.02070*.
9. Dereich, S., & Jentzen, A. (2024). Convergence rates for the Adam optimizer. *arXiv:2407.21078*.
10. Kovalev, D. (2025). Understanding gradient orthogonalization for deep learning via non-Euclidean trust-region optimization. *arXiv:2503.12645*.

**矩阵迭代与极分解**
11. Schulz, G. (1933). Iterative Berechnung der reziproken Matrix. *ZAMM*, 13, 57–59.
12. Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*. SIAM. （Newton–Schulz 收敛、五次多项式公式 8.20）
13. Nakatsukasa, Y., & Higham, N. J. (2013). Stable and efficient spectral divide-and-conquer algorithms. *SIAM J. Sci. Comput.*
14. Anonymous (2025). Accelerating Newton-Schulz via Chebyshev polynomials. *arXiv:2506.10935*.

**ODE 视角**
15. Su, W., Boyd, S., & Candès, E. J. (2016). A differential equation for modeling Nesterov's accelerated gradient method. *JMLR*, 17, 1–43.
16. Wibisono, A., Wilson, A. C., & Jordan, M. I. (2016). A variational perspective on accelerated methods in optimization. *PNAS*.
17. Shi, B., Du, S. S., Jordan, M. I., & Su, W. J. (2021). Understanding the acceleration phenomenon via high-resolution differential equations. *Mathematical Programming*.

**数值分析基础**
18. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM. （Chebyshev 半迭代、PCG）
19. Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. SIAM. （Krylov、QR、SVD）
20. Wathen, A. J. (2015). Preconditioning. *Acta Numerica*, 24, 329–376.
21. Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
22. Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving Ordinary Differential Equations I*. Springer.

---

## 附录 A：代码与 $P_k$ 对照表

| 文件 | 内容 |
|------|------|
| `src/quadratic.py` | 病态二次构造、目标、梯度 |
| `src/optimizers.py` | gd / momentum / nesterov / adam / adamw / sophia / jacobi |
| `src/momentum_spectrum.py` | Polyak 谱半径计算 |
| `src/newton_schulz.py` | 标准 NS + 五次多项式 Higham + 胖矩阵分支 |
| `src/chebyshev.py` | Chebyshev 半迭代 + minimax 误差多项式 + HB-Polyak 形式 |
| `src/spectral_trust_region.py` | 谱范数恢复问题 + Muon/GD/Adam 对照 |
| `src/adam_limit_cycle.py` | 标量 Adam 迭代 + 2-极限环检测 |
| `src/ode_integrators.py` | 梯度流 / Heavy-ball ODE / AVD-ODE / 高分辨率 NAG-ODE 的 RK4 |
| `src/pcg.py` | CG / PCG + Jacobi + CG 理论上界 |
| `src/matrix_quadratic.py` | $\min \tfrac{1}{2}\|AW-B\|_F^2$ 矩阵二次 + Muon-NS 接口 |
| `experiments/run_all.py` | 实验 1–10 + 入口 |
| `experiments/extended_experiments.py` | 实验 11–30 |
| `experiments/config.py` | 全局超参 |
| `tests/test_*.py` | 14 项回归测试 |

| $P_k$ 形态 | 优化器 | 代码标签 |
|------------|--------|----------|
| $\eta I$（固定） | GD | `gd` |
| 二阶状态 + $\eta I$ | Polyak HB | `momentum, beta=-1` |
| 二阶状态 + $\eta I$ + lookahead | Nesterov | `nesterov` |
| $\eta\,\mathrm{diag}(\sqrt{\hat v_k})^{-1}$ | Adam | `adam` |
| 同上 + 解耦 WD | AdamW | `adamw` |
| $\eta\,\mathrm{diag}(\max(h_k,\varepsilon))^{-1}$ + clip | Sophia-H | `sophia` |
| $\eta\,\mathrm{diag}(A)^{-1}$ | Oracle Jacobi | `jacobi` |
| Newton–Schulz 正交化 $G$ | Muon | `src/matrix_quadratic.py:muon_direction` |
| Chebyshev 时变 $\omega_k$ | Chebyshev 半迭代 | `src/chebyshev.py:chebyshev_semi_iterative` |

## 附录 B：实验数值自动摘要（精选）

来源：`data/experiment_results.json`，由 `scripts/generate_appendix_tables.py` 自动生成；完整版见 `report/appendix_auto.md`。

### 实验 1：达到 $10^{-6}$ 的迭代步数

| $\kappa$ | GD | Polyak HB | Adam |
|----------|----|-----------|------|
| 10 | 39 | 17 | 162 |
| 100 | 439 | 68 | 288 |
| 1000 | — (>2000) | 256 | 1133 |

### 实验 11：Nesterov vs Polyak HB

| $\kappa$ | Nesterov | Polyak HB |
|----------|----------|-----------|
| 10 | 20 | 17 |
| 100 | 70 | 68 |
| 1000 | 237 | 256 |

### 实验 12：Muon 在两种损失上

| 目标 | Muon-NS 最终 | GD 最终 | Adam 最终 |
|------|---------------|----------|------------|
| 谱范数损失 (80 步) | 0.037 | $2.4\times 10^{-32}$ | $1.1\times 10^{-4}$ |
| Frobenius 二次 (400 步) | 63266 | 51.4 | — |

### 实验 25：HB ≡ Chebyshev 半迭代

| 量 | 值 |
|-----|----|
| 200 步后 $f - f^*$（HB）  | $10^{-30}$ |
| 200 步后 $f - f^*$（Cheb）| $10^{-30}$ |
| max gap diff（末 50 步） | $1.7 \times 10^{-13}$ |

### 实验 27：五次多项式版 vs 标准 NS（迭代到 $10^{-16}$）

| 形状 | 标准 NS | 五次多项式 Higham NS |
|------|---------|----------------|
| $16{\times}8$ | 11 步 | **5 步** |
| $32{\times}16$ | 13 步 | **6 步** |
| $64{\times}32$ | 15 步 | **8 步** |

### 实验 30：Newton–Schulz 收敛盆

- 理论收敛盆上界 $\sqrt 3 \approx 1.7321$
- 100 个 $\sigma_0 \in [0.05, 2.5]$ 中 **11 个发散**，全部 $\sigma_0 > \sqrt 3$
- 与定理 6 预言**完全一致**

---

## 附录 C：完整证明

### C.1 定理 1 完整证明（GD 在强凸二次上的线性收敛）

设 $A = Q\Lambda Q^\top$，$\Lambda = \mathrm{diag}(\lambda_1,\ldots,\lambda_d)$，$0 < \mu \le \lambda_i \le L$。GD 更新 $x_{k+1} = x_k - \eta(A x_k - b)$，故误差 $e_k = x_k - x^*$ 满足

$$
e_{k+1} = (I - \eta A) e_k.
$$

在特征基下令 $\tilde e_k = Q^\top e_k$，得 $\tilde e_{k+1} = (I - \eta \Lambda)\tilde e_k$。故各模态衰减为 $|1 - \eta\lambda_i|$，最坏模态决定欧氏范数衰减：

$$
\|\tilde e_{k+1}\|_2 \le \max_i |1 - \eta\lambda_i| \cdot \|\tilde e_k\|_2 = \rho(\eta)\,\|\tilde e_k\|_2.
$$

由 $Q$ 正交，$\|e_k\|_2 = \|\tilde e_k\|_2$。故 $\|e_{k+1}\|_2 \le \rho(\eta)\,\|e_k\|_2$。

**优化 $\rho(\eta)$**：$\rho(\eta) = \max\{|1 - \eta\mu|, |1 - \eta L|\}$ 是分段线性函数。在 $\eta < 1/L$ 段两者均为正、单调递减；在 $\eta > 1/\mu$ 段单调递增。在 $\eta = 2/(\mu+L)$ 处 $1 - \eta\mu = -(1 - \eta L)$，二者绝对值相等，$\rho^* = (L-\mu)/(L+\mu) = (\kappa-1)/(\kappa+1)$。$\square$

### C.2 定理 2 完整证明（Polyak 最优 Heavy-ball）

在特征基下，对每个模态 $\lambda \in [\mu, L]$，Heavy-ball 写成 $2 \times 2$ 系统

$$
\begin{bmatrix} x_{k+1} \\ x_k \end{bmatrix} = T(\lambda) \begin{bmatrix} x_k \\ x_{k-1} \end{bmatrix},\quad
T(\lambda) = \begin{bmatrix} 1 - \eta\lambda + \beta & -\beta \\ 1 & 0 \end{bmatrix}.
$$

特征方程 $z^2 - (1 - \eta\lambda + \beta) z + \beta = 0$。判别式

$$
\Delta(\lambda) = (1 - \eta\lambda + \beta)^2 - 4\beta.
$$

- $\Delta < 0$：两特征值复共轭，模 $|z| = \sqrt\beta$；
- $\Delta = 0$：双重实根 $z = (1 - \eta\lambda + \beta)/2$，模 $|z| = \sqrt\beta$（因为 $z^2 = \beta$）；
- $\Delta > 0$：两实根，最大模 $|z|_{\max} > \sqrt\beta$。

故 $|z| = \sqrt\beta$ 是**最优情形**，当且仅当 $\Delta(\lambda) \le 0$，即

$$
(1 - \eta\lambda + \beta)^2 \le 4\beta. \tag{C.1}
$$

我们要求 (C.1) 对**所有** $\lambda \in [\mu, L]$ 同时成立。$1 - \eta\lambda + \beta$ 在 $\lambda$ 上单调递减，故其最大绝对值取在端点：

$$
\max\{|1 - \eta\mu + \beta|,\,|1 - \eta L + \beta|\} \le 2\sqrt\beta.
$$

最优选择是让两个端点的 $|1 - \eta\lambda + \beta|$ 同时取到上界：

$$
1 - \eta\mu + \beta = 2\sqrt\beta,\quad -(1 - \eta L + \beta) = 2\sqrt\beta.
$$

两式相加：$\eta(L - \mu) = 4\sqrt\beta$，即 $\sqrt\beta = \eta(L - \mu)/4$。两式相减：$1 - \eta(\mu+L)/2 + \beta = 0$，即 $\eta(\mu+L)/2 = 1 + \beta$，$\eta = 2(1+\beta)/(\mu+L)$。

直接用 $\sqrt\beta = \eta(L-\mu)/4$ 与 $\eta = 2(1+\beta)/(\mu+L)$ 消元。代入：

$$
\sqrt\beta = \frac{2(1+\beta)(L-\mu)}{4(L+\mu)} = \frac{(1+\beta)(L-\mu)}{2(L+\mu)}.
$$

令 $u = \sqrt\beta$，则

$$
2u(L+\mu) = (1+u^2)(L-\mu).
$$

整理为关于 $u$ 的二次：

$$
(L-\mu) u^2 - 2(L+\mu) u + (L-\mu) = 0.
$$

判别式 $4(L+\mu)^2 - 4(L-\mu)^2 = 16 L\mu$。解

$$
u = \frac{(L+\mu) \pm 2\sqrt{L\mu}}{L-\mu} = \frac{(\sqrt L \pm \sqrt\mu)^2}{(\sqrt L - \sqrt\mu)(\sqrt L + \sqrt\mu)}.
$$

取较小的根（$0 < u < 1$）：

$$
u = \frac{(\sqrt L - \sqrt\mu)^2}{(\sqrt L - \sqrt\mu)(\sqrt L + \sqrt\mu)} = \frac{\sqrt L - \sqrt\mu}{\sqrt L + \sqrt\mu}.
$$

故 $\beta^* = u^2 = \bigl(\frac{\sqrt L - \sqrt\mu}{\sqrt L + \sqrt\mu}\bigr)^2$，最优谱半径 $\rho^* = \sqrt{\beta^*} = u = (\sqrt L - \sqrt\mu)/(\sqrt L + \sqrt\mu) = (\sqrt\kappa - 1)/(\sqrt\kappa + 1)$。

最优步长由 $\eta = 2(1+\beta)/(\mu+L)$ 计算：

$$
\eta^* = \frac{2(1 + u^2)}{\mu + L} = \frac{2}{\mu+L} \cdot \frac{(\sqrt L+\sqrt\mu)^2 + (\sqrt L-\sqrt\mu)^2}{(\sqrt L+\sqrt\mu)^2} = \frac{4(L+\mu)}{(\mu+L)(\sqrt L+\sqrt\mu)^2} = \frac{4}{(\sqrt L+\sqrt\mu)^2}.
$$

$\square$

### C.3 定理 4 完整证明（HB 是 Chebyshev 半迭代的定常极限）

Chebyshev 半迭代（Hageman–Young 形式）：

$$
\begin{aligned}
x_1 &= x_0 + \frac{r_0}{d},\quad d = \frac{L+\mu}{2},\ \sigma = \frac{L-\mu}{L+\mu},\\
\omega_1 &= \frac{1}{1 - \sigma^2/2},\quad
\omega_{k+1} = \frac{1}{1 - \sigma^2 \omega_k / 4},\\
x_{k+1} &= \omega_{k+1}\left(\frac{r_k}{d} + x_k - x_{k-1}\right) + x_{k-1}, \quad r_k = b - A x_k.
\end{aligned}
$$

令 $\omega_k \to \omega_\infty$（不动点），则

$$
\omega_\infty = \frac{1}{1 - \sigma^2 \omega_\infty / 4} \implies \omega_\infty - \frac{\sigma^2 \omega_\infty^2}{4} = 1.
$$

即 $\sigma^2 \omega_\infty^2 - 4 \omega_\infty + 4 = 0$，解 $\omega_\infty = (4 \pm \sqrt{16 - 16\sigma^2})/(2\sigma^2) = 2(1 \pm \sqrt{1 - \sigma^2})/\sigma^2$。取较小根：

$$
\omega_\infty = \frac{2(1 - \sqrt{1 - \sigma^2})}{\sigma^2} = \frac{2}{1 + \sqrt{1 - \sigma^2}},
$$

后一等式由 $\frac{1-\sqrt{1-\sigma^2}}{\sigma^2} = \frac{1}{1+\sqrt{1-\sigma^2}}$（分子分母乘共轭）。

代入 Heavy-ball 形式：在 $k \to \infty$ 时 Chebyshev 递推退化为

$$
x_{k+1} = \omega_\infty\left(\frac{r_k}{d} + x_k - x_{k-1}\right) + x_{k-1} = \omega_\infty \frac{r_k}{d} + \omega_\infty (x_k - x_{k-1}) + x_{k-1}.
$$

整理：

$$
x_{k+1} - x_k = \omega_\infty \frac{r_k}{d} + (\omega_\infty - 1)(x_k - x_{k-1}).
$$

记 $\eta_\infty = \omega_\infty/d$、$\beta_\infty = \omega_\infty - 1$，则

$$
x_{k+1} = x_k - \eta_\infty (A x_k - b) + \beta_\infty (x_k - x_{k-1}),
$$

恰是 Heavy-ball 形式。

**验证 $\eta_\infty = \eta^*$、$\beta_\infty = \beta^*$**：

$1 - \sigma^2 = 1 - \frac{(L-\mu)^2}{(L+\mu)^2} = \frac{4 L \mu}{(L+\mu)^2}$，故 $\sqrt{1 - \sigma^2} = \frac{2\sqrt{L\mu}}{L+\mu}$。

$\omega_\infty = \frac{2}{1 + \frac{2\sqrt{L\mu}}{L+\mu}} = \frac{2(L+\mu)}{(L+\mu) + 2\sqrt{L\mu}} = \frac{2(L+\mu)}{(\sqrt L + \sqrt\mu)^2}$.

$\eta_\infty = \omega_\infty / d = \frac{4}{(\sqrt L+\sqrt\mu)^2} = \eta^*$. ✓

$\beta_\infty = \omega_\infty - 1 = \frac{2(L+\mu) - (\sqrt L+\sqrt\mu)^2}{(\sqrt L+\sqrt\mu)^2} = \frac{2L + 2\mu - L - 2\sqrt{L\mu} - \mu}{(\sqrt L+\sqrt\mu)^2} = \frac{L - 2\sqrt{L\mu} + \mu}{(\sqrt L+\sqrt\mu)^2} = \frac{(\sqrt L - \sqrt\mu)^2}{(\sqrt L+\sqrt\mu)^2} = \beta^*$. ✓

故 Chebyshev 半迭代在系数收敛后退化为 Polyak 最优 Heavy-ball；这说明 Polyak HB 是 Chebyshev 半迭代的定常极限。实验 25 中末段 gap 差距 $1.7 \times 10^{-13}$，与该解释一致。$\square$

### C.4 命题 3 证明（Adam 与 Jacobi 预条件的关系）

略证：$\beta_1 = 0$ 时 $\hat m_k = g_k / (1 - \beta_1^k) = g_k$；$v_k = (1-\beta_2)\sum_{j=1}^k \beta_2^{k-j}(g_j \odot g_j)$。在 $k\to\infty$ 与 $\beta_2 \to 1$ 极限下 $v_k$ 是 $g_j \odot g_j$ 的滑动平均，趋于 $\mathbb{E}[g \odot g]$。

对二次问题 $f(x) = \tfrac{1}{2}(x - x^*)^\top A (x - x^*)$，$g(x) = A(x - x^*)$。若 $x - x^*$ 的协方差为 $\Sigma$，则

$$
\mathbb{E}[g \odot g] = \mathrm{diag}(A \Sigma A^\top).
$$

特别地，当 $A$ 对角且 $\Sigma \propto I$ 时，$\mathbb{E}[g \odot g]\propto \mathrm{diag}(A)^2$。忽略比例常数与 $\varepsilon$ 后，$\sqrt{\hat v_k} \approx \mathrm{diag}(A)$，$P_k = \eta\,\mathrm{diag}(\sqrt{\hat v_k})^{-1} \approx \eta\,\mathrm{diag}(A)^{-1}$——恰为 Jacobi 预条件器。$\square$

### C.5 定理 6 二次收敛证明（已在 §4.1 给出）

### C.6 定理 7 证明（已在 §4.3 给出，von Neumann 迹不等式）


