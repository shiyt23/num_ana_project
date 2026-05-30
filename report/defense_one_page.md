# 答辩一页备忘（可直接做成 PPT）

**题目**：正交化梯度方法的数值分析视角：从 Newton–Schulz 极分解到 Muon 优化器及其统一框架
**作者**：赵泽霖（2023010848）、史云天（2023010836）

---

## 核心信息（30 秒）

统一模板 $x_{k+1} = x_k - P_k g_k$，**更进一步**：GD / signSGD / Muon = $\ell_2$ / $\ell_\infty$ / 谱范数下的**最速下降**（同一个 LMO，不同范数）。HB = Chebyshev 半迭代的冻结系数极限；Adam = 动态对角缩放（用梯度幅度而非曲率）；**Muon = Newton–Schulz 极分解 = 谱范数最速下降**（核心）。

---

## 图 1：经典—Heavy-ball 是 Chebyshev 半迭代的冻结系数极限

![exp25](../figures/exp25_hb_chebyshev.png)

**一句话**：Chebyshev 半迭代的时变系数 $\omega_k$ 单调收敛到 $1+\beta^* = 1.6694$（吻合 $10^{-7}$，中图）；Polyak Heavy-ball 就是把它冻结。即"动量加速 = 课内 Chebyshev minimax 的定常版本"。完整证明在附录 C.3。

---

## 图 2：核心—Muon 的奇异值均衡（结构性特征）

![exp31](../figures/exp31_muon_equalization.png)

**一句话**：让梯度条件数 $\kappa(G)$ 从 1 扫到 3162，Muon 更新 $-UV^\top$ 的条件数**恒为 1**（GD 更新继承 $\kappa(G)$）。这是 Muon 区别于 GD 的本质——**诚实说**：凸二次上 CG 最优、Muon 不更快，它的价值是这个结构性的"均衡"，在深度学习里有用。

---

## 图 3：极致—Newton–Schulz 收敛盆 $\sqrt 3$（三种归宿）

![exp30](../figures/exp30_ns_basin.png)

**一句话**：$(0,\sqrt3)$ 内 86 个初值全部收敛到**正确**极因子 $+U$；越过 $\sqrt3$ 后 21 个收敛到 $-U$（符号错）、13 个发散——盆是分形式的。定理 6 的局部分析从纸面变成肉眼可见。

---

## 预备追问表

| 问题 | 答 |
|------|-----|
| 为什么选 Muon？ | 2024 年最新优化器，核心是 Higham 教材里的极分解 NS 迭代——"纯前沿"对接"纯数值分析" |
| GD/signSGD/Muon 怎么统一？ | 都是 LMO $\arg\min_{\|d\|\le1}\langle g,d\rangle$，分别选 $\ell_2$/$\ell_\infty$/谱范数（定理 5，实验 32）|
| HB 与 Chebyshev 怎么等价？ | Chebyshev 时变 $\omega_k\to 1+\beta^*$，冻结系数即 HB（附录 C.3）|
| Muon 比 GD 快吗？ | **诚实：不**。凸二次上 CG 最优；Muon 价值是奇异值均衡（实验 31）+ 深度学习几何 |
| Adam = Jacobi 吗？ | **不等**。Adam 用梯度幅度、Jacobi 用曲率；对角 A 上 Adam $\kappa_{\rm eff}{\approx}12$、Jacobi$=1$（实验 4）|
| Adam 真的不收敛？ | Bock–Weiß 2022：最简凸 $f=x^2/2$ 上有 2-极限环；实验 26 复现到 $x^\pm=(0.025,-0.025)$ |
| 修了哪些问题？ | 一处真 bug（$\kappa_{\rm eff}$ 应为 $\kappa(P_kA)$）+ 多处过度声称 + 7 个不收敛实验——见 §7.4 |
| 为什么不做 NN？ | 聚焦可控二次/矩阵恢复，理论闭环；NN 需 GPU 集群，超出课程范围 |

---

*图片路径相对于本文件；导出 PPT 时请从 `figures/` 插入同名 PNG。*
