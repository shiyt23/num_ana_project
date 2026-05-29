# 答辩一页备忘（可直接做成 PPT）

**题目**：正交化梯度方法的数值分析视角：从 Newton–Schulz 极分解到 Muon 优化器及其统一框架
**作者**：赵泽霖（2023010848）、史云天（2023010836）

---

## 核心信息（30 秒）

统一模板 $x_{k+1} = x_k - P_k g_k$：GD = Richardson；HB / Nesterov = 谱加速；Adam = 动态 Jacobi；**Muon = Newton–Schulz 极分解 = 谱范数 trust-region 最速下降**（核心）。

---

## 图 1：经典—Heavy-ball ≡ Chebyshev 半迭代

![exp25](../figures/exp25_hb_chebyshev.png)

**一句话**：Polyak 1964 年的 Heavy-ball 在二次问题上与 Chebyshev 半迭代渐近等价（max gap diff $= 1.14\times 10^{-13}$），即"动量加速 = 课内 Chebyshev minimax"。完整证明在附录 C.3。

---

## 图 2：核心—Muon 在谱范数 trust-region 上严格下降

![exp28](../figures/exp28_muon_trust_region.png)

**一句话**：5 个种子 Muon 都从 1.08 下降到 0.028~0.037 的稳定带；Muon 优化的不是 Frobenius 几何，而是谱范数几何（定理 7：von Neumann 迹不等式取等）。

---

## 图 3：极致—Newton–Schulz 收敛盆 $\sqrt 3$

![exp30](../figures/exp30_ns_basin.png)

**一句话**：100 个初值 $\sigma_0 \in [0.05, 2.5]$ 中恰好 11 个发散，全部 $\sigma_0 > \sqrt 3$；定理 6 的局部二次收敛盆从纸面变成肉眼可见的事实。

---

## 预备追问表

| 问题 | 答 |
|------|-----|
| 为什么选 Muon？ | Muon 是 2024 年最新优化器，核心是 Higham 教材里的极分解 NS 迭代——把"纯前沿"和"纯数值分析"对接起来 |
| HB 与 Chebyshev 怎么等价的？ | Chebyshev 半迭代时变 $\omega_k$ 不动点解出来正好是 $1 + \beta^*$，定常极限即 HB |
| Muon 在 Frobenius 上发散？ | 不是 bug：谱范数 trust-region 与 Frobenius 几何错配，定理 7 蕴含 |
| Adam 真的不收敛？ | Bock-Weiß 2022：最简凸 $f = x^2/2$ 上仍有 2-极限环；E26 我们复现到 $x^\pm = (0.025, -0.025)$ |
| 修了哪些实验？ | exp11 Nesterov、exp12 Muon、exp15 Sophia、exp23 胖矩阵 NS、exp24 ablation、exp_kappa_scan、exp4——见报告 §6.4 修复纪事 |
| 为什么不做 NN？ | 聚焦可控二次/矩阵恢复，理论闭环；NN 实验需 GPU 集群，超出 8 周课程范围 |

---

*图片路径相对于本文件；导出 PPT 时请从 `figures/` 插入同名 PNG。*
