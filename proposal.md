---

# Proposal（选题申请）

---

## 选题 1：PageRank 的数值分析解释

### 一、预调研内容

#### 背景与问题
PageRank 是 Google 搜索引擎的核心算法，由 Brin & Page 于 1998 年提出。其数学本质是在一个巨大的有向图（网页超链接网络）上求列随机矩阵的主特征向量，而求解方法正是数值线性代数中经典的**幂迭代法（power iteration）**。这个问题天然地把“课本上的特征值迭代算法”和“现实世界中最著名的工程应用”连接起来，是数值分析课程 project 的理想选题。

#### 核心数学链条
整个问题的数学脉络清晰：

1.  **第一步：从网络到矩阵**
    给定 $n$ 个网页的超链接图，构造超链接矩阵 $H$，对每列归一化得到列随机矩阵 $P$（每列之和为 1）。处理“悬空节点”（dangling nodes，出度为零）后，再引入 teleportation 项，构造 **Google 矩阵**：
    $$G = \alpha P + (1 - \alpha)ev^T, \quad \alpha = 0.85$$
2.  **第二步：Perron-Frobenius 定理保证唯一性**
    $G$ 是正矩阵（每个元素大于零），因此满足 Perron-Frobenius 定理的条件：
    *   $G$ 有唯一最大实特征值 $\lambda_1 = 1$
    *   对应的特征向量各分量均为正，即 PageRank 向量 $\pi > 0$
    *   次大特征值满足 $|\lambda_2| < 1$
    可以证明，若 $P$ 的特征值为 $1, \lambda_2, \dots, \lambda_n$，则 Google 矩阵 $G = \alpha P + (1 - \alpha)ve^T$ 的特征值为 $1, \alpha\lambda_2, \dots, \alpha\lambda_n$。这个漂亮的谱结果直接给出了阻尼因子与收敛速度的精确关系。
3.  **第三步：幂迭代求主特征向量**
    幂迭代法从任意初始向量 $x^{(0)}$ 出发，重复计算 $x^{(k+1)} = Gx^{(k)} / \|Gx^{(k)}\|$，收敛速率由谱间隔 $|\lambda_2/\lambda_1|$ 决定，误差以几何级数衰减。
4.  **第四步：阻尼因子 $\alpha$ 的权衡**
    由于 Google 取 $\alpha = 0.85$，故 $|\lambda_2(G)| = \alpha = 0.85$，因此 $\alpha^{50} \approx 0.000296$，意味着约 50 步迭代即可达到足够精度——这与 Brin & Page 报告的实验结果吻合。$\alpha$ 越大，次大特征值越接近 1，收敛越慢；$\alpha$ 越小，矩阵越接近均匀分布，偏离真实链接结构。这是工程与数学的典型权衡。

#### 与课内知识的联系

| 课内内容 | PageRank 对应 |
| :--- | :--- |
| 特征值问题 | $G\pi = \lambda\pi$ |
| 幂迭代法 | 主 PageRank 迭代 |
| 谱半径与收敛 | $\lambda_2$ |
| 矩阵条件数 | 条件数与谱间隔倒数 |
| 不动点迭代 | PageRank = Markov 链平稳分布 |

#### 可以证明的核心定理
1.  **Perron-Frobenius 定理（正矩阵情形）**：完整证明可在约 1.5 页内完成，利用紧集连续函数有最大值 + 反证法。
2.  **Google 矩阵特征值公式**：$\lambda_k(G) = \alpha\lambda_k(P), k \ge 2$，直接证明只需 2-3 步代入计算。
3.  **幂迭代收敛率定理**：$\|x^{(k)} - \pi\| \le C \cdot |\lambda_2|^k$，用特征分解展开即可推导。

### 二、研究计划（6 周）

*   **第 1 周：建模**
    *   从有向图邻接矩阵出发，逐步构造超链接矩阵 $H \to$ 列随机矩阵 $P \to$ Google 矩阵 $G$
    *   处理悬空节点（将其补为均匀出链）
    *   手工构造 5 节点例子完整演示一遍迭代过程
*   **第 2 周：理论 I——Perron-Frobenius**
    *   陈述并完整证明 Perron-Frobenius 定理（正矩阵情形）
    *   区分正矩阵、非负不可约矩阵、本原矩阵的条件
    *   证明 Google 矩阵满足本原性条件
*   **第 3 周：理论 II——幂迭代收敛分析**
    *   幂迭代算法的误差递推公式推导
    *   谱间隔 $1 - |\lambda_2|$ 与收敛步数的定量关系
    *   证明 Google 矩阵特征值公式 $\lambda_k(G) = \alpha\lambda_k(P)$
*   **第 4 周：数值实验**
    *   Python 实现幂迭代，与 `networkx.pagerank` 结果对比
    *   在 20-50 节点随机有向图上测试 $\alpha \in \{0.70, 0.80, 0.85, 0.90, 0.95\}$ 的收敛曲线
    *   绘制 "$\alpha$ vs 迭代次数" 和 "谱间隔 vs 收敛速度" 图
    *   可选：用 Karate Club 网络数据集增加真实感
*   **第 5 周：写作**
    *   按结构撰写报告：引言 $\to$ 建模 $\to$ 定理与证明 $\to$ 算法 $\to$ 实验 $\to$ 结论
    *   目标页数 8 页左右（五号字）
*   **第 6 周：修改与完善**
    *   检查定理证明的严格性
    *   补充图表和示例
    *   完善参考文献格式

### 参考文献（8 篇，均超课本范围）
1. Brin, S. & Page, L. (1998). The anatomy of a large-scale hypertextual web search engine.
2. Bryan, K. & Leise, T. (2006). The \$25,000,000,000 eigenvector: the linear algebra behind Google. SIAM Review.
3. Langville, A. N. & Meyer, C. D. (2006). Google's PageRank and Beyond. Princeton UP.
4. Horn, R. A. & Johnson, C. R. (2013). Matrix Analysis (2nd ed.).（Perron-Frobenius 章节）
5. Ipsen, I. C. F. & Kirkland, S. (2006). Convergence analysis of a PageRank updating algorithm.
6. Gleich, D. F. (2015). PageRank beyond the web. SIAM Review.
7. Golub, G. H. & Van Loan, C. F. (2013). Matrix Computations (4th ed.).（幂法章节）
8. arXiv:2403.05198 — PageRank Power Iteration Convergence and Damping Factor Analysis.

### 三、给助教的邮件
**主题：数值分析 Project 选题申请——PageRank 的幂迭代收敛分析**

助教您好，
我是[姓名]，学号[学号]，想申请以"PageRank 的数值分析解释：幂法与谱理论"作为期末 Project 选题，写信简要说明一下选题动机和大致内容，希望您审阅。

选这个题目的起因是学完幂迭代法那一章之后，查资料发现 Google 的 PageRank 算法本质上就是在一个巨大的列随机矩阵上做幂迭代，求主特征向量。觉得这个联系很有意思——课本上讲的是一个相对抽象的数值方法，但它背后对应的是一个真实工程系统里每天在跑的东西。所以想以此为切入点，把课内的特征值迭代理论和 PageRank 的实际收敛行为连起来做一个系统的分析。

主要内容计划包括四个部分：
1. **建模**：从有向图邻接矩阵出发，逐步构造 Google 矩阵 $G = \alpha P + (1 - \alpha)ev^T$。
2. **Perron-Frobenius 定理**：证明主特征向量的唯一存在性。
3. **收敛率分析**：证明 $|\lambda_2(G)| = \alpha$ 及其对收敛速度的影响。
4. **数值实验**：用 Python 验证不同 $\alpha$ 下的收敛表现。

报告预计 8 页左右，会包含定理证明、算法伪代码、实验图表和手工小例子的演示。麻烦助教看一下这个方向是否合适，如有建议欢迎指正。谢谢！

---

## 选题 2：从经典迭代法到现代优化器的统一数值分析

### 一、预调研内容

#### 背景与问题
现代深度学习中最常用的优化器（Adam、AdamW、Sophia、Muon）在机器学习社区通常被视为"调参工具"，鲜少从数值分析角度加以审视。本选题的出路点是：**把现代优化器重新解读为数值迭代算法**，通过统一的迭代模板 $x_{k+1} = x_k - P_k g_k$ 将 GD、Momentum、Adam、Sophia、Muon 纳入同一框架。

#### 四类机制与数值分析对应
1.  **机制一：梯度下降 = 定常迭代 / Euler 离散**
    梯度流 $\dot{x} = -\nabla f(x)$ 的前向 Euler 离散就是 GD。在强凸函数上，GD 等价于针对 $Ax = b$ 的 Richardson 迭代，收敛率 $1 - \mu/L$ 直接由条件数 $\kappa = L/\mu$ 决定。
2.  **机制二：Momentum = 二阶 ODE 离散 + 谱加速**
    Heavy-ball 方法对应带阻尼的谐振子 ODE。Nesterov 加速可被推导为 $X'' + (3/t)X' + \nabla f(X) = 0$ 的离散化。谱半径分析显示其能将收敛率从 $(\kappa-1)/(\kappa+1)$ 降至 $(\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$。
3.  **机制三：Adam = 动态 Jacobi 预条件**
    Adam 的更新等价于：$x_{k+1} = x_k - \text{diag}(\hat{v}_k)^{-1/2} \cdot \hat{m}_k$。这和 Jacobi 预条件思想一致，只是预条件矩阵是根据梯度历史动态估计的。
4.  **机制四：Muon = Newton-Schulz 极分解迭代 + 谱范数 trust-region**
    Muon 优化器的核心是 Newton-Schulz 迭代：$X_{k+1} = \frac{1}{2}X_k(3I - X_k^T X_k)$。这是矩阵极分解的经典数值方法，具有二次收敛性。

#### 主要定理与证明难度估计

| 定理 | 证明难度 | 所需工具 |
| :--- | :--- | :--- |
| GD 在强凸函数上的线性收敛 | 初级 | Lipschitz 连续梯度不等式 |
| Momentum 最优谱半径 | 中级 | 伴随矩阵特征值分析 |
| Adam 等价于动态 Jacobi 预条件 | 中级 | 矩阵范数与预条件定义 |
| Newton-Schulz 迭代二次收敛 | 中级 | Newton 法局部分析 |
| 极分解 = 谱范数最速下降 | 高级 | 矩阵微分与凸分析 |

### 二、研究计划（8 周）

*   **第 1-2 周：理论框架与 GD / Momentum 分析**
    *   推导统一迭代模板，建立 $P_k$ 对应表
    *   完整证明 GD 在强凸函数上的线性收敛
    *   对二次函数进行 Momentum 谱半径推导
*   **第 3 周：Adam 的预条件分析**
    *   从 Jacobi 预条件出发，推导条件数改善公式
    *   将 Adam 写成动态 Jacobi 预条件形式
*   **第 4 周：Muon 与矩阵迭代**
    *   Newton-Schulz 迭代收敛证明（含收敛域分析）
    *   证明极分解等价于谱范数 trust-region 的最优解
*   **第 5-6 周：数值实验**
    *   病态二次函数实验（$\kappa \in \{10, 100, 1000\}$）：对比收敛轨迹
    *   Newton-Schulz 正交化精度实验
    *   有效预条件条件数 $\kappa(P_k^{-1} A)$ 随迭代步的演化
*   **第 7-8 周：写作与修改**
    *   全文写作，目标 10-12 页，完善图表

### 参考文献（12 篇核心）
1. Kingma & Ba (2015). Adam.
2. Liu, H. et al. (2023). Sophia.
3. Jordan, K. et al. (2024). Muon Optimizer. GitHub.
4. Kovalev, D. (2025). Gradient orthogonalization as trust-region optimization.
5. Su, W., Boyd, S., Candès, E. J. (2016). ODE for Nesterov's method. JMLR.
... (其余见截图列表)

### 三、给助教的邮件
**主题：数值分析 Project 选题申请——现代优化器的统一数值分析框架**

助教您好，
我是赵泽霖（学号2023010848），我与史云天同学（学号2023010836）一起组队完成本次数值分析 project。我们想申请以"经典迭代法与现代优化器的统一框架：预条件与收敛性分析"作为选题。

具体来说，我们想用统一迭代模板 $x_{k+1} = x_k - P_k g_k$ 来整理几类方法：
*   **GD** 对应 Richardson 迭代。
*   **Momentum** 对应二阶阻尼系统离散化。
*   **Adam** 本质是动态 Jacobi 预条件。
*   **Muon** 核心操作是 Newton-Schulz 矩阵迭代。

我们希望从谱性质角度给出解释，并以病态二次函数为例进行系统比较。如条件允许，我们也希望能针对小规模网络进行实验。请助教审阅。谢谢！

赵泽霖
2023010848