# 项目后续 TODO（2026-05-19 · 第三轮）

> **选题 2**：经典迭代法与现代优化器的统一数值分析  
> **完成对照（详细）**：[`re_TODO.md`](re_TODO.md)（含第二轮 E21/E24/κ 扫描、附录 C/D、`check_repro.sh` 等）  
> **本文件定位**：项目已接近**可提交**；下文仅列**仍未完成**、**本地需你操作**、以及**冲优秀/未来工作**时可选项。

---

## 一、当前状态（第三轮快照）

相对 `proposal.md` 与初版 TODO，**代码—实验—报告主链条已闭合**：

| 维度 | 状态 | 要点 |
|------|------|------|
| 优化器 | ✅ | `gd` / `momentum` / `nesterov` / `adam` / `adamw` / `sophia` / `jacobi` |
| 矩阵 Muon | ✅ | E12 + §3.4 trust-region 文字；Frobenius 不单调已在摘要/结论说明 |
| 实验 | ✅ | 1–10 + E11–E17 + E21 + E24 + E23 + `exp1b` + `multiseed` + **κ 连续扫描** |
| 工程 | ✅ | `config.py`、`requirements.txt`（带上界）、7 项 pytest、`scripts/check_repro.sh` |
| 报告 | ✅ | 摘要、AdamW §3.3、§4.12 扩展、**附录 C**（定理 1）、**附录 D**（答辩图+话术）、参考文献 12 篇 |
| 图表 | ✅ | 设计约 **26** 张 → `figures/`（以 `bash scripts/check_repro.sh` 为准） |

**尚未自动化**：`report/report.pdf`（本环境无 pandoc，需你本机导出）。

---

## 二、提交前必做（仅你本地可完成）

| 优先级 | 事项 | 命令 / 位置 |
|--------|------|-------------|
| **P0** | 一键复现通过 | `bash scripts/check_repro.sh` |
| **P0** | 导出 PDF | `pandoc report/report.md -o report/report.pdf --resource-path=.:report`（无 pandoc 则 `sudo apt install pandoc` 或用 Typora/VS Code 导出） |
| **P0** | 核对 PDF 内公式与图 | 重点：§4 各图、`exp12`/`exp21`/`exp_kappa_scan` 是否显示 |
| **P1** | 答辩 1 页 slide | 素材见 `report/report.md` **附录 D**（`exp1b` + `exp6` + `exp12` 三图一句） |
| **P2** | （可选）环境锁 | `conda env export --no-builds > environment.lock.yml` 随作业打包 |

完成以上即可视为**课程 project 交付闭环**。

---

## 三、仍可改进（非阻塞，按收益排序）

### 3.1 报告与答辩（不改代码）

- [ ] **附录 B 嵌入自动表**：将 `report/appendix_auto.md` 关键表粘贴进 `report.md` 附录 B，或提交时 PDF 附录附上该文件，避免助教只看正文而缺数值。
- [ ] **通读一遍「局限」话术**：结论 §5 + 附录 D + 本文 §五；预备追问：Adam $\kappa_{\mathrm{eff}}$、$\kappa=1000$、Muon Frobenius、Sophia 简化、PCG 更快。
- [ ] **页数/格式**：若课程要求 10–12 页五号字，用 PDF 页数统计；过长可把 E21/E24 图缩为半页或移附录。
- [ ] **双人分工说明**（若需）：README 或报告扉页注明谁负责理论/实验/写作（可选）。

### 3.2 实验深化（冲优秀 / 未来工作，**均未做**）

| 编号 | 内容 | 工作量 | 建议 |
|------|------|--------|------|
| **E16** | Chebyshev 半迭代 vs Heavy-ball | 高 | 写入报告「未来工作」即可；实现需 1 页推导 + 代码 |
| **E22** | $(\beta,\eta)$ 平面 $\rho$ 热力图 | 中 | 比实验 7 更直观；有 0.5 天可补 |
| **E12+** | 谱范数 / trust-region toy 矩阵损失 | 高 | 使 Muon 在目标上与几何一致；当前 trust-region 文字已够答辩 |
| **E18–E20** | MLP / MNIST / PyTorch Muon | 高 | **仅当课程强制 DL**；否则不必做 |

### 3.3 工程（低优先级）

| 项 | 说明 |
|----|------|
| 实验 10 等高线向量化 | 80×80 足够；仅当频繁重跑再改 |
| `environment.lock.yml` | 助教机复现失败时再补 |
| GitHub Actions | 无远程仓库可忽略 |
| 再拆 `expXX.py` | 维持 `extended_experiments.py` 即可 |

---

## 四、若答辩后被追问 — 可快速补强的方向

以下**不必现在做**，按提问类型选做：

1. **「Adam 和 AdamW 差多少？」**  
   已有 `exp1b` + §3.3；可口头强调：decoupled WD 不改变 $P_k$ 对角结构，只改有效目标。

2. **「为什么不用 Chebyshev？」**  
   答：与 Heavy-ball 同属加速族，E16 留作扩展；PCG（E17）已给 Krylov 上限。

3. **「模态实验说明什么？」**  
   指 E21：Polyak 使 $\mu,L$ 模态衰减率接近；GD 慢模态主导。

4. **「κ 扫描和实验 2 关系？」**  
   `exp_kappa_scan` 连迭代数与理论 $\rho$；实验 2 连谱半径公式。

5. **「能否演示非二次？」**  
   诚实答：聚焦可控二次模型；E14 已补噪声；Rosenbrock 可作未来工作一句。

---

## 五、已知局限（答辩直接用，无需改代码）

| 现象 | 一句话 |
|------|--------|
| $\kappa_{\mathrm{eff}}>\kappa(A)$ | Adam 早期预条件未稳定 |
| Adam $\kappa=1000$ 600 步未 tol | 对角预条件 + lr 敏感（E13） |
| Muon-NS 在 Frobenius 二次上 | 几何更新 ≠ Frobenius 最速下降 |
| Sophia | 教学简化版，非 Sophia-G |
| PCG vs Adam 迭代数 | Krylov 最优 vs 一阶实用 |

全文展开见 `report/report.md` 结论与附录 D。

---

## 六、快速勾选（提交当周）

- [x] `bash scripts/check_repro.sh`
- [x] 参考文献 12 篇 + AdamW 段落 + 附录 C 定理 1
- [x] 摘要 Muon / Adam 高 κ 表述
- [x] E21 + E24 + κ 扫描 + exp1b(含 AdamW)
- [x] 附录 D 答辩素材（报告内）
- [ ] **`report/report.pdf` 导出**
- [ ] **答辩 PPT（1 页，自附录 D）**
- [ ] （可选）`environment.lock.yml`
- [ ] （可选）附录 B 粘贴 `appendix_auto.md` 核心表

---

## 七、与 `re_TODO.md` 的分工

| 文件 | 用途 |
|------|------|
| **`re_TODO.md`** | 历史与第二轮逐项 [x]/[ ] 审计、新增文件列表、不合理预期说明 |
| **`TODO.md`（本文件）** | **当前还能做什么**： essentially PDF + PPT + 可选加分 |

---

*第三轮更新：核心开发已结束；后续以本地 PDF/PPT 与答辩准备为主。完成 §六 剩余项后可将对应 `[ ]` 改为 `[x]`。*
