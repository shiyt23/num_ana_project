# TODO 完成情况对照（re_TODO）

> 对照根目录 **`TODO.md`（2026-05-19 · 第三轮）**  
> **更新日期**：2026-05-19（第三轮）  
> **定位**：第三轮 TODO 写明「核心开发已结束」；本文件记录**本轮 agent 可完成项**与**仍需你本地完成项**。

---

## 一、第三轮 TODO 在说什么

| `TODO.md` 章节 | 含义 |
|----------------|------|
| §一 | 主链条已闭合（优化器 / 实验 / 报告 / 26+ 图） |
| §二 | **仅本地**：PDF、PPT、核对 PDF |
| §三 | 非阻塞改进 + 冲优秀可选（E16/E22/…） |
| §四–五 | 答辩话术、局限表（已在报告内） |
| §六 | 快速勾选：多数 [x]，剩 PDF/PPT/可选 |

**与旧版 `re_TODO.md`（第一、二轮）关系**：历史审计（E11–E24、附录 C/D、check_repro 等）**仍然有效**；第三轮不再重复逐条罗列，见 Git 历史或 `data/experiment_results.json`。

---

## 二、§二 提交前必做 — 完成情况

| 优先级 | 事项 | Agent | 你本地 |
|--------|------|-------|--------|
| **P0** | `bash scripts/check_repro.sh` | **[x]** 已通过（7 pytest，**27** 图） | 提交前再跑一遍 |
| **P0** | 导出 `report/report.pdf` | **[ ]** 无 pandoc，无法 sudo 安装 | **请你执行**（见下） |
| **P0** | 核对 PDF 公式与图 | — | **请你做** |
| **P1** | 答辩 1 页 slide | **[x]** 素材 `report/defense_one_page.md` | **请你做成 .pptx** |
| **P2** | `environment.lock.yml` | **[x]** 已 `conda env export` | 可选随作业提交 |

**PDF 命令**（任选其一）：

```bash
pandoc report/report.md -o report/report.pdf --resource-path=.:report
# 或 Typora / VS Code Markdown PDF 插件
```

---

## 三、§三 仍可改进 — 本轮处理

### 3.1 报告与答辩（不改代码）

| 条目 | 状态 | 说明 |
|------|------|------|
| 附录 B 嵌入自动表 | **[x]** | `appendix_b_embed.md` + **已写入 `report.md` 附录 B**；脚本生成双文件 |
| 通读局限话术 | **[x]** | 结论 §5、附录 D、`TODO.md` §五、`defense_one_page.md` 已对齐 |
| 页数/格式 | **[ ]** | 需你导出 PDF 后统计页数 |
| 双人分工 | **[x]** | `README.md` 已加分工说明 |

### 3.2 实验深化

| 编号 | 状态 | 说明 |
|------|------|------|
| **E22** β–η 热力图 | **[x]** | `figures/exp22_beta_eta_heatmap.png`；报告 §4.12 已补 |
| **E16** Chebyshev | **[ ]** | 仍建议「未来工作」 |
| **E12+** 谱范数 toy | **[ ]** | trust-region 文字已够答辩 |
| **E18–E20** DL | **[ ]** | 合理暂缓 |

### 3.3 工程

| 项 | 状态 |
|----|------|
| 实验 10 向量化 | **[ ]** 低优先级 |
| `environment.lock.yml` | **[x]** |
| GitHub Actions | **[ ]** |
| 再拆 exp 文件 | **[ ]** 不必 |

---

## 四、§四 答辩追问 — 材料是否齐备

| 追问 | 材料位置 |
|------|----------|
| Adam vs AdamW | §3.3 + `exp1b`（含 AdamW） |
| 为何不用 Chebyshev | 结论 + E17 PCG；E16 未做 |
| 模态实验 | **E21** + `exp21_eigenmode_decay.png` |
| κ 扫描 vs 实验 2 | `exp_kappa_scan.png` + 实验 2 |
| 非二次 | E14 噪声；诚实答聚焦二次 |

---

## 五、§五 局限话术

**无需改代码**；已分布在 `report/report.md` 结论、附录 D、`TODO.md` §五、`defense_one_page.md`。

---

## 六、§六 快速勾选（对照）

| 条目 | 状态 |
|------|------|
| [x] `check_repro.sh` | 本轮已跑通 |
| [x] 参考文献 12 + AdamW + 附录 C | 已有 |
| [x] 摘要 Muon / Adam 高 κ | 已有 |
| [x] E21 + E24 + κ + exp1b(AdamW) | 已有 |
| [x] 附录 D 答辩素材 | 已有 + **`defense_one_page.md`** |
| [ ] **`report/report.pdf`** | **你本地** |
| [ ] **答辩 PPT** | **你本地**（用 `defense_one_page.md`） |
| [x] （可选）`environment.lock.yml` | 已生成 |
| [x] （可选）附录 B 粘贴自动表 | **已嵌入 report.md** |

---

## 七、文件分工（TODO §七）

| 文件 | 用途 |
|------|------|
| **`re_TODO.md`（本文件）** | 第三轮：还能做什么 / 谁来做 / 本轮新增 |
| **`TODO.md`** | 当前收尾清单（essentially PDF + PPT） |

---

## 八、本轮新增/变更清单

| 路径 | 说明 |
|------|------|
| `experiments/extended_experiments.py` | +**E22** `exp22_beta_eta_heatmap` |
| `figures/exp22_beta_eta_heatmap.png` | 新图（总图数 **27**） |
| `scripts/generate_appendix_tables.py` | +`appendix_b_embed.md` 精简表 |
| `report/report.md` | 附录 B 嵌入表；§4 E22 |
| `report/defense_one_page.md` | 答辩一页（三图+话术） |
| `report/appendix_b_embed.md` | 自动生成精简附录 |
| `environment.lock.yml` | conda 导出 |
| `README.md` | 分工 + 答辩素材路径 |

---

## 九、仍未完成且 agent 无法代劳

1. **`report/report.pdf`** — 环境无 pandoc，且无 sudo。  
2. **`.pptx` 答辩幻灯片** — 已给 Markdown 素材，需你在 PowerPoint/WPS 中插图。  
3. **E16 / E18–E20** — 按第三轮 TODO 标为可选/未来工作。

---

## 十、建议你提交的 10 分钟 checklist

```bash
conda activate num_ana_opt
cd /path/to/num_ana_project
bash scripts/check_repro.sh
pandoc report/report.md -o report/report.pdf --resource-path=.:report
# 打开 report/defense_one_page.md → 做 1 页 PPT
# 打包：report.pdf + 代码 + figures/（或说明运行脚本生成）
```

---

*第三轮结论：**代码与报告主体已可提交**；剩余为 PDF、PPT 两项本地操作。完成後可在 `TODO.md` §六 将对应 `[ ]` 改为 `[x]`。*
