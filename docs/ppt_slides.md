# 中期答辩 PPT 内容规划

> 每页用 `---` 分隔，标注图表占位与建议布局

---

## Slide 1 — 封面

**布局**: 居中

- 题目：基于自进化技能系统与强化学习的越狱提示词生成方法研究
- 研究生：贾子潇
- 导师：徐冰
- 院（系）：计算学部
- 学科：电子信息-计算机科学与技术
- 学号：24S103188
- 日期：2026.6.22

---

## Slide 2 — 目录

**布局**: 左对齐列表

1. 课题背景与研究目的
2. 相关研究现状
3. 第一章 AHR-GRPO 方法与实验
4. 第二章 SESS 方法与实验
5. 第三章 Agentic-RL 方法设计（待开展）
6. 后期工作计划与进度安排
7. 总结

---

## Slide 3 — 研究背景：大模型安全与越狱攻击

**布局**: 左文右图

**要点**（3-4 条）:
- LLM 广泛应用 → 安全性问题日益突出
- 对齐技术（RLHF / Constitutional AI）无法完全消除风险
- 越狱攻击通过精心设计的 Prompt 绕过安全护栏
- 攻击研究意义：以攻促防，提升模型安全性

**图表占位**:
> `[图：越狱攻击示意图 — 原始 Prompt → 越狱 Prompt → 目标模型输出违规内容]`

---

## Slide 4 — 现有 Jailbreak 方法分类与局限

**布局**: 表格 + 右侧要点

**表格**:

| 方法类别 | 代表方法 | 局限性 |
|:---|:---|:---|
| 静态模板 | PAIR [1], AutoDAN [4] | 无法适应新防御 |
| 梯度优化 | GCG, AutoDAN-GA | 需白盒访问，成本高 |
| 多轮对话 | Crescendo | 缺乏知识迁移 |
| 角色扮演 | Persona | 效果不稳定 |

**右侧要点**:
- 核心瓶颈：效率低、泛化差、无知识积累

---

## Slide 5 — 四大研究挑战

**布局**: 四象限 / 2×2 卡片

| 挑战 | 描述 |
|:---|:---|
| **效率** | PAIR/TAP 需 1000+ API 调用/样本 |
| **泛化** | 攻击策略无法跨模型迁移 |
| **知识积累** | 成功策略无法复用到新案例 |
| **奖励设计** | ASR 奖励稀疏，二值信号方差大 |

---

## Slide 6 — 研究目的

**布局**: 居中大字 + 下方三点

**核心目标**: 系统研究越狱攻击方法与防御策略

三个子目标:
1. 推进 Jailbreak Prompt 合成边界，探索高效可迁移策略
2. 优化训练效率，实现低资源 MaaS 方案
3. 构建可复用 Skills 系统，提供开源攻击工具与数据合成管道

---

## Slide 7 — 三章方法总览

**布局**: 横向流程图（三模块递进）

```
[AHR-GRPO] ──→ [SESS] ──→ [Agentic-RL]
 自适应混合奖励    自进化技能系统    两者结合
 解决奖励稀疏      解决知识积累      协同优化
```

**要点**:
- 第一章：方差驱动自适应权重，动态平衡 ASR 与 Judge 奖励
- 第二章：可复用 Skills 模板 + 自进化机制
- 第三章：在自进化 Skills 环境中进行 RL 训练

**图表占位**:
> `[图：三章方法递进关系示意图]`

---

## Slide 8 — 进度总览

**布局**: 进度条 / 三列对比

| 章节 | 完成度 | 最佳 ASR | 关键进展 |
|:---|:---:|:---:|:---|
| AHR-GRPO | 85% | 33.2% (+2.4%) | 3 个实验全部完成 |
| SESS | 95% | 99.7% | 217 组实验完成 |
| Agentic-RL | 10% | — | 方法框架已设计 |

---

## Slide 9 — 章节分隔页：第一章 AHR-GRPO

**布局**: 居中大标题

**Adaptive Hybrid-Reward GRPO for Jailbreak Prompt Generation**

副标题：方差驱动的自适应奖励权重机制

---

## Slide 10 — AHR-GRPO 研究动机

**布局**: 左文右图

**问题**:
- 传统 GRPO 使用单一 ASR 奖励 → 稀疏、二值、方差大
- 引入 Judge Reward 可缓解稀疏性，但固定权重无法适应训练动态
- 直觉：ASR 全失败时依赖 Judge 引导，ASR 分化后回归结果导向

**图表占位**:
> `[图 1：自适应权重 Insight 示意图 — λ 随训练过程的变化曲线]`
> 文件：`figures/output/AHR_GRPO_06_summary.png` 或 report 中的图 1

---

## Slide 11 — AHR-GRPO 方法流程

**布局**: 全页流程图

**核心公式**:
$$\lambda_{\text{raw}} = \sigma\left(\alpha \cdot \log\frac{\sigma^2_{\text{ASR}}}{\sigma^2_{\text{proc}}} + \delta\right)$$

**三个关键机制**:
1. 双平滑：EMA + 惯性平滑，防止梯度震荡
2. 边界裁剪：λ ∈ [λ_min, λ_max]，防止信号坍缩
3. 冷启动：前 W 步固定 λ=0.5

**图表占位**:
> `[图 2：AHR-GRPO 整体方法流程图]`
> 文件：report 中的图 2

---

## Slide 12 — 实验 1：攻击策略 Prompt 筛选

**布局**: 左表右图

**实验设置**: 24 种 Jailbreak Prompt 模板 × 1000 条测试集

**Top-3 结果**:

| 策略 | ASR (%) | 类别 |
|:---|:---:|:---|
| hypothetical_scenario | 30.8 | 场景构造 |
| creative_writing | 28.3 | 场景构造 |
| role_playing | 25.0 | 角色扮演 |

**图表占位**:
> `[图 3：24 种攻击策略 ASR 柱状图]`
> 文件：`figures/output/AHR_GRPO_01_attack_prompt_selection.png`

---

## Slide 13 — 实验 2：Judge 维度对比

**布局**: 左文右图

**实验设置**: 3 策略 × 4 Judge 维度 = 12 组 GRPO 训练

**关键发现**:
- 最佳组合：hypothetical_scenario + idea_preservation → 32.3% (+1.5%)
- 通用 Judge 维度优于专用维度
- 所有 12 组均超越 baseline

**图表占位**:
> `[图 4：12 组 Judge 维度训练结果对比图]`
> 文件：`figures/output/AHR_GRPO_03_judge_dimension.png`

---

## Slide 14 — 实验 3：自适应权重验证

**布局**: 左图右表

**核心结果**:
- 自适应 β=0 → ASR 33.2%（+0.9% vs 实验 2 最佳）
- 消融：仅 ASR → 25.0%（-5.8%），仅 Judge → 28.5%（-2.3%）

**图表占位**:
> `[图 5：自适应 vs 固定权重 ASR 对比]`
> 文件：`figures/output/AHR_GRPO_04_adaptive_weight.png`

> `[图 6：Reward 消融对比]`
> 文件：`figures/output/AHR_GRPO_05_reward_ablation.png`

---

## Slide 15 — AHR-GRPO 核心结论

**布局**: 要点列表（3 条）

1. **混合 Reward >> 单一 Reward**：ASR + Judge 组合显著优于任一项单独使用
2. **自适应权重有效**：动态调整 λ 实现 ASR 与 Judge 最优平衡，额外 +0.9%
3. **通用 Judge 维度更稳定**：idea_preservation 平均提升 1.2%，优于专用维度

---

## Slide 16 — 章节分隔页：第二章 SESS

**布局**: 居中大标题

**Self-Evolving Skills System for Jailbreak Prompt Generation**

副标题：可复用攻击模板 + 自进化机制

---

## Slide 17 — SESS 研究动机

**布局**: 左文右图

**问题**:
- 现有方法每次攻击从头探索，无知识积累
- PAIR 等迭代方法效率低，无法跨案例复用
- 需要：可复用的攻击模板 + 自动进化机制

**图表占位**:
> `[图 7：SESS 整体方法流程图 — Cold Start → Evolution → Test]`
> 文件：report 中的图 7

---

## Slide 18 — SESS 方法设计

**布局**: 上图下文

**Skill 数据结构**:
- content（攻击模板）、source、success_rate、usage_count
- 检索评分：score = quality_score × (1 + 关键词匹配 × 0.3) × 类型匹配

**三阶段流程**:
1. Cold Start：基于 PAIR/AutoDAN 积累成功经验
2. Evolution：检索 + 使用 Skills，多种方式进化
3. Test：在 test 集上评估最终 ASR

**图表占位**:
> `[图 8：Skill 数据结构与检索机制]`
> 文件：report 中的图 8

> `[图 9：三阶段自进化流程]`
> 文件：report 中的图 9

---

## Slide 19 — SESS 消融实验总览（217 组）

**布局**: 四行表格

| Layer | 实验内容 | 组数 | 最佳 ASR | 关键发现 |
|:---:|:---|:---:|:---:|:---|
| L1 | 方法组合 Grid Search | 16 | 79.1% | single_call >> every_iteration (+12.7%) |
| L2 | 数据策略消融 | 36 | 80.0% | 数据量影响 <2%，配比影响显著 |
| L3 | DAN 模板验证 | 17 | 98.8% | DAN 无需 Cold Start，直接使用最优 |
| L4 | DAN 数据消融 | 12 | 99.7% | 数据量影响极小，差距仅 1.3% |

**图表占位**:
> `[图 10：SESS 四层实验结果汇总图]`
> 文件：`figures/output/SESS_01_sess_main_results.png`

---

## Slide 20 — SESS 迁移实验设计

**布局**: 左右分栏

**跨模型迁移**:
- 4 个目标模型：Qwen3-0.6B / 4B / 14B（同族）+ gpt-oss-20b（跨族）
- 6 种方法：no_rewrite, pair, autodan, deepinception, persona, pair_skills, autodan_skills

**跨数据集迁移**:
- 5 个数据集：default, advbench, harmbench_ctx, harmbench_std, jailbreakBench
- 累计 120 组实验

---

## Slide 21 — 同族模型迁移结果

**布局**: 大表格 + 右侧要点

| 模型 | PAIR | PAIR+SESS | Δ | AutoDAN | AutoDAN+SESS | Δ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-0.6B | 78.5% | **95.1%** | +16.6 | 80.2% | **96.8%** | +16.6 |
| Qwen3-4B | 72.3% | **86.7%** | +14.4 | 78.7% | **89.2%** | +10.5 |
| Qwen3-14B | 82.1% | **86.5%** | +4.4 | 85.5% | **90.3%** | +4.8 |

**要点**:
- SESS 对 PAIR 平均提升 +11.8pp，对 AutoDAN 平均 +10.6pp
- 较小模型提升更显著

**图表占位**:
> `[图 11：同族模型迁移效果对比]`
> 文件：`figures/output/SESS_04_same_family_transfer.png`

---

## Slide 22 — 跨族模型迁移结果

**布局**: 左表右图

| 方法 | gpt-oss-20b ASR |
|:---|:---:|
| no_rewrite | 0.14% |
| PAIR | 0.08% |
| AutoDAN | 6.0% |
| PAIR + SESS | 11.4% |
| **AutoDAN + SESS** | **32.5%** |

**要点**:
- 跨族迁移极具挑战，传统方法几乎失效
- AutoDAN + SESS 的 32.5% 是当前最佳跨族结果

**图表占位**:
> `[图 12：跨模型迁移效果（含跨族）]`
> 文件：`figures/output/SESS_04_cross_model_transfer.png`

---

## Slide 23 — 跨数据集迁移结果

**布局**: 左图右表

| 数据集 | no_rewrite | PAIR+SESS | AutoDAN+SESS |
|:---|:---:|:---:|:---:|
| default | 30.3% | 79.0% | 89.2% |
| advbench | 3.3% | 81.0% | — |
| harmbench_ctx | 79.0% | 98.0% | 96.5% |
| harmbench_std | 21.5% | 91.5% | 93.0% |
| jailbreakBench | 7.0% | 88.0% | — |

**图表占位**:
> `[图 13：跨数据集迁移效果]`
> 文件：`figures/output/SESS_03b_cross_dataset_transfer.png`

---

## Slide 24 — SESS 核心结论

**布局**: 要点列表（4 条）

1. **Skills 自进化有效**：从空库到 99.7% ASR，217 组实验验证
2. **DAN 模板是强力起点**：6 个 DAN 模板直接使用即达 98.8%，无需额外 Cold Start
3. **同族迁移稳定**：PAIR+SESS 平均 +11.8pp，AutoDAN+SESS 平均 +10.6pp
4. **跨族迁移是开放挑战**：最佳 32.5%，需新方法突破

---

## Slide 25 — 章节分隔页：第三章 Agentic-RL

**布局**: 居中大标题

**Agentic RL with Self-Evolving Skills System**

副标题：过程奖励 + 结果奖励 + 知识积累的协同优化（待开展）

---

## Slide 26 — Agentic-RL 方法设计

**布局**: 流程图 + 要点

**核心思路**: AHR-GRPO + SESS 结合

**三阶段训练**:
1. Phase 1：Skills 初始化（DAN 模板或演化 Skills）
2. Phase 2：RL 训练（Policy 在 Skills 环境中学习）
3. Phase 3：Skills 协同演化（Policy ↔ Skills 双向优化）

**奖励设计**:
- 结果奖励：ASR（稀疏）
- 过程奖励：Judge（稠密）
- 自适应权重：AHR-GRPO 方差驱动机制
- Skills 质量奖励：鼓励使用高质量 Skills

**图表占位**:
> `[图：Agentic-RL 方法框架图 — 待绘制]`

---

## Slide 27 — 后期工作计划

**布局**: 甘特图 / 时间轴

| 阶段 | 内容 | 时间 |
|:---|:---|:---:|
| 方法实现 | 搭建 Skills 环境 + 集成 AHR 机制 | 2 周 |
| 实验验证 | Baseline 对比 + 消融 + 迁移测试 | 4 周 |
| 结果分析 | λ 演化分析 + Skills 可视化 | 2 周 |
| 论文整理 | ICLR 格式 + Intro/RW/Conclusion | 持续 |

**图表占位**:
> `[图：甘特图 — 后续 8 周工作安排]`

---

## Slide 28 — 困难与应对

**布局**: 问题 → 对策（两列）

| 困难 | 应对策略 |
|:---|:---|
| 计算资源需求大 | 合理安排 GPU 调度，优先核心实验 |
| 高成本试错风险 | 参考 AHR-GRPO 稳定性机制 |
| 跨族迁移效果有限 | 作为研究发现分析原因，探索元学习 |
| 论文时间压力 | 尽快启动第三章实验，并行整理前两章 |

---

## Slide 29 — 总结

**布局**: 三列 + 底部总结

| AHR-GRPO (85%) | SESS (95%) | Agentic-RL (10%) |
|:---|:---|:---|
| 自适应混合奖励 | 自进化 Skills 系统 | 两者结合 |
| ASR 33.2% (+2.4%) | ASR 99.7% | 待实验验证 |
| 通用 Judge 维度有效 | 同族迁移 +11.8pp | 过程+结果协同优化 |

**总体完成度**: ~65%，如期完成可能性较高

---

## Slide 30 — 致谢

**布局**: 居中

**谢谢各位老师！**

恳请批评指正

---

# PPT 制作备注

## 图表文件索引

| 图号 | 内容 | 文件路径 |
|:---:|:---|:---|
| 图 1 | 自适应权重 Insight | report 图 1 |
| 图 2 | AHR-GRPO 方法流程 | report 图 2 |
| 图 3 | 24 种攻击策略 ASR | `figures/output/AHR_GRPO_01_attack_prompt_selection.png` |
| 图 4 | 12 组 Judge 维度对比 | `figures/output/AHR_GRPO_03_judge_dimension.png` |
| 图 5 | 自适应 vs 固定权重 | `figures/output/AHR_GRPO_04_adaptive_weight.png` |
| 图 6 | Reward 消融对比 | `figures/output/AHR_GRPO_05_reward_ablation.png` |
| 图 7 | SESS 整体流程 | report 图 7 |
| 图 8 | Skill 数据结构 | report 图 8 |
| 图 9 | 三阶段自进化流程 | report 图 9 |
| 图 10 | SESS 四层结果汇总 | `figures/output/SESS_01_sess_main_results.png` |
| 图 11 | 同族模型迁移 | `figures/output/SESS_04_same_family_transfer.png` |
| 图 12 | 跨模型迁移（含跨族） | `figures/output/SESS_04_cross_model_transfer.png` |
| 图 13 | 跨数据集迁移 | `figures/output/SESS_03b_cross_dataset_transfer.png` |

## 风格建议

- 配色：蓝色主色调（与论文图表一致）
- 字体：中文用黑体/微软雅黑，英文用 Arial/Calibri
- 每页要点不超过 5 条，字号 ≥ 18pt
- 表格用三线表样式
- 章节分隔页用深色背景 + 白色大字
