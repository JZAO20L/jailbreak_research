# SESS 论文写作、审稿与迭代计划

**论文标题**: SESS: Self-Evolving Skills for Jailbreak Prompt Generation  
**目标会议**: ICLR 2027 (截稿 ~2026-10) / AAAI 2027 (截稿 ~2026-08)  
**创建日期**: 2026-06-30  
**状态**: 规划阶段

---

## 一、现状评估

### 1.1 已完成工作

| 模块 | 完成度 | 关键数据 |
|------|--------|----------|
| SESS 方法设计 | 100% | 完整系统架构、Skill 数据结构、检索/进化/维护机制 |
| Layer 1-4 消融实验 | 100% | 81 组实验，最佳 ASR 99.7% |
| 迁移实验 | 100% | 120 组实验，4 模型 × 5 数据集 × 6 方法 |
| 中期报告 | 100% | 完整 LaTeX 项目、13 张图表、21 篇参考文献 |
| 代码与文档 | 100% | MECHANISM.md、README.md、完整代码库 |

### 1.2 现有资源

| 资源 | 位置 | 内容 |
|------|------|------|
| 实验数据 | `exp/layer{1-4}/`, `exp/transfer/` | 217 组实验 JSON 结果 |
| 图表 | `docs/figures/output/` | 13 张 PNG (AHR-GRPO 6张 + SESS 7张) |
| 参考文献 | `docs/reference.md` | 21 篇 BibTeX |
| 中期报告 | `docs/report.md` | 929 行，含完整实验描述 |
| 方法文档 | `MECHANISM.md` | 完整方法描述 + Mermaid 图 |
| 图表脚本 | `docs/figures/generate_figures_v3.py` | 可复用的 matplotlib 脚本 |

### 1.3 待补充工作

| 工作项 | 优先级 | 说明 |
|--------|--------|------|
| 论文格式 LaTeX 模板 | 高 | ICLR 2027 官方模板 |
| 方法框架图 (Figure 1) | 高 | 简化版系统架构，适合论文首页 |
| 实验结果重新整理 | 高 | 统一表格格式，补充统计检验 |
| 新图表：Skill 进化动态 | 中 | 展示 Skill 库随进化的质量变化 |
| 新图表：跨模型迁移热力图 | 中 | 4 模型 × 5 数据集热力图 |
| Case Study | 中 | 典型成功/失败案例分析 |
| 更多 Baseline 对比 | 低 | Metis、ASTRA 等最新方法（如有数据） |
| Limitations & Ethics | 高 | 安全研究论文必须 |

---

## 二、论文结构

### 2.1 大纲 (ICLR 格式，正文 10-12 页)

```
Abstract (250 words)
1. Introduction (1.5 页)
   - LLM 安全对齐与 jailbreak 威胁
   - 现有方法的三个局限：无知识积累、高查询成本、低迁移性
   - 我们的方法：Self-Evolving Skills System (SESS)
   - 核心贡献 (3 点)

2. Related Work (1.5 页)
   2.1 Jailbreak Attack Methods (PAIR, TAP, AutoDAN, DeepInception)
   2.2 RL-based Jailbreak (Jailbreak-R1, TROJail, xJailbreak)
   2.3 Self-Evolving Strategies (Metis, ASTRA, EvoSynth)
   2.4 Positioning (与本文方法的关系)

3. Method (2.5 页)
   3.1 Overview & Problem Formulation
   3.2 Skill Representation & Library
   3.3 Cold Start Phase
   3.4 Self-Evolution Mechanism
       3.4.1 Skill Retrieval
       3.4.2 Reflection & Extraction
       3.4.3 Update Strategies
   3.5 Library Maintenance

4. Experiments (3.5 页)
   4.1 Setup (models, datasets, baselines, metrics)
   4.2 Main Results (Layer 1-4 ablation)
       4.2.1 Method Combination (Layer 1)
       4.2.2 Data Strategy (Layer 2)
       4.2.3 Strong Prior: DAN Templates (Layer 3-4)
   4.3 Cross-Model Transfer
       4.3.1 Same-Family Transfer (Qwen series)
       4.3.2 Cross-Family Transfer (GPT-OSS-20B)
   4.4 Cross-Dataset Transfer
   4.5 Ablation: Evolution Necessity
   4.6 Case Study

5. Analysis & Discussion (1 页)
   5.1 Why DAN Templates Work Without Evolution
   5.2 Skill Quality Evolution Dynamics
   5.3 Cross-Family Transfer: Challenges & Insights
   5.4 Limitations

6. Conclusion (0.5 页)

Ethics Statement
References (appendix)
```

### 2.2 核心贡献 (3 点)

1. **Self-Evolving Skills System**: 首个将可复用攻击模板与自进化机制结合的 jailbreak 框架，实现知识积累与跨案例迁移
2. **Systematic Empirical Study**: 217 组实验的系统性消融分析，揭示方法组合、数据策略、强先验 (DAN) 的关键作用
3. **Cross-Model Transfer Analysis**: 在 4 个模型、5 个数据集上的 120 组迁移实验，量化分析同族/跨族迁移的边界

---

## 三、图表计划

### 3.1 需要制作的图表

| 编号 | 类型 | 内容 | 状态 | 优先级 |
|------|------|------|------|--------|
| Fig 1 | 方法框架图 | SESS 系统概览 (Cold Start → Evolution → Test) | 需重做 | 高 |
| Fig 2 | 流程图 | Skill 检索 + 攻击 + 反思循环 | 需重做 | 高 |
| Fig 3 | 柱状图 | Layer 1 方法消融对比 | 可复用 | 高 |
| Fig 4 | 热力图 | Layer 2 数据量 × 配比 ASR 矩阵 | 需新做 | 中 |
| Fig 5 | 柱状图 | Layer 3 DAN vs 非 DAN 对比 | 可复用 | 高 |
| Fig 6 | 分组柱状图 | 跨模型迁移对比 (4 模型 × 7 方法) | 需重做 | 高 |
| Fig 7 | 热力图 | 跨数据集迁移 (4 模型 × 5 数据集) | 需新做 | 高 |
| Fig 8 | 折线图 | Skill 质量演化动态 | 需新做 | 中 |
| Table 1 | 表格 | 主要结果 (Layer 1-4 最佳配置) | 需整理 | 高 |
| Table 2 | 表格 | 跨模型迁移详细数据 | 需整理 | 高 |
| Table 3 | 表格 | Baseline 对比 | 需整理 | 高 |
| Fig A1 | 消融图 | Evolution 必要性验证 | 可复用 | 低 |
| Fig A2 | 案例图 | Case Study 示例 | 需新做 | 中 |

### 3.2 图表风格要求

- 简化、清晰，聚焦核心结构和数据流
- 学术论文标准：矢量图 (PDF/EPS) 优先
- 配色方案：色盲友好 (避免红绿对比)
- 字体：与正文一致，标签清晰可读
- 工具：matplotlib (数据图) + TikZ/draw.io (框架图)

---

## 四、自动化写作流程

### 4.1 工作目录结构

```
paper_qwen/
├── PLAN.md                    ← 本文件
├── main.tex                   ← 论文主文件
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── method.tex
│   ├── experiments.tex
│   ├── analysis.tex
│   ├── conclusion.tex
│   └── appendix.tex
├── figures/
│   ├── system_overview.pdf    ← 方法框架图
│   ├── skill_retrieval.pdf    ← 检索流程图
│   ├── layer1_ablation.pdf    ← Layer 1 消融
│   ├── layer2_heatmap.pdf     ← Layer 2 热力图
│   ├── layer3_dan.pdf         ← Layer 3 DAN 对比
│   ├── transfer_models.pdf    ← 跨模型迁移
│   ├── transfer_heatmap.pdf   ← 跨数据集热力图
│   └── skill_evolution.pdf    ← Skill 质量演化
├── tables/
│   ├── main_results.tex
│   ├── transfer_results.tex
│   └── baseline_comparison.tex
├── references.bib
├── review/
│   ├── review_round1.md       ← 第 1 轮审稿意见
│   ├── review_round2.md       ← 第 2 轮审稿意见
│   ├── rebuttal.md            ← 审稿回复
│   └── revision_log.md        ← 修改日志
└── scripts/
    ├── generate_figures.py    ← 数据图生成脚本
    └── compile.sh             ← 编译脚本
```

### 4.2 写作流程 (6 阶段)

```
Stage 1: 数据整理与图表生成
  ├── 从 exp/ 提取所有实验数据
  ├── 统一格式，生成 LaTeX 表格
  ├── 生成所有数据图 (matplotlib → PDF)
  └── 绘制方法框架图 (TikZ/draw.io → PDF)

Stage 2: 初稿撰写
  ├── 先写 Method 章节 (核心贡献)
  ├── 再写 Experiments 章节 (数据驱动)
  ├── 然后 Introduction (基于 Method + Experiments)
  ├── Related Work (从 reference.md 整理)
  ├── Analysis & Discussion
  ├── Conclusion + Abstract
  └── Appendix (补充实验、证明)

Stage 3: 内部审稿 (Subagent Review)
  ├── Reviewer 1: 方法论审稿 (方法创新性、技术深度)
  ├── Reviewer 2: 实验审稿 (实验充分性、统计分析)
  ├── Reviewer 3: 写作审稿 (逻辑流畅性、语言表达)
  └── Reviewer 4: 综合审稿 (整体评价、接收建议)

Stage 4: 修订迭代
  ├── 逐条回复审稿意见
  ├── 修改对应章节
  ├── 更新图表
  └── 第 2 轮审稿 (验证修改)

Stage 5: 终稿打磨
  ├── 全文一致性检查
  ├── 引用完整性检查
  ├── 格式合规检查 (ICLR 模板)
  ├── 语言润色
  └── 最终编译

Stage 6: 投稿准备
  ├── 生成 Supplementary Material
  ├── 整理代码开源包
  └── 撰写 Cover Letter
```

---

## 五、Subagent 审稿系统

### 5.1 审稿角色设计

| 角色 | 关注点 | 审稿风格 |
|------|--------|----------|
| **Reviewer 1 (Method)** | 方法创新性、技术贡献、与现有工作的区别 | 严格但公正，关注 novelty |
| **Reviewer 2 (Experiment)** | 实验设计、baselines、消融、统计检验 | 细致，要求充分实验 |
| **Reviewer 3 (Writing)** | 逻辑结构、语言表达、图表质量 | 注重可读性 |
| **Reviewer 4 (Meta)** | 综合评价、接收建议、优先级排序 | 综合前 3 位意见 |

### 5.2 审稿 Prompt 模板

#### Reviewer 1: 方法论审稿

```
You are Reviewer 1 for an ICLR submission. Your expertise is in LLM safety and jailbreak attacks.

Review the following paper section focusing on:
1. **Novelty**: Is the self-evolving skills approach genuinely novel compared to Metis, ASTRA, EvoSynth?
2. **Technical Depth**: Are the skill retrieval, reflection, and maintenance mechanisms well-designed?
3. **Clarity**: Is the method description clear and reproducible?
4. **Comparison**: How does this compare to existing self-evolving strategies?

Specific questions to address:
- What is the key technical insight that distinguishes this from prior work?
- Are there any methodological weaknesses or missing components?
- Is the skill representation sufficient? Could semantic retrieval improve performance?
- How does the cold start vs. full_evolve finding impact the claimed contribution?

Rate: Soundness (1-10), Novelty (1-10), Technical Quality (1-10)
Provide: 3 strengths, 3 weaknesses, specific improvement suggestions
```

#### Reviewer 2: 实验审稿

```
You are Reviewer 2 for an ICLR submission. Your expertise is in experimental design and evaluation.

Review the experiments section focusing on:
1. **Baselines**: Are all relevant baselines included? (PAIR, AutoDAN, TAP, DeepInception, Metis, ASTRA)
2. **Ablation Study**: Is the ablation comprehensive? Are all components validated?
3. **Transfer Analysis**: Is the cross-model/cross-dataset analysis rigorous?
4. **Statistical Significance**: Are results reported with confidence intervals or variance?
5. **Fairness**: Are comparisons fair? (same model, same data, same compute)

Specific questions:
- Why only Qwen-family models for same-family transfer? What about LLaMA, Mistral?
- The cross-family result (32.5% on GPT-OSS-20B) is low — is this a fundamental limitation?
- Are 1000 test samples sufficient? What about variance across random seeds?
- Missing: comparison with Metis/ASTRA under the same experimental conditions.
- The evolution contribution is only +0.9% — does this undermine the "self-evolving" claim?

Rate: Experimental Rigor (1-10), Completeness (1-10), Statistical Soundness (1-10)
Provide: 3 strengths, 3 weaknesses, specific additional experiments needed
```

#### Reviewer 3: 写作审稿

```
You are Reviewer 3 for an ICLR submission. Your expertise is in clear scientific communication.

Review the full paper focusing on:
1. **Structure**: Is the paper well-organized? Does the narrative flow logically?
2. **Introduction**: Does it clearly motivate the problem and contributions?
3. **Figures & Tables**: Are they clear, self-contained, and well-labeled?
4. **Language**: Is the writing clear, concise, and grammatically correct?
5. **Related Work**: Does it adequately position the work?

Specific checks:
- Can the main contribution be understood from the abstract alone?
- Does Figure 1 effectively communicate the system architecture?
- Are all tables readable without referring to the text?
- Is the paper within the page limit?
- Are there any inconsistencies between sections?

Rate: Clarity (1-10), Organization (1-10), Presentation (1-10)
Provide: 3 strengths, 3 weaknesses, specific writing improvements
```

#### Reviewer 4: 综合审稿 (Meta-Reviewer)

```
You are the Area Chair summarizing reviews for an ICLR submission.

Based on the three reviewer reports below, provide:
1. **Summary of Reviews**: Key agreements and disagreements
2. **Overall Assessment**: 
   - Overall Score (1-10)
   - Confidence (1-5)
   - Recommendation: Accept / Weak Accept / Weak Reject / Reject
3. **Key Issues**: The 2-3 most critical issues that determine acceptance
4. **Actionable Items**: Prioritized list of changes needed for acceptance
5. **Meta-Questions**: Questions for the authors to address in rebuttal

[Insert Reviewer 1-3 reports here]
```

### 5.3 迭代流程

```
Round 1: 初稿 → 4 位 Reviewer 审稿 → 汇总意见
  ↓
Round 2: 逐条修改 → 更新论文 → 第 2 轮审稿 (重点关注修改)
  ↓
Round 3: 最终打磨 → 格式检查 → 投稿
```

每轮迭代使用 subagent 并行执行 4 个 reviewer，然后汇总意见。

---

## 六、执行计划

### 6.1 阶段时间线

| 阶段 | 预计时间 | 产出 |
|------|----------|------|
| Stage 1: 数据整理 | 1 天 | 所有表格 + 数据图 |
| Stage 2: 初稿撰写 | 3-4 天 | 完整 LaTeX 初稿 |
| Stage 3: 第 1 轮审稿 | 0.5 天 | 4 份审稿报告 |
| Stage 4: 修订迭代 | 2 天 | 修改稿 + rebuttal |
| Stage 5: 终稿打磨 | 1 天 | 最终投稿版 |
| **总计** | **~8 天** | **完整论文** |

### 6.2 Subagent 分工

| Subagent | 任务 | 输入 | 输出 |
|----------|------|------|------|
| **data-agent** | 提取实验数据、生成表格 | exp/ 下的 JSON 文件 | LaTeX 表格文件 |
| **figure-agent** | 生成数据图 | 实验数据 + 图表脚本 | PDF 图表文件 |
| **writer-agent** | 撰写各章节 | 大纲 + 数据 + 中期报告 | LaTeX 章节文件 |
| **reviewer-agent-1** | 方法论审稿 | 完整论文 | 审稿报告 |
| **reviewer-agent-2** | 实验审稿 | 完整论文 | 审稿报告 |
| **reviewer-agent-3** | 写作审稿 | 完整论文 | 审稿报告 |
| **reviewer-agent-4** | 综合审稿 | 3 份审稿报告 | Meta-review |
| **editor-agent** | 修订论文 | 审稿意见 + 论文 | 修改后论文 |

### 6.3 关键决策点

| 决策 | 选项 | 建议 |
|------|------|------|
| AHR-GRPO 是否纳入 | 纳入 / 不纳入 | 不纳入，SESS 独立成文 |
| 目标会议 | ICLR / AAAI / 其他 | ICLR 2027 优先 (时间充裕) |
| 图表语言 | 中文 / 英文 | 英文 (国际会议) |
| 论文语言 | 中文 / 英文 | 英文 (ICLR/AAAI) |
| Baseline 补充 | 仅已有 / 补充新实验 | 先用已有数据，必要时补充 |

---

## 七、风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| Evolution 贡献仅 +0.9% | 削弱 "self-evolving" 核心 claim | 重新定位：强调 Skills 系统整体效果 (99.7%) 而非进化本身 |
| 跨族迁移效果有限 (32.5%) | 泛化性质疑 | 坦诚讨论，作为 research insight 呈现 |
| 缺少 Metis/ASTRA 对比 | Baseline 不充分 | Related Work 中详细对比方法差异，实验部分说明复现限制 |
| 论文叙事不够聚焦 | 审稿人困惑 | 明确一条主线：Skills 作为可复用攻击模板的有效性 |
| 图表质量不达标 | 影响第一印象 | 使用矢量图，统一风格，参考顶会论文图表 |

---

## 八、论文核心叙事

### 主线逻辑

```
问题：现有 jailbreak 方法缺乏知识积累，每次攻击从头探索
  ↓
洞察：成功攻击中蕴含可复用的策略模式，应被提取为 Skills
  ↓
方法：SESS — 自进化技能系统
  ├── Cold Start：初始 Skills 积累
  ├── Evolution：从攻击轨迹中提取/改进 Skills
  └── Maintenance：质量控制与库管理
  ↓
验证：217 组消融 + 120 组迁移实验
  ↓
发现：
  1. DAN 模板 + Skills 系统 → 99.7% ASR
  2. 同族迁移效果良好 (85-95%)
  3. 跨族迁移是开放挑战 (32.5%)
  4. Evolution 对强先验非必需，对弱先验有提升
```

### 核心 Claim 调整

鉴于 Evolution 贡献有限 (+0.9%)，论文核心 claim 应调整为：

**不是**: "自进化机制是核心贡献"  
**而是**: "可复用的 Skills 系统 + 强先验模板 是高效 jailbreak 的关键"

这样：
- DAN 模板直接达到 98.8% → 证明 Skills 系统 + 强先验有效
- Evolution 在弱先验场景下仍有价值 (+0.9%) → 完整讨论
- 跨模型迁移中 Skills 带来显著提升 (+15-26pp) → 证明泛化性

---

*Plan created: 2026-06-30*
