# SESS 论文三级大纲

**论文标题**: SESS: Self-Evolving Skills System for Jailbreak Prompt Generation  
**目标会议**: ICLR 2027 / AAAI 2027  
**日期**: 2026-07-06

---

## Abstract

- **背景**: 自动化 jailbreak 攻击是 LLM 安全评估的关键手段，但现有方法每次攻击独立探索，缺乏知识积累与经验复用
- **方法**: 提出 SESS（Self-Evolving Skills System），一个轻量级可插拔框架——从成功攻击轨迹中提取可复用的 attack skills as shortcuts，构建动态 skill 库，通过冷启动与自进化机制持续积累攻击经验；基于关键词匹配的检索系统兼顾效果与效率，统一使用 top-1 检索避免多 skill 混杂干扰
- **结果**: 在 4 个目标模型、5 个基准数据集上的大规模实验表明：(1) 同族模型迁移 ASR 达 85–100%，显著优于 baseline；(2) 跨族模型迁移提升 +26.5pp；(3) 跨数据集迁移提升 +3.7–23.5pp
- **贡献**: (1) 可迁移的 jailbreak 方法——evolved skills 可直接迁移至同族/跨族模型，无需重新训练；(2) 可用于数据合成——skill 库作为高质量 jailbreak prompt 模板，批量生成多样化攻击数据用于安全评估与对齐训练

---

## 1. Introduction

### 1.1 问题背景
- LLM 安全对齐（RLHF、Constitutional AI）使模型拒绝有害请求
- Jailbreak 攻击作为红队评估的核心手段，对保障 LLM 安全部署至关重要
- 自动化 jailbreak 方法（PAIR、AutoDAN 等）已取得显著进展

### 1.2 现有方法的局限
- **高调用成本**: PAIR 每次攻击需 ~20 次 LLM 调用，AutoDAN 需 ~50 次评估
- **无记忆**: 每次攻击独立优化，成功策略在攻击结束后丢弃
- **无有效知识管理**: 即使多次攻击成功，也无法系统性地积累和复用有效策略
- **不可迁移**: 针对一个模型/数据集优化的攻击无法迁移到其他场景

### 1.3 本文方法
- 在传统攻击方法（PAIR/AutoDAN）基础上增加 SESS 系统
- **轻量化可插拔**: SESS 作为外部 skill 库模块，不修改 base attack 内部逻辑，通过 skill 注入即可增强任意攻击方法
- **冷启动 + 自进化**: 从少量种子数据构建初始 skill 库，通过反思机制从攻击轨迹中持续提取和改进 skills
- **低成本**: 基于关键词匹配的检索系统，无需 embedding 模型或额外 GPU 资源；top-1 检索避免多 skill 混杂

### 1.4 实验结果概述
- 在同族模型（Qwen3-0.6B/4B/14B）上迁移 ASR 达 85–100%，显著优于 baseline（PAIR 55–88%，AutoDAN 75–92%）
- 跨族模型（gpt-oss-20b）迁移 ASR 达 30–39%，相对 AutoDAN baseline 提升 +26.5pp
- 跨数据集迁移提升 +3.7–23.5pp（5 个基准数据集）
- DAN 模板初始化 + skill 系统达到 99.7% ASR

### 1.5 贡献总结
1. 提出 SESS 框架：轻量级可插拔的自进化 skill 系统，支持冷启动、自进化、基于关键词的 skill 检索
2. 可迁移的 jailbreak 方法：evolved skills 可直接迁移至同族/跨族模型和跨数据集，无需重新训练
3. 可用于数据合成：skill 库作为高质量 jailbreak prompt 模板，可批量生成多样化攻击数据，用于安全评估与对齐训练
4. 大规模实验验证：4 模型 × 5 数据集 × 7+ 方法的系统性迁移评估

---

## 2. Related Work

### 2.1 基于编排的 Jailbreak Prompt 合成方法
- **PAIR** (Chao et al., 2023): attacker-target 双模型对话，迭代优化 jailbreak prompt，~20 次查询
- **TAP** (Mehrotra et al., 2023): 树搜索 + 剪枝，构建 prompt 搜索空间
- **DeepInception** (Li et al., 2023): 多层嵌套虚拟场景
- **总结**: 效果好但每次攻击独立探索，成本高、无记忆、无经验积累

### 2.2 基于进化的 Jailbreak Prompt 合成方法
- **自进化概念与 Skills 使用**: 介绍自进化策略在攻击领域的应用思路——从攻击反馈中提取可复用知识
- **AutoDAN** (Liu et al., 2023): 遗传算法 + DAN 模板，进化 prompt 种群（个体级别进化，非策略级别）
- **AutoDAN-Turbo** (Liu et al., 2024): 层次化遗传优化
- **GAP** (Guo et al., 2024): 遗传算法，无梯度
- **Metis** (Chen et al., 2026): 元认知诊断 + 策略原语组合进化
- **ASTRA** (Liu et al., 2026): 三层动态策略库，按有效性分类
- **EvoSynth** (Chen et al., 2026): 代码级攻击方法进化
- **本文差异性**:
  - vs. AutoDAN/GAP: 进化对象是**可复用的策略模板**（skill），而非个体 prompt
  - vs. Metis/ASTRA: 更轻量的 skill 表示 + 关键词检索（无 embedding），可插拔设计
  - 独特贡献: 系统性的迁移能力验证（4 模型 × 5 数据集），以及 skill 库用于数据合成的应用

---

## 3. Method

### 3.1 Overview
#### 3.1.1 Skills 在 Jailbreak 任务中的作用——Skills as Shortcuts
- 核心洞察: 成功攻击中蕴含可复用的策略模式（如指令覆盖、角色扮演、场景构建）
- Skills as shortcuts: 将 PAIR 等方法的高成功率但高迭代次数的攻击过程，通过从成功轨迹中提取 skills as shortcuts，实现：提升成功率 + 降低迭代次数
- 形式化定义: skill $s = (\text{content}, \text{source}, \boldsymbol{\sigma}, \text{patterns}, \text{harm\_type})$
- 质量评分: $q(s) = r_{\text{suc}} \times \sqrt{n_{\text{use}}}$

#### 3.1.2 系统架构——三大模块
- **Skill 库管理模块**: 存储、检索、维护（剪枝/合并/容量控制）
- **冷启动模块**: 从种子数据构建初始 skill 库
- **自进化模块**: 从攻击轨迹中持续提取和改进 skills
- 轻量化可插拔设计: SESS 作为外部模块，通过 skill 注入增强 base attack，不修改其内部逻辑

### 3.2 Cold Start 机制
#### 3.2.1 基于种子数据的初始 Skill 提取
- 在部分训练数据（cold start prompts）上运行 base attack（PAIR/AutoDAN）
- 攻击成功后，通过 LLM 反思提取可复用 skill 模板
- 提取流程: 成功轨迹 → LLM 分析 → 识别关键策略 → 生成紧凑 skill 模板 + 适用模式标注

#### 3.2.2 基于强先验模板的初始化
- 使用已知有效的 DAN 模板（6 个）直接作为初始 skills
- 包括: dan_mode（指令覆盖）、mcpt（角色）、devil（缩写）、conversation/actor_villain/fictional_world（场景）
- 实验发现: 6 个 DAN 模板即可达到 98.8% ASR，无需额外进化

### 3.3 自进化（Self-Evolution）机制
#### 3.3.1 成功反思与 Skill 提取
- 攻击成功时: 分析成功轨迹，提取新的可复用 skill
- $s_{\text{new}} = \textsc{Extract}(p, \pi^*, s_{\text{used}})$
- 提取内容: 关键攻击技术、通用性评估、紧凑模板 + 适用模式

#### 3.3.2 失败反思与 Skill 改进
- 攻击失败时: 分析拒绝模式，改进现有 skill
- $s' = \textsc{Refine}(p, \pi_{\text{fail}}, r_{\text{refuse}}, s_{\text{used}})$
- 识别拒绝模式: 意图检测、关键词触发、语义抽象

#### 3.3.3 更新策略
- success_only: 仅在成功时提取新 skill
- failure_only: 仅在失败时改进现有 skill
- both: 两者兼有（可能导致 skill 爆炸 >300）
- statistical: 仅在固定间隔执行维护（最稳定，推荐默认）

### 3.4 基于 Skill 库的检索系统
#### 3.4.1 检索机制设计
- 多因子评分: $\text{score}(s, p) = q(s) \cdot (1 + \alpha \cdot |K_s \cap K_p|) \cdot \delta_{\text{type}}$
- 三个因子: 质量分 $q(s)$（效果）、关键词重叠 $|K_s \cap K_p|$（相关性）、类型匹配 $\delta_{\text{type}}$（领域对齐）
- 兼顾效果与相关性

#### 3.4.2 关键词匹配 vs. 稠密向量检索
- 选择关键词匹配的动机:
  - 降低系统复杂度: 无需 embedding 模型或额外 GPU
  - 加快响应速度: 纯文本匹配，毫秒级
  - 可解释性: 检索结果可追溯到具体关键词和 harm type
  - 实验验证: 在 jailbreak 场景下关键词匹配已足够有效

#### 3.4.3 Top-1 检索策略
- 统一使用 top-1 检索结果（single_call 模式）
- 动机: 避免多 skills 混杂干扰攻击效果
- 实验验证: single_call（top-1）比 every_iteration（动态切换）高 +12.7% ASR
- every_iteration 模式的 fallback 机制: 连续失败时依次尝试 top-2、top-3

### 3.5 Skill 库维护
- 定期维护（每 $M=50$ 次攻击）:
  - 质量剪枝: 移除低质量 skills（usage ≥ 10 且 success_rate < 10%）
  - 长度截断: 超过 $L_{\max}=500$ 字符的 skill 截断
  - 相似性聚类: Jaccard 相似度 > 0.75 的 skills 聚类
  - 聚类合并: 保留最高质量 skill，合并统计信息
  - 容量控制: 维持 $|\mathcal{S}| \leq N_{\max}=100$

---

## 4. Experiments

### 4.1 实验设置
#### 4.1.1 模型配置
- Policy model（攻击 prompt 生成）: Qwen3-4B
- Guard model（攻击评估）: Qwen3Guard-Gen-4B
- 迁移目标模型: Qwen3-0.6B, Qwen3-4B, Qwen3-14B-FP8（同族）, gpt-oss-20b（跨族）

#### 4.1.2 数据集
- 训练集: WildJailbreak 10K，train/dev/test = 8K/1K/1K
- 迁移测试集: AdvBench, HarmBench (contextual + standard), JailbreakBench
- 共 5 个数据集，覆盖不同有害行为类别和难度

#### 4.1.3 Baselines
- no_rewrite: 直接发送有害 prompt
- PAIR: 迭代式 attacker-target 对话
- AutoDAN: 遗传算法 prompt 进化
- DeepInception: 多层嵌套场景
- Persona: 角色扮演攻击
- SESS 增强变体: pair_skills, autodan_skills

#### 4.1.4 评估指标
- ASR (Attack Success Rate): 目标模型输出被 guard 判定为 Unsafe 的比例
- 严格判定: 仅 Unsafe 计为成功，Controversial 计为失败
- 辅助指标: 平均迭代次数、skill 库规模

### 4.2 策略组合实验（同模型评估）
#### 4.2.1 方法组合消融（Layer 1, 16 组）
- 三轴消融: call_mode × extraction_mode × update_strategy
- 最佳配置: single_call + trajectory + statistical → 79.1% ASR, 28 skills
- 关键发现: single_call 比 every_iteration 高 +8.3%；trajectory 比 final_prompt 高 +2.5%

#### 4.2.2 数据策略消融（Layer 2, 36 组）
- 数据量（300/500/1000）× CS/Evo 比例（30%/20%/10%）
- 关键发现: 数据量影响 <2%（数据高效）；CS 比例影响更大（early 30% 最优）
- 最佳: 80.6% ASR（300 samples + early ratio）

#### 4.2.3 强先验: DAN 模板（Layer 3-4, 29 组）
- 6 个 DAN 模板初始化 + 不同进化模式
- 关键发现: full_evolve（无 CS）达 98.8% ASR，6 个模板全程未变
- Layer 4 最佳: 99.7% ASR（medium data + evo ratio）

### 4.3 迁移实验（主要结果）
#### 4.3.1 实验设计
- 4 个目标模型 × 5 个数据集 × 7+ 方法 = 120+ 组实验
- Skills 方法使用预计算 skills（pair_skills_28, autodan_skills_54），跳过 CS+Evo
- 评估 SESS 的迁移能力: 在一个模型上进化的 skills 直接用于其他模型

#### 4.3.2 同族模型迁移分析
- Qwen3-0.6B: pair_skills 达 85.5–100%（5 数据集），autodan_skills 达 100%
- Qwen3-4B: pair_skills 达 79–100%，autodan_skills 达 100%
- Qwen3-14B: pair_skills 达 77–91%，autodan_skills 达 99–100%
- 分析: 小模型更容易被攻击（安全对齐较弱）；skills 提升在 small model 上更显著
- **表 + 图**: 跨模型 ASR 对比柱状图 / 跨数据集热力图

#### 4.3.3 跨族模型迁移分析
- gpt-oss-20b（OpenAI 架构）: 所有方法大幅下降
- pair_skills: 8.3–13%（语义模式迁移失败）
- autodan_skills: 28.1–39%（结构模式部分迁移成功）
- 相对 AutoDAN baseline 提升 +26.5pp
- 分析: 结构型攻击模式（指令覆盖）比语义型模式（特定措辞）更具跨族泛化性

#### 4.3.4 跨数据集迁移分析
- 5 个数据集上的表现差异
- 最大提升: harmbench_standard (+16pp), jailbreakBench (+16pp)
- 最小提升: advbench (+3.7pp)（PAIR 本身已较高）
- 分析: 在有害类别明确的数据集上 skill 检索更有效

### 4.4 消融实验
#### 4.4.1 去除 Cold Start 的影响
- 对比: Layer 3 full_evolve（无 CS，仅 DAN 模板）vs. Layer 3 with CS
- 结果: full_evolve 反而更好（98.8% vs. 81–95%），说明强先验下 CS 引入噪声
- 弱先验下（Layer 1/2）: CS 提供必要的初始知识

#### 4.4.2 去除 Evolution 的影响
- 对比: Ablation 实验（跳过 Evolution，仅 CS skills）vs. Layer 1（完整流程）
- 结果: Evolution 平均贡献仅 +0.9%
- 分析: skill 系统架构（表示、检索、维护）是主要贡献，进化机制是辅助

---

## 5. Conclusion

### 5.1 方法总结
- SESS 是一个轻量级可插拔的自进化 skill 系统
- 通过 skills as shortcuts 将攻击经验转化为可复用模板
- 冷启动 + 自进化 + 关键词检索 + top-1 策略，构成完整框架

### 5.2 结果总结
- 同族迁移 85–100%，跨族迁移 +26.5pp，跨数据集 +3.7–23.5pp
- 强先验（DAN）+ skill 系统 → 99.7% ASR
- Skill 系统架构 > 进化机制本身

### 5.3 应用价值
- **可迁移攻击**: 一次进化，多模型/多数据集复用
- **数据合成**: skill 库作为高质量 jailbreak prompt 模板，批量生成多样化攻击数据用于安全评估与对齐训练

---

## 6. Limitations

### 6.1 针对强护栏模型攻击效果有限
- 跨族模型（gpt-oss-20b）ASR 最高仅 39%，远低于同族 85–100%
- 不同模型家族的安全对齐机制差异巨大，skills 的泛化能力受限

### 6.2 评估范围
- 同族迁移仅限 Qwen 系列，跨族仅测试 1 个模型
- 更多模型家族（LLaMA, Mistral, Claude）的迁移能力有待验证

### 6.3 检索机制的局限
- 关键词匹配在复杂语义场景下可能不够精确
- 未来可探索轻量级语义检索（如 BM25）作为增强

---

## References

[21+ 篇参考文献，覆盖 PAIR, TAP, AutoDAN, DeepInception, GAP, Metis, ASTRA, EvoSynth, Jailbreak-R1, TROJail, xJailbreak 等]

---

## Appendix

### A. 实验详细配置
- 模型参数、超参数、硬件配置
- 完整实验结果表（所有模型 × 所有数据集）

### B. Prompts
- Cold Start 提取 prompt
- 成功反思 prompt
- 失败反思 prompt
- DAN 模板完整内容（6 个）

### C. Case Study
#### C.1 完整轨迹分析
- 展示一个完整的攻击轨迹: 原始 prompt → skill 检索 → skill 注入 → 迭代攻击 → 成功/失败
- 对比: 有 skill vs. 无 skill 的轨迹差异（迭代次数、成功率）

#### C.2 Skills 分析
- 典型 skill 示例: 内容、使用次数、成功率、适用模式
- DAN 模板使用分布: dan_mode 394 次 (100%), mcpt 78 次 (100%), devil 28 次 (85.7%)
- 进化产生的 skill 示例: 从成功轨迹中提取的策略模板
- Skill 库质量演化: 质量分分布随进化的变化

### D. Skill 库统计
- 各层实验的 skill 库规模、来源分布、质量分布
- 维护操作统计: 剪枝/合并/容量控制次数
