# SESS 论文三级大纲 (v2)

**论文标题**: SESS: Self-Evolving Skills System for Jailbreak Prompt Generation  
**目标会议**: ICLR 2027 / AAAI 2027  
**日期**: 2026-07-06  
**变更说明**: 基于 review agent 建议修改，v1 → v2。修改处标注 `[v2 新增]` 或 `[v2 修改]`。

---

## Abstract

- **背景**: 自动化 jailbreak 攻击是 LLM 安全评估的关键手段，但现有方法每次攻击独立探索，缺乏知识积累与经验复用
- **方法**: 提出 SESS（Self-Evolving Skills System），一个轻量级可插拔框架——从成功攻击轨迹中提取可复用的 attack skills as shortcuts，构建动态 skill 库，通过冷启动与自进化机制持续积累攻击经验；基于 Lexical Anchoring 检索系统兼顾效果与效率，统一使用 top-1 检索避免多 skill 混杂干扰
- **结果** `[v4 修正]`: 在同族模型上达到 99.7% ASR；在跨族模型（gpt-oss）上，揭示了结构先验对迁移能力的决定性作用——无结构先验的纯自进化 skills 仅达 8–13% ASR，而以 DAN 模板为种子的自进化 skills 达 28–39%（+26.5pp over baseline），证明结构型攻击模式是跨族泛化的关键
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

### 1.3 本文方法 `[v2 修改：加入边际成本动机]`
- 在传统攻击方法（PAIR/AutoDAN）基础上增加 SESS 系统
- **轻量化可插拔的动机——降低红队评估边际成本 (Marginal Cost of Red-teaming)**:
  - 现有方法每次新攻击都需从头探索，边际成本恒定（~20-50 次 LLM 调用/次）
  - SESS 通过 skill 库积累攻击经验，使边际成本随使用递减：skill 库构建后，每次攻击仅需 1 次检索 + 1 次生成
  - 可插拔设计使 SESS 无需替换现有红队工具链，作为插件增强已有方法
- **轻量化可插拔**: SESS 作为外部 skill 库模块，不修改 base attack 内部逻辑，通过 skill 注入即可增强任意攻击方法
- **冷启动 + 自进化**: 从少量种子数据构建初始 skill 库，通过反思机制从攻击轨迹中持续提取和改进 skills
- **低成本**: 基于 Lexical Anchoring 检索，无需 embedding 模型或额外 GPU；top-1 检索避免多 skill 混杂

### 1.4 实验结果概述 `[v4 修正]`
- 在同族模型（Qwen3-0.6B/4B/14B）上迁移 ASR 达 85–100%，显著优于 baseline（PAIR 55–88%，AutoDAN 75–92%）
- 跨族模型（GPT-OSS-20B）迁移: 无结构先验的纯自进化 skills（pair_skills）为 8–13%，以 DAN 模板为种子的自进化 skills（autodan_skills）达 28–39%，两者均显著优于各自 baseline（分别 +11.9pp 和 +26.5pp）
- 跨数据集迁移提升 +3.7–23.5pp（5 个基准数据集）
- DAN 模板初始化 + skill 系统达到 99.7% ASR

### 1.5 贡献总结 `[v4 修正]`
1. 提出 SESS 框架：轻量级可插拔的自进化 skill 系统，支持冷启动、自进化、基于 Lexical Anchoring 的 skill 检索
2. 可迁移的 jailbreak 方法：evolved skills 可直接迁移至同族模型（85–100% ASR）和跨数据集，无需重新训练；跨族迁移中结构先验（DAN 模板种子）显著提升泛化能力（28–39% vs. 纯自进化 8–13%）
3. 可用于数据合成：skill 库作为高质量 jailbreak prompt 模板，可批量生成多样化攻击数据，用于安全评估与对齐训练
4. 大规模实验验证：4 模型 × 5 数据集 × 7+ 方法的系统性迁移评估，揭示了结构先验对跨族迁移的决定性作用

---

## 2. Related Work

### 2.1 基于编排的 Jailbreak Prompt 合成方法
- **PAIR** (Chao et al., 2023): attacker-target 双模型对话，迭代优化 jailbreak prompt，~20 次查询
- **TAP** (Mehrotra et al., 2023): 树搜索 + 剪枝，构建 prompt 搜索空间
- **DeepInception** (Li et al., 2023): 多层嵌套虚拟场景
- **Crescendo** `[v2 新增]`: 多轮渐进式引导攻击，通过逐步升级对话绕过安全护栏
- **总结**: 效果好但每次攻击独立探索，成本高、无记忆、无经验积累

### 2.2 基于进化的 Jailbreak Prompt 合成方法 `[v2 修改：加入分类树]`

**自进化概念与 Skills 使用**: 介绍自进化策略在攻击领域的应用思路——从攻击反馈中提取可复用知识，使攻击系统随经验积累而增强。

**分类树——三个进化层次**:

```
基于进化的 Jailbreak 方法
├── Prompt-level Evolution（进化字符串）
│   ├── AutoDAN (Liu et al., 2023): 遗传算法 + DAN 模板，进化 prompt 种群
│   ├── AutoDAN-Turbo (Liu et al., 2024): 层次化遗传优化
│   └── GAP (Guo et al., 2024): 遗传算法，无梯度
│   → 进化对象：单个 prompt 字符串，不可复用
│
├── Method-level Evolution（进化代码/策略库）
│   ├── Metis (Chen et al., 2026): 元认知诊断 + 策略原语组合进化
│   ├── ASTRA (Liu et al., 2026): 三层动态策略库，按有效性分类
│   └── EvoSynth (Chen et al., 2026): 代码级攻击方法进化
│   → 进化对象：代码/策略库，系统臃肿，部署复杂
│
└── Skill-level Routing (Ours)（进化可插拔 Skill 模板）
    └── SESS: 进化轻量级自然语言 Skill 模板 + 检索路由
    → 进化对象：可复用的 Skill 模板，轻量、高效、可插拔
```

**对比总结**:
| 维度 | Prompt-level | Method-level | Skill-level (Ours) |
|------|-------------|-------------|-------------------|
| 进化对象 | 单个 prompt | 代码/策略库 | 可复用 skill 模板 |
| 可复用性 | ❌ 不可复用 | ✅ 可复用 | ✅ 可复用 |
| 系统复杂度 | 低 | 高 | 低 |
| 可迁移性 | ❌ | 部分 | ✅ 跨模型/跨数据集 |
| 部署成本 | 低 | 高（需额外训练） | 低（即插即用） |

### 2.3 基于强化学习与训练的方法 `[v2 新增]`
- **Jailbreak-R1** (Guo et al., 2025): 三阶段 RL 框架（SFT 冷启动 → 探索预热 → 课程学习），需完整训练流程
- **TROJail** (Xiong et al., 2026): 轨迹级 RL + 双过程奖励（隐蔽性 + 有效性）
- **xJailbreak** (Lee et al., 2025): 表示空间引导的 RL jailbreak
- **RL-MTJail** (Chen et al., 2025): 软/硬标签奖励转换的多轮 jailbreak
- **总结**: 效果强但训练成本极高（需 GPU 小时级训练），且学到的策略不可解释、不可迁移。SESS 作为免训练的即插即用方案，与这些方法形成互补。

---

## 3. Method

### 3.1 Overview
#### 3.1.1 Skills 在 Jailbreak 任务中的作用——Skills as Shortcuts
- 核心洞察: 成功攻击中蕴含可复用的策略模式（如指令覆盖、角色扮演、场景构建）
- Skills as shortcuts: 将 PAIR 等方法的高成功率但高迭代次数的攻击过程，通过从成功轨迹中提取 skills as shortcuts，实现：提升成功率 + 降低迭代次数
- 形式化定义: skill $s = (\text{content}, \text{source}, \boldsymbol{\sigma}, \text{patterns}, \text{harm\_type})$
- 质量评分: $q(s) = r_{\text{suc}} \times \sqrt{n_{\text{use}}}$

**Skill Dilution 问题** `[v2 新增]`:
- 定义: 当 skill 库中低质量或冗余 skills 增多时，检索系统被"稀释"，返回高质量 skill 的概率下降
- 数学直觉: 设库中高质量 skills 占比为 $\rho$，随机检索命中高质量 skill 的概率为 $P(\text{hit}) = \rho$；当库膨胀至 $N \gg N_{\text{useful}}$ 时，$\rho \to 0$
- 在 top-1 检索下更为关键: 仅返回 1 个 skill，dilution 直接影响攻击质量
- 应对: 通过定期维护（剪枝/合并/容量控制）保持 $\rho$ 在高水平；实验验证 `both` 策略导致 skill 爆炸 (>300) 后 ASR 下降

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
- 注: cold start 阶段 skill 库为空，攻击为裸攻击（无 skill 引导），成功后才提取首个 skills

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
- both: 两者兼有（可能导致 skill 爆炸 >300，触发 Skill Dilution）
- statistical: 仅在固定间隔执行维护（最稳定，推荐默认）

### 3.4 基于 Skill 库的检索系统
#### 3.4.1 检索机制设计
- 多因子评分: $\text{score}(s, p) = q(s) \cdot (1 + \alpha \cdot |K_s \cap K_p|) \cdot \delta_{\text{type}}$
- 三个因子: 质量分 $q(s)$（效果）、关键词重叠 $|K_s \cap K_p|$（相关性）、类型匹配 $\delta_{\text{type}}$（领域对齐）
- 兼顾效果与相关性

#### 3.4.2 Lexical Anchoring vs. Semantic Retrieval `[v2 修改：提升学术表述]`
- **Lexical Anchoring（本文方案）**: 基于关键词重叠 + harm type 匹配的检索
  - 降低系统复杂度: 无需 embedding 模型或额外 GPU
  - 加快响应速度: 纯文本匹配，毫秒级
  - 可解释性: 检索结果可追溯到具体关键词和 harm type
  - 避免 Skill Dilution: 精确匹配减少低相关 skills 的干扰
  - 实验验证: 在 jailbreak 场景下 lexical anchoring 已足够有效

- **Semantic Retrieval（对比方案）**: 基于稠密向量相似度的检索
  - 需要 embedding 模型（额外 GPU 开销）
  - 可能引入语义相近但策略无关的 skills（加剧 Skill Dilution）
  - 在 jailbreak 场景下，攻击策略的区分度更多体现在关键词层面（如 "ignore instructions"、"role-play"），而非语义嵌入空间

- **选择 Lexical Anchoring 的理由**: jailbreak skill 的核心区分度在于策略关键词（instruction override, persona adoption），而非语义细粒度差异；lexical matching 足以捕获这些区分度，同时避免 embedding 开销和 dilution 风险

#### 3.4.3 Top-1 检索策略
- 统一使用 top-1 检索结果（single_call 模式）
- 动机: 避免多 skills 混杂干扰攻击效果（与 Skill Dilution 问题呼应）
- 实验验证: single_call（top-1）比 every_iteration（动态切换）高 +12.7% ASR
- every_iteration 模式的 fallback 机制: 连续失败时依次尝试 top-2、top-3

### 3.5 Skill 库维护
- 定期维护（每 $M=50$ 次攻击）:
  - 质量剪枝: 移除低质量 skills（usage ≥ 10 且 success_rate < 10%）
  - 长度截断: 超过 $L_{\max}=500$ 字符的 skill 截断
  - 相似性聚类: Jaccard 相似度 > 0.75 的 skills 聚类
  - 聚类合并: 保留最高质量 skill，合并统计信息
  - 容量控制: 维持 $|\mathcal{S}| \leq N_{\max}=100$
- 维护的核心目标: 对抗 Skill Dilution，保持 skill 库的 precision 和 recall

---

## 4. Experiments

### 4.1 实验设置
#### 4.1.1 模型配置
- Policy model（攻击 prompt 生成）: Qwen3-4B
- Guard model（攻击评估）: Qwen3Guard-Gen-4B
- 迁移目标模型: Qwen3-0.6B, Qwen3-4B, Qwen3-14B-FP8（同族）, GPT-OSS-20B（跨族）

#### 4.1.2 数据集
- **默认数据集（WildJailbreak）**: 从 WildTeaming~\citep{wildteaming2024}（allenai/wildjailbreak）中抽取 10,000 条 jailbreak prompts，划分为训练集 8,000 条 + 测试集 1,000 条 + 验证集 1,000 条
- 训练集进一步划分为 cold start 和 evolution 子集（比例根据实验配置调整）
- **迁移测试集**: AdvBench~\citep{zou2023universal}, HarmBench~\citep{chao2024harmbench} (contextual + standard), JailbreakBench~\citep{caswell2024jailbreakbench}
- 共 5 个数据集，覆盖不同有害行为类别和难度

#### 4.1.3 Baselines `[v2 修改：扩展 baseline 列表]`
- **无攻击基线**: no_rewrite（直接发送有害 prompt）
- **基于编排的方法**: PAIR, DeepInception, Persona, Crescendo `[v2 新增]`
- **基于进化的方法**: AutoDAN
- **基于 RL 的方法（理论对比）**: `[v2 新增]`
  - TAP, Jailbreak-R1, Metis, ASTRA 等需要完整训练流程或大量计算资源
  - 本文在实验中直接对比 PAIR/AutoDAN（最广泛使用的 baseline）
  - 对于 TAP/Metis/ASTRA 等，在 Related Work 中进行理论对比（见 §2.1-2.3），并引用其公开报告的 ASR 数据作为参考
- **SESS 增强变体**: pair_skills, autodan_skills

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

#### 4.3.3 跨族模型迁移分析 `[v4 修正]`
- GPT-OSS-20B（OpenAI 架构）: 所有方法大幅下降
- **结构先验对迁移能力的决定性作用**:
  - pair_skills_28（纯自进化 skills，无结构先验）: 8.3–13%（语义模式跨族迁移失败）
  - autodan_skills_54（DAN 模板为种子 + 自进化）: 28.1–39%（结构模式跨族迁移成功）
  - 相对 baseline 提升: pair_skills +11.9pp over pair, autodan_skills +26.5pp over autodan
- **核心发现**:
  - 无结构先验的纯自进化 skills 在同族迁移中有效（85–100%），但跨族迁移能力有限（8–13%）
  - 以 DAN 模板为种子的自进化 skills 跨族迁移效果显著提升（28–39%），证明结构型攻击模式是跨族泛化的关键
  - 自进化机制在 DAN 种子基础上进一步优化（autodan_skills_54 包含 DAN 模板 + 48 个进化产物），但核心泛化能力来自结构先验
- **Case Study**（详见 Appendix C.3）:
  - 对比 pair_skills（纯自进化）与 autodan_skills（DAN 种子 + 自进化）在 gpt-oss 上的攻击轨迹
  - 直观展示结构型 skill vs. 语义型 skill 的跨族泛化差异

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
- 冷启动 + 自进化 + Lexical Anchoring 检索 + top-1 策略，构成完整框架

### 5.2 结果总结 `[v4 修正]`
- 同族迁移 85–100%，跨族迁移中结构先验（DAN 种子）显著提升泛化能力（28–39% vs. 纯自进化 8–13%），跨数据集 +3.7–23.5pp
- 强先验（DAN）+ skill 系统 → 99.7% ASR
- Skill 系统架构 > 进化机制本身
- 结构先验是跨族泛化的关键：无结构先验的自进化 skills 在同族有效但跨族受限，DAN 种子 + 自进化显著提升跨族迁移

### 5.3 应用价值 `[v4 修正]`
- **可迁移攻击（同族）**: 一次进化，同族多模型复用（85–100% ASR）
- **可迁移攻击（跨族）**: 结构先验（DAN 种子）是跨族泛化的关键；纯自进化 skills 跨族受限（8–13%），DAN 种子 + 自进化显著提升（28–39%）；未来可探索更通用的结构型 skill 设计
- **数据合成**: skill 库作为高质量 jailbreak prompt 模板，批量生成多样化攻击数据用于安全评估与对齐训练

---

## 6. Limitations

### 6.1 对强防御模型的适应性 `[v4 修正]`
- 跨族模型（GPT-OSS-20B）上，纯自进化 skills（pair_skills）仅 8–13% ASR，DAN 种子 + 自进化（autodan_skills）达 28–39%
- 这说明结构先验对跨族迁移至关重要：无结构先验的自进化倾向于语义层面优化，难以突破结构层面的防御
- 当前 Skill 表示（自然语言模板）可能还未触及模型安全对齐的底层几何空间
- 更强的防御模型（如经过 adversarial training 的模型）可能需要更深层的 attack 策略
- Future Work: 
  - 探索如何引导自进化机制发现结构型攻击模式（而非仅优化语义）
  - 探索与表示空间分析结合的 skill 设计（如 xJailbreak 的思路）
  - 设计更通用的结构型 skill 种子，提升跨族泛化能力

### 6.2 评估范围
- 同族迁移仅限 Qwen 系列，跨族仅测试 1 个模型
- 更多模型家族（LLaMA, Mistral, Claude）的迁移能力有待验证

### 6.3 检索机制的局限
- Lexical Anchoring 在复杂语义场景下可能不够精确
- Future Work: 探索轻量级语义检索（如 BM25）作为增强

---

## References

[21+ 篇参考文献，覆盖 PAIR, TAP, AutoDAN, DeepInception, Crescendo, GAP, Metis, ASTRA, EvoSynth, Jailbreak-R1, TROJail, xJailbreak, RL-MTJail 等]

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

#### C.3 跨族迁移 Case Study `[v4 修正]`
- Case A: pair_skills（纯自进化，无结构先验）在 GPT-OSS-20B 上——分析失败原因（语义型 skill 无法跨族泛化）
- Case B: autodan_skills（DAN 种子 + 自进化）在 GPT-OSS-20B 上——分析成功原因（结构型 skill 跨族泛化）
- 对比分析: 结构先验 vs. 无结构先验的 skill 在策略层面的差异，直观展示为何 DAN 种子对跨族迁移至关重要

### D. Skill 库统计
- 各层实验的 skill 库规模、来源分布、质量分布
- 维护操作统计: 剪枝/合并/容量控制次数

### E. 与 RL-based 方法的理论对比 `[v2 新增]`
- 对比维度: 训练成本、可迁移性、可解释性、部署复杂度
- 引用 Jailbreak-R1、TROJail 等论文中报告的 ASR 数据作为参考

---

## v1 → v2 → v4 变更汇总

| # | 位置 | 变更内容 | 版本 | 状态 |
|---|------|----------|------|------|
| 1 | Abstract 结果 | 改为"首次揭示人工先验与自动进化的能力边界" | v2 | ✅ 采纳 |
| 1' | Abstract 结果 | 修正为"结构先验对迁移能力的决定性作用"，区分 pair_skills vs. autodan_skills | v4 | ✅ 修正 |
| 2 | §1.3 | 加入"边际成本 (Marginal Cost of Red-teaming)"动机 | v2 | ✅ 采纳 |
| 3 | §1.5 贡献 2 | 改为"揭示泛化边界" | v2 | ❌ 不采纳 |
| 3' | §1.5 贡献 2 | 修正为"结构先验显著提升跨族泛化能力" | v4 | ✅ 修正 |
| 4 | §2.3 | 新增"基于强化学习与训练的方法"小节 | v2 | ✅ 采纳 |
| 5 | §2.2 | 加入三层次分类树（Prompt/Method/Skill-level） | v2 | ✅ 采纳 |
| 6 | §3.1.1 | 加入 Skill Dilution 的形式化解释 | v2 | ✅ 采纳 |
| 7 | §3.4.2 | 改为 "Lexical Anchoring vs. Semantic Retrieval" | v2 | ✅ 采纳 |
| 8 | §4.1.3 | 加入 TAP/Crescendo/Metis/ASTRA 作为 baseline（理论对比） | v2 | ✅ 部分采纳 |
| 9 | §4.3.3 | 增加跨族迁移 Case Study 指引 | v2 | ✅ 采纳 |
| 9' | §4.3.3 | 修正为"结构先验对迁移能力的决定性作用"，明确 pair_skills vs. autodan_skills | v4 | ✅ 修正 |
| 10 | §6.1 | 扩展"对强防御模型的适应性"，加入底层几何空间讨论 | v2 | ✅ 采纳 |
| 10' | §6.1 | 修正为"结构先验对跨族迁移至关重要" | v4 | ✅ 修正 |

### v4 修正说明

v2 中存在一处关键逻辑矛盾：将 autodan_skills 错误描述为"纯人工先验"，实际上：
- **pair_skills_28**: 纯自进化 skills（从 PAIR 轨迹提取，无 DAN 模板），跨族 ASR 8–13%
- **autodan_skills_54**: DAN 模板为种子 + 自进化（54 个 skills = 6 DAN + 48 进化产物），跨族 ASR 28–39%

v4 修正了全文 7 处相关表述，统一为"结构先验对跨族迁移的决定性作用"：
1. Abstract
2. §1.4 实验结果概述
3. §1.5 贡献总结
4. §4.3.3 跨族模型迁移分析
5. §5.2 结果总结
6. §5.3 应用价值
7. §6.1 Limitations
8. Appendix C.3 Case Study
