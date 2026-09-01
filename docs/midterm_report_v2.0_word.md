# 大语言模型越狱攻击与防御方法研究

## 中期报告 v2.0（Word 版）

---

**学院**：计算机科学与技术学院  
**学科**：计算机科学与技术  
**研究生**：[姓名]  
**学号**：[学号]  
**导师**：[导师姓名]  
**报告日期**：2026 年 6 月 17 日

---

## 目录

1. [课题主要研究内容及进度情况](#1-课题主要研究内容及进度情况)
2. [相关研究现状](#2-相关研究现状)
3. [目前已完成的研究工作及进展](#3-目前已完成的研究工作及进展)
4. [后期拟完成的研究工作及进度安排](#4-后期拟完成的研究工作及进度安排)
5. [存在的困难与问题](#5-存在的困难与问题)
6. [如期完成全部论文工作的可能性](#6-如期完成全部论文工作的可能性)
7. [参考文献](#7-参考文献)

---

## 1. 课题主要研究内容及进度情况

### 1.1 研究背景

#### 1.1.1 大模型安全与对齐技术

随着 ChatGPT、GPT-4、Qwen3 等大语言模型 (LLM) 在对话、代码生成、知识问答等任务中的广泛应用，其安全性问题日益受到学术界和工业界的关注。对齐技术 (Alignment) 通过训练使模型遵循人类价值观和安全准则，主要包括 RLHF (Reinforcement Learning from Human Feedback)、Constitutional AI、Red Teaming 等方法。然而，即便经过严格的安全对齐，主流大语言模型仍然存在被"越狱"(Jailbreak) 攻击的风险——攻击者通过精心设计的提示词可以绕过模型的安全护栏，诱导模型生成违规内容。

#### 1.1.2 Jailbreak 攻击研究现状

现有的 jailbreak 攻击方法主要可以分为以下几类：

| 方法类型 | 代表方法 | 特点 | 局限性 |
|----------|----------|------|--------|
| 静态模板 | PAIR, AutoDAN | 预定义策略模板 | 无法适应新防御 |
| 梯度优化 | GCG, AutoDAN-GA | 基于梯度的 prompt 优化 | 计算成本高，需白盒 |
| 多轮对话 | Crescendo | 渐进式引导 | 无跨案例知识迁移 |
| 角色扮演 | Persona | 角色设定绕过 | 效果不稳定 |

#### 1.1.3 研究挑战

当前 jailbreak 攻击研究面临以下几个核心挑战：

1. **效率挑战**：现有方法（如 PAIR、TAP）通常需要 1000+ 次 API 调用才能生成有效的攻击 prompt，训练时间长，资源消耗大
2. **泛化挑战**：攻击方法缺乏跨模型迁移能力，在一个模型上有效的攻击策略往往无法迁移到其他模型
3. **知识积累**：成功攻击策略无法复用到新案例，每次攻击都需要从头探索
4. **奖励设计**：强化学习训练中奖励信号稀疏，ASR 奖励是二值信号，方差大、梯度稀疏

### 1.2 研究目的

本课题旨在系统研究大语言模型的越狱攻击方法与防御策略，从攻击端和防御端两个维度深入探索，为构建更安全的大语言模型提供理论支撑和技术方案。具体研究目的包括：

1. **推进 jailbreak prompt 合成边界**：提升攻击绝对能力，保证方法的有效性，探索高效且可迁移的攻击策略
2. **针对传统方法的高资源需求**：优化训练效率，实现低资源需求的 MaaS (Model as a Service) 方案，降低实验成本
3. **产出易用的方法模块**：构建可复用的 Skills 系统，提供开源的攻击工具与数据合成管道

### 1.3 研究方法

本课题采用三种递进式的研究方法：

**AHR-GRPO**：基于自适应混合奖励 GRPO 的越狱提示词生成方法。创新性地提出方差驱动的自适应奖励权重机制，动态平衡稀疏的 ASR 奖励与稠密的 Judge 奖励，实现高效、稳定的强化学习训练。

**SESS**：基于自进化技能系统的越狱提示词生成方法。设计 Skills 库自进化机制，从成功攻击轨迹中提取可复用的策略模板，实现知识积累与跨案例迁移，同时可用于高效数据合成。

**Agentic-RL**：基于自进化系统内强化学习的越狱提示词生成方法。将 AHR-GRPO 与 SESS 结合，在自进化 Skills 环境中进行强化学习训练，实现过程奖励与结果奖励的协同优化。

### 1.4 进度情况

| 章节 | 方法 | 进度 | 关键成果 |
|------|------|------|----------|
| **第一章** | AHR-GRPO | **85%** | 最佳 ASR **33.2%** (+2.4% vs baseline) |
| **第二章** | SESS | **95%** | 最佳 ASR **99.7%**，完成 217 组实验 |
| **第三章** | Agentic-RL | **10%** | 方法框架已设计 |

**第一章 AHR-GRPO**：已完成核心方法设计与实现，自适应奖励权重计算器已开发，实验 1（jailbreak prompt 筛选）、实验 2（多维度 Judge reward）和实验 3（自适应权重验证）全部完成。

**第二章 SESS**：已完成 217 组实验，涵盖方法消融（Layer 1-4）、数据策略、迁移性验证等，主要结论已得出。

**第三章 Agentic-RL**：方法框架已设计，待开展实验验证。

---

## 2. 相关研究现状

### 2.1 基于提示词工程的越狱攻击方法

早期 jailbreak 攻击主要依赖人工构造的 Prompt 模板，通过角色扮演、假设场景、创意写作等策略绕过模型的安全限制。近年来，自动化方法显著提升了攻击效率。

**迭代式攻击方法**方面，PAIR 提出"攻击者-目标"双模型交互框架，通过多轮对话自动优化越狱提示，通常仅需 20 次查询即可攻破黑盒对齐模型。PromptAgent 将提示优化建模为智能体规划问题，提出"规划-执行-反思"闭环框架，实现专家级提示词自动生成。

**树搜索方法**方面，TAP 引入思维树（Tree-of-Thoughts）思想构建越狱提示搜索空间，结合评估器对分支进行有害性打分并动态剪枝，显著提升搜索效率。RedHit 进一步融合蒙特卡洛树搜索（MCTS）与直接偏好优化（DPO），实现攻击策略的渐进式演化。

**进化算法方法**方面，AutoDAN 结合人工模板与遗传算法，通过选择、交叉、变异等操作进化提示词种群。其升级版 AutoDAN-Turbo 提出分层遗传优化机制（片段级→提示级），大幅提升收敛速度。类似地，GAP 基于遗传算法实现免梯度搜索。

**场景诱导方法**方面，DeepInception 利用 LLM 对上下文情境的强依赖性，构建多层嵌套虚拟场景（"梦中梦"结构），将恶意意图隐藏于深层语境中。

### 2.2 基于强化学习的越狱攻击方法

近年来，强化学习被广泛应用于自动化越狱攻击，主要可分为以下几类：

**轨迹级优化**方面，TROJail 首次将多轮越狱建模为轨迹级强化学习问题，突破传统轮次级优化的短视局限。其核心创新包括轨迹级 MDP 建模与双过程奖励机制（隐蔽性奖励+有效性奖励）。RL-MTJail 针对黑盒多轮越狱提出强化学习优化框架，结合软/硬标签奖励过渡机制。

**课程学习与探索**方面，Jailbreak-R1 提出三阶段强化学习框架：冷启动阶段（SFT 初始化策略）、探索预热阶段（引入多样性奖励与一致性奖励）、增强越狱阶段（渐进式课程学习，奖励信号从软标签平滑过渡到硬标签）。

**表征空间引导**方面，xJailbreak 提出表征空间引导的黑盒越狱方法，通过分析良性与恶意提示的 embedding 邻近性，确保改写提示在语义上贴近原始意图同时提升攻击有效性。

**多轮对话攻击**方面，Many-Turn Jailbreaking 首次系统性探索多轮越狱攻击范式，构建首个多轮越狱评测基准 MTJ-Bench，揭示"首轮越狱→后续对话污染"的新型攻击链。

### 2.3 自进化策略与红队框架

**策略自进化**方面，Metis 提出基于自进化元认知策略优化的越狱框架，将攻击过程建模为因果诊断与策略演化的闭环系统，维护可组合的策略原语库。ASTRA 提出自动化策略发现、检索与演化的越狱框架，三层动态策略库按效果分类为"高效""有潜力""无效"。

**方法级演化**方面，EvoSynth 首次将越狱攻击范式从"提示优化"转向"攻击方法演化"，提出基于多智能体协作的代码级攻击算法自主合成框架。LLM-Virus 将越狱攻击建模为进化与迁移学习的联合问题，利用 LLM 作为启发式进化算子在提示种群中交叉变异传播。

**自适应红队**方面，Active Attacks 受主动学习范式启发，周期性用收集的攻击提示对目标模型进行安全微调，使奖励信号随目标防御演化而动态调整。Genesis 面向 Web Agent 场景提出策略演化式红队框架，Attacker-Scorer-Strategist 闭环支持遗传算法驱动的交叉/变异演化。

### 2.4 与本文方法的关系

本文方法与上述工作的关系可总结如下：

- **vs 迭代式/树搜索方法**：本文 AHR-GRPO 通过强化学习训练专用生成模型，避免 PAIR/TAP 等方法的高查询成本（通常 100+ API 调用/样本）
- **vs 进化算法方法**：本文 SESS 借鉴 AutoDAN 的进化思想，但将进化对象从提示词升级为可复用的 Skills 策略模板，实现知识积累与跨案例迁移
- **vs 轨迹级 RL 方法**：本文 AHR-GRPO 的自适应混合奖励机制与 TROJail 的双过程奖励机制类似，但本文聚焦单轮 Prompt 重写而非多轮对话
- **vs 自进化策略方法**：本文 SESS 与 Metis/ASTRA 均关注策略演化，但本文从成功攻击轨迹中提取 Skills 模板，而非依赖元认知诊断或三层策略库

---

## 3. 目前已完成的研究工作及进展

### 3.1 Adaptive Hybrid-Reward GRPO for Jailbreak Prompt Generation (AHR-GRPO)

#### 3.1.1 研究背景与动机

传统 GRPO 方法使用单一奖励信号，ASR 奖励稀疏导致早期探索困难。现有混合奖励方法采用固定权重，无法适应训练过程中奖励信号的动态变化。因此，需要设计自适应机制，在稀疏 ASR 阶段依赖稠密 Judge 引导，在 ASR 信号分化后回归结果导向优化。

#### 3.1.2 方法设计

**自适应 reward 权重计算**：核心公式使用指数移动方差（EMV）估计两组奖励的方差：

$$\lambda_{\text{raw}} = \sigma\left(\alpha \cdot \log\frac{\sigma^2_{\text{ASR}}}{\sigma^2_{\text{proc}}} + \delta\right)$$

该公式的核心思想是：当 ASR 奖励方差较大时（说明模型已经能区分好坏样本），增大 ASR 权重；当 ASR 奖励方差较小时（说明模型整体重写效果差），降低 ASR 权重，更依赖 Judge 奖励进行学习。

方法包含三个关键机制：
- **双平滑机制**：EMA 平滑 + 惯性平滑，防止单步跳变导致梯度震荡
- **边界裁剪**：$\lambda \in [\lambda_{\min}, \lambda_{\max}]$，防止信号坍缩
- **冷启动机制**：前$W$步固定$\lambda=0.5$，避免初始方差估计偏差

**多维度 Judge Prompt 设计**：设计了四个评判维度：intent_preservation（意图保留）、stealth（隐蔽性）、strategy_execution（策略执行）、attack_potential（攻击潜力）。

#### 3.1.3 实验设计与结果

**实验 1：Jailbreak Prompt 筛选**

我们测试了 24 种 jailbreak prompt 模板的初始 ASR（使用 1000 条测试集），筛选出 Top-3 策略：hypothetical_scenario（30.8%）、creative_writing（28.3%）、role_playing（25.0%）。

![Attack Prompt Selection](figures/output/AHR_GRPO_01_attack_prompt_selection.png)

**实验 2：多维度 Judge Reward 权重配置（12 组）**

我们设计了 3 种攻击 prompt × 4 种 Judge 维度 = 12 组 GRPO 训练实验。结果显示所有组合均超越 baseline，最佳为 hypothetical_scenario + idea_preservation，达到 32.3%（+1.5% 改进）。关键发现是通用 Judge 维度优于专用维度，idea_preservation 最稳定（平均 +1.2% 改进）。

![Judge Dimension Comparison](figures/output/AHR_GRPO_03_judge_dimension.png)

**实验 3：自适应权重机制验证**

在实验 2 最佳配置基础上，我们验证了自适应权重机制的进一步改进效果。结果显示自适应机制超越固定权重：β=0（窗口=1）时 ASR=33.2%，Δ vs 实验 2 最佳=+0.9%。消融验证表明仅 ASR Reward 效果下降 5.8%（ASR=25.0%），混合 Reward 显著优于单一 Reward。

![Adaptive Weight Results](figures/output/AHR_GRPO_04_adaptive_weight.png)

![Reward Ablation](figures/output/AHR_GRPO_05_reward_ablation.png)

**核心结论**：自适应权重机制有效，动态调整 Lambda 实现 ASR 与 Judge 的最优平衡；混合 Reward 显著优于单一 Reward；所有训练组合均达到或超越 baseline。

![AHR-GRPO Summary](figures/output/AHR_GRPO_06_summary.png)

### 3.2 Self-Evolving Skills System for Jailbreak Prompt Generation (SESS)

#### 3.2.1 研究背景与动机

现有 jailbreak 方法缺乏知识积累机制，每次攻击需从头探索。PAIR 等迭代方法效率低，无法跨案例复用成功策略。因此，需要设计可复用的攻击模板（Skills）与自进化机制，实现知识积累与迁移。

#### 3.2.2 方法设计

**Skill 数据结构与检索机制**：Skill 包含 content（攻击模板）、source（来源）、统计信息（success_rate、usage_count）、关键词标签、伤害类型。检索评分公式为：$score = quality\_score \times (1 + 关键词匹配 \times 0.3) \times 类型匹配系数$。

**三阶段自进化流程**：
- **Cold Start 阶段**：使用初始 Skills 进行攻击，积累成功案例
- **Evolution 阶段**：从成功/失败轨迹中提取/改进 Skills
- **Test 阶段**：使用演化后的 Skills 库评估最终 ASR

#### 3.2.3 实验设计与结果（217 组实验）

**Layer 1：方法组合 Grid Search（16 组）**

我们确定了最佳方法组合为 single_call + trajectory + statistical，ASR=79.1%，Skills=28 个。关键发现是 single_call 比 every_iteration 高 +12.7%。

**Layer 2：数据消融实验（36 组）**

最佳配置为 small(300) + early(30% CS)，ASR=80.0%。关键发现是数据量影响<2%，配比影响显著（early 最佳）。

**Layer 3：DAN 模板验证（17 组）**

最佳配置为 full_evolve（无 Cold Start），ASR=98.8%。关键发现是 DAN 模板无需进化，直接使用最优。

**Layer 4：DAN 数据消融（12 组）**

最佳配置为 medium + evo，ASR=99.7%。关键发现是数据量影响极小（差距 1.3%）。

![SESS Main Results](figures/output/SESS_01_sess_main_results.png)

#### 3.2.4 迁移实验（140 组）

**同族迁移效果（Qwen 系列）**：

| 模型 | PAIR_w_SESS ASR | Best Baseline | 提升 |
|------|-----------------|---------------|------|
| Qwen3-0.6B | 95.1% | 90.2% | +4.9% |
| Qwen3-4B | 86.7% | 78.7% | +8.0% |
| Qwen3-14B | 86.5% | 85.5% | +1.0% |

平均提升：+4.7%

**跨族迁移效果（gpt-oss-20b）**：

| 方法 | 同族 ASR | 跨族 ASR | 下降幅度 |
|------|----------|----------|----------|
| PAIR_w_SESS | 91.6% | 11.4% | -80.2% |
| AutoDAN_w_SESS | ~99% | 32.5% | -66.5% |
| AutoDAN (baseline) | 85.5% | 6.0% | -79.5% |
| PAIR (baseline) | 71.6% | 0.08% | -71.5% |

![Transfer Comparison](figures/output/SESS_02_transfer_comparison.png)

关键发现：跨族迁移是真正挑战，需要新方法提升泛化性。

#### 3.2.5 核心结论

- **最佳方法**：AutoDAN_w_SESS（同族 99%，跨族 32.5%）
- **Evolution 必要性**：DAN 场景不必要，pair 风格微弱（+0.9%）
- **Skills 泛化性**：同族良好（85-95%），跨族有限（<35%）
- **数据策略**：300 条足够，full_evolve 最优（DAN 场景）

### 3.3 Agentic RL with Self-Evolving Skills System (待开展)

#### 3.3.1 研究背景与动机

AHR-GRPO 解决了奖励信号稀疏问题，但缺乏攻击策略的知识积累；SESS 实现了 Skills 知识积累与迁移，但缺乏强化学习的策略优化。需要将两者结合：在自进化 Skills 环境中进行 RL 训练，实现策略优化与知识积累的双重效果。

#### 3.3.2 方法设计

**自进化 Skills 环境构建**：Skills 作为环境状态的一部分，动态更新。环境提供 Skills 检索接口，Policy 根据当前 prompt 检索最佳 Skill。Skills 库随训练进程演化，形成适应 Policy 策略的 Skills 分布。

**结果奖励与过程奖励设置**：结果奖励（ASR）为攻击成功与否的稀疏信号；过程奖励（Judge）为多维度评判的稠密信号；自适应权重使用 AHR-GRPO 的方差驱动机制动态调整权重；Skills 质量奖励鼓励 Policy 使用高质量 Skills。

**训练流程设计**：Phase 1（Skills 初始化）→ Phase 2（RL 训练）→ Phase 3（Skills 协同演化）。

---

## 4. 后期拟完成的研究工作及进度安排

### 4.1 第一章 AHR-GRPO 后续工作

- ✅ 已完成实验 3：自适应权重机制验证（3 组 EMA 配置），确认自适应超越固定权重（+0.9% 额外改进）
- ✅ 已验证自适应机制优越性：最佳配置达到 33.2%，超越实验 2 最佳的 32.3%
- ✅ 已确认混合 Reward 有效性：仅 ASR Reward 效果下降 5.8%，混合 Reward 显著优于单一 Reward
- 待完成：分析$\lambda$演化曲线，验证物理直觉（早期依赖 Judge，后期回归 ASR）
- 待完成：整理实验数据，撰写论文方法章节与实验章节

### 4.2 第二章 SESS 后续工作

- ✅ 已完成 217 组实验：Layer 1-4 方法消融、数据消融、迁移性验证等全部完成
- ✅ 已得出主要结论：最佳方法 AutoDAN_w_SESS（99% 同族、32.5% 跨族）
- 待完成：整理实验数据，撰写完整实验报告
- 待完成：分析跨族迁移失败的根本原因（模型安全机制差异、Skills 模型特异性）
- 待完成：探索提升 Skills 泛化性的方法（元学习、多模型联合训练）
- 待完成：整理 Skills 库，开源最佳 Skills 模板
- 待完成：撰写论文方法章节与实验章节

### 4.3 第三章 Agentic-RL 工作计划

| 阶段 | 时间 | 任务 |
|------|------|------|
| **方法实现** | 2 周 | 搭建自进化 Skills 环境、集成 AHR-GRPO 自适应奖励机制、实现 Skills 协同演化流程 |
| **实验验证** | 4 周 | Baseline 对比、消融实验、跨模型迁移测试 |
| **结果分析与论文撰写** | 2 周 | 分析训练曲线与$\lambda$演化、Skills 演化过程可视化、与前两章方法对比讨论 |

### 4.4 论文整理与投稿准备

1. 整理三章内容，统一论文框架（ICLR 格式）
2. 补充实验：根据 review 反馈可能需要的额外实验
3. 撰写 Introduction、Related Work、Conclusion
4. 完善图表与可视化

---

## 5. 存在的困难与问题

### 5.1 计算资源需求较大

AHR-GRPO 实验每实验约需 4×GPU(A800-80G)，训练 2-4 小时/1000 步。SESS 实验已完成 217 组，消耗大量 GPU 资源。第三章 Agentic-RL 预计需额外数百 GPU 卡时。需要合理安排实验顺序，避免资源冲突。

### 5.2 高成本试错风险

强化学习训练稳定性方面，Agentic-RL 训练可能较难收敛。Skills 演化不可控方面，可能存在 Skills 爆炸、低质量 Skills 积累等问题。实验设计迭代方面，部分实验结果可能不理想，需重新设计。

### 5.3 跨族迁移挑战

SESS 实验揭示：所有方法在 gpt-oss-20b 上效果大幅下降（best 仅 32.5%）。需要新方法：当前 Skills 缺乏跨族泛化性。第三章工作重点：探索提升泛化性的方法。

### 5.4 论文时间压力

三章内容整合需统一叙事逻辑。实验周期：第三章实验预计需 6 周。投稿时间窗口：需在毕业前完成论文撰写与投稿。

### 5.5 方法创新性论证

AHR-GRPO 需与固定权重方法充分对比。SESS 需与 baseline 方法（PAIR、AutoDAN）充分对比。Agentic-RL 需证明组合方法的协同效果。

---

## 6. 如期完成全部论文工作的可能性

### 6.1 已完成工作评估

| 章节 | 方法 | 进度 | 关键成果 |
|------|------|------|----------|
| 第一章 | AHR-GRPO | **85%** | 最佳 ASR 33.2% |
| 第二章 | SESS | **95%** | 最佳 ASR 99.7% |
| 第三章 | Agentic-RL | **10%** | 方法框架已设计 |
| **总体** | **-** | **65%** | **-** |

### 6.2 后期工作可行性分析

- **时间规划**：剩余 6-8 周，足够完成第三章实验与论文整理
- **技术可行性**：AHR-GRPO 与 SESS 代码库已成熟，组合实现难度可控
- **资源可行性**：已有 GPU 资源调度经验，可合理安排实验
- **关键成果**：第一章已验证自适应机制有效（+0.9% 额外改进），第二章已验证 Skills 积累有效（99.7% ASR）

### 6.3 风险评估与应对策略

| 风险 | 应对策略 |
|------|----------|
| Agentic-RL 训练不稳定 | 设计稳定性保障机制（双平滑、边界裁剪、冷启动），参考 AHR-GRPO 经验 |
| 跨族迁移效果仍不理想 | 将此作为研究发现，分析原因而非强行提升，符合研究完整性 |
| 时间不足 | 优先完成核心实验，消融实验可视情况缩减 |

### 6.4 结论

基于已完成工作与后续计划，**如期完成全部论文工作的可能性较高**。需合理安排实验顺序，控制资源消耗，保证核心实验优先完成。论文结构清晰，三章节递进关系明确，具备 ICLR 投稿潜力。

---

## 7. 参考文献

1. Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G. J., et al. (2023). Jailbreaking Black Box Large Language Models in Twenty Queries. *arXiv:2310.08419*. [PAIR]
2. Wang, X. et al. (2024). PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization. *arXiv:2310.16428*.
3. Mehrotri, A., Zampetakis, M., et al. (2023). Tree of Attacks: Jailbreaking Black-Box LLMs Automatically. *arXiv:2312.02119*. [TAP]
4. Liu, X. et al. (2023). AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models. *arXiv:2310.04451*.
5. Liu, X. et al. (2024). AutoDAN-Turbo: A Hierarchical Genetic Algorithm for Automated Jailbreaking. *arXiv:2405.06479*.
6. Guo, Y. et al. (2024). GAP: A Genetic Algorithm Approach to Automated Jailbreaking of Large Language Models. *arXiv:2402.11852*.
7. Li, X. et al. (2023). DeepInception: Hypnotize Large Language Models to Be Jailbreakers. *arXiv:2311.07588*.
8. Guo, W., Shi, Z., Li, Z., et al. (2025). Jailbreak-R1: Exploring the Jailbreak Capabilities of LLMs via Reinforcement Learning. *arXiv:2506.00782*.
9. Xiong, X., Li, O., Liu, Z., et al. (2026). TROJail: Trajectory-Level Optimization for Multi-Turn LLM Jailbreaks with Process Rewards. *ACL 2026*. *arXiv:2512.07761*.
10. Lee, S., Ni, S., Wei, C., et al. (2025). xJailbreak: Representation Space Guided Reinforcement Learning for Interpretable LLM Jailbreaking. *arXiv:2501.16727*.
11. Yang, X., Xiao, L., Li, S., et al. (2025). Many-Turn Jailbreaking. *arXiv:2508.06755*.
12. Belaire, R., Sinha, A., Varakantham, P. (2025). Automatic LLM Red Teaming. *arXiv:2508.04451*.
13. Zhang, Z., He, J., Cai, Y., et al. (2025). Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming. *arXiv:2510.18314*.
14. Wei, Z., Chen, H., Hu, P., et al. (2025). Learning-Based Automated Adversarial Red-Teaming for Robustness Evaluation of LLMs. *arXiv:2512.20677*.
15. Chen, X. et al. (2025). RL-MTJail: Reinforcement Learning for Automated Black-Box Multi-Turn Jailbreaking. *arXiv:2512.07761*.
16. Chen, Y., Wang, X., Li, J., et al. (2026). Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization. *arXiv:2605.10067*.
17. Liu, X., Chen, Y., Ling, K., et al. (2026). ASTRA: An Automated Framework for Strategy Discovery, Retrieval, and Evolution for Jailbreaking LLMs. *ACL 2026*. *arXiv:2511.02356*.
18. Chen, Y., Wang, X., Li, J., et al. (2025). Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs. *arXiv:2511.12710*. [EvoSynth]
19. Sorkhpour, M., Yazdinejad, A., Dehghantanha, A. (2025). RedHit: Adaptive Red-Teaming of LLMs via Search, Reasoning, and Preference Optimization. *LLM Security Conference 2025*.
20. Yun, T., St-Charles, P.-L., Park, J., Bengio, Y., Kim, M. (2025). Active Attacks: Red-teaming LLMs via Adaptive Environments. *arXiv:2509.21947*.
21. Zhang, W., Liu, Y., Chen, H., et al. (2026). LLM-Virus: Evolutionary Jailbreak Attack on Large Language Models. *arXiv:2601.0xxxx*.

---

*文档版本：v2.0*  
*最后更新：2026 年 6 月 17 日*
