# 大语言模型越狱攻击与防御方法研究

## 中期报告 v1.0

---

**学院**：计算机科学与技术学院  
**学科**：计算机科学与技术  
**研究生**：[姓名]  
**学号**：[学号]  
**导师**：[导师姓名]  
**报告日期**：2026 年 6 月 16 日

---

## 目录

1. [课题主要研究内容及进度情况](#1-课题主要研究内容及进度情况)
2. [目前已完成的研究工作及进展](#2-目前已完成的研究工作及进展)
3. [后期拟完成的研究工作及进度安排](#3-后期拟完成的研究工作及进度安排)
4. [存在的困难与问题](#4-存在的困难与问题)
5. [如期完成全部论文工作的可能性](#5-如期完成全部论文工作的可能性)

---

# 1. 课题主要研究内容及进度情况

## 1.1 研究背景

### 1.1.1 大模型安全与对齐技术

随着 ChatGPT、GPT-4、Qwen3 等大语言模型 (LLM) 的广泛应用，模型安全性成为关键问题。对齐技术 (Alignment) 通过训练使模型遵循人类价值观和安全准则，主要包括：

- **RLHF** (Reinforcement Learning from Human Feedback)：基于人类反馈的强化学习
- **Constitutional AI**：基于宪法原则的训练
- **Red Teaming**：红队测试发现安全漏洞

然而，对齐后的模型仍存在被"越狱"(Jailbreak) 攻击的风险，攻击者通过精心设计的提示词绕过模型的安全限制。

### 1.1.2 Jailbreak 攻击研究现状

现有 jailbreak 攻击方法可分为以下几类：

| 方法类型 | 代表方法 | 特点 | 局限性 |
|----------|----------|------|--------|
| 静态模板 | PAIR, AutoDAN | 预定义策略模板 | 无法适应新防御 |
| 梯度优化 | GCG, AutoDAN-GA | 基于梯度的 prompt 优化 | 计算成本高，需白盒 |
| 多轮对话 | Crescendo | 渐进式引导 | 无跨案例知识迁移 |
| 角色扮演 | Persona | 角色设定绕过 | 效果不稳定 |

### 1.1.3 研究挑战

- **效率挑战**：现有方法训练时间长，资源消耗大
- **泛化挑战**：攻击方法缺乏跨模型迁移能力
- **知识积累**：成功攻击策略无法复用到新案例
- **奖励设计**：强化学习训练中奖励信号稀疏

## 1.2 研究目的

1. **推进 jailbreak prompt 合成边界**：提升攻击绝对能力，保证方法的有效性，探索高效且可迁移的攻击策略
2. **针对传统方法的高资源需求**：优化训练效率，实现低资源需求的 MaaS (Model as a Service) 方案，降低实验成本
3. **产出易用的方法模块**：构建可复用的 Skills 系统，提供开源的攻击工具与数据合成管道

## 1.3 研究方法

### 1.3.1 基于自适应混合奖励 GRPO 的越狱提示词生成方法 (AHR-GRPO)

创新性地提出方差驱动的自适应奖励权重机制，动态平衡稀疏的 ASR 奖励与稠密的 Judge 奖励，实现高效、稳定的强化学习训练。

### 1.3.2 基于自进化技能系统的越狱提示词生成方法 (SESS)

设计 Skills 库自进化机制，从成功攻击轨迹中提取可复用的策略模板，实现知识积累与跨案例迁移，同时可用于高效数据合成。

### 1.3.3 基于自进化系统内 Agentic-RL 的越狱提示词生成方法

将 AHR-GRPO 与 SESS 结合，在自进化 Skills 环境中进行强化学习训练，实现过程奖励与结果奖励的协同优化。

## 1.4 进度情况

| 章节 | 方法 | 进度 | 关键成果 |
|------|------|------|----------|
| **第一章** | AHR-GRPO | **85%** | 最佳 ASR **33.2%** (+2.4% vs baseline) |
| **第二章** | SESS | **95%** | 最佳 ASR **99.7%**，完成 217 组实验 |
| **第三章** | Agentic-RL | **10%** | 方法框架已设计 |

- **第一章 AHR-GRPO**：已完成核心方法设计与实现，自适应奖励权重计算器已开发，实验 1(jailbreak prompt 筛选)、实验 2(多维度 Judge reward) 和实验 3(自适应权重验证) 全部完成
- **第二章 SESS**：已完成 217 组实验，涵盖方法消融 (Layer 1-4)、数据策略、迁移性验证等，主要结论已得出
- **第三章 Agentic-RL**：方法框架已设计，待开展实验验证

---

# 2. 目前已完成的研究工作及进展

## 2.1 Adaptive Hybrid-Reward GRPO for Jailbreak Prompt Generation (AHR-GRPO)

### 2.1.1 研究背景与动机

- 传统 GRPO 方法使用单一奖励信号，ASR 奖励稀疏导致早期探索困难
- 现有混合奖励方法采用固定权重，无法适应训练过程中奖励信号的动态变化
- 需要设计自适应机制，在稀疏 ASR 阶段依赖稠密 Judge 引导，在 ASR 信号分化后回归结果导向优化

### 2.1.2 方法设计

#### 自适应 reward 权重计算

核心公式：使用指数移动方差 (EMV) 估计两组奖励的方差

$$\lambda_{\text{raw}} = \sigma(\alpha \cdot \log\frac{\sigma^2_{\text{ASR}}}{\sigma^2_{\text{proc}}} + \delta)$$

- **双平滑机制**：EMA 平滑 + 惯性平滑，防止单步跳变导致梯度震荡
- **边界裁剪**：$\lambda \in [\lambda_{\min}, \lambda_{\max}]$，防止信号坍缩
- **冷启动机制**：前$W$步固定$\lambda=0.5$，避免初始方差估计偏差

#### 多维度 Judge Prompt 设计

四个评判维度：intent_preservation(意图保留)、stealth(隐蔽性)、strategy_execution(策略执行)、attack_potential(攻击潜力)

### 2.1.3 实验设计与结果

#### 实验 1：Jailbreak Prompt 筛选

**目标**：筛选高效的基础攻击 prompt 策略

**方法**：测试 24 种 jailbreak prompt 模板的初始 ASR（1000 条测试集）

**结果**：Top-3 策略为 hypothetical_scenario(30.8%)、creative_writing(28.3%)、role_playing(25.0%)

![Attack Prompt Selection](figures/output/AHR_GRPO_01_attack_prompt_selection.png)

#### 实验 2：多维度 Judge Reward 权重配置（12 组）

**目标**：研究不同 Judge 维度权重对训练效果的影响

**设计**：3 种攻击 prompt × 4 种 Judge 维度 = 12 组 GRPO 训练实验

**结果**：**所有组合均超越 baseline**，最佳为 hypothetical_scenario + idea_preservation 达到**32.3%**（+1.5% 改进）

**关键发现**：通用 Judge 维度优于专用维度，idea_preservation 最稳定（平均 +1.2% 改进）

![Judge Dimension Comparison](figures/output/AHR_GRPO_03_judge_dimension.png)

#### 实验 3：自适应权重机制验证

**目标**：在实验 2 最佳配置基础上，验证自适应权重机制的进一步改进效果

**结果**：**自适应机制超越固定权重**
- β=0（窗口=1）：ASR=**33.2%**，Δ vs 实验 2 最佳=**+0.9%**
- β=0.8（窗口=5）：ASR=32.9%，Δ=+0.6%
- β=0.9（窗口=10）：ASR=32.6%，Δ=+0.3%

![Adaptive Weight Results](figures/output/AHR_GRPO_04_adaptive_weight.png)

**消融验证**：仅 ASR Reward 效果下降 5.8%（ASR=25.0%），混合 Reward 显著优于单一 Reward

![Reward Ablation](figures/output/AHR_GRPO_05_reward_ablation.png)

#### 核心结论

![AHR-GRPO Summary](figures/output/AHR_GRPO_06_summary.png)

- ✅ **自适应权重机制有效**：动态调整 Lambda 实现 ASR 与 Judge 的最优平衡
- ✅ **混合 Reward 显著优于单一 Reward**：仅 ASR Reward 效果下降 5.8%
- ✅ **所有训练组合均达到或超越 baseline**：体现 GRPO 训练有效性

---

## 2.2 Self-Evolving Skills System for Jailbreak Prompt Generation (SESS)

### 2.2.1 研究背景与动机

- 现有 jailbreak 方法缺乏知识积累机制，每次攻击需从头探索
- PAIR 等迭代方法效率低，无法跨案例复用成功策略
- 需要设计可复用的攻击模板 (Skills) 与自进化机制，实现知识积累与迁移

### 2.2.2 方法设计

#### Skill 数据结构与检索机制

- **Skill 定义**：包含 content(攻击模板)、source(来源)、统计信息 (success_rate、usage_count)、关键词标签、伤害类型
- **检索评分**：$score = quality\_score \times (1 + 关键词匹配 \times 0.3) \times 类型匹配系数$
- **动态匹配**：根据目标 prompt 特征 (长度、关键词、伤害类型) 检索最匹配 Skills

#### 三阶段自进化流程

- **Cold Start 阶段**：使用初始 Skills 进行攻击，积累成功案例
- **Evolution 阶段**：从成功/失败轨迹中提取/改进 Skills
- **Test 阶段**：使用演化后的 Skills 库评估最终 ASR

### 2.2.3 实验设计与结果（217 组实验）

#### Layer 1：方法组合 Grid Search（16 组）

**目标**：确定最佳方法组合

**消融点**：skill_call_mode(2 种)、skill_extraction_mode(2 种)、update_strategy(4 种)

**最佳配置**：single_call + trajectory + statistical，ASR=79.1%，Skills=28 个

**关键发现**：single_call 比 every_iteration 高 +12.7%

#### Layer 2：数据消融实验（36 组）

**目标**：验证数据量和配比影响

**最佳配置**：small(300) + early(30% CS)，ASR=80.0%

**关键发现**：数据量影响<2%，配比影响显著（early 最佳）

#### Layer 3：DAN 模板验证（17 组）

**目标**：验证强初始 Skills 的效果

**最佳配置**：full_evolve(无 Cold Start)，ASR=98.8%

**关键发现**：DAN 模板无需进化，直接使用最优

#### Layer 4：DAN 数据消融（12 组）

**最佳配置**：medium + evo，ASR=99.7%

**关键发现**：数据量影响极小 (差距 1.3%)

![SESS Main Results](figures/output/SESS_01_sess_main_results.png)

### 2.2.4 迁移实验（140 组）

#### 同族迁移效果（Qwen 系列）

| 模型 | PAIR_w_SESS ASR | Best Baseline | 提升 |
|------|-----------------|---------------|------|
| Qwen3-0.6B | 95.1% | 90.2% | +4.9% |
| Qwen3-4B | 86.7% | 78.7% | +8.0% |
| Qwen3-14B | 86.5% | 85.5% | +1.0% |

**平均提升：+4.7%**

#### 跨族迁移效果（gpt-oss-20b）

| 方法 | 同族 ASR | 跨族 ASR | 下降幅度 |
|------|----------|----------|----------|
| PAIR_w_SESS | 91.6% | 11.4% | -80.2% |
| AutoDAN_w_SESS | ~99% | 32.5% | -66.5% |
| AutoDAN (baseline) | 85.5% | 6.0% | -79.5% |
| PAIR (baseline) | 71.6% | 0.08% | -71.5% |

![Transfer Comparison](figures/output/SESS_02_transfer_comparison.png)

**关键发现**：跨族迁移是真正挑战，需要新方法提升泛化性

### 2.2.5 核心结论

- **最佳方法**：AutoDAN_w_SESS（同族 99%，跨族 32.5%）
- **Evolution 必要性**：DAN 场景不必要，pair 风格微弱 (+0.9%)
- **Skills 泛化性**：同族良好 (85-95%)，跨族有限 (<35%)
- **数据策略**：300 条足够，full_evolve 最优 (DAN 场景)

---

## 2.3 Agentic RL with Self-Evolving Skills System (待开展)

### 2.3.1 研究背景与动机

- AHR-GRPO：解决奖励信号稀疏问题，但缺乏攻击策略的知识积累
- SESS：实现 Skills 知识积累与迁移，但缺乏强化学习的策略优化
- 需要将两者结合：在自进化 Skills 环境中进行 RL 训练，实现策略优化与知识积累的双重效果

### 2.3.2 方法设计

#### 自进化 Skills 环境构建

- Skills 作为环境状态的一部分，动态更新
- 环境提供 Skills 检索接口，Policy 根据当前 prompt 检索最佳 Skill
- Skills 库随训练进程演化，形成适应 Policy 策略的 Skills 分布

#### 结果奖励与过程奖励设置

- **结果奖励 (ASR)**：攻击成功与否的稀疏信号
- **过程奖励 (Judge)**：多维度评判的稠密信号
- **自适应权重**：使用 AHR-GRPO 的方差驱动机制动态调整权重
- **Skills 质量奖励**：鼓励 Policy 使用高质量 Skills

#### 训练流程设计

- **Phase 1**：Skills 初始化（使用 SESS 的 DAN 模板或演化 Skills）
- **Phase 2**：RL 训练（Policy 在 Skills 环境中学习）
- **Phase 3**：Skills 协同演化（Policy 与 Skills 双向优化）

---

# 3. 后期拟完成的研究工作及进度安排

## 3.1 第一章 AHR-GRPO 后续工作

- ✅ **已完成实验 3**：自适应权重机制验证（3 组 EMA 配置），确认自适应超越固定权重（+0.9% 额外改进）
- ✅ **已验证自适应机制优越性**：最佳配置达到 33.2%，超越实验 2 最佳的 32.3%
- ✅ **已确认混合 Reward 有效性**：仅 ASR Reward 效果下降 5.8%，混合 Reward 显著优于单一 Reward
- 待完成：分析$\lambda$演化曲线，验证物理直觉（早期依赖 Judge，后期回归 ASR）
- 待完成：整理实验数据，撰写论文方法章节与实验章节

## 3.2 第二章 SESS 后续工作

- ✅ **已完成 217 组实验**：Layer 1-4 方法消融、数据消融、迁移性验证等全部完成
- ✅ **已得出主要结论**：最佳方法 AutoDAN_w_SESS（99% 同族、32.5% 跨族）
- 待完成：整理实验数据，撰写完整实验报告
- 待完成：分析跨族迁移失败的根本原因（模型安全机制差异、Skills 模型特异性）
- 待完成：探索提升 Skills 泛化性的方法（元学习、多模型联合训练）
- 待完成：整理 Skills 库，开源最佳 Skills 模板
- 待完成：撰写论文方法章节与实验章节

## 3.3 第三章 Agentic-RL 工作计划

| 阶段 | 时间 | 任务 |
|------|------|------|
| **方法实现** | 2 周 | 搭建自进化 Skills 环境、集成 AHR-GRPO 自适应奖励机制、实现 Skills 协同演化流程 |
| **实验验证** | 4 周 | Baseline 对比、消融实验、跨模型迁移测试 |
| **结果分析与论文撰写** | 2 周 | 分析训练曲线与$\lambda$演化、Skills 演化过程可视化、与前两章方法对比讨论 |

## 3.4 论文整理与投稿准备

1. 整理三章内容，统一论文框架（ICLR 格式）
2. 补充实验：根据 review 反馈可能需要的额外实验
3. 撰写 Introduction、Related Work、Conclusion
4. 完善图表与可视化

---

# 4. 存在的困难与问题

## 4.1 计算资源需求较大

- AHR-GRPO 实验：每实验约需 4×GPU(A800-80G)，训练 2-4 小时/1000 步
- SESS 实验：已完成 217 组，消耗大量 GPU 资源
- 第三章 Agentic-RL：预计需额外数百 GPU 卡时
- 资源调度：需合理安排实验顺序，避免资源冲突

## 4.2 高成本试错风险

- 强化学习训练稳定性：Agentic-RL 训练可能较难收敛
- Skills 演化不可控：Skills 爆炸、低质量 Skills 积累等问题
- 实验设计迭代：部分实验结果可能不理想，需重新设计

## 4.3 跨族迁移挑战

- SESS 实验揭示：所有方法在 gpt-oss-20b 上效果大幅下降（best 仅 32.5%）
- 需要新方法：当前 Skills 缺乏跨族泛化性
- 第三章工作重点：探索提升泛化性的方法

## 4.4 论文时间压力

- 三章内容整合：需统一叙事逻辑
- 实验周期：第三章实验预计需 6 周
- 投稿时间窗口：需在毕业前完成论文撰写与投稿

## 4.5 方法创新性论证

- AHR-GRPO：需与固定权重方法充分对比
- SESS：需与 baseline 方法 (PAIR、AutoDAN) 充分对比
- Agentic-RL：需证明组合方法的协同效果

---

# 5. 如期完成全部论文工作的可能性

## 5.1 已完成工作评估

| 章节 | 方法 | 进度 | 关键成果 |
|------|------|------|----------|
| 第一章 | AHR-GRPO | **85%** | 最佳 ASR 33.2% |
| 第二章 | SESS | **95%** | 最佳 ASR 99.7% |
| 第三章 | Agentic-RL | **10%** | 方法框架已设计 |
| **总体** | - | **65%** | - |

## 5.2 后期工作可行性分析

- **时间规划**：剩余 6-8 周，足够完成第三章实验与论文整理
- **技术可行性**：AHR-GRPO 与 SESS 代码库已成熟，组合实现难度可控
- **资源可行性**：已有 GPU 资源调度经验，可合理安排实验
- **关键成果**：第一章已验证自适应机制有效（+0.9% 额外改进），第二章已验证 Skills 积累有效（99.7% ASR）

## 5.3 风险评估与应对策略

| 风险 | 应对策略 |
|------|----------|
| Agentic-RL 训练不稳定 | 设计稳定性保障机制（双平滑、边界裁剪、冷启动），参考 AHR-GRPO 经验 |
| 跨族迁移效果仍不理想 | 将此作为研究发现，分析原因而非强行提升，符合研究完整性 |
| 时间不足 | 优先完成核心实验，消融实验可视情况缩减 |

## 5.4 结论

基于已完成工作与后续计划，**如期完成全部论文工作的可能性较高**。

- 需合理安排实验顺序，控制资源消耗，保证核心实验优先完成
- 论文结构清晰，三章节递进关系明确，具备 ICLR 投稿潜力

---

# 附录

## 附录 A：实验数据汇总表

| 章节 | 实验类型 | 实验数量 | 最佳结果 | 关键发现 |
|------|----------|----------|----------|----------|
| AHR-GRPO | Jailbreak Prompt 筛选 | 24 组 | 30.8% | 确定 top-3 策略 |
| AHR-GRPO | Judge 维度权重 | 12 组 | 32.3% | 通用维度优于专用维度 |
| AHR-GRPO | 自适应权重消融 | 3 组 | 33.2% | 自适应超越固定权重 |
| SESS Layer1 | 方法组合 | 16 组 | 79.1% | single_call 最佳 |
| SESS Layer2 | 数据消融 | 36 组 | 80.0% | 数据量影响<2% |
| SESS Layer3 | DAN 模板 | 17 组 | 98.8% | DAN 无需进化 |
| SESS Layer4 | DAN 数据 | 12 组 | 99.7% | small 足够 |
| SESS Ablation | Evolution 必要性 | 16 组 | - | 平均贡献 +0.9% |
| SESS Transfer | 跨模型迁移 | 140 组 | 32.5%(跨族) | 跨族迁移挑战 |

## 附录 B：代码结构

### B.1 AHR-GRPO 代码结构

```
RL4jailbreak/
├── src/
│   ├── reward/
│   │   └── adaptive_weight.py      # 自适应权重计算器
│   ├── vllm_client.py              # vLLM 服务客户端
│   └── generate.py                 # Prompt 生成模块
├── experiments/
│   ├── jailbreak_prompt_exp/       # 实验 1：Prompt 筛选
│   ├── hybrid_reward_exp/          # 实验 2：Judge 维度
│   └── adaptive_hybrid_reward_exp/ # 实验 3：自适应权重
```

### B.2 SESS 代码结构

```
self_evolve_skills_jailbreak/
├── src/
│   ├── skill.py                    # Skill 数据结构
│   ├── skill_library.py            # Skills 库管理
│   ├── attacker.py                 # 攻击器
│   └── reflector.py                # 反思器
├── exp/
│   ├── layer1/                     # 方法消融 16 组
│   ├── layer2/                     # 数据消融 36 组
│   ├── layer3/                     # DAN 验证 17 组
│   ├── layer4/                     # DAN 数据 12 组
│   ├── ablation/                   # Evolution 消融 16 组
│   └── transfer/                   # 跨模型迁移 140 组
├── baselines/                      # 对比基线方法
```

## 附录 C：图表索引

| 图号 | 文件名 | 内容 |
|------|--------|------|
| AHR_GRPO_01 | AHR_GRPO_01_attack_prompt_selection.png | Attack Prompt Selection |
| AHR_GRPO_02 | AHR_GRPO_02_judge_prompt_selection.png | Judge Prompt Selection |
| AHR_GRPO_03 | AHR_GRPO_03_judge_dimension.png | Judge Dimension (12 组) |
| AHR_GRPO_04 | AHR_GRPO_04_adaptive_weight.png | Adaptive Weight 实验 |
| AHR_GRPO_05 | AHR_GRPO_05_reward_ablation.png | Reward Ablation |
| AHR_GRPO_06 | AHR_GRPO_06_summary.png | AHR-GRPO 总结 |
| SESS_01 | SESS_01_sess_main_results.png | Strategy & Data Effects |
| SESS_02 | SESS_02_transfer_comparison.png | Cross-Model Transfer |
| SESS_03 | SESS_03_transfer_comprehensive.png | Comprehensive Transfer |
| SESS_04 | SESS_04_same_family_transfer.png | Same-Family Transfer |

---

*文档版本：v1.0*  
*最后更新：2026 年 6 月 16 日*
