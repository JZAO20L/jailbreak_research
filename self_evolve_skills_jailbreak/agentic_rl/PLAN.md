# Agentic-RL for Jailbreak: 完整规划文档

**版本**: 1.1
**日期**: 2026-08-03
**定位**: 毕业论文第三章 + 顶会论文（ NeurIPS / ICLR / ACL 投稿 ）
**基础**: SESS (self_evolve_skills_jailbreak) + AHR-GRPO (RL4jailbreak)

**相关文档**:
- [方法设计](./METHOD_DESIGN.md) - 核心方法详细设计
- [进度追踪](../paper/README.md) - SESS 论文进度

---

## 📌 核心设计决策（2026-08-03 确认）

### 训练什么能力？

**答案**: 训练 **Skill Selection + Skill Adaptation** 两个能力

1. **Skill Selection**: 从候选 skills 中选择最合适的一个（离散 action）
2. **Skill Adaptation**: 将选中的 skill 改写以适配当前 prompt（文本生成）

### 为什么不训练 jailbreak prompt 生成？

- Jailbreak 生成可以由单独的 rewrite model 完成
- 聚焦于 "选对工具 + 用好工具" 才是真正的 Agentic 能力
- 与 SESS 完美衔接：SESS 提供 skill library，Agentic-RL 学会利用它

### 训练哪个阶段？

**答案**: **两阶段训练**（Curriculum Learning）

- **Stage 1**: 标准 RL（类似 AHR-GRPO，无 skill 输入）→ 学会基础 jailbreak 能力
- **Stage 2**: Agentic RL（有 skill 输入）→ 学会 skill selection + adaptation

---

## 一、研究定位与核心贡献

### 1.1 问题定义

**核心问题**：如何通过强化学习训练一个 **Agentic Attacker**，使其在多轮交互中自主选择和进化攻击策略，实现对目标LLM的高效越狱？

**与现有方法的本质区别**：

| 方法 | 范式 | 策略空间 | 知识积累 | 自适应 |
|------|------|----------|----------|--------|
| PAIR | 多轮对话 | 固定攻击模板 | ❌ | ❌ |
| AutoDAN | 遗传搜索 | DAN模板库 | ❌ | ❌ |
| SESS (Ch.2) | 检索+进化 | Skills库 (固定策略) | ✅ 规则驱动 | ✅ 检索驱动 |
| AHR-GRPO (Ch.1) | RL训练 | 隐式 (模型参数) | ✅ 参数化 | ❌ 固定reward |
| **Agentic-RL (本文)** | **RL+Agentic** | **Skills as Actions** | **✅ 参数+规则** | **✅ 双重** |

### 1.2 核心贡献 (面向论文)

1. **Agentic Attacker 架构**：将 Skills 库作为 Agent 的 action space，通过 RL 训练 policy 学会选择、组合、进化 Skills
2. **Adaptive Multi-dimensional Reward**：继承 AHR-GRPO 的自适应权重思路，扩展为多维度奖励（ASR + Judge + Skill Quality + Diversity）
3. **Skill-Policy 协同进化**：训练过程中 policy 和 skill library 共同进化，形成正反馈循环
4. **跨模型迁移**：在 Qwen 系列上训练，迁移到 gpt-oss、LLaMA 等跨族模型

### 1.3 论文标题 (暂定)

- "Agentic Jailbreak: Reinforcement Learning with Self-Evolving Skills for Adaptive Red Teaming"
- "From Prompt Engineering to Agent Engineering: Skill-Guided RL for LLM Jailbreaking"

---

## 二、技术架构

### 2.1 整体框架

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Agentic-RL Training Framework                        │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │  Policy Model π_θ (Qwen3-4B + LoRA)                               │      │
│  │                                                                   │      │
│  │  Input: [harmful_prompt | retrieved_skills | attack_history]      │      │
│  │  Output: [skill_selection] + [jailbreak_prompt]                   │      │
│  │                                                                   │      │
│  │  Action Space:                                                    │      │
│  │    a_1 = select_skill(skill_library, prompt_features)             │      │
│  │    a_2 = generate_jailbreak(original_prompt, selected_skill)      │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                │                                            │
│                                ↓                                            │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │  Environment (Multi-turn Interaction)                             │      │
│  │                                                                   │      │
│  │  Turn 1: Agent → [skill + jailbreak_prompt] → Target → Guard     │      │
│  │  Turn 2: Agent observes feedback → refine → Target → Guard       │      │
│  │  ...                                                              │      │
│  │  Turn T: success (Unsafe) or failure → compute rewards            │      │
│  │                                                                   │      │
│  │  Components:                                                      │      │
│  │    • Target Model: gpt-oss-20b / Qwen3-4B                        │      │
│  │    • Guard Model: Qwen3Guard-Gen-4B                               │      │
│  │    • Judge Model: Qwen3-4B (复用 Target)                          │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                │                                            │
│                                ↓                                            │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │  Adaptive Multi-dimensional Reward                                │      │
│  │                                                                   │      │
│  │  R_total = λ₁·R_asr + λ₂·R_judge + λ₃·R_skill + λ₄·R_diversity │      │
│  │                                                                   │      │
│  │  λᵢ adaptively adjusted based on variance ratios (AHR-GRPO ext.) │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                │                                            │
│                                ↓                                            │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │  Skill Library Update (SESS Mechanism)                            │      │
│  │                                                                   │      │
│  │  Every N steps:                                                   │      │
│  │    • Extract new skills from successful trajectories              │      │
│  │    • Evolve existing skills from failure analysis                 │      │
│  │    • Prune low-quality skills, merge similar ones                 │      │
│  └───────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent Action Space 设计

#### 方案A: 两阶段 Action (推荐)

```
Step 1 - Skill Selection:
  Input:  harmful_prompt + skill_library (top-k candidates)
  Output: selected_skill (or "none")

Step 2 - Prompt Generation:
  Input:  harmful_prompt + selected_skill + (optional) history
  Output: jailbreak_prompt

Step 3 (optional) - Iterative Refinement:
  Input:  target_response + guard_feedback + previous_prompt
  Output: refined_jailbreak_prompt
```

#### 方案B: 单阶段 Action (简单基线)

```
Input:  harmful_prompt + retrieved_skill_content
Output: jailbreak_prompt (conditioned on skill)
```

### 2.3 攻击方法集成

| 攻击方法 | 集成方式 | Skill 来源 |
|----------|----------|------------|
| **DAN + SESS** | 将 DAN 模板作为初始 Skills，RL 学习选择和精化 | 6 个 DAN 模板 |
| **PAIR + SESS** | PAIR 的多轮反思作为 Skill 进化的反馈信号 | 从 PAIR 轨迹提取 |
| **纯 SESS** | 仅使用检索+进化，无 RL 训练 (消融基线) | 自进化 Skills |

---

## 三、奖励建模 (继承 AHR-GRPO)

### 3.1 多维度奖励函数

```python
R_total = λ₁(t) · R_asr + λ₂(t) · R_judge + λ₃(t) · R_skill + λ₄(t) · R_diversity
```

#### R_asr: 攻击成功率奖励
```
R_asr = 1.0  if Guard → Unsafe
R_asr = 0.5  if Guard → Controversial
R_asr = 0.0  if Guard → Safe
```

#### R_judge: 多维度 Judge 评分 (复用 SESS Judge 基建)
```
R_judge = weighted_mean(
    idea_preservation,    # 原始意图保留程度
    stealthiness,         # 攻击隐蔽程度
    naturalness,          # 语言自然度
    strategy_execution,   # 攻击策略执行效果
)
```

#### R_skill: Skill 质量奖励 (新增)
```
R_skill = f(skill_quality, skill_novelty)
  - skill_quality:  被选 skill 的 history success rate
  - skill_novelty:  是否是新提取/进化的 skill
  → 鼓励使用高质量 Skills，同时探索新 Skills
```

#### R_diversity: 策略多样性奖励 (新增)
```
R_diversity = 1 - similarity(current_skill, recent_k_skills)
  → 避免 policy 坍缩到单一策略
```

### 3.2 自适应权重调整 (继承 AHR-GRPO)

```python
# 对每对奖励 (R_i, R_j)，基于方差比动态调整权重
ratio_ij = Var(R_i) / (Var(R_j) + eps)
λ_raw_ij = sigmoid(α · log(ratio_ij) + δ)
λ_i = ema(λ_raw_ij)  # EMA 平滑
λ_i = clip(λ_i, λ_min, λ_max)

# 归一化确保 Σλ = 1
```

---

## 四、对比基线设计

### 4.1 基线方法

| 编号 | 方法 | 描述 | 目的 |
|------|------|------|------|
| B1 | **GRPO-ASR** | 仅使用 R_asr 的 GRPO | 证明多维度 reward 必要性 |
| B2 | **GRPO-Judge** | 仅使用 R_judge 的 GRPO | 证明 outcome reward 必要性 |
| B3 | **GRPO-Fixed** | 固定比例 (λ=0.5) 的混合 GRPO | 证明自适应权重必要性 |
| B4 | **DAPO** | Direct Alignment Policy Optimization | SOTA RL 方法对比 |
| B5 | **GSPO** | Group Sequence Policy Optimization | 序列级优化对比 |
| B6 | **SESS (no RL)** | 纯 SESS，无 RL 训练 | 证明 RL 训练增益 |
| B7 | **PAIR** | 经典多轮攻击 (无 Skills) | 传统方法基线 |
| B8 | **AutoDAN** | DAN 模板 + 遗传搜索 | 模板方法基线 |

### 4.2 消融实验

| 编号 | 消融点 | 变量 |
|------|--------|------|
| A1 | Skill Integration | w/ vs w/o skill conditioning |
| A2 | Adaptive Weight | AHR vs Fixed vs Single |
| A3 | Skill Update | 协同进化 vs 固定库 vs 无库 |
| A4 | Attack Method | DAN+SESS vs PAIR+SESS vs Pure SESS |
| A5 | Target Model | Qwen3-4B vs gpt-oss-20b |
| A6 | Multi-turn | 1-turn vs 3-turn vs 5-turn |
| A7 | Reward Dimensions | 4维 vs 3维 vs 2维 |

---

## 五、数据与评估基建 (复用 SESS)

### 5.1 数据集

| 数据集 | 来源 | 规模 | 用途 |
|--------|------|------|------|
| SESS train | `self_evolve_skills_jailbreak/data/train_prompts.json` | 1000 | RL 训练 |
| SESS test | `self_evolve_skills_jailbreak/data/test_prompts.json` | 1000 | 评估 |
| SESS cold_start | `self_evolve_skills_jailbreak/data/cold_start_prompts.json` | 200 | Skill 冷启动 |
| SESS evolution | `self_evolve_skills_jailbreak/data/evolution_prompts.json` | 800 | Skill 进化 |

### 5.2 目标模型

| 模型 | 家族 | 角色 | 备注 |
|------|------|------|------|
| Qwen3-4B | Qwen (同族) | 主训练目标 | 与 policy 同模型 |
| gpt-oss-20b | GPT (跨族) | 迁移目标 | 验证泛化性 |

### 5.3 评估指标

| 指标 | 定义 | 计算方式 |
|------|------|----------|
| **ASR** | 攻击成功率 | Guard 判定 Unsafe 的比例 |
| **Avg Iterations** | 平均攻击轮次 | 成功攻击的平均迭代数 |
| **Skill Utilization** | Skill 使用率 | 使用 Skill 的攻击比例 |
| **Skill Success Rate** | Skill 成功率 | 使用 Skill 后攻击成功比例 |
| **Transfer ASR** | 迁移攻击成功率 | 跨模型攻击的 ASR |
| **Diversity Score** | 策略多样性 | 使用 Skills 的熵值 |
| **Judge Score** | 平均 Judge 评分 | 多维度加权平均分 |

### 5.4 评估 Pipeline (复用 SESS)

```
复用自 SESS:
├── VLLMClient (src/vllm_client.py)
├── Guard 评估逻辑 (attacker.py → _evaluate_attack)
├── Skill Library 评估 (skill_library.py → get_stats)
└── 实验脚本框架 (scripts/pipeline.py)
```

---

## 六、代码架构设计

### 6.1 目录结构 (统一在 src/ 下)

```
self_evolve_skills_jailbreak/
├── src/                              # 所有源码统一在 src/ 下
│   ├── skill.py                      # Skill 数据结构 (SESS 复用)
│   ├── skill_library.py              # Skills 库管理 (SESS 复用+扩展)
│   ├── attacker.py                   # SESS 攻击器 (保留)
│   ├── reflector.py                  # 反思器 (SESS 复用)
│   ├── data_config.py                # 数据配置 (SESS 复用)
│   │
│   ├── reward/                       # [新增] 奖励函数模块
│   │   ├── __init__.py
│   │   ├── asr_reward.py             # ASR 奖励
│   │   ├── judge_reward.py           # Judge 奖励
│   │   ├── skill_reward.py           # Skill 质量奖励
│   │   ├── diversity_reward.py       # 多样性奖励
│   │   └── adaptive_weight.py        # 自适应权重 (从 RL4jailbreak 迁移+扩展)
│   │
│   └── agentic_rl/                   # [新增] Agentic-RL 核心模块
│       ├── __init__.py
│       ├── agent.py                  # Agentic Attacker 实现
│       ├── environment.py            # 多轮交互环境
│       ├── policy.py                 # Policy 模型封装 (LoRA)
│       ├── trainer.py                # GRPO/DAPO/GSPO 训练器
│       ├── rollout.py                # Rollout 生成
│       ├── skill_updater.py          # Skill 库周期更新
│       └── config.py                 # 训练配置
│
├── baselines/                        # [新增] 对比基线实现
│   ├── grpo_asr.py                   # B1: 纯 ASR GRPO
│   ├── grpo_judge.py                 # B2: 纯 Judge GRPO
│   ├── grpo_fixed.py                 # B3: 固定比例 GRPO
│   ├── dapo.py                       # B4: DAPO
│   ├── gspo.py                       # B5: GSPO
│   ├── sess_no_rl.py                 # B6: 纯 SESS (无 RL)
│   └── pair.py                       # B7: PAIR
│
├── scripts/                          # [保留+新增]
│   ├── pipeline.py                   # SESS 原始 pipeline (保留)
│   ├── start_guard.sh                # Guard 服务 (复用)
│   ├── start_target.sh               # Target 服务 (复用)
│   ├── start_policy.sh               # Policy 服务 (扩展)
│   ├── train_agentic_rl.py           # [新增] 主训练入口
│   ├── eval_agentic_rl.py            # [新增] 评估脚本
│   └── run_experiments.sh            # [新增] 实验批量运行
│
├── data/                             # [保留] SESS 数据 (完全复用)
│   ├── cold_start_prompts.json
│   ├── evolution_prompts.json
│   ├── test_prompts.json
│   └── train_prompts.json
│
├── skills/                           # [保留] SESS Skills 库
│   └── skills_library.json
│
├── exp/                              # [保留+新增] 实验目录
│   ├── layer1-4/                     # SESS 原有实验 (保留)
│   ├── agentic_rl/                   # [新增] Agentic-RL 实验
│   │   ├── main_exp/                 # 主实验
│   │   ├── ablation/                 # 消融实验
│   │   ├── transfer/                 # 迁移实验
│   │   └── output/                   # 实验输出
│   └── COMPLETE_EXPERIMENT_SUMMARY.md
│
├── agentic_rl/                       # [规划文档] 仅存放规划和文档
│   ├── PLAN.md                       # 本文件
│   └── docs/                         # 方法文档
│       ├── METHOD.md
│       ├── EXPERIMENTS.md
│       └── RESULTS.md
│
└── paper/                            # [保留] 论文相关
```

### 6.2 代码复用策略

| 模块 | SESS 现有 | Agentic-RL 复用/扩展 |
|------|-----------|---------------------|
| Skill | `src/skill.py` | 完全复用，新增 `embedding` 字段 |
| SkillLibrary | `src/skill_library.py` | 复用检索逻辑，新增 `update_from_rollout()` |
| Attacker | `src/attacker.py` | 保留，新增 `src/agentic_rl/agent.py` |
| Reflector | `src/reflector.py` | 完全复用 |
| VLLMClient | 项目共享 | 完全复用 |
| Guard 评估 | `src/attacker._evaluate_attack()` | 抽取为 `src/reward/asr_reward.py` |
| Judge Prompts | 从 RL4jailbreak 迁移 | 复用 + 扩展 `strategy_execution` 维度 |
| Adaptive Weight | `RL4jailbreak/src/reward/adaptive_weight.py` | 迁移到 `src/reward/adaptive_weight.py` 并扩展 |

### 6.3 依赖管理

```
已有依赖 (pyproject.toml):
├── trl>=0.12          # GRPO trainer
├── peft>=0.13         # LoRA
├── vllm>=0.6.3        # Model serving
├── transformers       # Model loading
└── swanlab>=0.7.18    # Experiment tracking

新增依赖:
├── openai             # gpt-oss API (迁移实验)
└── sentence-transformers  # Skill embedding (可选)
```

---

## 七、实验计划

### 7.1 实验分阶段推进

#### Phase 1: 基础设施搭建 (1-2 周)

- [ ] 迁移 `adaptive_weight.py` 到 SESS 目录
- [ ] 实现 `src/reward/` 模块 (ASR, Judge, Skill, Diversity)
- [ ] 实现 `agentic_rl/environment.py` 多轮交互环境
- [ ] 实现 `agentic_rl/agent.py` Agentic Attacker
- [ ] 整合 Skill Library 的周期更新逻辑

#### Phase 2: 主实验 (2-3 周)

- [ ] 实现 GRPO 训练器 (`agentic_rl/trainer.py`)
- [ ] 训练配置: Qwen3-4B + LoRA, 1000 steps
- [ ] 攻击方法: DAN+SESS (主实验)
- [ ] Target: Qwen3-4B (同族)
- [ ] 运行 B1-B8 所有基线
- [ ] 评估所有指标

#### Phase 3: 消融实验 (1-2 周)

- [ ] A1-A7 所有消融实验
- [ ] PAIR+SESS 变体
- [ ] 不同攻击方法对比

#### Phase 4: 迁移实验 (1 周)

- [ ] Target: gpt-oss-20b
- [ ] 在 Qwen 训练的 policy 迁移到 GPT
- [ ] 对比 SESS 的迁移结果 (32.5% baseline)

#### Phase 5: 论文撰写 (并行)

- [ ] Method 章节
- [ ] Experiment 章节
- [ ] 图表生成

### 7.2 GPU 资源规划

| GPU | 角色 | 配置 |
|-----|------|------|
| GPU 0 | Guard | Qwen3Guard-Gen-4B, TP=1 |
| GPU 1-2 | Target | Qwen3-4B/gpt-oss, TP=2 |
| GPU 3-4 | Policy Training | Qwen3-4B + LoRA, TP=2 |

训练时使用 5 卡 (或 4 卡 colocate)。

### 7.3 预期实验量

| 类别 | 实验数 | 每个实验训练步数 |
|------|--------|------------------|
| 主实验 (B1-B8) | 8 | 1000 steps |
| 消融实验 (A1-A7) | ~15 | 1000 steps |
| 迁移实验 | ~6 | eval only |
| 超参搜索 (可选) | ~10 | 500 steps |
| **总计** | **~40** | - |

---

## 八、与论文章节的映射

### 8.1 毕业论文第三章

```
第三章 Agentic Jailbreak: 基于自进化技能与强化学习的自适应越狱攻击

3.1 引言
    - 第二章 (SESS) 的局限性：Skills 进化缺乏端到端优化
    - 本章动机：用 RL 训练 policy 来学会选择和进化 Skills

3.2 相关工作
    - RL for LLM Alignment (GRPO, DAPO, GSPO)
    - Agent-based Red Teaming
    - Skill Learning

3.3 方法
    3.3.1 问题建模: MDP formulation
    3.3.2 Agentic Attacker 架构
    3.3.3 Adaptive Multi-dimensional Reward
    3.3.4 Skill-Policy 协同进化

3.4 实验
    3.4.1 实验设置
    3.4.2 主实验结果
    3.4.3 消融实验
    3.4.4 迁移性分析

3.5 总结
```

### 8.2 顶会论文结构

```
1. Introduction (2 pages)
2. Related Work (1.5 pages)
3. Method (3 pages)
   3.1 Problem Formulation
   3.2 Agentic Attacker with Skill Library
   3.3 Adaptive Multi-dimensional Reward
   3.4 Skill-Policy Co-evolution
4. Experiments (3.5 pages)
   4.1 Setup
   4.2 Main Results
   4.3 Ablation Studies
   4.4 Transferability Analysis
   4.5 Skill Evolution Analysis
5. Conclusion (0.5 pages)
```

---

## 九、关键决策点

### 9.1 需要确认的设计选择

| 编号 | 决策点 | 选项 | 建议 |
|------|--------|------|------|
| D1 | RL 算法 | GRPO / DAPO / GSPO | 先用 GRPO (TRL 支持最好) |
| D2 | Action Space | 两阶段 vs 单阶段 | 先单阶段 baseline，后扩展 |
| D3 | Skill 更新频率 | 每步 vs 每 N 步 vs 每 epoch | 每 100 步 |
| D4 | 多维度权重 | 自适应 vs 手动调参 | 自适应 (继承 AHR-GRPO) |
| D5 | Policy Model | Qwen3-4B vs 更大模型 | Qwen3-4B (资源限制) |
| D6 | Target Model | 先 Qwen 还是先 gpt-oss | 先 Qwen (同族)，后 gpt-oss (迁移) |
| D7 | Skill 注入方式 | prefix vs in-context | 复用 SESS 的 prefix 方式 |

### 9.2 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| RL 训练不稳定 | 高 | 先用固定 reward 验证 pipeline |
| Skill Library 爆炸 | 中 | 复用 SESS 的 maintenance 机制 |
| Reward hacking | 中 | 增加 diversity reward 约束 |
| 跨族迁移效果差 | 低 | SESS 已验证 32.5% baseline，有提升空间 |
| GPU 资源不足 | 中 | 缩减 batch size，增加 gradient accumulation |

---

## 十、开发 Roadmap

### Week 1-2: 基础设施

```
Day 1-3:  代码结构搭建 + adaptive_weight 迁移
Day 4-7:  reward 模块实现 (ASR, Judge, Skill, Diversity)
Day 8-10: environment + agent 实现
Day 11-14: 单步 debug + 小规模验证
```

### Week 3-4: 主实验

```
Day 15-17: GRPO trainer 实现 + pipeline 联调
Day 18-20: 主实验运行 (B1-B4)
Day 21-24: 更多基线 + 初步结果分析
Day 25-28: 参数调优
```

### Week 5-6: 消融 + 迁移

```
Day 29-35: 消融实验 (A1-A7)
Day 36-38: 迁移实验 (gpt-oss)
Day 39-42: 结果整理 + 可视化
```

### Week 7-8: 论文

```
Day 43-49: Method + Experiment 章节
Day 50-56: 图表 + 补充材料
```

---

## 附录A: 与现有代码的接口定义

### A.1 Skill Library 扩展接口

```python
class SkillLibrary:
    # 现有方法 (保留)
    def retrieve(self, prompt: str, top_k: int = 1) -> List[Skill]
    def add_skill(self, skill: Skill) -> bool
    def run_maintenance(self) -> Dict

    # 新增方法
    def update_from_rollout(self, trajectories: List[Dict]):
        """从 RL rollout 轨迹中更新 Skills"""
        for traj in trajectories:
            if traj["success"]:
                # 提取新 skill
                new_skill = self.extract_skill(traj)
                self.add_skill(new_skill)
            else:
                # 进化失败 skill
                self.evolve_skill(traj)

    def get_skill_embeddings(self) -> np.ndarray:
        """获取所有 skill 的 embedding (用于相似度检索)"""
        pass

    def retrieve_by_embedding(self, prompt_emb: np.ndarray, top_k: int) -> List[Skill]:
        """基于 embedding 相似度检索 (替代关键词匹配)"""
        pass
```

### A.2 Agent 接口

```python
class AgenticAttacker:
    def __init__(self, policy_model, skill_library, config):
        self.policy = policy_model
        self.skill_library = skill_library

    def select_skill(self, prompt: str) -> Skill:
        """Agent 选择 skill"""
        candidates = self.skill_library.retrieve(prompt, top_k=3)
        # policy 决定选择哪个
        selected = self.policy.skill_selection(prompt, candidates)
        return selected

    def generate_attack(self, prompt: str, skill: Skill) -> str:
        """Agent 生成 jailbreak prompt"""
        return self.policy.generate(prompt, skill)

    def attack(self, prompt: str, max_turns: int = 5) -> AttackResult:
        """完整多轮攻击流程"""
        pass
```

### A.3 Reward 接口

```python
def compute_rewards(
    prompts: List[str],
    completions: List[str],
    skills_used: List[Skill],
    targets_responses: List[str],
    guard_labels: List[str],
) -> Dict[str, List[float]]:
    """计算多维度奖励

    Returns:
        {
            "asr": List[float],
            "judge": List[float],
            "skill": List[float],
            "diversity": List[float],
            "total": List[float],  # adaptive weighted sum
        }
    """
    pass
```
