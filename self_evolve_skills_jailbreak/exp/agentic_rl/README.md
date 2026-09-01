# Agentic-RL 实验计划（ms-swift 多轮版本）

> 第三章：基于 ms-swift 多轮 GRPO 的 Skill-Conditioned Agentic 越狱攻击

## 1. 研究问题

**核心空白**：skill library × 多轮 RL 训练无人占据。SESS 的 skill 选择和适配是启发式规则，无法从失败中学习。

**本章贡献**：
1. **真正的 Agentic 攻击**：模型学会在多轮交互中选择、适配、调整策略
2. **GYM 环境建模**：将 jailbreak 任务建模为标准 gym 环境
3. **多轮 GRPO 训练**：使用 ms-swift 的原生多轮支持

---

## 2. 方法概述

### 2.1 Agent 架构

```
┌─────────────────────────────────────────────────┐
│          JailbreakEnv (GYM 环境)                 │
│                                                   │
│  reset(prompt):                                   │
│    - 加载 harmful prompt + skill library          │
│    - 返回初始 observation                          │
│                                                   │
│  step(action):                                    │
│    - action = (skill_idx, adapted_content)        │
│    - 构建 attack_prompt                           │
│    - 发送到 target model                          │
│    - 评估 guard model                             │
│    - 返回 (next_obs, reward, done, info)          │
└─────────────────────────────────────────────────┘

多轮交互流程：
  Turn 1: Agent 选择 skill + 适配 → 攻击 → 失败
  Turn 2: Agent 看到反馈 → 选择不同 skill → 攻击 → 失败
  Turn 3: Agent 调整策略 → 攻击 → 成功！
```

### 2.2 与 SESS 的区别

| | SESS | Chapter 3 (ms-swift) |
|---|---|---|
| Skill 选择 | 启发式规则（关键词匹配） | **模型学习** |
| Skill 适配 | 无 / 简单 rewrite 模板 | **模型学习** |
| 失败后策略 | 固定 rewrite 逻辑 | **模型根据反馈决定** |
| 停止条件 | 固定 max_iterations | **模型自主决定** |
| 训练方式 | 无（规则系统） | **多轮 RL** |

### 2.3 技术栈

- **训练框架**：ms-swift（原生支持多轮 GRPO + GYM 环境）
- **多轮调度**：`GYMScheduler`（管理多轮交互）
- **推理引擎**：vLLM server mode + async engine（高效多轮采样）
- **环境接口**：标准 GYM（`reset()` / `step()`）

---

## 3. 实验矩阵

### 3.1 主实验：多轮 vs 单轮

| 实验 | 脚本 | 配置 | 说明 |
|------|------|------|------|
| E1 | `exp01_single_turn.sh` | max_turns=1 | 单轮 baseline |
| E2 | `exp02_multi_turn_3.sh` | max_turns=3 | 多轮（3 轮） |
| E3 | `exp03_multi_turn_5.sh` | max_turns=5 | 多轮（5 轮） |
| E4 | `exp04_multi_turn_adaptive.sh` | 模型自主决定 | 自适应停止 |

### 3.2 消融实验

| 实验 | 脚本 | 变量 | 证明什么 |
|------|------|------|----------|
| A1 | `exp05_no_skill.sh` | 无 skill 库 | skill 库的价值 |
| A2 | `exp06_process_reward.sh` | 过程奖励 vs 最终奖励 | 过程奖励的价值 |
| A3 | `exp07_skill_size.sh` | skill 库大小（10/30/54） | skill 库规模的影响 |

### 3.3 对比实验

| 实验 | 脚本 | 对比对象 | 说明 |
|------|------|----------|------|
| C1 | `exp08_vs_sess.sh` | SESS（规则系统） | RL vs 规则 |
| C2 | `exp09_vs_ahr_grpo.sh` | AHR-GRPO（Ch.1） | 多轮 vs 单轮 RL |

---

## 4. GPU 配置

### 8-GPU 模式（推荐）

```
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│  GPU 0  │  GPU 1  │  GPU 2  │  GPU 3  │  GPU 4  │  GPU 5  │  GPU 6  │  GPU 7  │
├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│  Guard  │ Target  │ vLLM    │ vLLM    │ Train   │ Train   │ Train   │ Train   │
│ (Guard) │(Qwen3)  │ Rollout │ Rollout │  (TP=2) │  (TP=2) │  (TP=2) │  (TP=2) │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

- GPU 0: Guard server（评估）
- GPU 1: Target server（攻击目标）
- GPU 2-3: vLLM rollout server（多轮采样）
- GPU 4-7: Training（TP=4）

### 4-GPU 模式

```
┌─────────┬─────────┬─────────┬─────────┐
│  GPU 0  │  GPU 1  │  GPU 2  │  GPU 3  │
├─────────┼─────────┼─────────┼─────────┤
│  Guard  │ Target  │ vLLM    │ Train   │
│ (Guard) │(Qwen3)  │ Rollout │  (TP=1) │
└─────────┴─────────┴─────────┴─────────┘
```

---

## 5. 实现计划

### Phase 1: 基础设施（Week 1）

1. **JailbreakEnv 实现**
   - `reset(prompt)`: 初始化状态
   - `step(action)`: 执行攻击，返回反馈
   - 集成 skill library、target model、guard model

2. **Reward Function**
   - ASR reward（最终）
   - 可选：过程奖励（每轮尝试）

3. **ms-swift 集成**
   - 注册 JailbreakEnv
   - 配置 GYMScheduler
   - 测试单轮训练

### Phase 2: 主实验（Week 2）

1. **单轮 baseline**
   - max_turns=1
   - 验证基础 pipeline

2. **多轮训练**
   - max_turns=3, 5
   - 验证多轮学习

3. **自适应停止**
   - 模型自主决定何时停止

### Phase 3: 消融 + 对比（Week 3）

1. **消融实验**
   - 无 skill 库
   - 过程奖励
   - skill 库大小

2. **对比实验**
   - vs SESS（规则系统）
   - vs AHR-GRPO（单轮 RL）

### Phase 4: 评估 + 论文（Week 4）

1. **全量评估**
   - 5 benchmarks × multiple models
   - 跨族迁移测试

2. **论文撰写**
   - 方法章节
   - 实验章节
   - 图表生成

---

## 6. 预期结果

### 主实验（default benchmark, Qwen3-4B target）

| 方法 | 预期 ASR | 说明 |
|------|----------|------|
| SESS (medium_evo) | 99.7% | 第二章最优，规则选择 |
| AHR-GRPO (Ch.1) | 33.2% | 第一章最优，单轮 RL |
| Single-turn RL | ~35% | 单轮 baseline |
| Multi-turn RL (3) | ~50% | 多轮学习 |
| **Multi-turn RL (5)** | **~60%** | 充分多轮交互 |
| Multi-turn + Process Reward | ~65% | 过程奖励辅助 |

### 迁移实验（gpt-oss-20b target）

| 方法 | 预期 ASR | 说明 |
|------|----------|------|
| SESS | 32.5% | 第二章基线 |
| **Multi-turn RL** | **~50%** | 跨族迁移提升 |

---

## 7. 文件结构

```
exp/agentic_rl/
├── README.md                    # 本文档
├── src/
│   ├── jailbreak_env.py         # GYM 环境实现
│   ├── reward_functions.py      # 奖励函数
│   └── plugin.py                # ms-swift 插件
├── scripts/
│   ├── common.sh                # 共享函数
│   ├── exp01_single_turn.sh     # 单轮 baseline
│   ├── exp02_multi_turn_3.sh    # 多轮 3 轮
│   ├── exp03_multi_turn_5.sh    # 多轮 5 轮
│   ├── exp04_multi_turn_adaptive.sh  # 自适应停止
│   ├── exp05_no_skill.sh        # 消融：无 skill
│   ├── exp06_process_reward.sh  # 消融：过程奖励
│   ├── exp07_skill_size.sh      # 消融：skill 库大小
│   ├── exp08_vs_sess.sh         # 对比：vs SESS
│   ├── exp09_vs_ahr_grpo.sh     # 对比：vs AHR-GRPO
│   └── eval_all.sh              # 全量评估
└── output/                      # 实验输出
```

---

## 8. 关键配置

每个脚本开头可配置：

```bash
# GPU 配置
NUM_GPUS=8
GUARD_GPU=0
TARGET_GPU=1
ROLLOUT_GPUS="2,3"
TRAIN_GPUS="4,5,6,7"

# 模型路径
BASE_MODEL="/home/tiger/models/Qwen/Qwen3-4B"
GUARD_MODEL="/home/tiger/models/Qwen/Qwen3Guard-Gen-4B"
TARGET_MODEL="/home/tiger/models/Qwen/Qwen3-4B"

# 端口
GUARD_PORT=8001
TARGET_PORT=8002
ROLLOUT_PORT=8003

# 训练超参
MAX_TURNS=5
MAX_STEPS=500
LEARNING_RATE=1e-5
NUM_GENERATIONS=8
```

---

## 9. 注意事项

1. **ms-swift 版本**：需要最新版本（支持多轮 GRPO + GYM）
2. **vLLM 版本**：需要 0.18.0+（支持 async engine）
3. **多轮采样**：使用 async engine 提高效率
4. **Loss mask**：环境反馈部分不计算 loss
5. **SwanLab**：实验追踪自动记录

---

## 10. TODO

- [ ] 实现 JailbreakEnv
- [ ] 实现 Reward Function
- [ ] 注册 ms-swift 插件
- [ ] 测试单轮训练
- [ ] 测试多轮训练
- [ ] 验证 gpt-oss-20b 可用性
