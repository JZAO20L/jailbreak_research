# Agentic Jailbreak: 多轮 Agent 越狱攻击

> 毕业论文第三章:基于 PAIR 框架的 Agent 式越狱攻击

## 研究目标

在 PAIR (Prompt Automatic Iterative Refinement) 框架基础上，用 Agent 替代固定的 rewrite 逻辑，实现：
- **Agent 分析**：理解攻击失败原因
- **动作选择**：从 SESS 提取的有效 skills 中选择最合适的
- **记忆机制**：可选的长期记忆，积累攻击经验
- **多轮 RL 训练**：通过强化学习优化 Agent 策略

## 核心设计

### Agent Loop

```
while turn < max_turns:
    # 1. 分析（Analysis）
    analysis = agent.analyze(prompt, history, memory)

    # 2. 动作选择（Action Selection）
    action = agent.select_action(analysis, available_skills, history)
    # action = {skill_idx, adapted_content, reasoning}

    # 3. 执行攻击
    attack_prompt = action.adapted_content + "\n\n" + prompt
    target_response = gpt_oss(attack_prompt)
    guard_label = guard(attack_prompt, target_response)

    # 4. 更新记忆
    memory.update({turn, action, response, success})

    # 5. 检查是否成功
    if guard_label == "Unsafe":
        reward = 1.0
        break

    turn += 1
```

### 与 SESS 的关系

| 维度 | SESS（第二章） | Chapter 3（Agent） |
|------|---------------|-------------------|
| **Skill 来源** | 自进化生成 | **复用 SESS 提取的有效 skills** |
| **Skill 选择** | 启发式规则 | **Agent 学习** |
| **Skill 适配** | 无 / 简单 rewrite | **Agent 学习** |
| **失败后策略** | 固定 rewrite | **Agent 根据反馈调整** |
| **训练方式** | 无（规则系统） | **多轮 RL** |

### 与 PAIR 的区别

| 维度 | PAIR | Chapter 3 Agent |
|------|------|-----------------|
| **Rewrite 逻辑** | 固定的 judge + rewrite | **Agent 分析 + 决策** |
| **Skill 使用** | 无 | **SESS skills** |
| **记忆机制** | 无 | **可选的长期记忆** |
| **训练方式** | 无 | **多轮 RL** |

## 技术栈

- **训练框架**：ms-swift（原生支持多轮 GRPO + GYM 环境）
- **推理引擎**：vLLM server mode + async engine
- **Target 模型**：Qwen3-4B-SafeRL（安全强化模型）
- **Guard 模型**：Qwen3Guard-Gen-4B
- **奖励设计**：ASR only（不需要 Judge）

## 当前状态

- ✅ 单轮 Agent、多轮 Agent 实现完成,冒烟测试通过(详见 [LOG](docs/LOG.md))
- ⏳ 纯 Agent 全量 ASR 基准**未跑**(只有少量冒烟用例)
- ⏳ RL 训练(ms-swift GRPO)**未开始**
- ⚠️ 存在 import bug 与 2-GPU 适配等阻塞项,见 [TODO](docs/TODO.md)

## 目录结构与文档导航

```
agentic_jailbreak/
├── README.md              # 本文件(项目概述 + 环境准备)
├── docs/                  # 文档
│   ├── README.md          # 设计文档 + FAQ
│   ├── TODO.md            # 唯一待办清单
│   └── LOG.md             # 唯一事件时间线
├── src/                   # 源代码(见 src/README.md)
│   ├── env.py             # JailbreakEnv（GYM 环境）
│   ├── agent.py           # Agent 实现
│   ├── memory.py          # 记忆机制
│   ├── rewards.py         # 奖励函数
│   ├── train.py           # 训练脚本
│   └── eval.py            # 评估脚本
├── scripts/               # 实验脚本(见 scripts/README.md)
│   ├── common.sh          # 共享函数
│   ├── exp01_single_turn.sh
│   ├── exp02_multi_turn_3.sh
│   ├── exp03_multi_turn_5.sh
│   └── eval_all.sh
├── data/                  # 数据文件(见 data/README.md)
│   └── skills.json        # 从 SESS 复制的有效 skills
├── configs/               # 配置文件(见 configs/README.md)
│   └── accelerate_config.yaml
└── output/                # 实验输出(见 output/README.md)
```

## 环境准备

### 模型与依赖

| 项 | 路径/版本 |
|----|-----------|
| Target | `/home/tiger/models/Qwen/Qwen3-4B-SafeRL` |
| Guard | `/home/tiger/models/Qwen/Qwen3Guard-Gen-4B` |
| Policy | `/home/tiger/models/Qwen/Qwen3-4B` |
| Skills | `data/skills.json`(54 个,来自 SESS Layer 4) |
| 虚拟环境 | `/home/tiger/jailbreak_research/.venv`(vLLM 0.18.0, ms-swift 4.3.2) |
| 硬件 | 2×A100 80GB |

### 服务端口(现状)

- GPU 0:SafeRL(Target)8000(训练与评估共用)
- GPU 1:Policy 8001 或 Guard 8002——**2 卡需分时复用,见 TODO P0**

## GPU 部署(4/8 卡参考布局)

```
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│  GPU 0  │  GPU 1  │  GPU 2  │  GPU 3  │  GPU 4  │  GPU 5  │  GPU 6  │  GPU 7  │
├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│  Guard  │ Target  │ vLLM    │ vLLM    │ Train   │ Train   │ Train   │ Train   │
│ (Guard) │(GPT-OSS)│ Rollout │ Rollout │  (TP=2) │  (TP=2) │  (TP=2) │  (TP=2) │
│  8001   │  8002   │  8003   │         │         │         │         │         │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

## 快速开始

```bash
# 1. 准备数据(从 SESS 复制 skills)
bash scripts/prepare_data.sh

# 2. 启动 vLLM servers(见 scripts/start_servers.sh)
bash scripts/start_servers.sh

# 3. 冒烟测试
python scripts/test_single_turn.py
python scripts/test_multi_turn.py

# 4. 全量评估(需先修 TODO P0 阻塞项)
bash scripts/eval_all.sh
```

## 参考

- [SESS 第二章](../self_evolve_skills_jailbreak/)
- [AHR-GRPO 第一章](../RL4jailbreak/)
- [PAIR 论文](https://arxiv.org/abs/2310.08414)
- [ms-swift 多轮 GRPO](../../docs/swift/grpo_multi_turn.md)