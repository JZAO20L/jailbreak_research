# Agentic Jailbreak: 多轮 Agent 越狱攻击

> 毕业论文第三章:PAIR 骨架 + Skills 的对话式 Agent 越狱(2026-09-02 口径定稿)

## 核心方法形态(定稿口径)

**Agent 核心 = PAIR 骨架 + Skills,LLM 自选(skill_decide)**,与 SESS(第二章)一脉相承。

- **Skill 库**:统一使用 **10-skill 精选库** `exp/skill_asr_sweep/seed_skills_top10.json`
  (54 库按实测单调用 ASR 筛选 top-10;此前"skill 负资产"结论的根因是 54 库全局 top-5 候选与 prompt 不对齐)
- **动作空间(每轮二选一)**:自由改写攻击 prompt(PAIR 式)或 `Selection: Skill [i]` + `Adapted Strategy:`(skill 调用+适配)
- **对话式 harness**:system → user(初始指令+skill 候选) → assistant(动作) → user(guard 反馈) 消息累积,默认 10 轮
- **主结果口径**:单轨迹 10 轮(beam-2 降级为 test-time scaling 附录分析)

### 对话式 Agent Loop

```
messages = [system, user(初始指令 + skill 候选)]
while turn < max_turns and not success:
    # 1. policy 生成动作(一次调用, thinking 模式)
    action_text = policy(messages)        # 自由改写 或 Selection+Adapted Strategy

    # 2. 解析并执行攻击
    attack_prompt = adapted_content + "\n\n" + prompt
    target_response = target(attack_prompt)
    guard_label = guard(attack_prompt, target_response)

    # 3. 环境反馈作为新 user 消息(上下文维护方式 = B 轴消融对象)
    messages += [assistant(action_text), user(guard 反馈)]
    #   C1 全量累积(现状) / C2 分层工作记忆(--work_memory) / C3 滑动窗口(--ctx_window N)

reward = 1.0 if any_turn_success else 0.0   # ASR-only
```

### 与 SESS 的关系

| 维度 | SESS(第二章) | Chapter 3(Agent) |
|------|-------------|------------------|
| **Skill 来源** | 自进化生成 | **复用 SESS 提取的有效 skills(精选 top-10)** |
| **Skill 选择** | 启发式规则 | **LLM 自选(skill_decide,RL 后学习)** |
| **Skill 适配** | 无 / 简单 rewrite | **LLM 生成 Adapted Strategy** |
| **失败后策略** | 固定 rewrite | **Agent 根据 guard 反馈调整** |
| **训练方式** | 无(规则系统) | **多轮 RL(GRPO)± RFT 冷启动** |

### 与 PAIR 的区别

| 维度 | PAIR | Chapter 3 Agent |
|------|------|-----------------|
| **Rewrite 逻辑** | 固定的 judge + rewrite prompt | **Agent 自由改写或 skill 调用(二选一动作空间)** |
| **Skill 使用** | 无 | **10-skill 精选库** |
| **反馈信号** | judge 打分 | **Qwen3Guard 结构化反馈** |
| **训练方式** | 无 | **多轮 RL ± RFT** |

## 技术栈

- **训练框架**：ms-swift 4.3.2（async GYM 环境 `plugin.py`,多轮 GRPO / SFT）
- **推理引擎**：vLLM server mode（client timeout 1000s,12 路并发采集验证过）
- **Target 模型**：Qwen3-4B-SafeRL(安全强化模型,`--enable-thinking`)
- **Guard 模型**：Qwen3Guard-Gen-4B
- **奖励设计**：ASR only(现协议);C 轴将扩展过程/效率/AHR 自适应组合
- **硬件**:4×V100-SXM2-32GB(V100 不支持 bf16,显式 `--dtype float16`)

## 当前状态(2026-09-03)

- ✅ 核心方法形态定稿:PAIR 骨架 + 10-skill,`skill_decide`,对话式 10 轮(协议已切齐 eval/env/plugin/build/采集脚本)
- ✅ 基座基线与消融:no_skill 全量 23.2%(5 轮)、skill_decide 全量 19.0%(5 轮,54 库)、TAP 7.2%、Crescendo 1.8%、PAIR 1.8%
- ✅ GRPO 链路验证:exp01/02/03(单/3/5 轮)500 步全量完成;RFT+GRPO 复训 200/200 步(暴露 reward 饥饿 → 10 轮 + num_generations=16 缓解)
- ✅ RFT v3 采集:split A 1000 条 skill_decide@10skills 10 轮(1000s timeout 事故已恢复,finalize 收尾中,ASR ~8-9%)
- 🔄 **B 轴上下文消融**(前置):C1 全量 / C2 工作记忆 / C3 滑动窗口,base + test C 前 300(`run_baxis_ctx_ablation.sh`)
- ⏳ A 轴 M1/M2/M3 训练臂 **HOLD 至 B 轴定稿 harness**;C 轴 reward 组合在 A 轴后
- 详见 [TODO](docs/TODO.md) 三轴规划与 [LOG](docs/LOG.md) 事件时间线

## 目录结构与文档导航

```
agentic_jailbreak/
├── README.md              # 本文件(项目概述 + 环境准备)
├── docs/                  # 文档
│   ├── README.md          # 设计文档 + FAQ
│   ├── TODO.md            # 唯一待办清单(三轴实验规划)
│   ├── LOG.md             # 唯一事件时间线
│   └── RSI_DESIGN.md      # Phase 5 RSI 递归自我改进设计
├── src/                   # 源代码(见 src/README.md)
│   ├── conv_eval.py       # 对话式评估核心(定稿 harness:消息累积/skill_decide 解析/C2 C3 上下文)
│   ├── env.py             # JailbreakEnv(GYM 环境;skill 候选质量池)
│   ├── plugin.py          # ms-swift async GYM 插件(DEFAULT_ENV_CONFIG=定稿协议)
│   ├── working_memory.py  # C2 分层工作记忆(前轮总结+末轮完整反馈)
│   ├── agent.py           # legacy Agent(无状态 harness,保留)
│   ├── rewards.py         # 奖励函数(C 轴扩展点,现 ASR-only)
│   └── eval.py            # 评估入口(--mode conversational / --ctx_window / --work_memory)
├── scripts/               # 实验脚本(见 scripts/README.md)
│   ├── common.sh          # 共享函数
│   ├── start_servers_v100.sh  # 4×V100 三服务(guard/target/policy)
│   ├── eval_conv.sh       # 对话式评估
│   ├── run_rft_collect_conv.sh        # RFT v3 采集(split A 1000 条,skill_decide@10skills)
│   ├── build_rft_data_conv.py         # conv 轨迹 → 全轨迹 SFT 样本
│   ├── rft_sft_conv.sh                # RFT SFT 训练
│   ├── run_baxis_ctx_ablation.sh      # B 轴上下文消融(C1/C2/C3)
│   ├── finalize_rft_collect_20260903.sh  # 采集收尾(补跑+merge+SFT)
│   ├── exp01/02/03_*.sh   # GRPO 单/3/5 轮
│   └── exp04_grpo_10turn.sh           # A 轴 M1 GRPO@10(待跑)
├── exp/                   # 消融实验产出
│   ├── skill_asr_sweep/   # 54 库单调用 ASR 扫描 + seed_skills_top10.json(定稿 skill 库)
│   └── beam_pilot/        # beam 消融
├── data/                  # 数据文件
│   └── skills.json        # 54 skills(SESS Layer 4)
├── baselines/             # TAP / Crescendo / PAIR baseline 实现
└── output/                # 实验输出(rft_collect_conv_10turn/ 等)
```

## 环境准备

### 模型与依赖

| 项 | 路径/版本 |
|----|-----------|
| Target | `/home/tiger/models/Qwen/Qwen3-4B-SafeRL` |
| Guard | `/home/tiger/models/Qwen/Qwen3Guard-Gen-4B` |
| Policy | `/home/tiger/models/Qwen/Qwen3-4B`(thinking-enabled 生成) |
| Skill 库 | `exp/skill_asr_sweep/seed_skills_top10.json`(定稿,10 个) |
| 虚拟环境 | `/home/tiger/jailbreak_research/.venv`(vLLM 0.18.0, ms-swift 4.3.2) |
| 硬件 | 4×V100-SXM2-32GB |

### 数据隔离(A/B/C 两两不重叠,08-26 修正)

| split | 来源 | 用途 |
|-------|------|------|
| A | train[1000:2000] | RFT 采集种子(`rft_seed_train1000.json`) |
| B | train[0:1000] | GRPO 训练(`grpo_data.jsonl`) |
| C | test(test_prompts.json,10k) | 评估 |

## GPU 部署(4×V100 现行布局)

```
┌───────────────┬────────────────┬────────────────┬──────────────┐
│     GPU 0     │      GPU 1     │      GPU 2     │     GPU 3    │
├───────────────┼────────────────┼────────────────┼──────────────┤
│ Guard 8001    │ Target 8002    │ Policy 8003    │ 训练         │
│ Qwen3Guard-4B │ Qwen3-4B-SafeRL│ Qwen3-4B       │ (SFT/GRPO)   │
│ max_len 8192  │ max_len 8192   │ max_len 24576  │ GRPO 时 GPU2 │
│               │                │                │ 换 rollout   │
└───────────────┴────────────────┴────────────────┴──────────────┘
```

注意:policy `max_model_len 24576`(16384 时出现过 input+max_tokens 越界崩溃,差 1 token 即崩)。

## 快速开始

```bash
# 1. 启动三服务(guard=8001 / target=8002 / policy=8003)
bash scripts/start_servers_v100.sh

# 2. 对话式评估(基座,skill_decide@10skills)
bash scripts/eval_conv.sh

# 3. RFT 采集(split A,12 路并发)
bash scripts/run_rft_collect_conv.sh

# 4. B 轴上下文消融(C1/C2/C3 × test C 前 300)
bash scripts/run_baxis_ctx_ablation.sh

# 5. A 轴训练臂(B 轴定稿后)
bash scripts/exp04_grpo_10turn.sh   # M1 只 GRPO
bash scripts/rft_sft_conv.sh        # M2 RFT SFT
```

## 参考

- [SESS 第二章](../self_evolve_skills_jailbreak/)
- [AHR-GRPO 第一章](../RL4jailbreak/)
- [PAIR 论文](https://arxiv.org/abs/2310.08414)
- [ms-swift 多轮 GRPO](../../docs/swift/grpo_multi_turn.md)