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

reward = Σ per-turn(1.0 if any_success else 0.0)   # ASR-only 累加（0~10，env.step 每轮信号被 trainer 累加）
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

- **训练框架**：ms-swift 4.3.2（async GYM 环境 `plugin.py`；多轮 GRPO/DAPO/SFT；LoRA，`tuner_type=lora`）
- **推理引擎**：vLLM server mode（client timeout 1000s）
- **Target 模型**：Qwen3-4B（plain，现役；早期 SafeRL target 已切换）
- **Guard 模型**：Qwen3Guard-Gen-4B
- **奖励设计**：ASR only（每轮 success 累加，0~10）；C 轴将扩展过程/效率/AHR 自适应组合
- **显存治理**：`--use_liger_kernel true`（防 logits 物化）+ PDB=1 安全批几何（PDB=2 曾触发 74G/80G 红线，见 LOG 09-16/17）
- **现役节点**：4×A100-80GB（调试节点，即将到期迁移；早期 4×V100-32GB 需 fp16 + `src/v100_attn_patch.py`）

## 当前状态（2026-09-17）

- ✅ **A 轴后训练配方定稿**（六臂 M0-M5，test C 300）：**本任务下不开动态采样的 vanilla GRPO 纯负收益**（M1 −3.3pp / M3 −4.3pp / M5 长训 −8.7pp；训练量/epoch 非主因）——**DAPO 唯一正收益**（M4 **64.33%**）。完整结论见 [REPORT 单元 8](docs/REPORT.md)
- ✅ B 轴上下文定稿（C1 全量累积）;对话式 `skill_decide@10skill@10 轮` 主协议;基座 M0 54.0% / TAP 7.2% / Crescendo 1.8%
- ✅ M5 长臂负结果（1.0 epoch = **48.0%**，低于起点 M3）：ckpt / 训练曲线 / 评估数据全部入库（防服务器释放）
- ⏭️ **下一步（服务器迁移后）**：**M6 = DAPO 更多步数续训**（M4 ckpt 起，方案 B）→ C 轴奖励组合（R1+P/E/λ）
- 📦 关键产物：`checkpoints/`（LoRA 血缘链 + 最新档 + 重建说明）、`exp/results/curves|a_axis/`（论文绘图数据）、`scripts/m5_train_cmd_v5.sh`（续跑模板）、[PLAYBOOK](docs/PLAYBOOK_CH3_EXPERIMENTS.md)（全命令手册）
- 迁移指引：新节点 `git clone` 本仓库 → `checkpoints/M5b_grpo_10turn/README.md` 重建权重链 → [PLAYBOOK](docs/PLAYBOOK_CH3_EXPERIMENTS.md) 起服务续跑
- 详见 [TODO](docs/TODO.md)（待办）、[LOG](docs/LOG.md)（事件时间线）、[REPORT](docs/REPORT.md)（研究结论）

## 目录结构与文档导航

```
agentic_jailbreak/
├── README.md              # 本文件(项目概述 + 环境准备 + 迁移指引)
├── docs/                  # 文档
│   ├── TODO.md            # 唯一待办清单(三轴 + M6/迁移待办)
│   ├── LOG.md             # 唯一事件时间线(含事故/排障全记录)
│   ├── REPORT.md          # 唯一研究结论清单(单元 1-8;A 轴结论=单元 8)
│   ├── PLAYBOOK_CH3_EXPERIMENTS.md  # 第三章执行手册(从零到全命令)
│   └── RSI_DESIGN.md      # Phase 5 RSI 递归自我改进设计
├── checkpoints/           # LoRA 血缘链入库(仅 adapter)
│   ├── M2_rft_sft_conv10turn_e2 / M3_grpo_10turn / M4_dapo_10turn
│   └── M5b_grpo_10turn/   # v5-ckpt600(最新档) + v4pdb2-ckpt50(血缘) + README(重建命令)
├── exp/results/           # 论文数据归档(防服务器释放)
│   ├── curves/            # 训练曲线 jsonl(M3/M4/M5)+ 绘图说明
│   └── a_axis/            # 六臂评估 summary.json
├── src/                   # 源代码(见 src/README.md)
│   ├── conv_eval.py / env.py / plugin.py   # 定稿 harness(对话式 / skill_decide / C1)
│   ├── logps_chunk_patch.py   # A100 显存诊断补丁(plugin 加载)
│   ├── v100_attn_patch.py     # V100 SDPA 修复(legacy 节点用)
│   ├── rewards.py         # 奖励函数(C 轴扩展点,现 ASR-only)
│   ── ...                # (working_memory/agent/eval 等见 src/README.md)
├── scripts/               # 实验脚本(见 scripts/README.md)
│   ├── eval_conv.sh       # 对话式评估(必带 RUN_TAG 防覆盖)
│   ├── m5_train_cmd_v5.sh # 当前训练模板(PDB=1/G=8/liger/save50;迁移续跑用)
│   ── exp04_grpo_10turn.sh / rft_sft_conv.sh / build_rft_data_conv.py / ...
├── data/                  # 数据(grpo_data.jsonl=训练 B 段 1000 条; skills.json=54 skills)
├── baselines/             # TAP / Crescendo / PAIR baseline 实现
└── output/                # 实验输出(gitignore;训练 ckpt 每 50 步自动存档)
```

## 环境准备

### 模型与依赖

| 项 | 路径/版本 |
|----|-----------|
| Target | `model/Qwen3-4B`（plain，现役 8002） |
| Guard | `model/Qwen3Guard-Gen-4B`（现役 8001） |
| Policy/训练基座 | `model/Qwen3-4B`（thinking-enabled 生成） |
| Skill 库 | `exp/skill_asr_sweep/seed_skills_top10.json`（定稿 10 个） |
| 虚拟环境 | `jailbreak_research/.venv`（vLLM + ms-swift 4.3.2 + liger-kernel 0.8.2） |

迁移到新节点：`git clone` 本仓库 → 按 `checkpoints/M5b_grpo_10turn/README.md` 重建权重链 → 按 [PLAYBOOK](docs/PLAYBOOK_CH3_EXPERIMENTS.md) 起服务/续跑。

### 数据隔离(A/B/C 两两不重叠,08-26 修正)

| split | 来源 | 用途 |
|-------|------|------|
| A | train[1000:2000] | RFT 采集种子(`rft_seed_train1000.json`) |
| B | train[0:1000] | GRPO 训练(`grpo_data.jsonl`) |
| C | test(test_prompts.json,10k) | 评估 |

## GPU 部署（4 卡布局，A100 现役）

```
┌───────────────┬────────────────┬────────────────────┬──────────────┐
│     GPU 0     │      GPU 1     │       GPU 2        │     GPU 3    │
├───────────────────────────────┼────────────────────┼──────────────┤
│ Guard 8001    │ Target 8002    │ 训练时: rollout    │ 训练 trainer │
│ Qwen3Guard-4B │ Qwen3-4B plain │  8004(swift)       │ GRPO/DAPO    │
│               │                │ 评估时: policy     │              │
│               │                │  8003(vllm serve)  │              │
└───────────────┴────────────────┴────────────────────┴──────────────┘
```

注意：①评估用 8003 **标准 vllm serve**，训练用 8004 **swift rollout**（带 communicator，裸 vllm serve 报 404）；②`max_model_len 32768`（24576 时代差 1 token 越界崩过）；③**长任务必须 `setsid`**（未套 setsid 的训练曾被会话清理误杀，LOG 09-15）；④vLLM 残留 `VLLM::EngineCore` 需按 PID 单独清理（见 TODO 踩坑记录）。

## 快速开始

```bash
# 1. 起服务(guard=8001 / target=8002；训练另起 rollout 8004 / 评估另起 policy 8003)
bash scripts/start_servers_v100.sh        # 旧 V100 节点;A100 节点命令见 PLAYBOOK §1

# 2. 对话式评估（示例:ckpt-600;RUN_TAG 必带,防多臂结果互相覆盖）
RUN_TAG=m5_600 EVAL_WORKERS=12 bash scripts/eval_conv.sh skill_decide 300 10 2 10

# 3. 训练（当前模板 = v5:PDB=1/G=8/liger/save50；setsid 必加）
CUDA_VISIBLE_DEVICES=3 setsid nohup bash scripts/m5_train_cmd_v5.sh > output/logs/m5_train.log 2>&1 &

# 4. 续跑/迁移:权重重建见 checkpoints/M5b_grpo_10turn/README.md；完整命令见 PLAYBOOK §2.6b
```

## 参考

- [REPORT 研究结论](docs/REPORT.md)（单元 8 = A 轴定稿）
- [PLAYBOOK 执行手册](docs/PLAYBOOK_CH3_EXPERIMENTS.md)（从零到全命令）
- [SESS 第二章](../self_evolve_skills_jailbreak/)
- [AHR-GRPO 第一章](../RL4jailbreak/)
- [PAIR 论文](https://arxiv.org/abs/2310.08414)