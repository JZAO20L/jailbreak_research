# Agentic Jailbreak 实验结果（截至 2026-08-18）

> 第三章核心方法：对话式(Agentic)多轮攻击 Agent + GRPO 强化学习训练。
> 本目录组织实验相关产物（结果副本、配置、复现说明）；原始完整产物见 `output/`（路径在下文"产物索引"）。

## 1. 实验设置摘要

| 项 | 配置 |
|---|---|
| Harness 形态 | 对话式 Agent：消息真实累积，system → user(初始观察) → assistant(攻击动作) → user(环境反馈) → ... |
| 训练算法 | GRPO（ms-swift 4.3.2，`--rlhf_type grpo` + `gym_scheduler`/`jailbreak_env`），无 critic，组内相对 advantage |
| 训练对象 | Qwen3-4B + LoRA（policy）；采样经 vLLM rollout；target=Qwen3-4B-SafeRL；guard=Qwen3Guard-Gen-4B |
| 奖励 | ASR-only（稀疏 0/1：guard 判 Unsafe 且轮内成功=1.0，否则 0），`beta=0.05` KL 正则 |
| 动作空间 | no_skill 自由生成攻击文本（skill 注入经实验证明为负资产，已被弃用） |
| 数据 | 1000 条有害 prompt（`output/grpo_data.jsonl`），训练与评估同数据集，全量评估 |
| 训练超参 | 500 步、num_generations=8、per_device_batch 2、grad_accum 4、lr 1e-5、max_completion_length 2048 |
| 评估 | `eval_conv.sh`：对话式协议（与训练一致），temp 0.7；beam-2 为测试时 2 候选放大（任一成功即成功） |

## 2. 主结果矩阵（全量 1000 条，对话式同口径，ASR）

| 配置 | 单次 | beam-2 | 训练耗时 |
|---|---|---|---|
| 未训练，1 轮 | 9.0% | — | — |
| **exp01：RL，1 轮** | **10.0%** | 17.1% | 2h26m |
| 未训练，3 轮（RL 基线） | 23.2% | ~35%（前 100 子集） | — |
| **exp02：RL，3 轮** | **25.8%** | **35.2%** | 6h27m |
| **exp03：RL，5 轮** | **33.9%** | **44.4%** | 10h53m |

- 结论：RL 增益随 max_turns 单调放大（10.0 → 25.8 → 33.9）；5 轮单次超越预定目标 23.2%，beam-2 超越 35% 目标，均超过 AutoDAN 19% 参照。

## 3. 多轮曲线（RL vs 未训练）

```
ASR
40% +             beam: 17.1 ─ 35.2 ─ 44.4
   +
30% +                      exp02 25.8 ─ exp03 33.9
   +
20% +
   +     exp01 10.0
10% +    ── 未训练: 1轮 9.0 ─ 3轮 23.2 ─（5 轮同口径基线待补）
   +
 0% +────┬─────────┬─────────┬─────────┬─────────
        1 轮       3 轮       5 轮
```

未训练 5 轮同口径基线尚未测量（仅有 AV1 select+adapt 无状态 11.1% 与 AV2 无状态 5 轮前 100 22% 作参考）。

## 4. 与既有基线/消融对比（设计空间）

### 4.1 对话式前 100 条（2026-08-13，未训练模型，3 轮）

| 变体 | ASR | 结论 |
|---|---|---|
| conv no_skill | 25.0% | 主方法形态（消息累积使反思有效，+7pp vs 无状态） |
| conv no_skill beam-2 | 35.0% | beam 放大收敛 ~35% |
| conv select+adapt | 5.0% | skill 注入恒为负资产 → 弃用 |

### 4.1b Skill 调用时机消融（2026-08-18，未训练 base，3 轮）

| 变体 | 前 100 | 全量 1000 | 说明 |
|---|---|---|---|
| no_skill（主方法对照） | 25.0% | 23.2% | — |
| skill_once（轨迹只调一次） | 9.0% | — | 强制调用（即使只一次）仍负资产 |
| skill_every_turn（每轮调用） | 6.0% | — | 同旧 select_adapt 语义 |
| skill_decide（LLM 自决） | 28.0% | 19.0% | 前 100 正收益为子集噪声；全量口径仍低于 no_skill（-4.2pp） |

结论：skill 注入负资产的根源是**强制候选不对齐**（54 个 skill 中多数 quality=0，top-5 quality 检索全局最优而非按 prompt 对齐）；LLM 自决可部分规避（-4pp vs -18pp）但未转正。可选后续：similarity 检索 + skill_decide 重测、动作使用率分析。

### 4.2 无状态 harness（legacy，2026-08-12，未训练模型）

| 变体 | 3 轮（前 100 / 全量） | 消融 |
|---|---|---|
| AV2 No-Skill | 18% / 23.2% | 自由生成的威力（> select+adapt 5%） |
| AV6 No-Skill + Beam-2 | 35%（前 100） | Beam 是最大杠杆 |
| AV1 Select+Adapt | 5% / 9.1%(v1 3 轮全量) | 基线 |
| AV3 Select-Only | 3% | 适配有价值但整体弱 |
| AV4 Beam（select 版） | 8% | — |
| AV5 Sim-Retrieval | 6% | quality vs similarity 检索 |

### 4.3 外部参照

直发 0% / PAIR 0% / AutoDAN 19% / AV1 多轮曲线 4.3 → 9.1 → 11.1 → 12.5%(10 轮子集)。

## 5. 产物索引

| 产物 | 路径 |
|---|---|
| exp01 checkpoint（LoRA） | `output/single_turn_agent/v0-20260816-143432/checkpoint-500/` |
| exp01 合并权重 | `output/single_turn_agent/merged-500/` |
| exp01 评估结果 | `output/eval_results/conv_no_skill_1turn_rl_exp01/`（beam: `conv_no_skill_beam_1turn/`） |
| exp02 checkpoint（LoRA） | `output/multi_turn_3_agent/v0-20260816-195521/checkpoint-500/` |
| exp02 合并权重 | `output/multi_turn_3_agent/merged-500/` |
| exp02 评估结果 | `output/eval_results/conv_no_skill_3turn_rl_exp02/`（beam: `conv_no_skill_beam_3turn/`） |
| exp03 checkpoint（LoRA） | `output/multi_turn_5_agent/v0-20260817-083321/checkpoint-500/` |
| exp03 合并权重 | `output/multi_turn_5_agent/merged-500/` |
| exp03 评估结果 | `output/eval_results/conv_no_skill_5turn_rl_exp03/`（beam: `conv_no_skill_beam_5turn_rl_exp03/`） |
| 训练数据 | `output/grpo_data.jsonl`（1000 条） |
| 事件记录 | `docs/LOG.md`、`docs/TODO.md` |
| 结果副本 | `exp/results/*.json`（本目录） |

## 6. 复现命令

```bash
# 1. 启动服务（4-GPU 布局必须在 source 前声明）
NUM_GPUS=4 bash scripts/start_servers.sh

# 2. 训练（exp01/02/03 仅 max_turns 不同）
bash scripts/exp01_single_turn.sh     # max_turns=1
bash scripts/exp02_multi_turn_3.sh    # max_turns=3
bash scripts/exp03_multi_turn_5.sh    # max_turns=5

# 3. 评估（评估需标准 vLLM policy：先 swift export --merge_lora + vllm serve）
bash scripts/eval_conv.sh no_skill 1000 3        # 单次
bash scripts/eval_conv.sh no_skill_beam 1000 3 2 # beam-2
```

已知坑（详见 LOG 08-16/17）：`start_servers.sh` 需 `NUM_GPUS=4` 前置声明；训练命令需显式 `--generation_batch_size`；swift rollout 偶发忽略 `--port`（用 `--load_args false` 或 `ROLLOUT_PORT` 覆盖）；评估的 VLLMClient 需要标准 `/v1/models` 端点 → 用 merge_lora 后 `vllm serve`。

## 7. 待办/后续

- [ ] 未训练 5 轮同口径基线（补全主矩阵对照）
- [ ] 奖励建模扩展：过程/效率/多样性 reward（R1 ASR-only 已完成）
- [ ] 多 benchmark 评估（advbench / harmbench / jailbreakBench）
- [ ] 对比 SESS（第二章）、PAIR、AHR-GRPO（第一章）
- [ ] REPORT 单元 8 定稿
## 8. 单流重测（2026-08-25，beam_agent --beams 1，全量 1000, seed=42）

与双流（08-24 beam_pilot）同源同 seed，仅流数不同：

| 配置 | 单流 (B=1) | 双流 (B=2) | beam 增益 |
|---|---|---|---|
| no_skill | **10.7%** | 14.7% | +4.0pp |
| skills step | **15.9%** | 23.3% | +7.4pp |
| skills trajectory | **15.4%** | 21.5% | +6.1pp |
| skill 增益（流内） | +5.2pp | +8.6pp | — |

- 结论 1:单流下 skill 已是正资产——"skill 负资产"旧结论（conv 口径 19% vs 23.2%）不成立
- 结论 2:beam 放大 skill 收益（+4.0→+8.6pp），存在 beam×skill 协同
- 口径警示:conv_eval(3 轮对话)与 beam_agent(4 层独立凭据)不可直接比

## 9. 搜索类基线：Crescendo / TAP（2026-08-25/26，全量 1000）

| 基线 | 实现 | 配置 | ASR |
|---|---|---|---|
| Crescendo | `baselines/crescendo.py`（官方 Automated-Multi-Turn-Jailbreaks 参考） | 8 轮, 43.9s/样本 | **1.8%** |
| TAP | `baselines/tap.py`（官方 RICommunity/TAP 参考） | depth6/branch2/prune1, ≤12 调用/样本 | **7.2%** |

- 统一判定:guard strict Unsafe;数据同 `test_prompts.json` 体系;详细见 `baselines/experiments/{crescendo,tap}_asr/README.md`
- 对照:所有未训练 agent 变体（单流 10.7%~15.9%）均高于 TAP；RL 后 33.9%（2 倍以上）

## 10. RFT 冷启动 + GRPO 复训（2026-08-26，进行中）

流水线（数据隔离版）:
1. RFT 数据:beam_pilot 成功轨迹→前缀式对话 SFT 样本（`build_rft_data.py`,train 种子集 369 条轨迹）
2. SFT:ms-swift sft + LoRA 3 epochs → merge → `rft_sft_merged/`
3. GRPO 复训:`rft_grpo.sh`（500 步/5 轮/ASR-only/lr=1e-5,与 exp03 全同配置,初始权重=冷启动模型）
4. 评估:merge → deployment 8003 → eval_conv.sh 5 轮单次/beam-2 → 与纯 GRPO(33.9%/44.4%) 对比

数据隔离：RFT 种子=train[1000:2000]、GRPO 训练=train[0:1000]、评估=test，两两不重叠（修正了早期 RFT 用 test 集种子的泄漏问题）

状态:GRPO 复训运行中（见 TODO.md 09-26 节与 LOG.md）

### 10.1 RFT 冷启动评估结果（2026-08-29，V100 fp16）

RFT(SFT) 模型（beam 成功轨迹 → 对话式 SFT，2192 样本/3 epochs）5 轮评估：

| 口径 | RFT(SFT) | 未训练 | 纯 GRPO(E3) |
|---|---|---|---|
| 单次 | **6.8%** | 3轮 23.2% / 1轮 9.0% | 33.9% |
| beam-2 | **5.1%** | — | 44.4% |

**结论：RFT 冷启动（beam→conv SFT）在本设置为负收益**。候选原因：
1. **协议失配**（主因）：RFT 数据来自 beam 协议轨迹（每轮独立生成+历史摘要），监督时为对话式协议（消息累积）。SFT 学到的"上下文→动作"映射与评估上下文结构不匹配
2. **分布坍缩/过拟合**：3 epochs 在 2192 样本上，模型收敛到有限成功模板，多样性/探索下降
3. system prompt 若不一致会加剧失配（SFT 用 plugin SYSTEM_PROMPT）

**修正方向**（未执行）：
- 用对话式协议（conv_eval）的成功轨迹直接做 RFT 数据（消除协议失配）
- 降低 epochs（1-2）或加 eval 集早停，防坍缩
- 先小规模验证 RFT 模型在 conv 单轮上的行为（判断是协议问题还是过拟合）

相关：GRPO 复训在本机 V100 上被 swift 4.3.2 trainer 早期崩溃阻塞（9 次失败），待算力银行/白名单通道解决后执行。
