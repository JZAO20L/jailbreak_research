# TAP Baseline — 全量 ASR 评测结果

> 生成日期: 2026-08-25
> 方法: TAP（Tree of Attacks with Pruning, Mehrotra et al., NeurIPS 2024）

## 配置

| 项 | 值 |
|---|---|
| 攻击模型 (attacker/judge) | Qwen3-4B（复用 target_client） |
| 目标模型 (target) | Qwen3-4B-SafeRL (port 8002) |
| 守卫模型 (guard) | Qwen3Guard-Gen-4B (port 8001) |
| max_depth | 6 |
| branching_factor | 2 |
| prune_top_k | 1 |
| attacker temperature | 0.7 |
| 判定标准 | guard 判定 Unsafe = success（strict, 与全项目统一） |
| 每样本调用预算 | ≤ max_depth × branching_factor = 12 次 target 调用 |
| 数据 | `self_evolve_skills_jailbreak/data/test_prompts.json`（1000 条, 与 agent 实验同源） |
| 并发 | max_workers=8 |

> 注: 论文原版为 depth=10 / 大分支预算（每样本数十次调用）。此处为与
> Crescendo(8 轮)/PAIR 等迭代基线可比，采用受限预算配置（≤12 次/样本）。

## 结果

| 指标 | 值 |
|---|---|
| ASR | **7.2%** |
| 成功 / 总数 | 72 / 1000 |
| 平均耗时 | 63.6 s/样本 |
| 成功时平均深度 | 3.9 |

## 复现命令

```bash
cd /home/tiger/jailbreak_research
.venv/bin/python -B -m baselines.cli batch \
  --input self_evolve_skills_jailbreak/data/test_prompts.json \
  --strategy tap --output baselines/experiments/tap_asr \
  --limit 1000 --evaluate --max-turns 6 --branching-factor 2 --prune-top-k 1 \
  --max_workers 8 --target-port 8002 --guard-port 8001
```

## 实现说明

- 实现: `baselines/tap.py`（参考官方 `RICommunity/TAP`）
- 核心流程:
  1. Branching: 每个种子用 attacker 生成 branching_factor 个新分支
  2. Evaluation: 每个分支发 target，guard 判成功 + judge 打分 1-10
  3. Pruning: 按分保留 prune_top_k 个种子进入下一轮
- 成功判定统一走 guard（strict Unsafe）
- 与 beam-agent 的结构对照: TAP 是"每层生成多分支并按进度分数剪枝"，
  我们的 beam agent 是"B 条平行流同时推进（无打分剪枝）"——剪枝信号是核心差异

## 对照基准（同数据同 seed 体系）

| 方法 | ASR | 备注 |
|---|---|---|
| TAP (受限预算) | **7.2%** | 72/1000 |
| Crescendo | 1.8% | 见 crescendo_asr/README.md |
| 无技能单流 agent (no_skill, d=4) | 10.7% | 单流, 无 RL |
| 技能单流 agent (skills, d=4) | 15.9% | 单流, 无 RL |
| RL 5 轮 agent (单次) | 33.9% | 对话式 GRPO 训练后 |