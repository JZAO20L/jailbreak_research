# Jailbreak Rewrite Prompt Generator

训练 Jailbreak 攻击生成器的项目，支持两种训练模式。

> **工作目录**: 所有命令从 `jailbreak_research/jailbreak_rewrite_prompt_generator` 目录执行

---

## 项目概述

**训练目标**: 学习从有害 prompt 生成攻击 prompt

**两种模式**:
- **instruction_only**: seed → rewrite_instruction（可解释）
- **end_to_end**: seed → jailbreak_prompt（端到端）

**训练策略**: SFT + RL

---

## 目录结构

```
jailbreak_rewrite_prompt_generator/
├── README.md
│
├── data/                    # 训练数据
│   ├── pair_skill/
│   │   ├── sft_train.jsonl  (2448)
│   │   ├── rl_train.jsonl   (2448)
│   │   └ test.jsonl         (1225)
│   └── autodan_skill/
│       ├── sft_train.jsonl  (3189)
│       ├── rl_train.jsonl   (3189)
│       └ test.jsonl         (1596)
│
├── scripts/                 # 通用脚本
│   ├── configs/
│   ├── eval/                # 评测脚本
│   └── training/
│
├── exp/                     # 实验目录
│   ├── data_extraction/     # ✅ 已完成
│   ├── sft/                 # 🔄 进行中
│   └ rl/                    # ⏳ 待实现
│
└── docs/
```

---

## 快速开始

### 1. Base 模型评测

```bash
bash exp/sft/scripts/eval_base_model.sh
```

### 2. SFT 训练

```bash
# 训练改写指令生成器
bash exp/sft/scripts/train_instruction.sh --use_lora --epochs 2

# 训练端到端攻击生成器
bash exp/sft/scripts/train_end2end.sh --use_lora --epochs 2
```

### 3. 评测 SFT 模型

```bash
bash scripts/eval/run_eval.sh --generator_model sft --generator_path exp/sft/results/xxx/
```

---

## 训练模式对比

| 模式 | 流程 | 优点 | 缺点 |
|------|------|------|------|
| **instruction_only** | seed → instruction → executor → prompt | 可解释性强 | 流程复杂 |
| **end_to_end** | seed → prompt | 简单直接 | 缺少中间过程 |

---

## 数据分割（40:40:20）

| 用途 | pair_skill | autodan_skill | 合计 |
|------|------------|---------------|------|
| SFT | 2448 | 3189 | 5637 |
| RL | 2448 | 3189 | 5637 |
| Test | 1225 | 1596 | 2821 |

---

## GPU 配置

| 任务 | GPU |
|------|-----|
| SFT 训练 | 4 GPU 分布式 |
| 评测 | 4 GPU (Gen, Exec, Target, Guard) |

---

## 文档索引

- [实验总览](exp/README.md)
- [SFT 实验](exp/sft/README.md)
- [RL 实验](exp/rl/README.md)

---

## 实验进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 0: 数据提取 | ✅ 完成 | 14k 条数据 |
| Phase 1: SFT | 🔄 进行中 | 两种模式训练 |
| Phase 2: RL | ⏳ 待开始 | 待 SFT 完成 |

---

*最后更新：2026-06-11*