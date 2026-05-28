# Layer 1: 核心方法消融实验

Grid Search 实验目录，用于验证 Skills 系统的最佳方法组合。

## 实验设计

**组合数**: 2 × 2 × 4 = 16 种

| 消融点 | 选项 | 说明 |
|--------|------|------|
| **A: skill_call_mode** | single_call / every_iteration | Skill 调用时机 |
| **B: skill_extraction_mode** | final_prompt / trajectory | Skill 总结粒度 |
| **C: update_strategy** | success_only / failure_only / both / statistical | Skill 进化策略 |

## 数据配置

数据通过 `extract_data.py` 预抽取，存储在 `self_evolve_skills_jailbreak/data/`:

| 数据集 | 文件 | 数量 | 来源 | 用途 |
|--------|------|------|------|------|
| Cold Start | `cold_start_prompts.json` | 200 条 | train.jsonl 前1000条的20% | 生成初始 Skills |
| Evolution | `evolution_prompts.json` | 800 条 | train.jsonl 前1000条的80% | Skills 进化 |
| Test | `test_prompts.json` | 1000 条 | 完整 test.jsonl | 最终 ASR 计算 |
| Eval | 从 Test 抽样 | 100 条 | Test 集随机抽样 | 中间评估 |

## 并发配置

**轨迹级并发**: 每个攻击轨迹（PAIR+Skills 完整迭代）并发执行

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_workers` | 8 | 并发攻击轨迹数 |

并发适用于所有阶段:
- **Cold Start**: 200 prompts 并发攻击
- **Evolution**: 每个 epoch 800 prompts 并发攻击
- **Test**: 1000 prompts 并发攻击
- **Intermediate Eval**: 100 prompts 并发评估

## 目录结构

```
layer1/
├── run_layer1.sh              # Bash 实验启动脚本
├── scripts/
│   ├── summarize_layer1.py    # 结果汇总脚本
│   └── generate_report.py     # Markdown 报告生成
├── results/                   # 实验结果输出
│   ├── result_*.json          # 每个组合的详细结果
│   ├── layer1_summary_*.json  # 汇总 + 消融分析
│   └── layer1_report_*.md     # Markdown 报告
├── logs/                      # vLLM 服务日志
│   ├── guard.log
│   └── target.log
└── README.md                  # 本文件
```

## 快速开始

### 1. 启动服务

```bash
# GPU 0: Guard (Qwen3Guard-Gen-4B, port 8002)
bash RL4jailbreak/scripts/start_guard.sh

# GPU 1-2: Target (Qwen3-4B, port 8001)
bash RL4jailbreak/scripts/start_target.sh
```

### 2. 运行实验

```bash
# 完整实验（16 组，全量数据，默认并发）
bash run_layer1.sh --skip_launch

# 高并发（根据服务承载能力调整）
bash run_layer1.sh --skip_launch --max_workers 16

# 限制测试数据量（快速验证）
bash run_layer1.sh --skip_launch --test_limit 100 --eval_limit 20

# 只运行单个组合（调试）
bash run_layer1.sh --skip_launch --single every_iteration trajectory both

# 断点续跑
bash run_layer1.sh --skip_launch --resume_from 8
```

### 3. 查看结果

```bash
# 查看 Markdown 报告
cat results/layer1_report_*.md

# 查看 JSON 汇总
cat results/layer1_summary_*.json | python -m json.tool
```

## 完整参数

```bash
bash run_layer1.sh [OPTIONS]

Options:
  --test_limit N       Test 数据量限制（默认使用全部 1000 条）
  --eval_limit N       中间评估数据量（每 epoch 结束后评估，默认 100）
  --num_epochs N       进化轮数（默认 3）
  --max_iterations N   最大攻击迭代次数（默认 10）
  --max_workers N      轨迹级并发数（默认 8）
  --skip_launch        跳过服务启动（连接已有服务）
  --resume_from N      从第 N 个实验开始（断点续跑）
  --single A B C       只运行单个组合（调试用）
  --guard_gpu N        Guard GPU 编号（默认 0）
  --target_gpu N,M     Target GPU 编号（默认 1,2）
```

## 实验流程

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: 启动 vLLM Servers                                            │
│   Guard:  GPU 0, Port 8002, Qwen3Guard-Gen-4B                        │
│   Target: GPU 1-2, Port 8001, Qwen3-4B (TP=2)                        │
├─────────────────────────────────────────────────────────────────────┤
│ Step 2: 运行 Grid Search (16 组，并发执行)                           │
│   每组实验:                                                          │
│     1. Phase Cold Start → 并发攻击 200 prompts → 生成初始 Skills    │
│     2. Phase Evolution → 并发攻击 800 prompts × 3 epochs            │
│     3. Phase Test → 并发攻击 1000 prompts → 计算 ASR                │
│     中间评估: 每 epoch 结束后并发评估 100 条                         │
├─────────────────────────────────────────────────────────────────────┤
│ Step 3: 统计结果                                                     │
│   - summarize_layer1.py → JSON 汇总                                  │
│   - generate_report.py → Markdown 报告                               │
├─────────────────────────────────────────────────────────────────────┤
│ Step 4: 关闭服务                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 结果示例

### Markdown 报告结构

```markdown
# Layer 1 Grid Search Experiment Report

## Experiment Overview
- Total combinations: 16

## Results Summary
| Method | ASR | Avg Iterations | Skill Count |
|--------|-----|----------------|-------------|
| every_iteration_trajectory_both | 45.2% | 3.5 | 25 |
| ... | ... | ... | ... |

## Best Combination
**every_iteration_trajectory_both**
- ASR: 45.2%
- Average Iterations: 3.5

## Ablation Analysis
### A: skill_call_mode
| Mode | Avg ASR | Count |
| every_iteration | 42.1% | 8 |
| single_call | 38.5% | 8 |

### B: extraction_mode
| Mode | Avg ASR | Count |
| trajectory | 41.8% | 8 |
| final_prompt | 39.2% | 8 |

### C: update_strategy
| Strategy | Avg ASR | Count |
| both | 43.5% | 4 |
| ... | ... | ... |

## Evolution Curves
每 epoch 的 ASR 变化趋势...
```

## 预期输出

| 指标 | 说明 | 目标 |
|------|------|------|
| **ASR** | Attack Success Rate | 越高越好 |
| **Avg Iterations** | 平均攻击迭代次数 | 越低越好（效率高） |
| **Skill Count** | 最终 Skills 数量 | 适中（有效但不冗余） |
| **Evolution Curve** | ASR vs epoch 变化 | 观察进化趋势 |

## 性能估算

基于并发数 `max_workers=8`:

| 阶段 | 数据量 | 单轨迹耗时 | 并发耗时 |
|------|--------|------------|----------|
| Cold Start | 200 | ~30s | ~12.5min |
| Evolution (1 epoch) | 800 | ~30s | ~50min |
| Test | 1000 | ~30s | ~62.5min |
| **单组实验** | - | - | ~3.5h |
| **16组 Grid Search** | - | - | ~56h |

可根据服务承载能力调整 `max_workers`。

## 下一步

Layer 1 完成后，使用最佳组合运行 Layer 2 数据消融：

```bash
cd ../layer2
bash run_layer2.sh --best_method "every_iteration trajectory both"
```