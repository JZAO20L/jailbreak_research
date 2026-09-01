# AHR-GRPO Paper Experiments

基于 ms-swift GRPO 框架的论文级对比实验，验证自适应混合奖励（Adaptive Hybrid Reward）的有效性。

## 核心贡献

1. **自适应奖励建模**：根据 ASR 和 Judge 奖励的方差比动态调整权重 λ
2. **多模型交互流程**：Policy → Target → Guard + Judge → 自适应更新

## 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GPU 0 (推理服务)                             │
│  ┌─────────────────────┐    ┌─────────────────────┐                │
│  │  Guard Server       │    │  Target&Judge Server│                │
│  │  Qwen3Guard-Gen-4B  │    │  Qwen3-4B           │                │
│  │  Port: 8001         │    │  Port: 8002         │                │
│  │  max_model_len: 8k  │    │  max_model_len: 8k  │                │
│  └─────────────────────┘    └─────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
                              ▲ HTTP
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU 1-2 (Policy 训练 - colocate)                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Policy Model (Qwen3-4B) + vLLM (TP=2)                       │  │
│  │  - LoRA training                                              │  │
│  │  - Generates rewritten prompts                                │  │
│  │  - Reward functions call GPU 0 servers via HTTP               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 数据流

```
Dataset (original_prompt)
        │
        ▼
┌───────────────────┐
│  Policy Model     │  GPU 1-2, colocate
│  generates k=8    │
│  rewritten prompts│
└───────────────────┘
        │
        ▼ completions (rewritten prompts)
┌───────────────────────────────────────────────────────────────────┐
│  Reward Functions (called by ms-swift)                            │
│                                                                   │
│  1. Send rewritten prompt → Target server (GPU 0, port 8002)     │
│     → Get response                                                │
│                                                                   │
│  2. Send response → Guard server (GPU 0, port 8001)              │
│     → Get ASR score (0/0.5/1.0)                                   │
│                                                                   │
│  3. Send response + original_prompt → Judge server (GPU 0)       │
│     → Get quality score (0~1)                                     │
│                                                                   │
│  4. Combine: final_reward = λ × ASR + (1-λ) × Judge             │
│     (λ is adaptive based on variance ratio)                       │
└───────────────────────────────────────────────────────────────────┘
```

## 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **num_samples** | 1000 | 从 10k 数据集中随机抽取的样本数 |
| **num_epochs** | 2 | 训练轮数 |
| **per_device_batch** | 32 | 每卡 batch size |
| **num_gpus** | 2 | 训练 GPU 数 |
| **grad_accum** | 2 | 梯度累积步数 |
| **num_generations** | 8 | GRPO 组大小 |
| **max_steps** | 自动计算 | `ceil(samples × epochs / (batch × gpus × accum))` |

**默认配置计算示例：**
- 1000 × 2 / (32 × 2 × 2) = **62 steps**

### 使用示例

```bash
# 默认：1000 条数据，2 epochs
bash train.sh ahr_grpo

# 自定义数据量和 epoch
bash train.sh ahr_grpo --num_samples 2000 --num_epochs 3

# 查看自动计算的 steps
bash train.sh ahr_grpo --num_samples 500 --num_epochs 1
# → 500 × 1 / (32 × 2 × 2) = 16 steps
```

### 模型 Context Length

| 模型 | max_model_len | max_tokens (生成) |
|------|---------------|------------------|
| Policy | 4096 | 2048 |
| Target | 8192 | 2048 |
| Guard | 8192 | 256 |
| Judge | 8192 | 256 |

## 实验设计

### 主实验：验证 AHR-GRPO 整体有效性

| 实验 | 奖励配置 | 说明 |
|------|----------|------|
| `baseline` | 无训练 | 原始 prompt 直接攻击（参考基准） |
| `pure_asr` | 仅 ASR 奖励 | 验证单一奖励信号 |
| `pure_judge` | 仅 Judge 奖励 | 验证单一奖励信号 |
| `fixed_hybrid` | 固定 1:1 混合 | 验证混合奖励 vs 单一奖励 |
| `ahr_grpo` | **自适应混合（Ours）** | 动态调整 λ |

### 消融实验：验证自适应权重的必要性

| 实验 | EMA β | 窗口大小 | 说明 |
|------|-------|----------|------|
| `ablation_beta0` | 0.0 | 1（无记忆） | 响应最快，易震荡 |
| `ablation_beta08` | 0.8 | ~5 | 中等记忆 |
| `ablation_beta09` | 0.9 | ~10 | 长记忆，更稳定 |

## 文件结构

```
paper_ahr_grpo/
├── README.md                    # 本文档
├── config.py                    # 统一配置文件
├── reward_functions.py          # 自定义奖励函数
│   ├── asr_reward()             # ASR 奖励（Target + Guard）
│   ├── judge_reward()           # Judge 奖励（质量评估）
│   ├── fixed_hybrid_reward()    # 固定 1:1 混合
│   └── adaptive_hybrid_reward() # 自适应混合（核心）
├── plugin.py                    # ms-swift 插件注册（从环境变量读取配置）
├── rollout.sh                   # vLLM 推理服务启动（GPU 0）
├── train.sh                     # 训练启动脚本（GPU 1-2）
├── eval.sh                      # 评估脚本
├── summarize_results.py         # 结果汇总
└── output/                      # 实验输出
    ├── logs/                    # 推理服务日志
    └── <experiment_type>/       # 训练输出
```

## 快速开始

### 1. 启动推理服务（GPU 0）

```bash
cd /home/tiger/jailbreak_research/RL4jailbreak

# 启动 Guard 和 Target&Judge 服务
bash experiments/paper_ahr_grpo/rollout.sh start

# 检查服务状态
bash experiments/paper_ahr_grpo/rollout.sh status

# 查看日志
tail -f experiments/paper_ahr_grpo/output/logs/guard_rollout.log
tail -f experiments/paper_ahr_grpo/output/logs/target_judge_rollout.log
```

### 2. 运行实验（GPU 1-2）

```bash
# 主实验
bash experiments/paper_ahr_grpo/train.sh baseline          # 基线（无需训练）
bash experiments/paper_ahr_grpo/train.sh pure_asr          # 仅 ASR 奖励
bash experiments/paper_ahr_grpo/train.sh pure_judge        # 仅 Judge 奖励
bash experiments/paper_ahr_grpo/train.sh fixed_hybrid      # 固定 1:1 混合
bash experiments/paper_ahr_grpo/train.sh ahr_grpo          # 自适应混合（Ours）

# 消融实验
bash experiments/paper_ahr_grpo/train.sh ablation_beta0    # β=0.0
bash experiments/paper_ahr_grpo/train.sh ablation_beta08   # β=0.8
bash experiments/paper_ahr_grpo/train.sh ablation_beta09   # β=0.9
```

### 3. SwanLab 监控

训练自动记录到 SwanLab：
- **Project**: `AHR_GRPO_Paper`
- **Run Name**: `{experiment_type}_steps250_bs32_lr1e-05`

访问 https://swanlab.cn 查看训练曲线。

### 4. 评估结果

```bash
# 评估基线（使用原始 prompt）
bash experiments/paper_ahr_grpo/eval.sh baseline --use_original_prompt

# 评估训练后的模型
bash experiments/paper_ahr_grpo/eval.sh ahr_grpo \
    experiments/paper_ahr_grpo/output/ahr_grpo_steps250_bs32_lr1e-05/final_lora
```

### 5. 汇总结果

```bash
python experiments/paper_ahr_grpo/summarize_results.py
```

## 自适应奖励参数

| 参数 | 值 | 说明 |
|------|-----|------|
| EMA Beta | 0.9 | 指数移动平均系数 |
| Alpha | 2.0 | 方差比敏感度 |
| Delta | -2.0 | Sigmoid 偏置 |
| Lambda Min | 0.2 | ASR 权重下界 |
| Lambda Max | 0.8 | ASR 权重上界 |

## 攻击 Prompt 策略

使用实验 1 筛选出的最佳策略：`hypothetical_scenario`（假设性场景框架）

## Judge 评分维度

使用**多维度混合评估**（multi_dimensional）：
- **Idea Preservation**：改写后 prompt 保留原始攻击意图的程度
- **Stealthiness**：改写后 prompt 的隐蔽程度

最终 Judge 分数为两个维度的平均值：`COMBINED = (IDEA + STEALTH) / 2`

## 预期结果

基于前期实验（TODO.md / experiment_results.md）：

| 方法 | 预期 ASR | Δ vs Baseline |
|------|----------|---------------|
| Baseline | ~30.8% | - |
| Pure ASR | ~25.0% | -5.8% |
| Pure Judge | ~28.5% | -2.3% |
| Fixed Hybrid (1:1) | ~32.3% | +1.5% |
| **AHR-GRPO (Ours)** | **~33.2%** | **+2.4%** |

## 注意事项

1. **端口冲突**：确保 `rollout.sh` 使用的端口（8001, 8002）未被占用
2. **显存不足**：如遇到 OOM，可降低 `vllm_gpu_memory_utilization`
3. **训练一致性**：确保 `rollout.sh` 和 `train.sh` 使用相同的端口配置
4. **结果复现**：设置 `--seed 42` 保证可复现性

## 相关文档

- [ms-swift GRPO 文档](/home/tiger/docs/swift/grpo.md)
- [实验结果汇总](/home/tiger/jailbreak_research/docs/experiment_results.md)
- [中期报告](/home/tiger/jailbreak_research/docs/midterm_report_v2.0_word.md)
