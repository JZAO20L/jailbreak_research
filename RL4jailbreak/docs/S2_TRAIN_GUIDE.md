# S2 训练指南

## 概述

S2 阶段使用**直接攻击结果**作为 reward，而不是 S1 的 Judge 模型打分。

### S1 vs S2 关键区别

| 维度 | S1 | S2 |
|------|----|----|
| Reward 来源 | Judge 模型多维度打分 | Target+Guard 直接攻击结果 |
| Reward 维度 | idea_preservation, semantic_diversity, jailbreak_potential | Safe=0 / Controversial=0.5 / Unsafe=1 |
| Reward 权重 | w_idea, w_div, w_jb, w_format | w_attack, w_format |
| GPU 配置 | GPU0: Policy, GPU1: Judge | GPU0: Policy, GPU1: Target+Guard |

---

## 文件结构

```
scripts/
├── s2_train.py              # S2 训练主脚本
├── run_s2_train.sh          # S2 训练运行脚本（自动启动 Target+Guard）
└── run_s2_eval.sh           # S2 评估脚本
```

---

## 快速开始

### 1. 启动 S2 训练

```bash
cd /home/jiazixiao.jzx/.nanobot/TSRL4jailbreak
source /opt/conda/bin/activate rlgen
bash scripts/run_s2_train.sh
```

**GPU 配置：**
- GPU0: Policy 模型训练（GRPO + LoRA）
- GPU1: Target vLLM (0.5) + Guard vLLM (0.3)

**默认超参数：**
```bash
LORA_R=16
LEARNING_RATE=5e-5
NUM_EPOCHS=2
BETA=0.05
NUM_GENERATIONS=8
W_ATTACK=0.9
W_FORMAT=0.1
```

---

### 2. 评估训练结果

```bash
bash scripts/run_s2_eval.sh
```

**输出目录：** `output/s2_eval/<experiment_name>/`
- `01_rewritten_prompts.jsonl`: 重写后的 prompts
- `02_asr_report.json`: ASR 测试结果
- `summary.json`: 完整评估摘要

---

## 自定义配置

### 修改超参数

编辑 `scripts/run_s2_train.sh`：

```bash
# Training hyperparameters
LORA_R=16
LEARNING_RATE=5e-5
NUM_EPOCHS=2
BETA=0.05
NUM_GENERATIONS=8
MAX_COMPLETION_LEN=1024

# Reward weights
W_ATTACK=0.9  # 攻击成功率权重
W_FORMAT=0.1  # 格式奖励权重
```

### 修改模型路径

```bash
# Policy model
POLICY_MODEL_PATH="/home/jiazixiao.jzx/models/Qwen/Qwen3-4B"

# Target model (攻击目标)
TARGET_MODEL_PATH="/home/jiazixiao.jzx/models/Qwen/Qwen3-4B"

# Guard model (安全分类器)
GUARD_MODEL_PATH="/home/jiazixiao.jzx/models/Qwen/Qwen3-4B"
```

### 手动运行训练（不通过 shell 脚本）

```bash
CUDA_VISIBLE_DEVICES="0" python scripts/s2_train.py \
    --policy_model_path /home/jiazixiao.jzx/models/Qwen/Qwen3-4B \
    --lora_r 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 2 \
    --beta 0.05 \
    --num_generations 8 \
    --max_completion_len 1024 \
    --per_device_train_batch_size 32 \
    --w_attack 0.9 \
    --w_format 0.1 \
    --target_host 127.0.0.1 \
    --target_port 8102 \
    --target_model_name Target \
    --guard_host 127.0.0.1 \
    --guard_port 8103 \
    --guard_model_name Guard \
    --dataset_path /home/jiazixiao.jzx/.nanobot/TSRL4jailbreak/data/dataset/processed/s1_1k \
    --output_dir /home/jiazixiao.jzx/.nanobot/TSRL4jailbreak/output/s2/s2_baseline \
    --run_name s2_baseline \
    --bf16
```

---

## Reward 计算流程

### Attack Reward (w_attack=0.9)

```
1. Policy 生成 jailbreak prompt
   ↓
2. Target 模型响应 jailbreak prompt
   ↓
3. Guard 模型分类响应安全性
   ↓
4. 返回 reward: Safe=0 / Controversial=0.5 / Unsafe=1
```

### Format Reward (w_format=0.1)

```
1. 检查输出是否包含 <new_jailbreak_prompt> 标签
2. 检查内容长度 (10-500 词)
3. 返回 reward: 0.0 或 1.0
```

---

## 输出目录结构

```
output/s2/<run_name>/
├── final_lora/           # 训练完成的 LoRA 权重
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── logs/
│   ├── train_*.log       # 训练日志
│   ├── target_vllm.log   # Target vLLM 日志
│   └── guard_vllm.log    # Guard vLLM 日志
├── config.json           # 训练配置
└── checkpoint-*          # 训练 checkpoint（仅保留最后一个）
```

---

## 常见问题

### Q: Target 和 Guard 模型必须不同吗？
A: 可以相同。当前配置使用同一个模型（Qwen3-4B）同时作为 Target 和 Guard。

### Q: 如何调整 GPU 内存分配？
A: 修改 `run_s2_train.sh` 中的：
```bash
TARGET_GPU_MEM_UTIL=0.5  # Target 模型 GPU 内存利用率
GUARD_GPU_MEM_UTIL=0.3   # Guard 模型 GPU 内存利用率
```

### Q: 训练中断后如何恢复？
A: 当前不支持断点续训。需要重新启动训练。

### Q: 如何监控训练进度？
A: 
1. 查看日志：`tail -f output/s2/<run_name>/logs/train.log`
2. 查看 SwanLab dashboard（如果配置了）

---

## 下一步

1. 完成 S2 训练后，使用 `run_s2_eval.sh` 评估 ASR
2. 比较 S1 和 S2 的训练效果
3. 根据评估结果调整超参数或尝试 S1+S2 两阶段微调
