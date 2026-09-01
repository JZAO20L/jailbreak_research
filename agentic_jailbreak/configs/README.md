# Agentic Jailbreak 配置文件

## 目录结构

```
configs/
└── accelerate_config.yaml    # Accelerate 配置（多卡训练）
```

## accelerate_config.yaml

用于多卡分布式训练的配置。

**使用方式**：
```bash
accelerate launch --config_file configs/accelerate_config.yaml train.py
```

**配置内容**：
- 分布式类型：MULTI_GPU
- 混合精度：bf16
- GPU 数量：根据实际可用 GPU 自动检测

## 其他配置

训练超参数在各实验脚本中配置：

```bash
# 在 exp01_single_turn.sh 等脚本中
MAX_TURNS=5
MAX_STEPS=500
LEARNING_RATE=1e-5
NUM_GENERATIONS=8
PER_DEVICE_BATCH=2
GRAD_ACCUM=4
```

## GPU 配置

根据可用 GPU 数量自动调整：

```bash
# 8-GPU 模式
NUM_GPUS=8
GUARD_GPU=0
TARGET_GPU=1
ROLLOUT_GPUS="2,3"
TRAIN_GPUS="4,5,6,7"

# 4-GPU 模式
NUM_GPUS=4
GUARD_GPU=0
TARGET_GPU=1
ROLLOUT_GPUS="2"
TRAIN_GPUS="3"
```
