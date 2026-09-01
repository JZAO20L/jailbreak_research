#!/bin/bash
# RL 训练脚本 - 使用 GRPO + 自适应权重奖励建模
# 资源分配方案：2 GPU 训练 + 1 GPU vLLM server + 1 GPU Guard/Target

set -e

# 设置环境变量
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 参数配置
TRAIN_DATA_PATH="data/processed/triples/rl_train_pool.json"
DATA_SIZE=2000
SFT_MODEL_PATH="models/sft/sft_small"  # SFT 训练后的模型
OUTPUT_DIR="models/rl/rl_after_sft"
NUM_EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=1e-5
MAX_COMPLETION_LENGTH=256
NUM_GENERATIONS=4

# vLLM 配置
VLLM_MODE="server"  # server 或 colocate
VLLM_SERVER_HOST="localhost"
VLLM_SERVER_PORT=8000
VLLM_GPU_MEMORY_UTILIZATION=0.7

# Reward 配置
REWARD_WEIGHTS_SIMILARITY=0.5
REWARD_WEIGHTS_ASR=0.5
ADAPTIVE_REWARD=true

echo "========================================="
echo "RL 训练配置 (GRPO)"
echo "========================================="
echo "训练数据: $TRAIN_DATA_PATH"
echo "数据量: $DATA_SIZE"
echo "模型起点: $SFT_MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Learning rate: $LEARNING_RATE"
echo "Max completion length: $MAX_COMPLETION_LENGTH"
echo "Num generations: $NUM_GENERATIONS"
echo "vLLM 模式: $VLLM_MODE"
echo "自适应奖励: $ADAPTIVE_REWARD"
echo "========================================="

# 检查数据是否存在
if [ ! -f "$TRAIN_DATA_PATH" ]; then
    echo "错误: 训练数据不存在: $TRAIN_DATA_PATH"
    echo "请先运行数据处理脚本准备训练数据"
    exit 1
fi

# 检查 SFT 模型是否存在
if [ ! -d "$SFT_MODEL_PATH" ]; then
    echo "错误: SFT 模型不存在: $SFT_MODEL_PATH"
    echo "请先运行 SFT 训练脚本"
    exit 1
fi

# 方案1: Server Mode（推荐）
# 需要：
# - GPU 0: vLLM server
# - GPU 1-2: 训练 (2 GPU)
# - GPU 3: Guard + Target (部署)

if [ "$VLLM_MODE" = "server" ]; then
    echo "启动 vLLM Server (GPU 0)..."

    # 启动 vLLM server（在 GPU 0）
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
        --model "$SFT_MODEL_PATH" \
        --tensor_parallel_size 1 \
        --port "$VLLM_SERVER_PORT" \
        --gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" &
    VLLM_PID=$!

    echo "等待 vLLM server 启动..."
    sleep 30

    echo "启动 RL 训练 (GPU 1-2)..."

    # 使用 GPU 1-2 进行训练
    CUDA_VISIBLE_DEVICES=1,2 accelerate launch \
        --config_file scripts/configs/accelerate_config_2gpu.yaml \
        scripts/training/train_rl.py \
        --train_data_path "$TRAIN_DATA_PATH" \
        --data_size "$DATA_SIZE" \
        --model_name_or_path "$SFT_MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs "$NUM_EPOCHS" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LEARNING_RATE" \
        --max_completion_length "$MAX_COMPLETION_LENGTH" \
        --num_generations "$NUM_GENERATIONS" \
        --use_vllm \
        --vllm_mode "server" \
        --vllm_server_host "$VLLM_SERVER_HOST" \
        --vllm_server_port "$VLLM_SERVER_PORT" \
        --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        --reward_weights_similarity "$REWARD_WEIGHTS_SIMILARITY" \
        --reward_weights_asr "$REWARD_WEIGHTS_ASR" \
        --adaptive_reward \
        --logging_steps 10 \
        --save_steps 200 \
        --seed 42

    # 清理 vLLM server
    echo "停止 vLLM server..."
    kill $VLLM_PID

# 方案2: Colocate Mode（vLLM 和训练共享 GPU）
# 使用 GPU 0-3 进行训练，vLLM 共享内存
else
    echo "启动 RL 训练 (Colocate Mode, GPU 0-3)..."

    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --config_file scripts/configs/accelerate_config_4gpu.yaml \
        scripts/training/train_rl.py \
        --train_data_path "$TRAIN_DATA_PATH" \
        --data_size "$DATA_SIZE" \
        --model_name_or_path "$SFT_MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs "$NUM_EPOCHS" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LEARNING_RATE" \
        --max_completion_length "$MAX_COMPLETION_LENGTH" \
        --num_generations "$NUM_GENERATIONS" \
        --use_vllm \
        --vllm_mode "colocate" \
        --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        --reward_weights_similarity "$REWARD_WEIGHTS_SIMILARITY" \
        --reward_weights_asr "$REWARD_WEIGHTS_ASR" \
        --adaptive_reward \
        --logging_steps 10 \
        --save_steps 200 \
        --seed 42
fi

echo "========================================="
echo "RL 训练完成！"
echo "模型保存在: $OUTPUT_DIR"
echo "========================================="