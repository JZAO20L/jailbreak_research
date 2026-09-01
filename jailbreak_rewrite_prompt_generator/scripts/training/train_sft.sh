#!/bin/bash
# SFT 训练脚本 - 使用 4 GPU 并发训练

set -e

# 设置环境变量
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 参数配置
TRAIN_DATA_PATH="data/processed/triples/sft_train_triples.json"
DATA_SIZE="small"  # small(1000), medium(1500), large(2000)
MODEL_NAME="Qwen/Qwen3-4B"
OUTPUT_DIR="models/sft/sft_${DATA_SIZE}"
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
LEARNING_RATE=2e-5
MAX_SEQ_LENGTH=512

echo "========================================="
echo "SFT 训练配置"
echo "========================================="
echo "训练数据: $TRAIN_DATA_PATH"
echo "数据量: $DATA_SIZE"
echo "模型: $MODEL_NAME"
echo "输出目录: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Learning rate: $LEARNING_RATE"
echo "========================================="

# 检查数据是否存在
if [ ! -f "$TRAIN_DATA_PATH" ]; then
    echo "错误: 训练数据不存在: $TRAIN_DATA_PATH"
    echo "请先运行数据处理脚本准备训练数据"
    exit 1
fi

# 使用全部 4 GPU 进行 SFT 训练
echo "启动 SFT 训练 (使用 4 GPU)..."

accelerate launch \
    --config_file scripts/configs/accelerate_config_4gpu.yaml \
    scripts/training/train_sft.py \
    --train_data_path "$TRAIN_DATA_PATH" \
    --data_size "$DATA_SIZE" \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --packing \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --seed 42

echo "========================================="
echo "SFT 训练完成！"
echo "模型保存在: $OUTPUT_DIR"
echo "========================================="