#!/bin/bash
# =============================================================================
# Exp01: RFT 训练
# =============================================================================
#
# 在 RFT 数据上训练 skill-conditioned 攻击模型
#
# GPU 分配 (4-GPU):
#   GPU 0: Guard (可选，用于在线验证)
#   GPU 1: Target (可选，用于在线验证)
#   GPU 2-3: Training (accelerate, TP=2)
#
# Usage:
#   bash exp01_rft_train.sh
#
# =============================================================================

set -e

source "$(dirname "$0")/common.sh"

# =============================================================================
# 配置
# =============================================================================

EXP_NAME="rft"
OUTPUT_SUBDIR="$OUTPUT_DIR/$EXP_NAME"
TRAIN_DATA="$OUTPUT_DIR/rft_data/rft_train.jsonl"
VAL_DATA="$OUTPUT_DIR/rft_data/rft_val.jsonl"

log_section "Exp01: RFT 训练"
log_info "Train data: $TRAIN_DATA"
log_info "Output: $OUTPUT_SUBDIR"

# 检查数据
if [ ! -f "$TRAIN_DATA" ]; then
    log_error "训练数据不存在: $TRAIN_DATA"
    log_error "请先运行: bash exp00_data_prep.sh"
    exit 1
fi

mkdir -p "$OUTPUT_SUBDIR"

# =============================================================================
# 训练
# =============================================================================

log_info "开始 RFT 训练..."
log_info "GPU: $TRAIN_GPUS (TP=$TRAIN_TP)"

NUM_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
    --num_processes $NUM_TRAIN_GPUS \
    "$AGENTIC_RL_DIR/src/sft_train.py" \
    --train_data_path "$TRAIN_DATA" \
    --eval_data_path "$VAL_DATA" \
    --model_name_or_path "$BASE_MODEL" \
    --output_dir "$OUTPUT_SUBDIR" \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --packing \
    --bf16 \
    --logging_steps 10 \
    --save_steps 50

# =============================================================================
# 完成
# =============================================================================

log_section "RFT 训练完成"
log_info "LoRA 保存到: $OUTPUT_SUBDIR/final_lora/"

# 保存摘要
cat > "$OUTPUT_SUBDIR/summary.json" << EOF
{
    "experiment": "rft",
    "base_model": "$BASE_MODEL",
    "train_data": "$TRAIN_DATA",
    "output_dir": "$OUTPUT_SUBDIR",
    "lora_path": "$OUTPUT_SUBDIR/final_lora",
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "lora_r": 16,
    "lora_alpha": 32
}
EOF

log_info "下一步: bash exp02_grpo_asr.sh (或其他 GRPO 实验)"
