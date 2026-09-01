#!/bin/bash
# =============================================================================
# Exp07: 消融 - SFT 冷启动（非 RFT）
# =============================================================================
#
# 消融 A2: 使用传统 SFT（oracle skill content 作为 ground truth）
#           而非 RFT（ASR 过滤的成功样本）
#
# 这里复用 SESS Layer4 的成功轨迹来构建 SFT 数据
#
# Usage:
#   bash exp07_sft_coldstart.sh
#
# =============================================================================

set -e

source "$(dirname "$0")/common.sh"

EXP_NAME="ablation_sft_coldstart"
OUTPUT_SUBDIR="$OUTPUT_DIR/$EXP_NAME"
SFT_DATA_DIR="$OUTPUT_DIR/sft_data"

log_section "Exp07: 消融 - SFT 冷启动"
log_info "使用 SESS 轨迹构建 SFT 数据（oracle 注入）"

mkdir -p "$OUTPUT_SUBDIR" "$SFT_DATA_DIR"

# =============================================================================
# Step 1: 构建 SFT 数据（从 SESS 轨迹）
# =============================================================================

# 检查 descriptions 是否存在
if [ ! -f "$OUTPUT_DIR/descriptions/skill_descriptions.json" ]; then
    log_error "Descriptions 不存在，请先运行: bash exp00_data_prep.sh"
    exit 1
fi

# 使用 data_prep.py 的旧模式构建 SFT 数据
# 这里需要直接从 SESS 轨迹构建
python "$AGENTIC_RL_DIR/src/data_prep.py" \
    --step build \
    --trajectory_path "$SESS_DIR/exp/layer4/results/result_dan_data_medium_evo.json" \
    --output_dir "$SFT_DATA_DIR"

# =============================================================================
# Step 2: SFT 训练
# =============================================================================

log_section "SFT 训练（冷启动）"

NUM_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
    --num_processes $NUM_TRAIN_GPUS \
    "$AGENTIC_RL_DIR/src/sft_train.py" \
    --train_data_path "$SFT_DATA_DIR/rft_train.jsonl" \
    --eval_data_path "$SFT_DATA_DIR/rft_val.jsonl" \
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
    --bf16

log_section "Exp07 完成"
cat > "$OUTPUT_SUBDIR/summary.json" << EOF
{"experiment": "ablation_sft_coldstart", "ablation": "sft_instead_of_rft", "output_dir": "$OUTPUT_SUBDIR", "lora_path": "$OUTPUT_SUBDIR/final_lora"}
EOF

log_info "下一步: 用此 checkpoint 运行 GRPO（修改 exp05 的 --rft_model_path 指向此目录）"
