#!/bin/bash
# =============================================================================
# RFT 冷启动: 在 beam_pilot 成功轨迹上对 policy 做 SFT (LoRA)
# 数据: agentic_jailbreak/output/rft_data.jsonl (2175 条前缀式样本)
# 输出: agentic_jailbreak/output/rft_sft
# =============================================================================
set -e
source "$(dirname "$0")/common.sh"

EXP_DIR="$AGENTIC_DIR/output/rft_sft_train1000_v2"
mkdir -p "$EXP_DIR"

log_section "RFT SFT 冷启动 (GPU1)"
CUDA_VISIBLE_DEVICES=3 swift sft \
    --model "$BASE_MODEL" \
    --dataset "$AGENTIC_DIR/output/rft_data_train1000_v2.jsonl" \
    --tuner_type lora \
    --max_length 4096 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --fp16 true \
    --bf16 false \
    --output_dir "$EXP_DIR" \
    --report_to none

log_section "RFT SFT 完成"
ls "$EXP_DIR" | head