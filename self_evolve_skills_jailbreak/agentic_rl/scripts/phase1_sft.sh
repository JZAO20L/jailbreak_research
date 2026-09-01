#!/bin/bash
# Phase 1: RFT (Rejection Sampling Fine-Tuning) Training
#
# Usage:
#   bash scripts/phase1_sft.sh
#
# Prerequisites:
#   1. Run data_prep.py to generate rft_train.jsonl and rft_val.jsonl
#   2. GPU 0-1 available for training

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_ROOT/self_evolve_skills_jailbreak/agentic_rl"

TRAIN_DATA="output/rft_train.jsonl"
VAL_DATA="output/rft_val.jsonl"
OUTPUT_DIR="output/rft_checkpoint"
BASE_MODEL="/home/tiger/models/Qwen/Qwen3-4B"

echo "=== Phase 1: RFT Training ==="
echo "Train data: $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"
echo "Base model: $BASE_MODEL"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file configs/accelerate_2gpu.yaml \
    src/sft_train.py \
    --train_data_path "$TRAIN_DATA" \
    --eval_data_path "$VAL_DATA" \
    --model_name_or_path "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
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

echo "=== Phase 1 Complete ==="
echo "LoRA saved to: $OUTPUT_DIR/final_lora"
