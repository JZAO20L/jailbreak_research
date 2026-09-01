#!/bin/bash
# Phase 2: AHR-GRPO Training
#
# Usage:
#   bash scripts/phase2_grpo.sh
#
# Prerequisites:
#   1. Phase 1 RFT complete (output/rft_checkpoint/final_lora)
#   2. Target + Guard vLLM servers running on GPU 0
#   3. GPU 1-2 available for training

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_ROOT/self_evolve_skills_jailbreak/agentic_rl"

RFT_MODEL="output/rft_checkpoint/final_lora"
BASE_MODEL="/home/tiger/models/Qwen/Qwen3-4B"
TRAIN_DATA="output/rft_train.jsonl"
SKILL_LIBRARY="../exp/layer4/results/skills/skills_dan_data_medium_evo.json"
DESCRIPTIONS="output/skill_descriptions.json"
OUTPUT_DIR="output/grpo_checkpoint"

TARGET_PORT=8002
GUARD_PORT=8001

echo "=== Phase 2: AHR-GRPO Training ==="
echo "RFT model: $RFT_MODEL"
echo "Output: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=1,2 accelerate launch \
    --config_file configs/accelerate_2gpu.yaml \
    src/grpo_train.py \
    --rft_model_path "$RFT_MODEL" \
    --base_model_path "$BASE_MODEL" \
    --train_data "$TRAIN_DATA" \
    --skill_library_path "$SKILL_LIBRARY" \
    --description_path "$DESCRIPTIONS" \
    --output_dir "$OUTPUT_DIR" \
    --target_port $TARGET_PORT \
    --guard_port $GUARD_PORT \
    --max_steps 500 \
    --learning_rate 1e-5 \
    --num_generations 8 \
    --max_completion_len 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --ema_beta 0.9 \
    --logging_steps 1 \
    --save_steps 100

echo "=== Phase 2 Complete ==="
echo "LoRA saved to: $OUTPUT_DIR/final_lora"
