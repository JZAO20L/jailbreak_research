#!/bin/bash
# =============================================================================
# Exp05: GRPO-AHR (Ours: variance ratio + sigmoid + EMA)
# =============================================================================
#
# 主实验 E4: 自适应混合奖励（本文核心贡献）
#
# Usage:
#   bash exp05_grpo_ahr.sh
#
# =============================================================================

set -e

source "$(dirname "$0")/common.sh"

EXP_NAME="grpo_ahr"
OUTPUT_SUBDIR="$OUTPUT_DIR/$EXP_NAME"
RFT_MODEL="$OUTPUT_DIR/rft/final_lora"

log_section "Exp05: GRPO-AHR (Ours)"
log_info "Reward: Variance ratio + sigmoid + EMA (adaptive hybrid)"

if [ ! -d "$RFT_MODEL" ]; then
    log_error "RFT checkpoint 不存在: $RFT_MODEL"
    exit 1
fi

mkdir -p "$OUTPUT_SUBDIR"

# 启动 Servers
log_section "启动 vLLM Servers"
cleanup_gpus

CUDA_VISIBLE_DEVICES=$GUARD_GPU vllm serve "$GUARD_MODEL" \
    --port $GUARD_PORT --max-model-len 4096 --gpu-memory-utilization 0.9 \
    --trust-remote-code --dtype auto > "$OUTPUT_SUBDIR/guard.log" 2>&1 &
GUARD_PID=$!

CUDA_VISIBLE_DEVICES=$TARGET_GPU vllm serve "$TARGET_MODEL" \
    --port $TARGET_PORT --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --trust-remote-code --dtype auto > "$OUTPUT_SUBDIR/target.log" 2>&1 &
TARGET_PID=$!

wait_for_server $GUARD_PORT "Guard"
wait_for_server $TARGET_PORT "Target"

# 训练
log_section "GRPO 训练: reward_type=ahr (OURS)"
NUM_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
    --num_processes $NUM_TRAIN_GPUS \
    "$AGENTIC_RL_DIR/src/grpo_train.py" \
    --rft_model_path "$RFT_MODEL" \
    --base_model_path "$BASE_MODEL" \
    --skill_library_path "$SKILL_LIBRARY" \
    --description_path "$OUTPUT_DIR/descriptions/skill_descriptions.json" \
    --output_dir "$OUTPUT_SUBDIR" \
    --target_port $TARGET_PORT --guard_port $GUARD_PORT \
    --reward_type ahr \
    --max_steps 500 --learning_rate 1e-5 \
    --num_generations 8 --max_completion_len 1024 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --vllm_tensor_parallel_size $TRAIN_TP \
    --ema_beta 0.9 --alpha 2.0 --delta -2.0 \
    --lambda_min 0.2 --lambda_max 0.8 \
    --logging_steps 1 --save_steps 100

# 停止
kill $GUARD_PID $TARGET_PID 2>/dev/null || true
sleep 3

log_section "Exp05 完成 (核心实验)"
cat > "$OUTPUT_SUBDIR/summary.json" << EOF
{"experiment": "grpo_ahr", "reward_type": "ahr", "ema_beta": 0.9, "alpha": 2.0, "delta": -2.0, "lambda_range": [0.2, 0.8], "rft_model": "$RFT_MODEL", "output_dir": "$OUTPUT_SUBDIR", "lora_path": "$OUTPUT_SUBDIR/final_lora"}
EOF

log_info "下一步: bash exp06_no_skill.sh (消融) 或 bash exp09_eval_all.sh (评估)"
