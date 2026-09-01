#!/bin/bash
# =============================================================================
# Exp02: GRPO-ASR (单一 ASR reward)
# =============================================================================
#
# 主实验 E1: 仅使用 ASR reward 的 GRPO
#
# GPU 分配 (4-GPU):
#   GPU 0: Guard server (port 8001)
#   GPU 1: Target server (port 8002)
#   GPU 2-3: Training (vLLM colocate TP=2)
#
# Usage:
#   bash exp02_grpo_asr.sh
#
# =============================================================================

set -e

source "$(dirname "$0")/common.sh"

# =============================================================================
# 配置
# =============================================================================

EXP_NAME="grpo_asr"
OUTPUT_SUBDIR="$OUTPUT_DIR/$EXP_NAME"
RFT_MODEL="$OUTPUT_DIR/rft/final_lora"

log_section "Exp02: GRPO-ASR"
log_info "RFT model: $RFT_MODEL"
log_info "Output: $OUTPUT_SUBDIR"
log_info "Reward: ASR only"

# 检查 RFT checkpoint
if [ ! -d "$RFT_MODEL" ]; then
    log_error "RFT checkpoint 不存在: $RFT_MODEL"
    log_error "请先运行: bash exp01_rft_train.sh"
    exit 1
fi

mkdir -p "$OUTPUT_SUBDIR"

# =============================================================================
# 启动 Servers
# =============================================================================

log_section "启动 vLLM Servers"

cleanup_gpus

# Guard server (GPU 0)
CUDA_VISIBLE_DEVICES=$GUARD_GPU vllm serve "$GUARD_MODEL" \
    --port $GUARD_PORT \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --dtype auto \
    > "$OUTPUT_SUBDIR/guard.log" 2>&1 &
GUARD_PID=$!

# Target server (GPU 1)
CUDA_VISIBLE_DEVICES=$TARGET_GPU vllm serve "$TARGET_MODEL" \
    --port $TARGET_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --dtype auto \
    > "$OUTPUT_SUBDIR/target.log" 2>&1 &
TARGET_PID=$!

wait_for_server $GUARD_PORT "Guard"
wait_for_server $TARGET_PORT "Target"

# =============================================================================
# GRPO 训练
# =============================================================================

log_section "GRPO 训练: reward_type=asr"

NUM_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
    --num_processes $NUM_TRAIN_GPUS \
    "$AGENTIC_RL_DIR/src/grpo_train.py" \
    --rft_model_path "$RFT_MODEL" \
    --base_model_path "$BASE_MODEL" \
    --skill_library_path "$SKILL_LIBRARY" \
    --description_path "$OUTPUT_DIR/descriptions/skill_descriptions.json" \
    --output_dir "$OUTPUT_SUBDIR" \
    --target_port $TARGET_PORT \
    --guard_port $GUARD_PORT \
    --reward_type asr \
    --max_steps 500 \
    --learning_rate 1e-5 \
    --num_generations 8 \
    --max_completion_len 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --vllm_tensor_parallel_size $TRAIN_TP \
    --logging_steps 1 \
    --save_steps 100

# =============================================================================
# 停止 Servers
# =============================================================================

log_section "停止 Servers"
kill $GUARD_PID $TARGET_PID 2>/dev/null || true
sleep 3

# =============================================================================
# 完成
# =============================================================================

log_section "Exp02 完成"
log_info "LoRA 保存到: $OUTPUT_SUBDIR/final_lora/"

cat > "$OUTPUT_SUBDIR/summary.json" << EOF
{
    "experiment": "grpo_asr",
    "reward_type": "asr",
    "rft_model": "$RFT_MODEL",
    "output_dir": "$OUTPUT_SUBDIR",
    "lora_path": "$OUTPUT_SUBDIR/final_lora",
    "max_steps": 500,
    "learning_rate": 1e-5,
    "num_generations": 8
}
EOF

log_info "下一步: bash exp03_grpo_fixed.sh"
