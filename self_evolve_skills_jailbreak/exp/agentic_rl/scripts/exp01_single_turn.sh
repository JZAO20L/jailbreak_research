#!/bin/bash
# =============================================================================
# Exp01: 单轮 RL Baseline (ms-swift 多轮 GRPO, max_turns=1)
# =============================================================================
#
# 使用 ms-swift 的多轮 GRPO 训练，但限制为单轮交互
# 作为多轮实验的 baseline
#
# GPU 分配 (8-GPU):
#   GPU 0: Guard server
#   GPU 1: Target server
#   GPU 2-3: vLLM rollout server
#   GPU 4-7: Training (TP=4)
#
# Usage:
#   bash exp01_single_turn.sh
#
# =============================================================================

set -e

source "$(dirname "$0")/common.sh"

# =============================================================================
# 配置
# =============================================================================

EXP_NAME="single_turn_rl"
OUTPUT_SUBDIR="$OUTPUT_DIR/$EXP_NAME"

# 训练配置
MAX_TURNS=1
MAX_STEPS=500
LEARNING_RATE=1e-5
NUM_GENERATIONS=8
PER_DEVICE_BATCH=2
GRAD_ACCUM=4

# ms-swift 配置
PLUGIN_PATH="$AGENTIC_RL_DIR/src/plugin.py"
SKILL_LIBRARY="$SESS_DIR/exp/layer4/results/skills/skills_dan_data_medium_evo.json"
DESCRIPTIONS="$OUTPUT_DIR/descriptions/skill_descriptions.json"

log_section "Exp01: 单轮 RL Baseline"
log_info "Max turns: $MAX_TURNS"
log_info "Output: $OUTPUT_SUBDIR"

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

# vLLM rollout server (GPU 2-3)
ROLLOUT_GPUS="2,3"
CUDA_VISIBLE_DEVICES=$ROLLOUT_GPUS swift rollout \
    --model "$BASE_MODEL" \
    --vllm_tensor_parallel_size 2 \
    --port $ROLLOUT_PORT \
    --vllm_max_model_len 8192 \
    --vllm_gpu_memory_utilization 0.8 \
    > "$OUTPUT_SUBDIR/rollout.log" 2>&1 &
ROLLOUT_PID=$!

wait_for_server $GUARD_PORT "Guard"
wait_for_server $TARGET_PORT "Target"
wait_for_server $ROLLOUT_PORT "Rollout"

# =============================================================================
# 训练
# =============================================================================

log_section "ms-swift GRPO 训练 (单轮)"

TRAIN_GPUS="4,5,6,7"

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS swift rlhf \
    --rlhf_type grpo \
    --model "$BASE_MODEL" \
    --external_plugins "$PLUGIN_PATH" \
    --multi_turn_scheduler gym_scheduler \
    --env jailbreak_env \
    --env_config "{
        \"skill_library_path\": \"$SKILL_LIBRARY\",
        \"description_path\": \"$DESCRIPTIONS\",
        \"target_port\": $TARGET_PORT,
        \"guard_port\": $GUARD_PORT,
        \"max_turns\": $MAX_TURNS,
        \"top_k_skills\": 5
    }" \
    --reward_funcs asr_reward format_reward \
    --reward_weights 1.0 0.1 \
    --max_turns $MAX_TURNS \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port $ROLLOUT_PORT \
    --vllm_server_timeout 600 \
    --per_device_train_batch_size $PER_DEVICE_BATCH \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_generations $NUM_GENERATIONS \
    --max_completion_length 2048 \
    --bf16 true \
    --beta 0.05 \
    --output_dir "$OUTPUT_SUBDIR" \
    --report_to swanlab \
    --run_name "$EXP_NAME"

# =============================================================================
# 停止 Servers
# =============================================================================

log_section "停止 Servers"
kill $GUARD_PID $TARGET_PID $ROLLOUT_PID 2>/dev/null || true
sleep 3

# =============================================================================
# 完成
# =============================================================================

log_section "Exp01 完成"
log_info "Checkpoint: $OUTPUT_SUBDIR"

cat > "$OUTPUT_SUBDIR/summary.json" << EOF
{
    "experiment": "single_turn_rl",
    "max_turns": $MAX_TURNS,
    "max_steps": $MAX_STEPS,
    "learning_rate": $LEARNING_RATE,
    "output_dir": "$OUTPUT_SUBDIR"
}
EOF

log_info "下一步: bash exp02_multi_turn_3.sh"
