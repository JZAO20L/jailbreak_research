#!/bin/bash
# =============================================================================
# Exp01: 单轮 Agent Baseline (max_turns=1)
# =============================================================================
#
# 使用 ms-swift 的多轮 GRPO 训练，但限制为单轮交互
# 作为多轮实验的 baseline
#
# Usage:
#   bash scripts/exp01_single_turn.sh
#
# =============================================================================

# 4-GPU 机器:必须在 source common.sh 前声明(NUM_GPUS 决定 GPU 布局,默认 8)
NUM_GPUS=4

set -e

source "$(dirname "$0")/common.sh"

# =============================================================================
# 配置
# =============================================================================

EXP_NAME="single_turn_agent"
OUTPUT_SUBDIR="$OUTPUT_DIR/$EXP_NAME"

# 训练配置
MAX_TURNS=1
MAX_STEPS=500
LEARNING_RATE=1e-5
NUM_GENERATIONS=8
PER_DEVICE_BATCH=2
GRAD_ACCUM=4

# ms-swift 配置
PLUGIN_PATH="$AGENTIC_DIR/src/plugin.py"

log_section "Exp01: 单轮 Agent Baseline"
log_info "Max turns: $MAX_TURNS"
log_info "Output: $OUTPUT_SUBDIR"

mkdir -p "$OUTPUT_SUBDIR"

# =============================================================================
# 启动 Servers（如果未启动）
# =============================================================================

log_section "检查 Servers"

if ! curl -s "http://127.0.0.1:$GUARD_PORT/health" > /dev/null 2>&1; then
    log_info "Servers 未启动，正在启动..."
    bash "$(dirname "$0")/start_servers.sh"
else
    log_info "Servers 已就绪"
fi

# =============================================================================
# 训练
# =============================================================================

log_section "ms-swift GRPO 训练 (单轮)"

# 4-GPU 布局:guard=0 / target=1 / rollout=2 / train=3(见 common.sh)
TRAIN_GPUS="${TRAIN_GPUS:-3}"

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS swift rlhf \
    --rlhf_type grpo \
    --model "$BASE_MODEL" \
    --dataset "$AGENTIC_DIR/output/grpo_data.jsonl" \
    --external_plugins "$PLUGIN_PATH" \
    --multi_turn_scheduler gym_scheduler \
    --gym_env jailbreak_env \
    --use_gym_env true \
    --max_turns $MAX_TURNS \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port $ROLLOUT_PORT \
    --vllm_server_timeout 600 \
    --per_device_train_batch_size $PER_DEVICE_BATCH \
    --generation_batch_size $((PER_DEVICE_BATCH * NUM_GENERATIONS)) \
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
# 完成
# =============================================================================

log_section "Exp01 完成"
log_info "Checkpoint: $OUTPUT_SUBDIR"

cat > "$OUTPUT_SUBDIR/summary.json" << EOF
{
    "experiment": "single_turn_agent",
    "max_turns": $MAX_TURNS,
    "max_steps": $MAX_STEPS,
    "learning_rate": $LEARNING_RATE,
    "output_dir": "$OUTPUT_SUBDIR"
}
EOF

log_info "下一步: bash scripts/exp02_multi_turn_3.sh"
