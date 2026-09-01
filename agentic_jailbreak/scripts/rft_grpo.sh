#!/bin/bash
# =============================================================================
# RFT + GRPO: 冷启动模型上的对话式多轮 GRPO 复训 (对齐 exp03: 5轮/500步/ASR-only)
#
# 前置:
#   1) rft_sft.sh 完成; 2) LoRA 已 merge 到 rft_sft_merged
#   3) rollout 服务必须用 `swift rollout` 启动(带 swift communicator 端点,
#      裸 `vllm serve` 会在 init_communicator 报 404 Not Found):
#      CUDA_VISIBLE_DEVICES=<rollout_gpu> swift rollout --model <merged> \
#          --vllm_tensor_parallel_size 1 --port 8004 --vllm_max_model_len 8192 \
#          --vllm_gpu_memory_utilization 0.8
#   4) MASTER_PORT 不能落在 29500-29522 段(本机被系统预占, EADDRINUSE)
# Usage:
#   bash scripts/rft_grpo.sh [lr]
# =============================================================================
NUM_GPUS=4
set -e
source "$(dirname "$0")/common.sh"

EXP_NAME="multi_turn_5_agent_rft"
OUTPUT_SUBDIR="$OUTPUT_DIR/$EXP_NAME"
MERGED_DIR="$AGENTIC_DIR/output/rft_sft_train1000_v2_merged"
LR="${1:-1e-5}"
MAX_TURNS=5
MAX_STEPS=500
NUM_GENERATIONS=8
PER_DEVICE_BATCH=2
GRAD_ACCUM=4
PLUGIN_PATH="$AGENTIC_DIR/src/plugin.py"

log_section "RFT+GRPO: 冷启动模型 ($MERGED_DIR) 多轮训练 max_turns=$MAX_TURNS"

if [ ! -f "$MERGED_DIR/config.json" ]; then
    log_error "未找到 merged 模型: $MERGED_DIR (先运行 merge_rft_lora.sh)"
    exit 1
fi

mkdir -p "$OUTPUT_SUBDIR"

# 检查 guard/target 服务
for p in $GUARD_PORT $TARGET_PORT; do
    if ! curl -s "http://127.0.0.1:$p/health" > /dev/null 2>&1; then
        log_error "服务 $p 未就绪"
        exit 1
    fi
done

# 训练: GPU1 (rollout 由外部 `swift rollout` 服务提供, 8004)
TRAIN_GPUS="${TRAIN_GPUS:-3}"
export ROLLOUT_PORT=8004
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-43210}"  # 29500 段被系统预占, 避开

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS swift rlhf \
    --rlhf_type grpo \
    --model "$MERGED_DIR" \
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
    --learning_rate $LR \
    --num_generations $NUM_GENERATIONS \
    --max_completion_length 2048 \
    --fp16 true \
    --bf16 false \
    --beta 0.05 \
    --output_dir "$OUTPUT_SUBDIR" \
    --report_to swanlab \
    --run_name "$EXP_NAME"

log_section "RFT+GRPO 完成"
cat > "$OUTPUT_SUBDIR/summary.json" << EOF
{"experiment": "multi_turn_5_agent_rft", "max_turns": $MAX_TURNS, "max_steps": $MAX_STEPS, "lr": $LR, "base": "rft_sft_merged"}
EOF

log_info "下一步: bash scripts/eval_conv.sh no_skill_full 1000 5 1 (单次) 或 1000 5 2 (beam-2)"