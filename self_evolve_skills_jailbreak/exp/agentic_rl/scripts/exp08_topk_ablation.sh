#!/bin/bash
# =============================================================================
# Exp08: 消融 - Top-k 候选集大小
# =============================================================================
#
# 消融 A3: 不同 top-k 值（1, 3, 5）对性能的影响
#
# Usage:
#   bash exp08_topk_ablation.sh          # 运行所有 k 值
#   bash exp08_topk_ablation.sh --k 3    # 只运行 k=3
#
# =============================================================================

set -e

source "$(dirname "$0")/common.sh"

K_VALUES="${K_VALUES:-1 3 5}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --k) K_VALUES="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

log_section "Exp08: 消融 - Top-k 候选集大小"
log_info "K values: $K_VALUES"

for k in $K_VALUES; do
    EXP_NAME="ablation_topk_${k}"
    OUTPUT_SUBDIR="$OUTPUT_DIR/$EXP_NAME"
    RFT_MODEL="$OUTPUT_DIR/rft/final_lora"

    log_section "Top-k = $k"

    if [ ! -d "$RFT_MODEL" ]; then
        log_error "RFT checkpoint 不存在: $RFT_MODEL"
        exit 1
    fi

    mkdir -p "$OUTPUT_SUBDIR"

    # 启动 Servers
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
        --top_k_candidates $k \
        --max_steps 500 --learning_rate 1e-5 \
        --num_generations 8 --max_completion_len 1024 \
        --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
        --vllm_tensor_parallel_size $TRAIN_TP \
        --logging_steps 1 --save_steps 100

    # 停止
    kill $GUARD_PID $TARGET_PID 2>/dev/null || true
    sleep 3

    cat > "$OUTPUT_SUBDIR/summary.json" << EOF
{"experiment": "ablation_topk", "top_k": $k, "output_dir": "$OUTPUT_SUBDIR"}
EOF

    log_info "Top-k=$k 完成: $OUTPUT_SUBDIR"
done

log_section "Exp08 全部完成"
log_info "结果: $OUTPUT_DIR/ablation_topk_*/"
