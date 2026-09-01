#!/bin/bash
# =============================================================================
# Exp10: 消融 - 数据配比 (RFT:GRPO ratio)
# =============================================================================
#
# 消融 A4: 不同 RFT:GRPO 数据配比对性能的影响
#
# **重要**：控制总数据量不变（6000），只改变 RFT:GRPO 比例
#
# 数据划分（总量 8000，排除前 1000 条 SESS 数据）：
#   [1000:7000] 可用数据 6000 条
#
# 测试配置（总量固定 6000）：
#   - 1:2  (RFT=2000, GRPO=4000)  → [1000:3000] + [3000:7000]
#   - 1:1  (RFT=3000, GRPO=3000)  → [1000:4000] + [4000:7000]
#   - 2:1  (RFT=4000, GRPO=2000)  → [1000:5000] + [5000:7000]
#
# Usage:
#   bash exp10_data_ratio.sh
#
# =============================================================================

set -e

source "$(dirname "$0")/common.sh"

log_section "Exp10: 消融 - 数据配比（控制总量 6000）"
log_info "测试不同 RFT:GRPO 数据配比对性能的影响"

# 固定总量
TOTAL_DATA=6000
DATA_BASE=1000  # 跳过 SESS 使用的 [0:1000]

# 数据配比配置（总量固定 6000）
# 格式: "ratio_name:rft_size:grpo_size"
RATIOS=(
    "1:2:2000:4000"
    "1:1:3000:3000"
    "2:1:4000:2000"
)

for ratio_config in "${RATIOS[@]}"; do
    IFS=':' read -r ratio_name rft_size grpo_size <<< "$ratio_config"
    
    # 计算数据范围
    rft_start=$DATA_BASE
    rft_end=$((rft_start + rft_size))
    grpo_start=$rft_end
    grpo_end=$((grpo_start + grpo_size))
    
    EXP_NAME="ablation_ratio_${ratio_name//:/_}"
    OUTPUT_SUBDIR="$OUTPUT_DIR/$EXP_NAME"
    
    log_section "数据配比: $ratio_name (RFT=$rft_size, GRPO=$grpo_size, 总量=$((rft_size + grpo_size)))"
    log_info "RFT 范围: [$rft_start:$rft_end]"
    log_info "GRPO 范围: [$grpo_start:$grpo_end]"
    
    # 设置环境变量
    export RFT_DATA_START=$rft_start
    export RFT_DATA_END=$rft_end
    export GRPO_DATA_START=$grpo_start
    export GRPO_DATA_END=$grpo_end
    
    mkdir -p "$OUTPUT_SUBDIR"
    
    # Step 1: 生成 RFT 数据
    log_info "Step 1: 生成 RFT 数据 [$rft_start:$rft_end]"
    bash "$(dirname "$0")/exp00_data_prep.sh" \
        --start $rft_start --end $rft_end \
        --max_samples $rft_size \
        --suffix "ratio_${ratio_name//:/_}_rft"
    
    # Step 2: RFT 训练
    log_info "Step 2: RFT 训练"
    RFT_DATA_DIR="$OUTPUT_DIR/ratio_${ratio_name//:/_}_rft_data"
    RFT_OUTPUT="$OUTPUT_SUBDIR/rft"
    
    NUM_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
    
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
        --num_processes $NUM_TRAIN_GPUS \
        "$AGENTIC_RL_DIR/src/sft_train.py" \
        --train_data_path "$RFT_DATA_DIR/rft_train.jsonl" \
        --eval_data_path "$RFT_DATA_DIR/rft_val.jsonl" \
        --model_name_or_path "$BASE_MODEL" \
        --output_dir "$RFT_OUTPUT" \
        --num_epochs 3 \
        --learning_rate 2e-5 \
        --lora_r 16 \
        --lora_alpha 32 \
        --max_seq_length 2048 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --packing \
        --bf16
    
    RFT_MODEL="$RFT_OUTPUT/final_lora"
    
    # Step 3: GRPO 训练
    log_info "Step 3: GRPO 训练 [$grpo_start:$grpo_end]"
    
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
    
    # GRPO 训练
    GRPO_OUTPUT="$OUTPUT_SUBDIR/grpo"
    
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
        --num_processes $NUM_TRAIN_GPUS \
        "$AGENTIC_RL_DIR/src/grpo_train.py" \
        --rft_model_path "$RFT_MODEL" \
        --base_model_path "$BASE_MODEL" \
        --skill_library_path "$SKILL_LIBRARY" \
        --description_path "$OUTPUT_DIR/descriptions/skill_descriptions.json" \
        --output_dir "$GRPO_OUTPUT" \
        --target_port $TARGET_PORT --guard_port $GUARD_PORT \
        --reward_type ahr \
        --data_start $grpo_start \
        --data_end $grpo_end \
        --max_steps 500 --learning_rate 1e-5 \
        --num_generations 8 --max_completion_len 1024 \
        --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
        --vllm_tensor_parallel_size $TRAIN_TP \
        --logging_steps 1 --save_steps 100
    
    # 停止
    kill $GUARD_PID $TARGET_PID 2>/dev/null || true
    sleep 3
    
    # 保存摘要
    cat > "$OUTPUT_SUBDIR/summary.json" << EOF
{
    "experiment": "ablation_data_ratio",
    "ratio": "$ratio_name",
    "total_data": $TOTAL_DATA,
    "rft_range": [$rft_start, $rft_end],
    "rft_samples": $rft_size,
    "grpo_range": [$grpo_start, $grpo_end],
    "grpo_samples": $grpo_size,
    "output_dir": "$OUTPUT_SUBDIR"
}
EOF
    
    log_info "配比 $ratio_name 完成: $OUTPUT_SUBDIR"
done

log_section "Exp10 全部完成"
log_info "结果: $OUTPUT_DIR/ablation_ratio_*/"
log_info ""
log_info "对比维度：RFT:GRPO 比例（1:2 vs 1:1 vs 2:1），总量固定 6000"
