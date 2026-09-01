#!/bin/bash
# =============================================================================
# Exp09: 全量评估
# =============================================================================
#
# 在所有 benchmark 上评估所有实验的 checkpoint
#
# Benchmarks: default, advbench, harmbench_standard, harmbench_contextual, jailbreakBench
# Models: Qwen3-4B (同族), gpt-oss-20b (跨族, 可选)
#
# GPU 分配 (3-GPU):
#   GPU 0: Guard server
#   GPU 1: Target server
#   GPU 2: Policy server (with LoRA)
#
# Usage:
#   bash exp09_eval_all.sh
#   bash exp09_eval_all.sh --target_model gpt-oss-20b  # 跨族评估
#
# =============================================================================

set -e

source "$(dirname "$0")/common.sh"

# =============================================================================
# 参数
# =============================================================================

TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-Qwen3-4B}"
EVAL_EXPERIMENTS="${EVAL_EXPERIMENTS:-rft grpo_asr grpo_fixed grpo_ema grpo_ahr}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --target_model) TARGET_MODEL_NAME="$2"; shift 2;;
        --experiments) EVAL_EXPERIMENTS="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# 模型路径映射
declare -A MODEL_PATHS
MODEL_PATHS["Qwen3-4B"]="/home/tiger/models/Qwen/Qwen3-4B"
MODEL_PATHS["Qwen3-0.6B"]="/home/tiger/models/Qwen/Qwen3-0.6B"
MODEL_PATHS["Qwen3-14B-FP8"]="/home/tiger/models/Qwen/Qwen3-14B-FP8"
MODEL_PATHS["gpt-oss-20b"]="/home/tiger/models/openai-mirror/gpt-oss-20b"

TARGET_MODEL_PATH="${MODEL_PATHS[$TARGET_MODEL_NAME]}"
if [ -z "$TARGET_MODEL_PATH" ]; then
    log_error "Unknown target model: $TARGET_MODEL_NAME"
    exit 1
fi

# Benchmarks
BENCHMARKS=("default" "advbench" "harmbench_standard" "harmbench_contextual" "jailbreakBench")

# 评估结果目录
EVAL_BASE="$OUTPUT_DIR/eval_results/$TARGET_MODEL_NAME"

log_section "Exp09: 全量评估"
log_info "Target model: $TARGET_MODEL_NAME ($TARGET_MODEL_PATH)"
log_info "Experiments: $EVAL_EXPERIMENTS"
log_info "Benchmarks: ${BENCHMARKS[*]}"
log_info "Output: $EVAL_BASE"

# =============================================================================
# 启动 Servers
# =============================================================================

log_section "启动 vLLM Servers"
cleanup_gpus

# Guard (GPU 0)
CUDA_VISIBLE_DEVICES=0 vllm serve "$GUARD_MODEL" \
    --port $GUARD_PORT --max-model-len 4096 --gpu-memory-utilization 0.9 \
    --trust-remote-code --dtype auto > "$EVAL_BASE/guard.log" 2>&1 &
GUARD_PID=$!

# Target (GPU 1)
CUDA_VISIBLE_DEVICES=1 vllm serve "$TARGET_MODEL_PATH" \
    --port $TARGET_PORT --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --trust-remote-code --dtype auto > "$EVAL_BASE/target.log" 2>&1 &
TARGET_PID=$!

wait_for_server $GUARD_PORT "Guard"
wait_for_server $TARGET_PORT "Target"

# =============================================================================
# 评估每个实验
# =============================================================================

for exp_name in $EVAL_EXPERIMENTS; do
    LORA_PATH="$OUTPUT_DIR/$exp_name/final_lora"
    
    if [ ! -d "$LORA_PATH" ]; then
        log_error "Checkpoint 不存在: $LORA_PATH，跳过"
        continue
    fi
    
    log_section "评估: $exp_name"
    
    # 启动 Policy server (GPU 2)
    CUDA_VISIBLE_DEVICES=2 vllm serve "$BASE_MODEL" \
        --port $POLICY_PORT --max-model-len 8192 --gpu-memory-utilization 0.9 \
        --trust-remote-code --dtype auto \
        --enable-lora \
        > "$EVAL_BASE/${exp_name}_policy.log" 2>&1 &
    POLICY_PID=$!
    wait_for_server $POLICY_PORT "Policy"
    
    # 评估所有 benchmarks
    for benchmark in "${BENCHMARKS[@]}"; do
        EVAL_DIR="$EVAL_BASE/$exp_name/$benchmark"
        mkdir -p "$EVAL_DIR"
        
        # 确定数据路径
        case $benchmark in
            default)
                DATA_PATH="$SESS_DIR/data/test_prompts.json"
                ;;
            *)
                DATA_PATH="$PROJECT_ROOT/data/benchmark/${benchmark}.jsonl"
                ;;
        esac
        
        if [ ! -f "$DATA_PATH" ]; then
            log_error "数据不存在: $DATA_PATH，跳过"
            continue
        fi
        
        log_info "  $benchmark: $DATA_PATH"
        
        python "$AGENTIC_RL_DIR/src/eval.py" \
            --model_path "$BASE_MODEL" \
            --lora_path "$LORA_PATH" \
            --skill_library_path "$SKILL_LIBRARY" \
            --description_path "$OUTPUT_DIR/descriptions/skill_descriptions.json" \
            --test_data "$DATA_PATH" \
            --policy_port $POLICY_PORT \
            --target_port $TARGET_PORT \
            --guard_port $GUARD_PORT \
            --output_dir "$EVAL_DIR" \
            2>&1 | tee "$EVAL_DIR/eval.log"
    done
    
    # 停止 Policy server
    kill $POLICY_PID 2>/dev/null || true
    sleep 2
done

# =============================================================================
# 停止 Servers
# =============================================================================

log_section "停止 Servers"
kill $GUARD_PID $TARGET_PID 2>/dev/null || true
sleep 3

# =============================================================================
# 汇总结果
# =============================================================================

log_section "评估结果汇总"

echo ""
echo "============================================================================="
echo "评估结果: $TARGET_MODEL_NAME"
echo "============================================================================="
echo ""
printf "%-20s" "Experiment"
for b in "${BENCHMARKS[@]}"; do
    printf "%-15s" "$b"
done
echo ""
echo "-----------------------------------------------------------------------------"

for exp_name in $EVAL_EXPERIMENTS; do
    printf "%-20s" "$exp_name"
    for benchmark in "${BENCHMARKS[@]}"; do
        SUMMARY_FILE="$EVAL_BASE/$exp_name/$benchmark/summary.json"
        if [ -f "$SUMMARY_FILE" ]; then
            ASR=$(python -c "import json; print(f'{json.load(open(\"$SUMMARY_FILE\"))[\"asr\"]:.1%}')" 2>/dev/null || echo "N/A")
            printf "%-15s" "$ASR"
        else
            printf "%-15s" "-"
        fi
    done
    echo ""
done

echo ""
echo "详细结果: $EVAL_BASE/"
log_info "评估完成"
