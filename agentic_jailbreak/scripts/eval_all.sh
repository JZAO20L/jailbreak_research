#!/bin/bash
# =============================================================================
# 全量评估
# =============================================================================
#
# 在所有 benchmark 上评估所有实验的 checkpoint
#
# Usage:
#   bash scripts/eval_all.sh
#
# =============================================================================

set -e

source "$(dirname "$0")/common.sh"

# =============================================================================
# 参数
# =============================================================================

EVAL_EXPERIMENTS="${EVAL_EXPERIMENTS:-single_turn_agent multi_turn_3_agent multi_turn_5_agent}"

# Benchmarks
BENCHMARKS=("default" "advbench" "harmbench_standard" "harmbench_contextual" "jailbreakBench")

# 评估结果目录
EVAL_BASE="$OUTPUT_DIR/eval_results"

log_section "全量评估"
log_info "Experiments: $EVAL_EXPERIMENTS"
log_info "Benchmarks: ${BENCHMARKS[*]}"
log_info "Output: $EVAL_BASE"

# =============================================================================
# 检查 Servers
# =============================================================================

if ! curl -s "http://127.0.0.1:$GUARD_PORT/health" > /dev/null 2>&1; then
    log_info "Servers 未启动，正在启动..."
    bash "$(dirname "$0")/start_servers.sh"
fi

# =============================================================================
# 评估每个实验
# =============================================================================

for exp_name in $EVAL_EXPERIMENTS; do
    CHECKPOINT_DIR="$OUTPUT_DIR/$exp_name"
    
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        log_error "Checkpoint 不存在: $CHECKPOINT_DIR，跳过"
        continue
    fi
    
    log_section "评估: $exp_name"
    
    # 评估所有 benchmarks
    for benchmark in "${BENCHMARKS[@]}"; do
        EVAL_DIR="$EVAL_BASE/$exp_name/$benchmark"
        mkdir -p "$EVAL_DIR"
        
        # 确定数据路径
        case $benchmark in
            default)
                DATA_PATH="$PROJECT_ROOT/self_evolve_skills_jailbreak/data/test_prompts.json"
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
        
        python "$AGENTIC_DIR/src/eval.py" \
            --test_data "$DATA_PATH" \
            --skills_path "$SKILLS_PATH" \
            --policy_port $ROLLOUT_PORT \
            --target_port $TARGET_PORT \
            --guard_port $GUARD_PORT \
            --max_turns 5 \
            --output_dir "$EVAL_DIR" \
            2>&1 | tee "$EVAL_DIR/eval.log"
    done
done

# =============================================================================
# 汇总结果
# =============================================================================

log_section "评估结果汇总"

echo ""
echo "============================================================================="
echo "评估结果"
echo "============================================================================="
echo ""
printf "%-25s" "Experiment"
for b in "${BENCHMARKS[@]}"; do
    printf "%-15s" "$b"
done
echo ""
echo "-----------------------------------------------------------------------------"

for exp_name in $EVAL_EXPERIMENTS; do
    printf "%-25s" "$exp_name"
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
