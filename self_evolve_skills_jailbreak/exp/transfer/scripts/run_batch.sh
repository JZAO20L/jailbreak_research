#!/bin/bash
# =============================================================================
# Batch Transfer Experiment: 批量测试多个模型
# =============================================================================
#
# 对多个 target 模型，依次测试所有方法 × 所有数据集
#
# Usage:
#   bash run_batch.sh --models Qwen3-0.6B Qwen3-14B-FP8
#   bash run_batch.sh --all_models
#   bash run_batch.sh --models gemma-4-12B-it --mode baselines
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER_SCRIPT="$SCRIPT_DIR/run_transfer.sh"

# 所有可用模型
ALL_MODELS=("Qwen3-4B" "Qwen3-0.6B" "Qwen3-14B-FP8" "gemma-4-12B-it" "gpt-oss-20b")

# 解析参数
MODELS_TO_RUN=()
RUN_MODE="all"
METHOD=""
DATASET=""
SKIP_LAUNCH=false
KEEP_SERVERS=false  # 保持 Guard/Attacker 服务运行

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS_TO_RUN+=("$1")
                shift
            done
            ;;
        --all_models)
            MODELS_TO_RUN=("${ALL_MODELS[@]}")
            shift
            ;;
        --mode)
            RUN_MODE="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --skip_launch)
            SKIP_LAUNCH=true
            shift
            ;;
        --keep_servers)
            KEEP_SERVERS=true
            shift
            ;;
        --help|-h)
            echo "Usage: bash run_batch.sh [options]"
            echo ""
            echo "Options:"
            echo "  --models <name1> <name2> ...  Models to test"
            echo "  --all_models                  Test all available models"
            echo "  --mode <mode>                 Experiment mode: all, baselines, ours"
            echo "  --method <method>             Single method to run"
            echo "  --dataset <dataset>           Single dataset to run"
            echo "  --skip_launch                 Use existing servers"
            echo "  --keep_servers                Keep Guard/Attacker running between models"
            echo ""
            echo "Available models: ${ALL_MODELS[@]}"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 默认测试所有模型
if [ ${#MODELS_TO_RUN[@]} -eq 0 ]; then
    MODELS_TO_RUN=("${ALL_MODELS[@]}")
fi

# =============================================================================
# Print Plan
# =============================================================================

echo ""
echo "============================================================================"
echo "Batch Transfer Experiment"
echo "============================================================================"
echo ""
echo "Models to test: ${MODELS_TO_RUN[@]}"
echo "Total models: ${#MODELS_TO_RUN[@]}"
echo "Mode: $RUN_MODE"
echo "Method: ${METHOD:-all}"
echo "Dataset: ${DATASET:-all}"
echo "Skip launch: $SKIP_LAUNCH"
echo "Keep servers: $KEEP_SERVERS"
echo ""
echo "Estimated time per model: ~10 hours (30 experiments × 20 min)"
echo "Total estimated time: ~${#MODELS_TO_RUN[@]} × 10 hours"
echo "============================================================================"

# =============================================================================
# Run Experiments for Each Model
# =============================================================================

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_MODELS=()

for model in "${MODELS_TO_RUN[@]}"; do
    echo ""
    echo "************************************************************************"
    echo "Testing Model: $model"
    echo "************************************************************************"
    echo ""

    # 构建命令
    CMD="bash $TRANSFER_SCRIPT --target_model $model --mode $RUN_MODE"

    if [ -n "$METHOD" ]; then
        CMD="$CMD --method $METHOD"
    fi

    if [ -n "$DATASET" ]; then
        CMD="$CMD --dataset $DATASET"
    fi

    if [ "$SKIP_LAUNCH" = true ]; then
        CMD="$CMD --skip_launch"
    fi

    echo "Running: $CMD"
    echo ""

    # 执行
    if eval $CMD; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo ""
        echo "✓ Model $model completed successfully"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_MODELS+=("$model")
        echo ""
        echo "✗ Model $model FAILED"
    fi

    echo ""
    echo "Progress: $SUCCESS_COUNT success, $FAIL_COUNT failed"
    echo ""
done

# =============================================================================
# Final Summary
# =============================================================================

echo ""
echo "============================================================================"
echo "Batch Experiment Complete"
echo "============================================================================"
echo ""
echo "Total models tested: ${#MODELS_TO_RUN[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Failed models:"
    for m in "${FAILED_MODELS[@]}"; do
        echo "  - $m"
    done
fi

echo ""
echo "Results saved in: /home/tiger/jailbreak_research/self_evolve_skills_jailbreak/exp/transfer/results/"
echo ""
echo "View full summary:"
echo "  python $SCRIPT_DIR/summarize.py --full_matrix"
echo ""
echo "============================================================================"