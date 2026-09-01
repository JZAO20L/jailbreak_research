#!/bin/bash
# =============================================================================
# Evaluation Script for AHR-GRPO Paper Experiments
#
# Evaluates trained policy model on test set.
#
# Usage:
#   bash eval.sh <experiment_type> <lora_path> [options]
#
# Examples:
#   bash eval.sh ahr_grpo experiments/paper_ahr_grpo/output/ahr_grpo_steps500_lr1e-05/final_lora
#   bash eval.sh baseline --use_original_prompt
# =============================================================================

set -e

# =============================================================================
# Default Configuration
# =============================================================================
POLICY_MODEL="/home/tiger/models/Qwen/Qwen3-4B"
TARGET_MODEL="/home/tiger/models/Qwen/Qwen3-4B"
GUARD_MODEL="/home/tiger/models/Qwen/Qwen3Guard-Gen-4B"

TEST_DATA="/home/tiger/jailbreak_research/data/dataset/processed/10k/test.jsonl"

# Ports (must match rollout.sh)
GUARD_PORT=8001
TARGET_JUDGE_PORT=8002

# Attack prompt
ATTACK_PROMPT="hypothetical_scenario"

# Output
OUTPUT_DIR="experiments/paper_ahr_grpo/output"

# =============================================================================
# Parse Arguments
# =============================================================================
EXPERIMENT_TYPE=$1
shift || true

LORA_PATH=""
USE_ORIGINAL_PROMPT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --lora_path) LORA_PATH="$2"; shift 2 ;;
        --use_original_prompt) USE_ORIGINAL_PROMPT=true; shift ;;
        --guard_port) GUARD_PORT="$2"; shift 2 ;;
        --target_judge_port) TARGET_JUDGE_PORT="$2"; shift 2 ;;
        --attack_prompt) ATTACK_PROMPT="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/eval/${EXPERIMENT_TYPE}"
mkdir -p $EVAL_OUTPUT_DIR

echo "================================================================"
echo "AHR-GRPO Paper Experiment Evaluation"
echo "================================================================"
echo "Experiment Type: $EXPERIMENT_TYPE"
echo "LoRA Path:       ${LORA_PATH:-'(none, using base model)'}"
echo "Test Data:       $TEST_DATA"
echo "Output Dir:      $EVAL_OUTPUT_DIR"
echo "----------------------------------------------------------------"
echo "Ports:"
echo "  Guard:         $GUARD_PORT"
echo "  Target&Judge:  $TARGET_JUDGE_PORT"
echo "----------------------------------------------------------------"
echo "Attack Prompt:  $ATTACK_PROMPT"
echo "================================================================"

# =============================================================================
# Evaluation
# =============================================================================
if [ "$USE_ORIGINAL_PROMPT" = true ]; then
    echo ""
    echo "Evaluating with original prompt (baseline)..."
    python -m src.eval_baseline \
        --test_data $TEST_DATA \
        --target_port $TARGET_JUDGE_PORT \
        --guard_port $GUARD_PORT \
        --output_dir $EVAL_OUTPUT_DIR \
        2>&1 | tee $EVAL_OUTPUT_DIR/eval.log
else
    if [ -z "$LORA_PATH" ]; then
        echo "Error: --lora_path is required for non-baseline experiments."
        exit 1
    fi

    echo ""
    echo "Evaluating with trained policy..."
    python -m src.eval_trained \
        --policy_model $POLICY_MODEL \
        --lora_path $LORA_PATH \
        --target_port $TARGET_JUDGE_PORT \
        --guard_port $GUARD_PORT \
        --test_data $TEST_DATA \
        --attack_prompt $ATTACK_PROMPT \
        --output_dir $EVAL_OUTPUT_DIR \
        2>&1 | tee $EVAL_OUTPUT_DIR/eval.log
fi

echo ""
echo "Evaluation complete!"
echo "Results saved to: $EVAL_OUTPUT_DIR"
