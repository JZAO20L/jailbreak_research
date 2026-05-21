#!/bin/bash
# Experiment 3 Alternative: Simple Rule-Based Weight Parameter Ablation
#
# This script runs 3 experiments with different lambda bounds:
# 1. Fixed λ=1.0 (ASR-only, upper bound test)
# 2. Adaptive [0.8, 0.2] (ASR-preferred, high ASR weight when variance>0)
# 3. Adaptive [0.6, 0.4] (balanced, narrow range around 0.5)
#
# All experiments use Experiment 2 best combination:
# - Attack prompt: hypothetical_scenario
# - Judge dimension: idea_preservation
#
# Each experiment: 500 steps training + immediate evaluation on test set
#
# Usage:
#   bash experiments/simple_rule_weight_exp/exp.sh              # Run all 3 experiments
#   bash experiments/simple_rule_weight_exp/exp.sh 2            # Run only experiment 2
#
# Time estimate: ~50 min train + ~30 min eval = ~80 min per experiment × 3 = ~4 hours total

set -e

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT_DIR="$BASE_DIR/experiments/simple_rule_weight_exp"

# Model paths
POLICY_MODEL="/root/autodl-tmp/models/Qwen/Qwen3-4B"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Training data
TRAIN_DATA="$BASE_DIR/../data/dataset/processed/10k/train.jsonl"

# vLLM ports (must match running servers)
TARGET_PORT=8001
GUARD_PORT=8002

# Training hyperparameters (aligned with Exp2)
MAX_STEPS=500
LEARNING_RATE=1e-5
BETA=0.05
NUM_GENERATIONS=8
BATCH_SIZE=4
GRAD_ACCUM=4

# Attack & Judge (Exp2 best)
ATTACK_PROMPT="hypothetical_scenario"
JUDGE_PROMPT="idea_preservation_single"

# LoRA config
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET="q_proj,v_proj,k_proj,o_proj"

# Logging
SAVE_STEPS=100
SEED=42

# =============================================================================
# Experiment Definitions
# =============================================================================
# Format: experiment_id | run_name | parameters
#
# Experiment 1: Fixed λ=1.0 (ASR-only, tests pure ASR reward)
# Experiment 2: Adaptive [0.8, 0.2] (ASR-preferred, high weight when variance>0)
# Experiment 3: Adaptive [0.6, 0.4] (balanced, narrow range around 0.5)

EXPERIMENTS=(
    "exp1|fixed_1.0|--fixed_lambda 1.0"
    "exp2|adaptive_0.8_0.2|--lambda_min 0.2 --lambda_max 0.8"
    "exp3|adaptive_0.6_0.4|--lambda_min 0.4 --lambda_max 0.6"
)

# =============================================================================
# Helper Functions
# =============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_experiment() {
    local exp_id=$1
    local run_name=$2
    local params=$3

    log "=========================================="
    log "Starting Experiment: $exp_id ($run_name)"
    log "Parameters: $params"
    log "=========================================="

    # Create output directory
    local exp_output="$OUTPUT_DIR/$exp_id"
    mkdir -p "$exp_output"

    # Run training
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/simple_rule_weight_grpo.py" \
        --policy_model "$POLICY_MODEL" \
        --train_data "$TRAIN_DATA" \
        --output_dir "$exp_output" \
        --run_name "$run_name" \
        --attack_prompt "$ATTACK_PROMPT" \
        --judge_prompt "$JUDGE_PROMPT" \
        --max_steps $MAX_STEPS \
        --learning_rate $LEARNING_RATE \
        --beta $BETA \
        --num_generations $NUM_GENERATIONS \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_target_modules "$LORA_TARGET" \
        --target_port $TARGET_PORT \
        --guard_port $GUARD_PORT \
        --save_steps $SAVE_STEPS \
        --seed $SEED \
        --run_eval_after_train \
        $params

    log "Experiment $exp_id completed!"
    log "Results saved to: $exp_output"
    log "=========================================="
}

# =============================================================================
# Main Execution
# =============================================================================
log "Simple Rule-Based Weight Parameter Ablation - Experiment 3 Alternative"
log "Base directory: $BASE_DIR"
log "Output directory: $OUTPUT_DIR"
log ""
log "Experiment plan:"
log "  1. Fixed λ=1.0 (ASR-only, tests pure ASR reward)"
log "  2. Adaptive [0.8, 0.2] (ASR-preferred, high weight when variance>0)"
log "  3. Adaptive [0.6, 0.4] (balanced, narrow range around 0.5)"
log ""
log "Total experiments: ${#EXPERIMENTS[@]}"
log "Estimated time: ~80 min × ${#EXPERIMENTS[@]} = ~$(( 80 * ${#EXPERIMENTS[@]} / 60 )) hours"
log ""

# If a specific experiment ID is provided, run only that one
if [ -n "$1" ]; then
    log "Running only experiment $1"
    idx=$(( $1 - 1 ))
    if [ $idx -ge 0 ] && [ $idx -lt ${#EXPERIMENTS[@]} ]; then
        IFS='|' read -r exp_id run_name params <<< "${EXPERIMENTS[$idx]}"
        run_experiment "$exp_id" "$run_name" "$params"
    else
        log "ERROR: Invalid experiment ID: $1 (must be 1-${#EXPERIMENTS[@]})"
        exit 1
    fi
else
    # Run all experiments
    for exp_config in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_id run_name params <<< "$exp_config"
        run_experiment "$exp_id" "$run_name" "$params"

        # Optional: add a pause between experiments
        log "Waiting 10 seconds before next experiment..."
        sleep 10
    done
fi

log "All experiments completed!"
log "Results directory: $OUTPUT_DIR"

# =============================================================================
# Summary
# =============================================================================
log ""
log "=========================================="
log "Experiment Summary"
log "=========================================="
log "To evaluate results:"
log "  bash scripts/eval.py --lora_paths $OUTPUT_DIR/exp1/final_lora $OUTPUT_DIR/exp2/final_lora ..."
log ""
log "To analyze lambda history:"
log "  python $SCRIPT_DIR/summarize_results.py --output_dir $OUTPUT_DIR"
log "=========================================="
