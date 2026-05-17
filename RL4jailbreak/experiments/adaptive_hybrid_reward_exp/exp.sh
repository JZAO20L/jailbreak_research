#!/bin/bash
# =============================================================================
# Experiment 3: Adaptive Hybrid Reward GRPO - Execution Script
#
# This script runs 6 experiments with different EMA beta values:
#   EMA beta: 0, 0.5, 0.67, 0.8, 0.9, 0.95
#   Window size: 1, 2, 3, 5, 10, 20
#
# Each experiment:
#   1. Start Target + Guard services (GPU1)
#   2. Train Policy with adaptive reward (GPU0)
#   3. Evaluate trained Policy (start Policy with LoRA on GPU0)
#   4. Clean up for next experiment
#
# Usage:
#   bash exp.sh                     # Run all 6 experiments
#   bash exp.sh --ema_beta 0.95     # Run single experiment
#   bash exp.sh --attack_prompt hypothetical_scenario  # Use specific attack prompt
#   bash exp.sh --reset             # Clear checkpoints and start fresh
#   bash exp.sh --max_steps 1000    # Override max steps
#
# Reference: TODO.md (Experiment 3), NEW_IDEA.md
# =============================================================================

set -e

# =========================
# Configuration
# =========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Training data
TRAIN_DATA="${TRAIN_DATA:-$BASE_DIR/../data/dataset/processed/10k/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$BASE_DIR/../data/dataset/processed/10k/val.jsonl}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"

# Model paths
POLICY_MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
TARGET_MODEL="${TARGET_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B}"

# Ports
TARGET_PORT=8001
GUARD_PORT=8002
POLICY_PORT=8003

# vLLM config (根据 TODO.md: policy 4k, target/guard 8k)
VLLM_MAX_MODEL_LEN_POLICY=4096
VLLM_MAX_MODEL_LEN_TARGET=8192
VLLM_MAX_MODEL_LEN_GUARD=8192
VLLM_GPU_UTIL_TARGET=0.4
VLLM_GPU_UTIL_GUARD=0.4
VLLM_GPU_UTIL_POLICY=0.9

# Training config (from TODO.md)
MAX_STEPS="${MAX_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
BETA="${BETA:-0.05}"

# Adaptive reward config (fixed for all experiments except ema_beta)
ALPHA=2.0
DELTA=-1.0
LAMBDA_MIN=0.2
LAMBDA_MAX=0.8

# EMA beta values for ablation (from TODO.md)
# Window size = 1/(1-beta)
EMA_BETAS_DEFAULT=(0 0.5 0.67 0.8 0.9 0.95)
WINDOW_SIZES=("1" "2" "3" "5" "10" "20")

# Attack prompts (from Experiment 1 top-3)
ATTACK_PROMPTS_DEFAULT=("hypothetical_scenario" "creative_writing" "role_playing")

# Default: run all EMA beta experiments on single attack prompt
SELECTED_EMA_BETA=""
SELECTED_ATTACK_PROMPT=""
RESET_CKPT=false

# =========================
# Parse Arguments
# =========================
while [[ $# -gt 0 ]]; do
    case $1 in
        --ema_beta)
            SELECTED_EMA_BETA="$2"
            shift 2
            ;;
        --attack_prompt)
            SELECTED_ATTACK_PROMPT="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --reset)
            RESET_CKPT=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --ema_beta VALUE        Run single EMA beta experiment (default: all 6)"
            echo "                           Values: 0, 0.5, 0.67, 0.8, 0.9, 0.95"
            echo "  --attack_prompt NAME    Attack prompt strategy (default: hypothetical_scenario)"
            echo "  --max_steps N           Training steps (default: 1000)"
            echo "  --reset                 Clear checkpoints and start fresh"
            echo "  --help                  Show this help"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Set experiment parameters
if [ -n "$SELECTED_EMA_BETA" ]; then
    EMA_BETAS=("$SELECTED_EMA_BETA")
else
    EMA_BETAS=("${EMA_BETAS_DEFAULT[@]}")
fi

if [ -n "$SELECTED_ATTACK_PROMPT" ]; then
    ATTACK_PROMPTS=("$SELECTED_ATTACK_PROMPT")
else
    ATTACK_PROMPTS=("${ATTACK_PROMPTS_DEFAULT[@]:0:1}")  # Default: only top-1
fi

# =========================
# Helper Functions
# =========================
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

print_progress() {
    local current=$1
    local total=$2
    local pct=$((current * 100 / total))
    local filled=$((pct / 2))
    local empty=$((50 - filled))
    printf -v bar '%*s' "$filled" ''
    bar=${bar// /#}
    printf -v spaces '%*s' "$empty" ''
    printf "\r  [%s%s] %d%% (%d/%d)" "$bar" "$spaces" "$pct" "$current" "$total"
}

check_port_active() {
    local port=$1
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q "200"
}

wait_for_port() {
    local port=$1
    local timeout=120
    log "Waiting for port $port..."
    for i in $(seq 1 $timeout); do
        if check_port_active "$port"; then
            log "Port $port ready!"
            return 0
        fi
        sleep 2
    done
    log "ERROR: Port $port timeout!"
    return 1
}

# =========================
# Service Management
# =========================
start_target_service() {
    log "Starting Target service (port: $TARGET_PORT)..."
    if check_port_active "$TARGET_PORT"; then
        log "Target already running"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=1 nohup vllm serve "$TARGET_MODEL" \
        --host 127.0.0.1 --port $TARGET_PORT \
        --max-model-len $VLLM_MAX_MODEL_LEN_TARGET \
        --gpu-memory-utilization $VLLM_GPU_UTIL_TARGET \
        --served-model-name target \
        > "$OUTPUT_DIR/target_vllm.log" 2>&1 &

    wait_for_port "$TARGET_PORT"
}

start_guard_service() {
    log "Starting Guard service (port: $GUARD_PORT)..."
    if check_port_active "$GUARD_PORT"; then
        log "Guard already running"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=1 nohup vllm serve "$GUARD_MODEL" \
        --host 127.0.0.1 --port $GUARD_PORT \
        --max-model-len $VLLM_MAX_MODEL_LEN_GUARD \
        --gpu-memory-utilization $VLLM_GPU_UTIL_GUARD \
        --served-model-name guard \
        > "$OUTPUT_DIR/guard_vllm.log" 2>&1 &

    wait_for_port "$GUARD_PORT"
}

stop_policy_service() {
    log "Stopping Policy service..."
    pkill -f "vllm.*$POLICY_PORT" 2>/dev/null || true
    sleep 3
}

stop_all_services() {
    log "Stopping all services..."
    pkill -f "vllm.*$POLICY_PORT" 2>/dev/null || true
    pkill -f "vllm.*$TARGET_PORT" 2>/dev/null || true
    pkill -f "vllm.*$GUARD_PORT" 2>/dev/null || true
    sleep 5
}

ensure_target_guard_running() {
    log "============================================================"
    log "Checking Target + Guard services..."
    log "============================================================"
    
    start_target_service
    start_guard_service
    
    log "Target + Guard services ready!"
}

# =========================
# Checkpoint Management
# =========================
CKPT_FILE="$OUTPUT_DIR/exp_checkpoint.json"

load_checkpoint() {
    if [ -f "$CKPT_FILE" ] && [ "$RESET_CKPT" = false ]; then
        python3 -c "
import json
with open('$CKPT_FILE') as f:
    ckpt = json.load(f)
print(' '.join(ckpt.get('completed', [])))
" 2>/dev/null
    fi
}

save_checkpoint() {
    local completed_list="$1"
    mkdir -p "$OUTPUT_DIR"
    python3 -c "
import json
completed = '$completed_list'.split() if '$completed_list' else []
ckpt = {'completed': completed, 'max_steps': $MAX_STEPS}
with open('$CKPT_FILE', 'w') as f:
    json.dump(ckpt, f, indent=2)
"
}

# =========================
# Training Function
# =========================
run_training() {
    local ema_beta=$1
    local attack_prompt=$2
    local exp_key="ema${ema_beta}_${attack_prompt}"
    local exp_output="$OUTPUT_DIR/$exp_key"
    
    log "============================================================"
    log "Training: EMA beta=$ema_beta, Attack prompt=$attack_prompt"
    log "============================================================"
    
    mkdir -p "$exp_output"
    
    # Calculate window size
    if [ "$ema_beta" = "0" ]; then
        window_size=1
    else
        window_size=$(python3 -c "print(int(round(1/(1-$ema_beta))))")
    fi
    log "Window size: ~$window_size"
    
    # Run training
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/adaptive_hybrid_reward_grpo.py" \
        --ema_beta "$ema_beta" \
        --alpha "$ALPHA" \
        --delta "$DELTA" \
        --lambda_min "$LAMBDA_MIN" \
        --lambda_max "$LAMBDA_MAX" \
        --attack_prompt "$attack_prompt" \
        --train_data "$TRAIN_DATA" \
        --max_steps "$MAX_STEPS" \
        --learning_rate "$LEARNING_RATE" \
        --num_generations "$NUM_GENERATIONS" \
        --beta "$BETA" \
        --output_dir "$exp_output" \
        --target_port "$TARGET_PORT" \
        --guard_port "$GUARD_PORT" \
        --run_name "exp3_${exp_key}"
    
    log "Training complete: $exp_key"
}

# =========================
# Evaluation Function
# =========================
run_evaluation() {
    local ema_beta=$1
    local attack_prompt=$2
    local exp_key="ema${ema_beta}_${attack_prompt}"
    local exp_output="$OUTPUT_DIR/$exp_key"
    local lora_path="$exp_output/final_lora"
    
    # Check if LoRA exists
    if [ ! -d "$lora_path" ]; then
        log "[SKIP] Evaluation: $exp_key (LoRA not found)"
        return 1
    fi
    
    log "============================================================"
    log "Evaluation: EMA beta=$ema_beta, Attack prompt=$attack_prompt"
    log "============================================================"
    
    # Stop any existing Policy service
    stop_policy_service
    
    # Start Policy with LoRA
    log "Starting Policy service with LoRA..."
    CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$POLICY_MODEL" \
        --host 127.0.0.1 --port $POLICY_PORT \
        --max-model-len $VLLM_MAX_MODEL_LEN_POLICY \
        --gpu-memory-utilization $VLLM_GPU_UTIL_POLICY \
        --served-model-name policy \
        --enable-lora \
        --lora-modules policy_lora="$lora_path" \
        --max-lora-rank 32 \
        > "$exp_output/policy_vllm.log" 2>&1 &
    
    wait_for_port "$POLICY_PORT"
    
    # Run evaluation
    local eval_output="$exp_output/eval_results"
    mkdir -p "$eval_output"
    
    python "$BASE_DIR/scripts/eval.py" \
        --eval_path "$EVAL_DATA" \
        --lora_paths "$lora_path" \
        --base_model_path "$POLICY_MODEL" \
        --target_model_path "$TARGET_MODEL" \
        --guard_model_path "$GUARD_MODEL" \
        --policy_port "$POLICY_PORT" \
        --target_port "$TARGET_PORT" \
        --guard_port "$GUARD_PORT" \
        --output_root "$eval_output" \
        --run_name "eval_${exp_key}"
    
    # Stop Policy service
    stop_policy_service
    
    log "Evaluation complete: $exp_key"
}

# =========================
# Main Execution
# =========================
TOTAL_EXPS=$(( ${#EMA_BETAS[@]} * ${#ATTACK_PROMPTS[@]} ))

# Load checkpoint
COMPLETED_STR=$(load_checkpoint)
COMPLETED_COUNT=0
if [ -n "$COMPLETED_STR" ]; then
    COMPLETED_COUNT=$(echo "$COMPLETED_STR" | wc -w)
    log "Checkpoint: $COMPLETED_COUNT experiments completed"
fi

if [ "$RESET_CKPT" = true ]; then
    log "Reset checkpoint, starting fresh"
    COMPLETED_STR=""
    COMPLETED_COUNT=0
fi

log "============================================================"
log "Experiment 3: Adaptive Hybrid Reward GRPO"
log "============================================================"
log "Training data: $TRAIN_DATA"
log "Evaluation data: $EVAL_DATA"
log "Output directory: $OUTPUT_DIR"
log "EMA beta values: ${EMA_BETAS[*]}"
log "Attack prompts: ${ATTACK_PROMPTS[*]}"
log "Total experiments: $TOTAL_EXPS"
log "Completed: $COMPLETED_COUNT"
log "Remaining: $((TOTAL_EXPS - COMPLETED_COUNT))"
log "Max steps per experiment: $MAX_STEPS"
log "============================================================"

mkdir -p "$OUTPUT_DIR"

# Ensure Target + Guard running (shared across experiments)
ensure_target_guard_running

# Run experiments
EXP_IDX=0
COMPLETED_LIST=""
if [ -n "$COMPLETED_STR" ]; then
    COMPLETED_LIST="$COMPLETED_STR"
fi

for ema_beta in "${EMA_BETAS[@]}"; do
    for attack_prompt in "${ATTACK_PROMPTS[@]}"; do
        EXP_IDX=$((EXP_IDX + 1))
        
        EXP_KEY="ema${ema_beta}_${attack_prompt}"
        
        # Check if already completed
        if echo " $COMPLETED_LIST " | grep -q " $EXP_KEY "; then
            log "[SKIP] $EXP_KEY (checkpoint)"
            continue
        fi
        
        # Check if LoRA exists (from previous incomplete run)
        if [ -d "$OUTPUT_DIR/$EXP_KEY/final_lora" ]; then
            log "[SKIP] $EXP_KEY (LoRA exists)"
            COMPLETED_LIST="$COMPLETED_LIST $EXP_KEY"
            save_checkpoint "$COMPLETED_LIST"
            continue
        fi
        
        # Progress
        print_progress $((COMPLETED_COUNT + 1)) $TOTAL_EXPS
        log ""
        
        # Training
        run_training "$ema_beta" "$attack_prompt"
        
        # Evaluation
        run_evaluation "$ema_beta" "$attack_prompt"
        
        # Update checkpoint
        COMPLETED_LIST="$COMPLETED_LIST $EXP_KEY"
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
        save_checkpoint "$COMPLETED_LIST"
        
        log "[$EXP_IDX/$TOTAL_EXPS] Complete: $EXP_KEY"
    done
done

log ""
log "============================================================"
log "All experiments complete!"
log "============================================================"
log "Results in: $OUTPUT_DIR"
log ""

# Generate summary (if summarize script exists)
if [ -f "$SCRIPT_DIR/summarize_results.py" ]; then
    log "Generating summary report..."
    python "$SCRIPT_DIR/summarize_results.py" --output_dir "$OUTPUT_DIR"
fi

log "============================================================"