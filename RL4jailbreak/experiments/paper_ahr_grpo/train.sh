#!/bin/bash
# =============================================================================
# Training Script for AHR-GRPO Paper Experiments
#
# GPU Allocation (3 cards):
# - GPU 0: Guard + Target&Judge vLLM servers
# - GPU 1-2: Policy training (colocate mode, vLLM TP=2)
#
# Usage:
#   bash train.sh <experiment_type> [options]
#
# Examples:
#   bash train.sh ahr_grpo
#   bash train.sh ahr_grpo --num_samples 2000 --num_epochs 3
#   bash train.sh ablation_beta0 --ema_beta 0.0
# =============================================================================

set -e

# =============================================================================
# Default Configuration
# =============================================================================
POLICY_MODEL="/home/tiger/models/Qwen/Qwen3-4B"
SOURCE_DATA="/home/tiger/jailbreak_research/data/dataset/processed/10k/train.jsonl"

# Data & training schedule
NUM_SAMPLES=1000
NUM_EPOCHS=2

# Training hyperparameters (batch/step auto-calculated)
LEARNING_RATE=1e-5
BETA=0.05
NUM_GENERATIONS=8
PER_DEVICE_TRAIN_BATCH_SIZE=32
MAX_COMPLETION_LEN=2048
GRADIENT_ACCUMULATION_STEPS=2
NUM_GPUS=2

# LoRA configuration
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"

# vLLM configuration
VLLM_GPU_MEMORY_UTILIZATION=0.3
VLLM_TENSOR_PARALLEL_SIZE=2

# Adaptive reward configuration
EMA_BETA=0.9
ALPHA=2.0
DELTA=-2.0
LAMBDA_MIN=0.2
LAMBDA_MAX=0.8

# Ports (must match rollout.sh)
GUARD_PORT=8001
TARGET_JUDGE_PORT=8002

# Attack prompt and judge dimension
ATTACK_PROMPT="hypothetical_scenario"
JUDGE_PROMPT="multi_dimensional"  # Multi-dimensional: idea_preservation + stealthiness

# Output
OUTPUT_DIR="experiments/paper_ahr_grpo/output"
SEED=42
LOGGING_STEPS=1
SAVE_STEPS=50

# =============================================================================
# Parse Arguments
# =============================================================================
EXPERIMENT_TYPE=$1
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
        --num_epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --beta) BETA="$2"; shift 2 ;;
        --num_generations) NUM_GENERATIONS="$2"; shift 2 ;;
        --per_device_train_batch_size) PER_DEVICE_TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --max_completion_len) MAX_COMPLETION_LEN="$2"; shift 2 ;;
        --gradient_accumulation_steps) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
        --lora_r) LORA_R="$2"; shift 2 ;;
        --lora_alpha) LORA_ALPHA="$2"; shift 2 ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift 2 ;;
        --lora_target_modules) LORA_TARGET_MODULES="$2"; shift 2 ;;
        --vllm_gpu_memory_utilization) VLLM_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --vllm_tensor_parallel_size) VLLM_TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --ema_beta) EMA_BETA="$2"; shift 2 ;;
        --alpha) ALPHA="$2"; shift 2 ;;
        --delta) DELTA="$2"; shift 2 ;;
        --lambda_min) LAMBDA_MIN="$2"; shift 2 ;;
        --lambda_max) LAMBDA_MAX="$2"; shift 2 ;;
        --guard_port) GUARD_PORT="$2"; shift 2 ;;
        --target_judge_port) TARGET_JUDGE_PORT="$2"; shift 2 ;;
        --attack_prompt) ATTACK_PROMPT="$2"; shift 2 ;;
        --judge_prompt) JUDGE_PROMPT="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --logging_steps) LOGGING_STEPS="$2"; shift 2 ;;
        --save_steps) SAVE_STEPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Auto-calculate max_steps
# =============================================================================
# steps = ceil(num_samples * num_epochs / (batch_size * num_gpus * grad_accum))
EFFECTIVE_BATCH=$((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
MAX_STEPS=$(( (NUM_SAMPLES * NUM_EPOCHS + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH ))

# =============================================================================
# Create data subset
# =============================================================================
SUBSET_DIR="experiments/paper_ahr_grpo/output/data_subsets"
mkdir -p $SUBSET_DIR
TRAIN_DATA="${SUBSET_DIR}/train_n${NUM_SAMPLES}_s${SEED}.jsonl"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Creating data subset: ${NUM_SAMPLES} samples from ${SOURCE_DATA} (seed=${SEED})"
    python3 -c "
import random
random.seed($SEED)
with open('$SOURCE_DATA', 'r') as f:
    lines = f.readlines()
random.shuffle(lines)
with open('$TRAIN_DATA', 'w') as f:
    f.writelines(lines[:$NUM_SAMPLES])
print(f'Subset created: $TRAIN_DATA ({len(lines[:$NUM_SAMPLES])} lines)')
"
else
    echo "Using existing subset: $TRAIN_DATA ($(wc -l < $TRAIN_DATA) lines)"
fi

# =============================================================================
# Setup Output Directory
# =============================================================================
RUN_NAME="${EXPERIMENT_TYPE}_n${NUM_SAMPLES}_e${NUM_EPOCHS}_steps${MAX_STEPS}"
FULL_OUTPUT_DIR="${OUTPUT_DIR}/${RUN_NAME}"
mkdir -p $FULL_OUTPUT_DIR

# SwanLab configuration
export SWANLAB_PROJECT="AHR_GRPO_Paper"
export SWANLAB_RUN_NAME=$RUN_NAME

echo "================================================================"
echo "AHR-GRPO Paper Experiment Training"
echo "================================================================"
echo "Experiment Type: $EXPERIMENT_TYPE"
echo "Output Dir:      $FULL_OUTPUT_DIR"
echo "Run Name:        $RUN_NAME"
echo "----------------------------------------------------------------"
echo "Data Config:"
echo "  Source:             $SOURCE_DATA"
echo "  Num Samples:        $NUM_SAMPLES"
echo "  Num Epochs:         $NUM_EPOCHS"
echo "  Subset:             $TRAIN_DATA"
echo "----------------------------------------------------------------"
echo "Training Config:"
echo "  Max Steps:          $MAX_STEPS (auto-calculated)"
echo "  Effective Batch:    $EFFECTIVE_BATCH (${PER_DEVICE_TRAIN_BATCH_SIZE} × ${NUM_GPUS} GPUs × ${GRADIENT_ACCUMULATION_STEPS} accum)"
echo "  Learning Rate:      $LEARNING_RATE"
echo "  Beta (KL penalty):  $BETA"
echo "  Num Generations:    $NUM_GENERATIONS"
echo "  Max Completion Len: $MAX_COMPLETION_LEN"
echo "----------------------------------------------------------------"
echo "LoRA Config:"
echo "  Rank:      $LORA_R"
echo "  Alpha:     $LORA_ALPHA"
echo "  Dropout:   $LORA_DROPOUT"
echo "  Modules:   $LORA_TARGET_MODULES"
echo "----------------------------------------------------------------"
echo "vLLM Config:"
echo "  GPU Memory Utilization: $VLLM_GPU_MEMORY_UTILIZATION"
echo "  Tensor Parallel Size:   $VLLM_TENSOR_PARALLEL_SIZE"
echo "----------------------------------------------------------------"
echo "Adaptive Reward Config:"
echo "  EMA Beta:    $EMA_BETA"
echo "  Alpha:       $ALPHA"
echo "  Delta:       $DELTA"
echo "  Lambda:      [$LAMBDA_MIN, $LAMBDA_MAX]"
echo "----------------------------------------------------------------"
echo "Ports:"
echo "  Guard:         $GUARD_PORT"
echo "  Target&Judge:  $TARGET_JUDGE_PORT"
echo "----------------------------------------------------------------"
echo "Attack Prompt:   $ATTACK_PROMPT"
echo "Judge Dimension: $JUDGE_PROMPT"
echo "================================================================"

# =============================================================================
# Determine Reward Functions
# =============================================================================
case $EXPERIMENT_TYPE in
    baseline)
        echo "Baseline experiment - no training needed."
        echo "Use eval.sh to evaluate original prompt."
        exit 0
        ;;
    pure_asr)
        REWARD_FUNCS="asr_reward"
        ;;
    pure_judge)
        REWARD_FUNCS="judge_reward"
        ;;
    fixed_hybrid)
        REWARD_FUNCS="fixed_hybrid_reward"
        ;;
    ahr_grpo|ablation_beta0|ablation_beta08|ablation_beta09)
        REWARD_FUNCS="adaptive_hybrid_reward"
        ;;
    *)
        echo "Unknown experiment type: $EXPERIMENT_TYPE"
        echo "Available: baseline, pure_asr, pure_judge, fixed_hybrid, ahr_grpo, ablation_beta0, ablation_beta08, ablation_beta09"
        exit 1
        ;;
esac

echo "Reward Functions: $REWARD_FUNCS"

# =============================================================================
# Training Command
# =============================================================================
PLUGIN_PATH="$(pwd)/experiments/paper_ahr_grpo/plugin.py"

# Set environment variables for reward function configuration (read by plugin.py)
export EXPERIMENT_TYPE=$EXPERIMENT_TYPE
export EMA_BETA=$EMA_BETA
export ALPHA=$ALPHA
export DELTA=$DELTA
export LAMBDA_MIN=$LAMBDA_MIN
export LAMBDA_MAX=$LAMBDA_MAX
export GUARD_PORT=$GUARD_PORT
export TARGET_JUDGE_PORT=$TARGET_JUDGE_PORT
export ATTACK_PROMPT=$ATTACK_PROMPT
export JUDGE_PROMPT=$JUDGE_PROMPT
export NUM_GENERATIONS=$NUM_GENERATIONS
export TARGET_MAX_TOKENS=2048

echo ""
echo "Starting training..."
echo ""

# Training with ms-swift
CUDA_VISIBLE_DEVICES=1,2 \
swift rlhf \
    --rlhf_type grpo \
    --model $POLICY_MODEL \
    --train_dataset $TRAIN_DATA \
    --eval_dataset $VAL_DATA \
    --external_plugins $PLUGIN_PATH \
    --reward_funcs $REWARD_FUNCS \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --beta $BETA \
    --num_generations $NUM_GENERATIONS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --max_completion_length $MAX_COMPLETION_LEN \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lora_rank $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --vllm_tensor_parallel_size $VLLM_TENSOR_PARALLEL_SIZE \
    --output_dir $FULL_OUTPUT_DIR \
    --run_name $RUN_NAME \
    --seed $SEED \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --save_total_limit 2 \
    --save_strategy steps \
    --bf16 true \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --report_to swanlab \
    2>&1 | tee $FULL_OUTPUT_DIR/train.log

echo ""
echo "Training complete!"
echo "Output saved to: $FULL_OUTPUT_DIR"
echo "Logs: $FULL_OUTPUT_DIR/train.log"
