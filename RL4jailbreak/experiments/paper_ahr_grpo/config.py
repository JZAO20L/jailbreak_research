# -*- coding: utf-8 -*-
"""
AHR-GRPO Paper Experiments Configuration

Architecture:
    GPU 0:    Guard server (port 8001) + Target&Judge server (port 8002)
    GPU 1-2:  Policy training (colocate mode, vLLM TP=2)

Training:
    Dataset: 2000 samples × 2 epochs = 4000 prompt-processings
    Batch:   per_device=32 × 2 GPUs = 64 completions/step
    Groups:  num_generations=8 → 8 prompts/step
    Accum:   gradient_accumulation=2 → 16 prompts/update
    Steps:   4000 / 16 = 250 steps

Experiments:
1. Main: Baseline vs Pure ASR vs Pure Judge vs Fixed Hybrid vs AHR-GRPO
2. Ablation: Fixed 1:1 vs Adaptive β=0.0/0.8/0.9
3. Generalization: Cross-model/Cross-dataset (future)
"""

import os
from typing import Dict, Optional

# =============================================================================
# Model Paths
# =============================================================================
MODEL_BASE = "/home/tiger/models/Qwen"

POLICY_MODEL = f"{MODEL_BASE}/Qwen3-4B"
TARGET_MODEL = f"{MODEL_BASE}/Qwen3-4B"
GUARD_MODEL = f"{MODEL_BASE}/Qwen3Guard-Gen-4B"
JUDGE_MODEL = f"{MODEL_BASE}/Qwen3-4B"  # Same as target, served on same port

# =============================================================================
# Context Length
# =============================================================================
POLICY_MAX_MODEL_LEN = 4096
TARGET_MAX_MODEL_LEN = 8192
GUARD_MAX_MODEL_LEN = 8192
JUDGE_MAX_MODEL_LEN = 8192

# =============================================================================
# Training Configuration
# =============================================================================
# Data & schedule: controlled by --num_samples and --num_epochs in train.sh
# Steps auto-calculated: ceil(num_samples * num_epochs / (batch * gpus * accum))
NUM_SAMPLES = 1000
NUM_EPOCHS = 2
LEARNING_RATE = 1e-5
BETA = 0.05  # KL penalty
NUM_GENERATIONS = 8  # Group size for GRPO
PER_DEVICE_TRAIN_BATCH_SIZE = 32
MAX_COMPLETION_LEN = 2048
GRADIENT_ACCUMULATION_STEPS = 2
NUM_GPUS = 2

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "q_proj,v_proj,k_proj,o_proj"

# =============================================================================
# vLLM Configuration (Policy colocate mode on GPU 1-2)
# =============================================================================
VLLM_GPU_MEMORY_UTILIZATION = 0.3
VLLM_TENSOR_PARALLEL_SIZE = 2  # For policy training on GPU 1-2

# =============================================================================
# Adaptive Reward Configuration
# =============================================================================
EMA_BETA = 0.9  # Default: window~10
ALPHA = 2.0  # Variance ratio sensitivity
DELTA = -2.0  # Sigmoid bias
LAMBDA_MIN = 0.2
LAMBDA_MAX = 0.8

# =============================================================================
# Ports (GPU 0 servers)
# =============================================================================
GUARD_PORT = 8001
TARGET_JUDGE_PORT = 8002

# =============================================================================
# Dataset Paths
# =============================================================================
DATA_BASE = "/home/tiger/jailbreak_research/data/dataset/processed/2k"
TRAIN_DATA = f"{DATA_BASE}/train.jsonl"
VAL_DATA = f"{DATA_BASE}/val.jsonl"
TEST_DATA = f"{DATA_BASE}/test.jsonl"

# =============================================================================
# Attack Prompt & Judge Dimension
# =============================================================================
ATTACK_PROMPT = "hypothetical_scenario"  # Best from Exp1
JUDGE_PROMPT = "multi_dimensional"  # Multi-dimensional: idea_preservation + stealthiness

# =============================================================================
# Output
# =============================================================================
OUTPUT_DIR = "experiments/paper_ahr_grpo/output"
SEED = 42
LOGGING_STEPS = 1
SAVE_STEPS = 50

# =============================================================================
# SwanLab
# =============================================================================
SWANLAB_PROJECT = "AHR_GRPO_Paper"


def get_experiment_config(experiment_type: str) -> Dict:
    """
    Get configuration for specific experiment type.

    Args:
        experiment_type: One of:
            - "baseline": No training, original prompt
            - "pure_asr": Only ASR reward
            - "pure_judge": Only Judge reward
            - "fixed_hybrid": Fixed 1:1 hybrid reward
            - "ahr_grpo": Adaptive hybrid reward (ours)
            - "ablation_beta0": Adaptive with β=0.0
            - "ablation_beta08": Adaptive with β=0.8
            - "ablation_beta09": Adaptive with β=0.9

    Returns:
        Dict: Experiment configuration
    """
    base_config = {
        "policy_model": POLICY_MODEL,
        "target_model": TARGET_MODEL,
        "guard_model": GUARD_MODEL,
        "judge_model": JUDGE_MODEL,
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "attack_prompt": ATTACK_PROMPT,
        "judge_prompt": JUDGE_PROMPT,
        "output_dir": OUTPUT_DIR,
    }

    if experiment_type == "baseline":
        return {**base_config, "reward_type": "none"}

    elif experiment_type == "pure_asr":
        return {**base_config, "reward_type": "asr"}

    elif experiment_type == "pure_judge":
        return {**base_config, "reward_type": "judge"}

    elif experiment_type == "fixed_hybrid":
        return {**base_config, "reward_type": "fixed_hybrid", "lambda_fixed": 0.5}

    elif experiment_type == "ahr_grpo":
        return {
            **base_config,
            "reward_type": "adaptive_hybrid",
            "ema_beta": EMA_BETA,
            "alpha": ALPHA,
            "delta": DELTA,
            "lambda_min": LAMBDA_MIN,
            "lambda_max": LAMBDA_MAX,
        }

    elif experiment_type == "ablation_beta0":
        return {
            **base_config,
            "reward_type": "adaptive_hybrid",
            "ema_beta": 0.0,
            "alpha": ALPHA,
            "delta": DELTA,
            "lambda_min": LAMBDA_MIN,
            "lambda_max": LAMBDA_MAX,
        }

    elif experiment_type == "ablation_beta08":
        return {
            **base_config,
            "reward_type": "adaptive_hybrid",
            "ema_beta": 0.8,
            "alpha": ALPHA,
            "delta": DELTA,
            "lambda_min": LAMBDA_MIN,
            "lambda_max": LAMBDA_MAX,
        }

    elif experiment_type == "ablation_beta09":
        return {
            **base_config,
            "reward_type": "adaptive_hybrid",
            "ema_beta": 0.9,
            "alpha": ALPHA,
            "delta": DELTA,
            "lambda_min": LAMBDA_MIN,
            "lambda_max": LAMBDA_MAX,
        }

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def print_config():
    """Print configuration information."""
    print("=" * 70)
    print("AHR-GRPO Paper Experiments Configuration")
    print("=" * 70)
    print(f"Architecture:")
    print(f"  GPU 0:    Guard + Target&Judge servers")
    print(f"  GPU 1-2:  Policy training (colocate, TP=2)")
    print("-" * 70)
    print(f"Model Paths:")
    print(f"  Policy:  {POLICY_MODEL}")
    print(f"  Target:  {TARGET_MODEL}")
    print(f"  Guard:   {GUARD_MODEL}")
    print(f"  Judge:   {JUDGE_MODEL}")
    print("-" * 70)
    print(f"Training:")
    print(f"  Dataset:   2000 samples × 2 epochs")
    print(f"  Max Steps: {MAX_STEPS}")
    print(f"  LR:        {LEARNING_RATE}")
    print(f"  Batch:     {PER_DEVICE_TRAIN_BATCH_SIZE} × 2 GPUs = {PER_DEVICE_TRAIN_BATCH_SIZE * 2} completions/step")
    print(f"  Num Gen:   {NUM_GENERATIONS}")
    print(f"  Grad Acc:  {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Max Comp:  {MAX_COMPLETION_LEN}")
    print("-" * 70)
    print(f"LoRA:")
    print(f"  Rank:    {LORA_R}")
    print(f"  Alpha:   {LORA_ALPHA}")
    print(f"  Dropout: {LORA_DROPOUT}")
    print(f"  Modules: {LORA_TARGET_MODULES}")
    print("-" * 70)
    print(f"Adaptive Reward:")
    print(f"  EMA Beta: {EMA_BETA} (window ~{int(1/(1-EMA_BETA)) if EMA_BETA < 1 else 'inf'})")
    print(f"  Alpha:    {ALPHA}")
    print(f"  Delta:    {DELTA}")
    print(f"  Lambda:   [{LAMBDA_MIN}, {LAMBDA_MAX}]")
    print("-" * 70)
    print(f"Ports:")
    print(f"  Guard:        {GUARD_PORT}")
    print(f"  Target&Judge: {TARGET_JUDGE_PORT}")
    print("-" * 70)
    print(f"Dataset:")
    print(f"  Train: {TRAIN_DATA}")
    print(f"  Val:   {VAL_DATA}")
    print(f"  Test:  {TEST_DATA}")
    print("-" * 70)
    print(f"Attack Prompt:   {ATTACK_PROMPT}")
    print(f"Judge Dimension: {JUDGE_PROMPT}")
    print("-" * 70)
    print(f"SwanLab Project: {SWANLAB_PROJECT}")
    print(f"Output:          {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
