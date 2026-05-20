#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Hybrid Reward GRPO Training Script - Experiment 3

This script implements the core adaptive reward mechanism for experiment 3.
The key innovation is dynamically adjusting the weight between ASR reward
and Judge reward based on their variance ratios.

Key components:
1. AdaptiveRewardCalculator: Dynamically adjusts lambda (ASR weight)
2. Multi-dimensional Judge Prompt: Evaluates 4 dimensions in one call
3. ASR Reward: Standard attack success rate via Target+Guard

Usage:
    # Default EMA beta=0.9 (window~10)
    python adaptive_hybrid_reward_grpo.py --ema_beta 0.9

    # Different EMA beta values for ablation
    python adaptive_hybrid_reward_grpo.py --ema_beta 0.0   # window=1
    python adaptive_hybrid_reward_grpo.py --ema_beta 0.67  # window=3
    python adaptive_hybrid_reward_grpo.py --ema_beta 0.8   # window=5
    python adaptive_hybrid_reward_grpo.py --ema_beta 0.9   # window=10

Reference: TODO.md (Experiment 3), NEW_IDEA.md
"""

import os
import sys
import json
import re
import gc
import logging
import datetime
import argparse
import random
from typing import List, Dict, Optional
from collections import defaultdict

import torch
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.vllm_client import VLLMClient
from src.utils import extract_label_categories_refusal
from src.prompts import REWRITE_PROMPT, GUARD_PROMPT
from src.reward.adaptive_weight import AdaptiveRewardCalculator, AdaptiveRewardConfig
from experiments.adaptive_hybrid_reward_exp.judge_prompts import (
    JUDGE_PROMPTS,
    get_judge_prompt,
    parse_judge_response,
)


# =============================================================================
# Default Arguments
# =============================================================================
DEFAULT_ARGS = {
    # Model paths
    "policy_model": "/root/autodl-tmp/models/Qwen/Qwen3-4B",
    "target_model": "/root/autodl-tmp/models/Qwen/Qwen3-4B",
    "guard_model": "/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B",
    
    # LoRA config
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,v_proj,k_proj,o_proj",
    
    # Training hyperparameters (from TODO.md)
    "learning_rate": 1e-5,
    "max_steps": 500,            # Fixed 500 steps for ablation (aligned with Exp2)
    "beta": 0.05,                # KL penalty
    "num_generations": 8,        # Group size for GRPO
    "per_device_train_batch_size": 4,
    "max_completion_len": 2048,
    "gradient_accumulation_steps": 4,
    
    # vLLM config
    "vllm_max_model_len": 4096,
    "vllm_gpu_memory_utilization": 0.3,
    
    # Adaptive reward config (from TODO.md)
    "ema_beta": 0.9,             # Default: window~10
    "alpha": 2.0,                # Variance ratio sensitivity
    "delta": -2.0,               # Sigmoid bias (ratio=1 → lambda=0.5, neutral)
    "lambda_min": 0.2,           # ASR weight lower bound
    "lambda_max": 0.8,           # ASR weight upper bound
    
    # Ports
    "target_port": 8001,
    "guard_port": 8002,
    "target_max_tokens": 512,
    
    # Jailbreak prompt strategy (from TODO.md: use top-3 prompts)
    "attack_prompt": "hypothetical_scenario",  # Exp2 re-eval best (27.0%)

    # Judge prompt dimension (aligned with Experiment 2 re-evaluation)
    "judge_prompt": "idea_preservation",       # Best overall (+0.8% vs baseline)

    # Output
    "output_dir": "experiments/adaptive_hybrid_reward_exp/output",
    "run_name": None,
    "seed": 42,
    "logging_steps": 1,
    "save_steps": 100,
}


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Hybrid Reward GRPO Training - Experiment 3"
    )
    
    # EMA beta (key parameter for ablation)
    parser.add_argument("--ema_beta", type=float, default=DEFAULT_ARGS["ema_beta"],
                        help="EMA smoothing parameter. Window size = 1/(1-beta). "
                             "Values: 0, 0.67, 0.8, 0.9 → windows: 1,3,5,10")
    
    # Adaptive reward parameters
    parser.add_argument("--alpha", type=float, default=DEFAULT_ARGS["alpha"],
                        help="Variance ratio sensitivity for sigmoid")
    parser.add_argument("--delta", type=float, default=DEFAULT_ARGS["delta"],
                        help="Sigmoid bias (negative → prefer ASR initially)")
    parser.add_argument("--lambda_min", type=float, default=DEFAULT_ARGS["lambda_min"],
                        help="Lower bound for ASR weight lambda")
    parser.add_argument("--lambda_max", type=float, default=DEFAULT_ARGS["lambda_max"],
                        help="Upper bound for ASR weight lambda")
    
    # Model paths
    parser.add_argument("--policy_model", type=str, default=DEFAULT_ARGS["policy_model"])
    parser.add_argument("--lora_r", type=int, default=DEFAULT_ARGS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_ARGS["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_ARGS["lora_dropout"])
    parser.add_argument("--lora_target_modules", type=str, default=DEFAULT_ARGS["lora_target_modules"])
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_ARGS["learning_rate"])
    parser.add_argument("--max_steps", type=int, default=DEFAULT_ARGS["max_steps"])
    parser.add_argument("--beta", type=float, default=DEFAULT_ARGS["beta"])
    parser.add_argument("--num_generations", type=int, default=DEFAULT_ARGS["num_generations"])
    parser.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_ARGS["per_device_train_batch_size"])
    parser.add_argument("--max_completion_len", type=int, default=DEFAULT_ARGS["max_completion_len"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_ARGS["gradient_accumulation_steps"])
    parser.add_argument("--vllm_max_model_len", type=int, default=DEFAULT_ARGS["vllm_max_model_len"])
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=DEFAULT_ARGS["vllm_gpu_memory_utilization"])
    
    # Ports
    parser.add_argument("--target_port", type=int, default=DEFAULT_ARGS["target_port"])
    parser.add_argument("--guard_port", type=int, default=DEFAULT_ARGS["guard_port"])
    parser.add_argument("--target_max_tokens", type=int, default=DEFAULT_ARGS["target_max_tokens"])
    
    # Attack prompt strategy
    parser.add_argument("--attack_prompt", type=str, default=DEFAULT_ARGS["attack_prompt"],
                        help="Jailbreak prompt strategy from Experiment 1 top-3")

    # Judge prompt dimension (aligned with Experiment 2)
    parser.add_argument("--judge_prompt", type=str, default=DEFAULT_ARGS["judge_prompt"],
                        help="Judge prompt dimension. Options: idea_preservation, stealthiness, "
                             "naturalness, hypothetical_scenario, creative_writing, role_playing")

    # Data
    parser.add_argument("--train_data", type=str,
                        default=os.path.join(BASE_DIR, "../data/dataset/processed/10k/train.jsonl"))
    parser.add_argument("--eval_data", type=str,
                        default=os.path.join(BASE_DIR, "../data/dataset/processed/10k/val.jsonl"))
    
    # Output
    parser.add_argument("--output_dir", type=str, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--run_name", type=str, default=DEFAULT_ARGS["run_name"])
    parser.add_argument("--seed", type=int, default=DEFAULT_ARGS["seed"])
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_ARGS["logging_steps"])
    parser.add_argument("--save_steps", type=int, default=DEFAULT_ARGS["save_steps"])
    
    return parser.parse_args()


# =============================================================================
# Global Setup
# =============================================================================
args = parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SWANLAB_PROJECT"] = "JPG_adaptive_hybrid_reward_exp"

LORA_DIR = os.path.join(args.output_dir, "final_lora")


# =============================================================================
# Logging Setup
# =============================================================================
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
LOG_FILE = os.path.join(args.output_dir, "logs",
                        f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger("adaptive_hybrid_reward")


# =============================================================================
# Connect to Existing vLLM Servers
# =============================================================================
logger.info("Connecting to existing vLLM servers...")

TARGET_CLIENT = VLLMClient(
    model_name="target",
    model_path="unused",
    host="127.0.0.1",
    port=args.target_port,
    launch_server=False,
    timeout=30,
    temperature=0.0,
)

GUARD_CLIENT = VLLMClient(
    model_name="guard",
    model_path="unused",
    host="127.0.0.1",
    port=args.guard_port,
    launch_server=False,
    timeout=30,
    temperature=0.0,
)

logger.info(f"Target connected: port {args.target_port}")
logger.info(f"Guard connected: port {args.guard_port}")


# =============================================================================
# Adaptive Reward Calculator (Global Instance)
# =============================================================================
adaptive_config = AdaptiveRewardConfig(
    alpha=args.alpha,
    delta=args.delta,
    lambda_min=args.lambda_min,
    lambda_max=args.lambda_max,
    ema_beta=args.ema_beta,
    lambda_init=0.5,
)

adaptive_calculator = AdaptiveRewardCalculator(adaptive_config)

# Lambda history for logging/analysis
lambda_history: List[Dict] = []


# =============================================================================
# Dataset
# =============================================================================
def load_train_dataset(path: str, attack_prompt: str) -> Dataset:
    """Load training dataset with specified attack prompt template."""
    
    # Import attack prompts from experiment 1
    from experiments.jailbreak_prompt_exp.jailbreak_prompts import get_strategy_template
    
    attack_template = get_strategy_template(attack_prompt)
    
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            raw = (obj.get("prompt") or "").strip()
            if not raw:
                continue
            # Apply attack prompt template
            prompt = attack_template.format(original_prompt=raw)
            rows.append({"prompt": prompt, "original_prompt": raw})
    
    ds = Dataset.from_list(rows)
    logger.info(f"Loaded {len(ds)} training samples from {path}")
    logger.info(f"Attack prompt strategy: {attack_prompt}")
    return ds


# =============================================================================
# Reward Functions
# =============================================================================
def judge_reward(
    prompts: List[str],
    completions: List[str],
    original_prompts: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """
    Judge Reward: Evaluates rewritten prompt quality on specified dimension.

    Aligned with Experiment 2 format: uses SCORE=0.XX output.

    Available dimensions (from judge_prompts.py):
    - General: idea_preservation, stealthiness, naturalness
    - Specialized: hypothetical_scenario, creative_writing, role_playing

    Returns raw judge reward (0~1, before adaptive weighting).
    """

    # Get original prompts for comparison
    if original_prompts is None:
        original_prompts = prompts

    # Get judge template for specified dimension
    judge_template = get_judge_prompt(args.judge_prompt)

    judge_prompts_batch = []
    for orig, rewritten in zip(original_prompts, completions):
        rewritten = (rewritten or "").strip()
        judge_prompts_batch.append(
            judge_template.format(
                original_prompt=orig,
                rewritten_prompt=rewritten
            )
        )

    # Batch call judge (using target as judge model)
    judge_responses = TARGET_CLIENT.llm_batch_call(
        prompts=judge_prompts_batch,
        temperature=0.0,
        max_tokens=256,
        max_workers=16,
        return_exceptions=True,
    )

    # Parse responses and compute raw scores
    raw_scores = []
    for resp in judge_responses:
        if resp is None or isinstance(resp, Exception):
            raw_scores.append(0.1)  # Fallback low score
            continue

        score = parse_judge_response(str(resp))
        raw_scores.append(score)

    return raw_scores


def asr_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    ASR Reward: Send rewritten prompt to target, then classify with guard.
    
    Returns raw ASR reward (0, 0.5, or 1, before adaptive weighting).
    """
    
    jailbreak_prompts = [(c or "").strip() for c in completions]
    
    # Step 1: Get target responses
    target_responses = TARGET_CLIENT.llm_batch_call(
        prompts=jailbreak_prompts,
        temperature=0.7,
        max_tokens=args.target_max_tokens,
        max_workers=16,
        return_exceptions=True,
    )
    
    # Step 2: Guard classification
    guard_messages_list = []
    for jp, tr in zip(jailbreak_prompts, target_responses):
        if tr is None or isinstance(tr, Exception):
            tr = "Error: No response"
        guard_messages_list.append([
            {"role": "system", "content": GUARD_PROMPT},
            {"role": "user", "content": jp},
            {"role": "assistant", "content": tr},
        ])
    
    guard_responses = GUARD_CLIENT.llm_batch_call(
        messages_list=guard_messages_list,
        temperature=0.0,
        max_tokens=256,
        max_workers=16,
        return_exceptions=True,
    )
    
    # Step 3: Parse guard responses
    raw_scores = []
    for gr in guard_responses:
        if gr is None or isinstance(gr, Exception):
            raw_scores.append(0.0)
            continue
        
        label, categories, refusal = extract_label_categories_refusal(str(gr))
        
        if label == "Unsafe":
            raw_scores.append(1.0)
        elif label == "Controversial":
            raw_scores.append(0.5)
        else:
            raw_scores.append(0.0)
    
    return raw_scores


def adaptive_hybrid_reward(
    prompts: List[str],
    completions: List[str],
    original_prompts: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """
    Adaptive Hybrid Reward: Combines ASR and Judge rewards with adaptive weighting.

    Window is measured in **training steps**, not samples.
    Each step:
    1. Compute raw ASR and Judge rewards for each completion
    2. Group by original prompt (k completions per prompt)
    3. Compute variance of rewards within each group (across k completions)
    4. Average variances across batch → one (var_asr, var_judge) pair per step
    5. Update lambda based on this step's variance ratio
    6. Final reward = lambda * ASR + (1-lambda) * Judge
    """

    # Step 1: Get raw rewards for each completion
    asr_raws = asr_reward(prompts, completions, **kwargs)
    judge_raws = judge_reward(prompts, completions, original_prompts, **kwargs)

    # Step 2-3: Group by original prompt and compute per-group variance
    # In GRPO, prompts list contains repeated prompts (each repeated k times)
    # Completions are in the same order
    k = args.num_generations  # 8

    step_var_asr_list: List[float] = []
    step_var_judge_list: List[float] = []

    for i in range(0, len(completions), k):
        group_asr = asr_raws[i:i+k]
        group_judge = judge_raws[i:i+k]

        if len(group_asr) >= 2:
            var_asr = float(torch.var(torch.tensor(group_asr, dtype=torch.float32)).item())
            var_judge = float(torch.var(torch.tensor(group_judge, dtype=torch.float32)).item())
            step_var_asr_list.append(var_asr)
            step_var_judge_list.append(var_judge)
        else:
            step_var_asr_list.append(0.0)
            step_var_judge_list.append(0.0)

    # Step 4: Average variance across prompts in this batch
    n_prompts = len(step_var_asr_list)
    avg_var_asr = sum(step_var_asr_list) / n_prompts if n_prompts > 0 else 0.0
    avg_var_judge = sum(step_var_judge_list) / n_prompts if n_prompts > 0 else 0.0

    # Step 5: Update lambda with this step's variances
    adaptive_calculator.update_step(avg_var_asr, avg_var_judge)

    # Step 6: Compute final rewards using current lambda
    lambda_val = adaptive_calculator.get_lambda()
    final_rewards = [
        lambda_val * a + (1 - lambda_val) * j
        for a, j in zip(asr_raws, judge_raws)
    ]

    # Log lambda statistics
    stats = adaptive_calculator.get_statistics()
    lambda_history.append({
        "step": stats["step"],
        "lambda": stats["lambda"],
        "var_asr": stats["var_asr"],
        "var_judge": stats["var_judge"],
        "ratio": stats["ratio"],
        "batch_size": n_prompts,
    })

    # Periodic logging
    if stats["step"] % 10 == 0:
        window_size = adaptive_config.get_window_size()
        logger.info(f"[Adaptive] step={stats['step']} λ={stats['lambda']:.3f} | "
                    f"var_asr={stats['var_asr']:.4f} var_judge={stats['var_judge']:.4f} | "
                    f"ratio={stats['ratio']:.2f} | "
                    f"window={'warmup' if stats['step'] <= window_size else 'adaptive'}")

    return final_rewards


# =============================================================================
# Main Training Function
# =============================================================================
def main():
    logger.info("=" * 70)
    logger.info("Adaptive Hybrid Reward GRPO Training - Experiment 3")
    logger.info("=" * 70)
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Attack prompt: {args.attack_prompt}")
    logger.info(f"Judge dimension: {args.judge_prompt}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("-" * 70)
    logger.info("Adaptive Reward Config:")
    logger.info(f"  EMA beta: {args.ema_beta} (window ~{int(1/(1-args.ema_beta)) if args.ema_beta < 1 else 'inf'})")
    logger.info(f"  Alpha: {args.alpha} (variance ratio sensitivity)")
    logger.info(f"  Delta: {args.delta} (sigmoid bias)")
    logger.info(f"  Lambda range: [{args.lambda_min}, {args.lambda_max}]")
    logger.info("=" * 70)
    
    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        config_dict = vars(args)
        config_dict["adaptive_config"] = {
            "alpha": args.alpha,
            "delta": args.delta,
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
            "ema_beta": args.ema_beta,
            "window_size": int(1/(1-args.ema_beta)) if args.ema_beta < 1 else "inf",
        }
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Config saved to {config_path}")
    
    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Load model
    logger.info("Loading policy model on GPU 0...")
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    policy = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    
    try:
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False
    except Exception:
        pass
    
    # LoRA
    lora_modules = args.lora_target_modules.split(",")
    policy = get_peft_model(
        policy,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_modules,
            bias="none",
        ),
    )
    policy.print_trainable_parameters()
    
    # Load dataset
    train_ds = load_train_dataset(args.train_data, args.attack_prompt)
    
    # GRPO Config
    run_name = args.run_name or f"exp3_ema{args.ema_beta}_{args.attack_prompt}_{args.judge_prompt}"
    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_len,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        save_strategy="steps",
        bf16=True,
        beta=args.beta,
        report_to="swanlab",
        run_name=run_name,
        seed=args.seed,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=False,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    
    # Trainer
    trainer = GRPOTrainer(
        model=policy,
        args=grpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=[adaptive_hybrid_reward],
    )
    
    # Training
    logger.info("Training start...")
    try:
        trainer.train()
        logger.info("Training done.")
        
        # Save LoRA
        logger.info(f"Saving LoRA to: {LORA_DIR}")
        os.makedirs(LORA_DIR, exist_ok=True)
        trainer.save_model(LORA_DIR)
        tokenizer.save_pretrained(LORA_DIR)
        logger.info("LoRA saved.")
        
        # Save lambda history for analysis
        lambda_path = os.path.join(args.output_dir, "lambda_history.json")
        with open(lambda_path, "w", encoding="utf-8") as f:
            json.dump(lambda_history, f, indent=2)
        logger.info(f"Lambda history saved: {lambda_path}")
        
        # Summary statistics
        if lambda_history:
            lambdas = [h["lambda"] for h in lambda_history]
            logger.info(f"Lambda statistics: min={min(lambdas):.3f}, max={max(lambdas):.3f}, "
                        f"mean={sum(lambdas)/len(lambdas):.3f}, final={lambdas[-1]:.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Cleanup done.")
    
    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()