#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Rule-Based Hybrid Reward GRPO Training Script - Experiment 3 (Alternative)

This script implements a simplified adaptive reward mechanism as an alternative
to the EMA-based approach. The key innovation is using direct rules instead of
windowed EMA for weight adjustment.

Key differences from adaptive_hybrid_reward_exp:
1. No EMA smoothing → immediate response to variance changes
2. No sigmoid transformation → linear mapping, more interpretable
3. No warmup period → rules apply from step 1
4. 4 explicit states: both_zero, asr_zero, judge_zero, both_nonzero

Key components:
1. SimpleRuleCalculator: Direct rule-based lambda adjustment
2. Multi-dimensional Judge Prompt: Evaluates specified dimension
3. ASR Reward: Standard attack success rate via Target+Guard

Based on Experiment 2 best combination:
- Attack prompt: hypothetical_scenario (27.0% after training)
- Judge dimension: idea_preservation (+0.8% vs baseline)

Usage:
    # Default lambda bounds [0.2, 0.8]
    python simple_rule_weight_grpo.py

    # Tighter bounds [0.3, 0.7]
    python simple_rule_weight_grpo.py --lambda_min 0.3 --lambda_max 0.7

    # More extreme bounds [0.1, 0.9]
    python simple_rule_weight_grpo.py --lambda_min 0.1 --lambda_max 0.9

    # Fixed weight (control experiment)
    python simple_rule_weight_grpo.py --fixed_lambda 0.5
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
from src.reward.simple_rule_weight import SimpleRuleCalculator, SimpleRuleConfig
from experiments.hybrid_reward_exp.judge_prompts import (
    JUDGE_PROMPTS,
    get_judge_template,
)
import re


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

    # Training hyperparameters (aligned with Exp2)
    "learning_rate": 1e-5,
    "max_steps": 500,            # Fixed 500 steps
    "beta": 0.05,                # KL penalty
    "num_generations": 8,        # Group size for GRPO
    "per_device_train_batch_size": 4,
    "max_completion_len": 2048,
    "gradient_accumulation_steps": 4,

    # vLLM config
    "vllm_max_model_len": 4096,
    "vllm_gpu_memory_utilization": 0.3,

    # Simple rule config
    "lambda_min": 0.2,           # ASR weight lower bound
    "lambda_max": 0.8,           # ASR weight upper bound
    "fixed_lambda": None,        # If set, use fixed weight (no adaptation)

    # Ports
    "target_port": 8001,
    "guard_port": 8002,
    "target_max_tokens": 512,

    # Jailbreak prompt strategy (Exp2 best: hypothetical_scenario + idea_preservation = 27.0%)
    "attack_prompt": "hypothetical_scenario",

    # Judge prompt dimension (Exp2 best: idea_preservation)
    "judge_prompt": "idea_preservation",

    # Output
    "output_dir": "experiments/simple_rule_weight_exp/output",
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
        description="Simple Rule-Based Hybrid Reward GRPO Training - Experiment 3 Alternative"
    )

    # Lambda bounds (key parameters for ablation)
    parser.add_argument("--lambda_min", type=float, default=DEFAULT_ARGS["lambda_min"],
                        help="Lower bound for ASR weight lambda (default: 0.2)")
    parser.add_argument("--lambda_max", type=float, default=DEFAULT_ARGS["lambda_max"],
                        help="Upper bound for ASR weight lambda (default: 0.8)")
    parser.add_argument("--fixed_lambda", type=float, default=DEFAULT_ARGS["fixed_lambda"],
                        help="If set, use fixed lambda instead of adaptive rules "
                             "(e.g., 0.5 for 1:1, 0.8 for ASR-heavy)")

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

    # Attack prompt strategy (Exp2 best)
    parser.add_argument("--attack_prompt", type=str, default=DEFAULT_ARGS["attack_prompt"],
                        help="Jailbreak prompt strategy from Experiment 2 best")

    # Judge prompt dimension (Exp2 best)
    parser.add_argument("--judge_prompt", type=str, default=DEFAULT_ARGS["judge_prompt"],
                        help="Judge prompt dimension from Experiment 2 best")

    # Data
    parser.add_argument("--train_data", type=str,
                        default=os.path.join(BASE_DIR, "../data/dataset/processed/10k/train.jsonl"))
    parser.add_argument("--eval_data", type=str,
                        default=os.path.join(BASE_DIR, "../data/dataset/processed/10k/test.jsonl"))

    # Output
    parser.add_argument("--output_dir", type=str, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--run_name", type=str, default=DEFAULT_ARGS["run_name"])
    parser.add_argument("--seed", type=int, default=DEFAULT_ARGS["seed"])
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_ARGS["logging_steps"])
    parser.add_argument("--save_steps", type=int, default=DEFAULT_ARGS["save_steps"])

    # Eval config
    parser.add_argument("--run_eval_after_train", action="store_true",
                        help="Run evaluation on test set after training completes")
    parser.add_argument("--eval_policy_port", type=int, default=8003,
                        help="Port for policy model vLLM server during eval")

    return parser.parse_args()


# =============================================================================
# Global Setup
# =============================================================================
args = parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SWANLAB_PROJECT"] = "JPG_simple_rule_weight_exp"

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
logger = logging.getLogger("simple_rule_weight")


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
# Simple Rule Calculator (Global Instance)
# =============================================================================
if args.fixed_lambda is not None:
    logger.info(f"Using FIXED lambda = {args.fixed_lambda} (no adaptation)")
    adaptive_calculator = None
    lambda_val = args.fixed_lambda
else:
    simple_config = SimpleRuleConfig(
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        lambda_init=0.5,
    )
    adaptive_calculator = SimpleRuleCalculator(simple_config)
    lambda_val = adaptive_calculator.get_lambda()
    logger.info(f"Using SIMPLE RULE-BASED adaptive lambda")
    logger.info(f"  Lambda range: [{args.lambda_min}, {args.lambda_max}]")

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

    Uses Exp2 format: SCORE=0.XX output.

    Returns raw judge reward (0~1, before weighting).
    """

    # Get original prompts for comparison
    if original_prompts is None:
        original_prompts = prompts

    # Get judge template for specified dimension
    judge_template = get_judge_template(args.judge_prompt)

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

        # Parse SCORE=0.XX format
        score = 0.1  # Default fallback
        try:
            match = re.search(r"SCORE=([0-9]+\.[0-9]+)", str(resp))
            if match:
                score = float(match.group(1))
                score = max(0.0, min(1.0, score))
        except:
            pass

        raw_scores.append(score)

    return raw_scores


def asr_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    ASR Reward: Send rewritten prompt to target, then classify with guard.

    Returns raw ASR reward (0, 0.5, or 1, before weighting).
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


def simple_rule_hybrid_reward(
    prompts: List[str],
    completions: List[str],
    original_prompts: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """
    Simple Rule-Based Hybrid Reward: Combines ASR and Judge rewards.

    If fixed_lambda is set, uses constant weight.
    Otherwise, uses SimpleRuleCalculator for adaptive weighting based on variance.
    """

    # Step 1: Get raw rewards for each completion
    asr_raws = asr_reward(prompts, completions, **kwargs)
    judge_raws = judge_reward(prompts, completions, original_prompts, **kwargs)

    # Step 2: Determine lambda (weight for ASR)
    if args.fixed_lambda is not None:
        # Fixed weight mode
        lambda_val = args.fixed_lambda
    else:
        # Adaptive mode: compute variances and update lambda
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

        # Average variance across prompts in this batch
        n_prompts = len(step_var_asr_list)
        avg_var_asr = sum(step_var_asr_list) / n_prompts if n_prompts > 0 else 0.0
        avg_var_judge = sum(step_var_judge_list) / n_prompts if n_prompts > 0 else 0.0

        # Update lambda based on simple rules
        adaptive_calculator.update_step(avg_var_asr, avg_var_judge)
        lambda_val = adaptive_calculator.get_lambda()

        # Log lambda statistics
        stats = adaptive_calculator.get_statistics()
        lambda_history.append({
            "step": stats["step"],
            "lambda": stats["lambda"],
            "var_asr": stats["var_asr"],
            "var_judge": stats["var_judge"],
        })

        # Periodic logging
        if stats["step"] % 10 == 0:
            logger.info(f"[SimpleRule] step={stats['step']} λ={stats['lambda']:.3f} | "
                        f"var_asr={stats['var_asr']:.4f} var_judge={stats['var_judge']:.4f}")

    # Step 3: Compute final rewards
    final_rewards = [
        lambda_val * a + (1 - lambda_val) * j
        for a, j in zip(asr_raws, judge_raws)
    ]

    return final_rewards


# =============================================================================
# Main Training Function
# =============================================================================
def main():
    logger.info("=" * 70)
    logger.info("Simple Rule-Based Hybrid Reward GRPO Training - Experiment 3 Alternative")
    logger.info("=" * 70)
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Attack prompt: {args.attack_prompt} (Exp2 best)")
    logger.info(f"Judge dimension: {args.judge_prompt} (Exp2 best)")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("-" * 70)

    if args.fixed_lambda is not None:
        logger.info(f"Mode: FIXED WEIGHT λ={args.fixed_lambda}")
    else:
        logger.info(f"Mode: SIMPLE RULE-BASED ADAPTIVE")
        logger.info(f"  Lambda range: [{args.lambda_min}, {args.lambda_max}]")

    logger.info("=" * 70)

    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        config_dict = vars(args)
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
    run_name = args.run_name or (
        f"exp3_simple_rule_λ{args.lambda_min}-{args.lambda_max}"
        if args.fixed_lambda is None
        else f"exp3_fixed_λ{args.fixed_lambda}"
    )
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
        reward_funcs=[simple_rule_hybrid_reward],
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
        if lambda_history:
            lambda_path = os.path.join(args.output_dir, "lambda_history.json")
            with open(lambda_path, "w", encoding="utf-8") as f:
                json.dump(lambda_history, f, indent=2)
            logger.info(f"Lambda history saved: {lambda_path}")

            # Summary statistics
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

    # =================================================================
    # Post-Training Evaluation (if requested)
    # =================================================================
    if args.run_eval_after_train:
        logger.info("=" * 70)
        logger.info("Starting post-training evaluation...")
        logger.info("=" * 70)

        eval_output_dir = os.path.join(args.output_dir, "eval_results")
        os.makedirs(eval_output_dir, exist_ok=True)

        # Import eval script
        eval_script_path = os.path.join(BASE_DIR, "scripts", "eval.py")

        import subprocess
        eval_cmd = [
            sys.executable, eval_script_path,
            "--eval_path", args.eval_data,
            "--lora_paths", LORA_DIR,
            "--strategy_name", args.attack_prompt,
            "--base_model_path", args.policy_model,
            "--target_model_path", args.policy_model,  # target uses same base
            "--guard_model_path", "/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B",
            "--policy_port", str(args.eval_policy_port),
            "--target_port", str(args.target_port),
            "--guard_port", str(args.guard_port),
            "--output_root", eval_output_dir,
            "--run_name", f"eval_{run_name}",
        ]

        logger.info(f"Eval command: {' '.join(eval_cmd)}")

        try:
            result = subprocess.run(
                eval_cmd,
                cwd=BASE_DIR,
                capture_output=False,  # Stream to console
                check=True,
            )
            logger.info("Post-training evaluation completed successfully!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Post-training evaluation failed: {e}")
        except Exception as e:
            logger.error(f"Failed to run evaluation: {e}")

    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
