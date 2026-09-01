"""
Phase 2: GRPO 训练

基于 SFT checkpoint，使用自适应混合奖励进一步优化 skill-conditioned 攻击模型。
使用 TRL GRPOTrainer。

数据流:
  harmful prompt + skill candidates -> policy generates adapted skill
  -> adapted_skill + prompt -> Target -> Guard -> R_asr
  -> (original, adapted) -> Judge -> R_judge
  -> adaptive hybrid -> GRPO update

Usage:
    accelerate launch --config_file configs/accelerate_2gpu.yaml grpo_train.py \
        --sft_model_path output/sft_checkpoint/final_lora \
        --train_data output/sft_train.jsonl \
        --output_dir output/grpo_checkpoint
"""

import os
import sys
import json
import gc
import random
import logging
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Optional

import torch
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RL4jailbreak"))

from self_evolve_skills_jailbreak.agentic_rl.src.rewards import (
    adaptive_hybrid_reward,
    asr_reward_fn,
    judge_reward_fn,
    save_lambda_history,
    init_clients,
    NUM_GENERATIONS,
)
from self_evolve_skills_jailbreak.agentic_rl.src.utils import (
    format_user_prompt,
    SYSTEM_PROMPT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

DEFAULT_CONFIG = {
    "rft_model_path": None,
    "base_model_path": "/home/tiger/models/Qwen/Qwen3-4B",
    "train_data": None,
    "skill_library_path": None,
    "description_path": None,
    "output_dir": "output/grpo_checkpoint",

    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,

    "learning_rate": 1e-5,
    "max_steps": 500,
    "beta": 0.05,
    "num_generations": 8,
    "per_device_train_batch_size": 4,
    "max_completion_len": 1024,
    "gradient_accumulation_steps": 4,

    "vllm_gpu_memory_utilization": 0.3,

    "target_port": 8002,
    "guard_port": 8001,

    "ema_beta": 0.9,
    "seed": 42,
    "logging_steps": 1,
    "save_steps": 100,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Agentic GRPO Training")

    parser.add_argument("--rft_model_path", type=str, default=DEFAULT_CONFIG["rft_model_path"],
                        help="Path to RFT LoRA checkpoint (if None, use base model)")
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_CONFIG["base_model_path"])
    parser.add_argument("--train_data", type=str, default=DEFAULT_CONFIG["train_data"],
                        help="Training data JSONL (with skill candidates)")
    parser.add_argument("--skill_library_path", type=str, default=DEFAULT_CONFIG["skill_library_path"])
    parser.add_argument("--description_path", type=str, default=DEFAULT_CONFIG["description_path"])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])

    parser.add_argument("--lora_r", type=int, default=DEFAULT_CONFIG["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_CONFIG["lora_alpha"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--max_steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--beta", type=float, default=DEFAULT_CONFIG["beta"])
    parser.add_argument("--num_generations", type=int, default=DEFAULT_CONFIG["num_generations"])
    parser.add_argument("--per_device_train_batch_size", type=int,
                        default=DEFAULT_CONFIG["per_device_train_batch_size"])
    parser.add_argument("--max_completion_len", type=int, default=DEFAULT_CONFIG["max_completion_len"])
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=DEFAULT_CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--vllm_gpu_memory_utilization", type=float,
                        default=DEFAULT_CONFIG["vllm_gpu_memory_utilization"])

    parser.add_argument("--target_port", type=int, default=DEFAULT_CONFIG["target_port"])
    parser.add_argument("--guard_port", type=int, default=DEFAULT_CONFIG["guard_port"])
    parser.add_argument("--ema_beta", type=float, default=DEFAULT_CONFIG["ema_beta"])

    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_CONFIG["logging_steps"])
    parser.add_argument("--save_steps", type=int, default=DEFAULT_CONFIG["save_steps"])

    # 数据范围参数（避免 RFT/GRPO 数据泄露）
    parser.add_argument("--data_start", type=int, default=4000,
                        help="Skip first N samples (default: 4000, after RFT data)")
    parser.add_argument("--data_end", type=int, default=8000,
                        help="Stop at this index (default: 8000)")

    return parser.parse_args()


# =============================================================================
# Dataset
# =============================================================================

def build_grpo_dataset(
    train_data_path: str,
    skill_library_path: str,
    description_path: str,
    top_k: int = 3,
    data_start: int = 4000,
    data_end: int = 8000,
) -> Dataset:
    """
    构建 GRPO 训练数据集。

    每条数据包含:
    - prompt: 完整的 user prompt（包含 Harmful Prompt + Candidate Strategies）
    - original_prompt: 原始有害 prompt（用于 reward 计算）

    如果 train_data_path 提供了 SFT 格式的数据，从中提取。
    否则从原始 harmful prompts + skill library 构建。

    Args:
        data_start: 跳过前 N 条数据（避免与 RFT 数据重叠）
        data_end: 数据结束索引
    """
    from self_evolve_skills_jailbreak.src.skill_library import SkillLibrary

    skill_library = SkillLibrary(
        storage_path="/tmp/grpo_skills.json",
        max_skills=200,
    )
    skill_library.load_from_file(skill_library_path)

    with open(description_path, "r", encoding="utf-8") as f:
        descriptions = json.load(f)

    rows = []

    if train_data_path and os.path.exists(train_data_path):
        with open(train_data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < data_start:
                    continue
                if data_end is not None and i >= data_end:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line)
                original_prompt = obj.get("original_prompt", "")
                if not original_prompt:
                    messages = obj.get("messages", [])
                    for msg in messages:
                        if msg["role"] == "user":
                            import re
                            match = re.search(
                                r'## Harmful Prompt\s*\n(.+?)(?=\n\n## |\Z)',
                                msg["content"], re.DOTALL
                            )
                            if match:
                                original_prompt = match.group(1).strip()
                            break

                if not original_prompt:
                    continue

                candidates = skill_library.retrieve(original_prompt, top_k=top_k)
                candidate_info = [
                    {"name": c.name, "description": descriptions.get(c.name, f"攻击策略: {c.name}")}
                    for c in candidates
                ]
                user_prompt = format_user_prompt(original_prompt, candidate_info)

                rows.append({
                    "prompt": user_prompt,
                    "original_prompt": original_prompt,
                })
    else:
        default_data = PROJECT_ROOT / "data" / "dataset" / "processed" / "10k" / "train.jsonl"
        with open(default_data, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < data_start:
                    continue
                if data_end is not None and i >= data_end:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line)
                original_prompt = obj.get("prompt", "").strip()
                if not original_prompt:
                    continue

                candidates = skill_library.retrieve(original_prompt, top_k=top_k)
                candidate_info = [
                    {"name": c.name, "description": descriptions.get(c.name, f"攻击策略: {c.name}")}
                    for c in candidates
                ]
                user_prompt = format_user_prompt(original_prompt, candidate_info)

                rows.append({
                    "prompt": user_prompt,
                    "original_prompt": original_prompt,
                })

    ds = Dataset.from_list(rows)
    logger.info(f"Built GRPO dataset: {len(ds)} samples (range [{data_start}:{data_end}])")
    return ds


# =============================================================================
# Reward Wrapper
# =============================================================================

def make_reward_fn(num_generations: int):
    """创建适配 TRL GRPOTrainer 签名的 reward function。"""

    def reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        return adaptive_hybrid_reward(
            completions, prompts,
            num_generations=num_generations,
            **kwargs,
        )

    return reward_fn


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = vars(args)
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Config saved to {config_path}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    init_clients(target_port=args.target_port, guard_port=args.guard_port)

    base_model_path = args.base_model_path
    if args.rft_model_path and os.path.exists(args.rft_model_path):
        base_model_path = args.base_model_path
        logger.info(f"Loading from base model + RFT LoRA: {args.rft_model_path}")
        load_rft = True
    else:
        logger.info(f"No RFT checkpoint, training from base model: {base_model_path}")
        load_rft = False

    logger.info("Loading policy model...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))

    policy = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if load_rft:
        from peft import PeftModel
        policy = PeftModel.from_pretrained(policy, args.rft_model_path)
        logger.info(f"Loaded RFT LoRA from {args.rft_model_path}")

    try:
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False
    except Exception:
        pass

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    if not load_rft:
        policy = get_peft_model(policy, lora_config)
    policy.print_trainable_parameters()

    train_ds = build_grpo_dataset(
        args.train_data,
        args.skill_library_path,
        args.description_path,
        data_start=args.data_start,
        data_end=args.data_end,
    )

    reward_fn = make_reward_fn(args.num_generations)

    run_name = f"agentic_grpo_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

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
        vllm_tensor_parallel_size=num_gpus,
    )

    trainer = GRPOTrainer(
        model=policy,
        args=grpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
    )

    logger.info("=" * 70)
    logger.info("Agentic GRPO Training Start")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Num generations: {args.num_generations}")
    logger.info(f"  Max completion len: {args.max_completion_len}")
    logger.info(f"  EMA beta: {args.ema_beta}")
    logger.info("=" * 70)

    try:
        trainer.train()
        logger.info("Training done.")

        lora_dir = output_dir / "final_lora"
        lora_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(lora_dir))
        tokenizer.save_pretrained(str(lora_dir))
        logger.info(f"LoRA saved to {lora_dir}")

        save_lambda_history(str(output_dir / "lambda_history.json"))

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Agentic GRPO Training Complete!")


if __name__ == "__main__":
    main()
