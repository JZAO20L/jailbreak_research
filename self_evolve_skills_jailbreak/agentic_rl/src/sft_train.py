"""
Phase 1: RFT (Rejection Sampling Fine-Tuning) 训练

基于 RFT 轨迹数据，训练 skill-conditioned 攻击模型。
RFT 轨迹由 trajectory_generator.py 生成：基础模型生成 N 个候选，
多维度评估后过滤 ASR=1.0 的样本。

使用 TRL SFTTrainer + LoRA。

数据格式（JSONL）：
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}], ...}

Usage:
    accelerate launch --config_file configs/accelerate_2gpu.yaml sft_train.py \
        --train_data_path output/rft_train.jsonl \
        --eval_data_path output/rft_val.jsonl \
        --output_dir output/rft_checkpoint
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Agentic SFT Training")

    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str,
                        default="/home/tiger/models/Qwen/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--packing", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def format_for_sft(example):
    """
    将 ChatML messages 转换为 SFT 训练格式。

    SFTTrainer 需要 "text" 字段（完整对话文本）或 "prompt"+"completion" 字段。
    我们用 "messages" 字段让 SFTTrainer 自动处理 chat template。
    """
    return {"messages": example["messages"]}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    logger.info(f"Loading training data from {args.train_data_path}")
    train_dataset = load_dataset("json", data_files=args.train_data_path, split="train")
    train_dataset = train_dataset.map(format_for_sft)
    logger.info(f"Train samples: {len(train_dataset)}")

    eval_dataset = None
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        eval_dataset = load_dataset("json", data_files=args.eval_data_path, split="train")
        eval_dataset = eval_dataset.map(format_for_sft)
        logger.info(f"Eval samples: {len(eval_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        gradient_checkpointing=True,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        completion_only_loss=True,
        report_to="swanlab",
        run_name="agentic_sft",
        model_init_kwargs={"torch_dtype": torch.bfloat16},
    )

    logger.info(f"SFT config: epochs={args.num_epochs}, lr={args.learning_rate}, "
                f"lora_r={args.lora_r}, max_seq_len={args.max_seq_length}")

    trainer = SFTTrainer(
        model=args.model_name_or_path,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_dir = output_dir / "final_lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    logger.info(f"SFT training complete. LoRA saved to {lora_dir}")


if __name__ == "__main__":
    main()
