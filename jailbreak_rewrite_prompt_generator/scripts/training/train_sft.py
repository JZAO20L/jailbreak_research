#!/usr/bin/env python3
"""
SFT 训练脚本 - Rewrite Prompt Generator 基础能力学习

使用 TRL SFTTrainer 进行训练，支持多 GPU 并发训练。

数据格式：
- 输入：seed_prompt (原始有害 prompt)
- 输出：rewrite_prompt (改写指令)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


def parse_args():
    parser = argparse.ArgumentParser(description="SFT 训练 Rewrite Prompt Generator")

    # 数据参数
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="训练数据路径（三元组数据）")
    parser.add_argument("--eval_data_path", type=str, default=None,
                        help="评估数据路径（可选）")
    parser.add_argument("--data_size", type=str, default="small",
                        choices=["small", "medium", "large", "custom"],
                        help="数据量配置")
    parser.add_argument("--custom_data_size", type=int, default=None,
                        help="自定义数据量（仅当 data_size='custom' 时有效）")

    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B",
                        help="基础模型名称或路径")
    parser.add_argument("--use_peft", action="store_true",
                        help="是否使用 PEFT (LoRA)")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="models/sft",
                        help="模型输出目录")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练 epoch 数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="每 GPU batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--packing", action="store_true",
                        help="是否使用 packing")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="是否使用 gradient checkpointing")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="是否使用 bf16")

    # 其他参数
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="模型保存步数")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="最多保存模型数量")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    return parser.parse_args()


def load_and_prepare_data(args):
    """加载并准备训练数据"""

    # 加载三元组数据
    dataset = load_dataset("json", data_files=args.train_data_path, split="train")

    # 根据数据量配置选择数据
    data_sizes = {
        "small": 1000,
        "medium": 1500,
        "large": 2000,
        "custom": args.custom_data_size or 1000
    }

    target_size = data_sizes[args.data_size]
    if len(dataset) > target_size:
        # 选择高质量数据（这里简化处理，实际应该有质量评估）
        dataset = dataset.select(range(target_size))

    print(f"加载训练数据: {len(dataset)} 条")
    print(f"数据格式示例:")
    print(dataset[0])

    return dataset


def format_dataset(example):
    """格式化数据为 SFT 格式"""

    # 原始数据格式: (seed_prompt, rewrite_prompt, final_prompt)
    # SFT 格式: prompt-completion format
    # prompt: seed_prompt
    # completion: rewrite_prompt

    return {
        "prompt": example["seed_prompt"],
        "completion": example["rewrite_prompt"]
    }


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 加载数据
    train_dataset = load_and_prepare_data(args)

    # 格式化数据
    train_dataset = train_dataset.map(format_dataset)

    # PEFT 配置
    peft_config = None
    if args.use_peft:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        print(f"使用 LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    # SFT 配置
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        # completion_only_loss=True (默认对 prompt-completion 数据为 True)
        completion_only_loss=True,
        # 模型初始化参数
        model_init_kwargs={
            "torch_dtype": torch.bfloat16 if args.bf16 else torch.float32,
        }
    )

    print(f"\nSFT 配置:")
    print(f"  - 训练数据量: {len(train_dataset)}")
    print(f"  - Epochs: {args.num_epochs}")
    print(f"  - Batch size: {args.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Max sequence length: {args.max_seq_length}")
    print(f"  - 使用 PEFT: {args.use_peft}")
    print(f"  - 使用 BF16: {args.bf16}")
    print(f"  - 输出目录: {args.output_dir}")

    # 创建 SFT Trainer
    trainer = SFTTrainer(
        model=args.model_name_or_path,
        args=sft_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    # 开始训练
    print(f"\n开始 SFT 训练...")
    trainer.train()

    # 保存最终模型
    print(f"\n保存模型到 {args.output_dir}")
    trainer.save_model(args.output_dir)

    # 保存 tokenizer
    trainer.processing_class.save_pretrained(args.output_dir)

    print(f"\nSFT 训练完成！")


if __name__ == "__main__":
    main()