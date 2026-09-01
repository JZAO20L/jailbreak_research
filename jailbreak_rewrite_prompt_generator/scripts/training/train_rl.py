#!/usr/bin/env python3
"""
RL 训练脚本 - 使用 GRPO + 自适应权重奖励建模

资源分配方案（4 GPU）：
- 方案1 (推荐): 2 GPU 训练 + 1 GPU vLLM server + 1 GPU Guard/Target
- 方案2: 4 GPU 训练 + colocate mode（vLLM 共享训练 GPU）

奖励建模：
- 稠密奖励：与 GT rewrite prompt 的相似度
- 稀疏奖励：最终攻击成功率 (ASR)
- 自适应权重动态调整
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="RL 训练 Rewrite Prompt Generator")

    # 数据参数
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="RL 训练数据路径")
    parser.add_argument("--eval_data_path", type=str, default=None,
                        help="评估数据路径")
    parser.add_argument("--data_size", type=int, default=2000,
                        help="RL 训练数据量")

    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="基础模型路径（通常是 SFT 训练后的模型）")
    parser.add_argument("--sft_model_path", type=str, default=None,
                        help="SFT 模型路径（作为起点）")

    # vLLM 参数
    parser.add_argument("--use_vllm", action="store_true", default=True,
                        help="是否使用 vLLM 加速生成")
    parser.add_argument("--vllm_mode", type=str, default="server",
                        choices=["colocate", "server"],
                        help="vLLM 模式")
    parser.add_argument("--vllm_server_host", type=str, default="localhost",
                        help="vLLM server 地址")
    parser.add_argument("--vllm_server_port", type=int, default=8000,
                        help="vLLM server 端口")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.7,
                        help="vLLM GPU 内存利用率")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM tensor parallel size")

    # GRPO 参数
    parser.add_argument("--output_dir", type=str, default="models/rl",
                        help="模型输出目录")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="训练 epoch 数（RL 通常只训练 1 epoch）")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="每 GPU batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率")
    parser.add_argument("--max_completion_length", type=int, default=256,
                        help="最大生成长度")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="每个 prompt 生成的数量")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="KL divergence 权重（DeepSeekMath 推荐 0.0）")
    parser.add_argument("--scale_rewards", type=str, default="batch",
                        choices=["none", "batch", "group"],
                        help="奖励缩放策略")

    # Reward 建模参数
    parser.add_argument("--reward_weights_similarity", type=float, default=0.5,
                        help="相似度奖励初始权重")
    parser.add_argument("--reward_weights_asr", type=float, default=0.5,
                        help="ASR 奖励初始权重")
    parser.add_argument("--adaptive_reward", action="store_true", default=True,
                        help="是否使用自适应权重")

    # Guard/Target 部署参数
    parser.add_argument("--guard_url", type=str, default=None,
                        help="Guard 模型 API 地址")
    parser.add_argument("--target_url", type=str, default=None,
                        help="Target 模型 API 地址")
    parser.add_argument("--executor_model", type=str, default=None,
                        help="Executor 模型路径（用于执行 rewrite prompt）")

    # 其他参数
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="模型保存步数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    return parser.parse_args()


def calculate_similarity_reward(completions: List[str], ground_truths: List[str], **kwargs) -> List[float]:
    """
    计算稠密奖励：生成的 rewrite prompt 与 GT 的相似度

    使用简单的字符级相似度（实际应用中可以使用更好的相似度计算方法）
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        # 简化的相似度计算（实际应该使用 embedding-based similarity）
        # 这里使用 Jaccard 相似度作为示例
        completion_tokens = set(completion.lower().split())
        gt_tokens = set(gt.lower().split())

        if not completion_tokens or not gt_tokens:
            rewards.append(0.0)
        else:
            intersection = len(completion_tokens & gt_tokens)
            union = len(completion_tokens | gt_tokens)
            similarity = intersection / union if union > 0 else 0.0
            rewards.append(similarity)

    return rewards


def calculate_asr_reward(prompts: List[str], completions: List[str],
                         guard_url: str = None, target_url: str = None,
                         executor_model: str = None, **kwargs) -> List[float]:
    """
    计算稀疏奖励：最终攻击成功率

    这个奖励需要：
    1. 用生成的 rewrite prompt 和 seed prompt 生成新的 jailbreak prompt
    2. 测试新 prompt 的 ASR

    这是一个异步函数，因为它需要调用外部 API
    """
    # 这里简化处理，实际应该：
    # 1. 调用 executor 模型生成新 prompt
    # 2. 调用 target + guard 测试 ASR

    # 简化版本：随机返回奖励（实际应该实现完整流程）
    rewards = [np.random.random() for _ in completions]
    return rewards


class AdaptiveRewardWeights:
    """自适应权重奖励机制"""

    def __init__(self, initial_weights: Dict[str, float]):
        self.weights = initial_weights
        self.step = 0

    def update(self, step: int):
        """根据训练步骤动态调整权重"""
        self.step = step

        # 简化的自适应策略：
        # 初期：similarity 权重较高（学习基础策略）
        # 后期：asr 权重提升（优化实际效果）

        if step < 500:
            # 初期阶段
            self.weights["similarity"] = 0.7
            self.weights["asr"] = 0.3
        elif step < 1000:
            # 中期阶段
            self.weights["similarity"] = 0.5
            self.weights["asr"] = 0.5
        else:
            # 后期阶段
            self.weights["similarity"] = 0.3
            self.weights["asr"] = 0.7

    def get_weights(self) -> Dict[str, float]:
        return self.weights


def create_combined_reward_func(args):
    """创建组合奖励函数（相似度 + ASR）"""

    adaptive_weights = None
    if args.adaptive_reward:
        adaptive_weights = AdaptiveRewardWeights({
            "similarity": args.reward_weights_similarity,
            "asr": args.reward_weights_asr
        })

    def combined_reward_func(prompts, completions, ground_truth=None,
                             log_metric=None, log_extra=None, **kwargs):
        """组合奖励函数"""

        # 计算相似度奖励
        similarity_rewards = calculate_similarity_reward(
            completions, ground_truth if ground_truth else prompts
        )

        # 计算 ASR 奖励
        asr_rewards = calculate_asr_reward(
            prompts, completions,
            guard_url=args.guard_url,
            target_url=args.target_url,
            executor_model=args.executor_model
        )

        # 动态调整权重
        if adaptive_weights:
            # 这里的 step 信息需要从 trainer_state 传递
            weights = adaptive_weights.get_weights()
        else:
            weights = {
                "similarity": args.reward_weights_similarity,
                "asr": args.reward_weights_asr
            }

        # 组合奖励
        combined_rewards = []
        for sim_r, asr_r in zip(similarity_rewards, asr_rewards):
            combined_r = weights["similarity"] * sim_r + weights["asr"] * asr_r
            combined_rewards.append(combined_r)

        # 日志记录
        if log_metric:
            log_metric("reward/similarity/mean", np.mean(similarity_rewards))
            log_metric("reward/asr/mean", np.mean(asr_rewards))
            log_metric("reward/combined/mean", np.mean(combined_rewards))

        if log_extra:
            log_extra("similarity_reward", similarity_rewards)
            log_extra("asr_reward", asr_rewards)

        return combined_rewards

    return combined_reward_func


def load_and_prepare_data(args):
    """加载并准备 RL 训练数据"""

    dataset = load_dataset("json", data_files=args.train_data_path, split="train")

    # 选择数据量
    if len(dataset) > args.data_size:
        dataset = dataset.select(range(args.data_size))

    print(f"加载 RL 训练数据: {len(dataset)} 条")

    return dataset


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载数据
    train_dataset = load_and_prepare_data(args)

    # 创建组合奖励函数
    reward_func = create_combined_reward_func(args)

    # GRPO 配置
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        beta=args.beta,
        scale_rewards=args.scale_rewards,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        # vLLM 配置
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )

    print(f"\nGRPO 配置:")
    print(f"  - 训练数据量: {len(train_dataset)}")
    print(f"  - Batch size: {args.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Max completion length: {args.max_completion_length}")
    print(f"  - Num generations: {args.num_generations}")
    print(f"  - 使用 vLLM: {args.use_vllm}")
    print(f"  - vLLM 模式: {args.vllm_mode}")
    print(f"  - 自适应奖励: {args.adaptive_reward}")
    print(f"  - 输出目录: {args.output_dir}")

    # 创建 GRPO Trainer
    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        args=grpo_config,
        reward_funcs=reward_func,
        train_dataset=train_dataset,
    )

    # 开始训练
    print(f"\n开始 GRPO RL 训练...")
    trainer.train()

    # 保存最终模型
    print(f"\n保存模型到 {args.output_dir}")
    trainer.save_model(args.output_dir)

    print(f"\nRL 训练完成！")


if __name__ == "__main__":
    main()