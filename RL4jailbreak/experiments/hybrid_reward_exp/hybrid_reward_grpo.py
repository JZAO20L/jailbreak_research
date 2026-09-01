#!/usr/bin/env python3
"""
混合奖励 GRPO 训练脚本 - 实验2 (重做版本)

根据 TODO.md 实验重做 - 实验2:
- 使用单一多维度judge prompt，JSON格式输出
- 维度: intent_preservation, stealth, strategy_execution, attack_potential
- 权重配置: 4单一维度 + 1均匀维度 = 5种
- 攻击prompt: 3种 (来自实验1 top3)
- ASR reward 和 Judge reward 1:1混合
- 每个实验训练1000步
- 总实验数: 3攻击prompt × 5权重配置 = 15个

使用方式:
    # 默认配置 (creative_writing + uniform权重)
    python hybrid_reward_grpo.py

    # 指定攻击prompt和权重配置
    python hybrid_reward_grpo.py --attack_prompt hypothetical_scenario --weight_config intent_only

    # 其他参数
    python hybrid_reward_grpo.py --attack_prompt role_playing --weight_config stealth_only --max_steps 1000
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

import torch
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.vllm_client import VLLMClient
from src.utils import extract_label_categories_refusal
from src.prompts import GUARD_PROMPT
from experiments.hybrid_reward_exp.judge_prompts import (
    JUDGE_MULTI_DIMENSION,
    DIMENSION_NAMES,
    WEIGHT_CONFIGS,
    WEIGHT_CONFIG_NAMES,
    ATTACK_PROMPTS,
    parse_judge_response,
    compute_weighted_score,
    get_attack_template,
)
from experiments.jailbreak_prompt_exp.jailbreak_prompts import get_strategy_template

# =============================================================================
# 默认参数 (根据 TODO.md)
# =============================================================================
DEFAULT_ARGS = {
    # 模型配置
    "policy_model": "/home/tiger/models/Qwen/Qwen3-4B",
    "target_model": "/home/tiger/models/Qwen/Qwen3-4B",  # Judge复用target
    "guard_model": "/home/tiger/models/Qwen/Qwen3Guard-Gen-4B",
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,v_proj,k_proj,o_proj",

    # 训练超参数 (根据 TODO.md)
    "learning_rate": 1e-5,
    "max_steps": 1000,           # 固定1000步
    "beta": 0.05,
    "num_generations": 8,
    "per_device_train_batch_size": 2,
    "max_completion_len": 2048,
    "gradient_accumulation_steps": 4,
    "vllm_max_model_len": 4096,  # policy使用4k

    # 奖励权重 (固定 1:1)
    "asr_weight": 0.5,
    "judge_weight": 0.5,

    # 端口
    "target_judge_port": 8001,   # target和judge共用
    "guard_port": 8002,
    "target_max_tokens": 512,

    # 实验2核心参数
    "attack_prompt": "creative_writing",      # 默认top1策略
    "weight_config": "uniform",               # 默认均匀权重

    # 输出
    "output_dir": "experiments/hybrid_reward_exp/output",
    "run_name": None,
    "seed": 42,
    "logging_steps": 1,
    "save_steps": 100,
}


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="混合奖励 GRPO 训练脚本 - 实验2")

    # 实验参数 (核心)
    parser.add_argument("--attack_prompt", type=str, default=DEFAULT_ARGS["attack_prompt"],
                        choices=ATTACK_PROMPTS,
                        help=f"攻击prompt策略 (来自实验1 top3): {ATTACK_PROMPTS}")
    parser.add_argument("--weight_config", type=str, default=DEFAULT_ARGS["weight_config"],
                        choices=WEIGHT_CONFIG_NAMES,
                        help=f"Judge维度权重配置: {WEIGHT_CONFIG_NAMES}")

    # 模型 & LoRA
    parser.add_argument("--policy_model", type=str, default=DEFAULT_ARGS["policy_model"])
    parser.add_argument("--lora_r", type=int, default=DEFAULT_ARGS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_ARGS["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_ARGS["lora_dropout"])
    parser.add_argument("--lora_target_modules", type=str, default=DEFAULT_ARGS["lora_target_modules"])

    # 训练超参数
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_ARGS["learning_rate"])
    parser.add_argument("--max_steps", type=int, default=DEFAULT_ARGS["max_steps"])
    parser.add_argument("--beta", type=float, default=DEFAULT_ARGS["beta"])
    parser.add_argument("--num_generations", type=int, default=DEFAULT_ARGS["num_generations"])
    parser.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_ARGS["per_device_train_batch_size"])
    parser.add_argument("--max_completion_len", type=int, default=DEFAULT_ARGS["max_completion_len"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_ARGS["gradient_accumulation_steps"])
    parser.add_argument("--vllm_max_model_len", type=int, default=DEFAULT_ARGS["vllm_max_model_len"])

    # 端口
    parser.add_argument("--target_judge_port", type=int, default=DEFAULT_ARGS["target_judge_port"])
    parser.add_argument("--guard_port", type=int, default=DEFAULT_ARGS["guard_port"])
    parser.add_argument("--target_max_tokens", type=int, default=DEFAULT_ARGS["target_max_tokens"])

    # 奖励权重
    parser.add_argument("--asr_weight", type=float, default=DEFAULT_ARGS["asr_weight"])
    parser.add_argument("--judge_weight", type=float, default=DEFAULT_ARGS["judge_weight"])

    # 数据集
    parser.add_argument("--train_data", type=str,
                        default=os.path.join(BASE_DIR, "../data/dataset/processed/10k/train.jsonl"))

    # 输出
    parser.add_argument("--output_dir", type=str, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--run_name", type=str, default=DEFAULT_ARGS["run_name"])
    parser.add_argument("--seed", type=int, default=DEFAULT_ARGS["seed"])
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_ARGS["logging_steps"])
    parser.add_argument("--save_steps", type=int, default=DEFAULT_ARGS["save_steps"])

    return parser.parse_args()


# =============================================================================
# 全局设置
# =============================================================================
args = parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["SWANLAB_PROJECT"] = "JPG_exp2_judge_prompt_new"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["WANDB_DISABLED"] = "true"  # 禁用wandb避免protobuf冲突

LORA_DIR = os.path.join(args.output_dir, "final_lora")


# =============================================================================
# 日志设置
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
logger = logging.getLogger("exp2_hybrid_reward")


# =============================================================================
# 连接已有vLLM服务
# =============================================================================
logger.info("Connecting to existing vLLM servers...")

TARGET_JUDGE_CLIENT = VLLMClient(
    host="127.0.0.1",
    port=args.target_judge_port,
    launch_server=False,
    timeout=300,  # 增加到300秒
    temperature=0.0,
)

GUARD_CLIENT = VLLMClient(
    host="127.0.0.1",
    port=args.guard_port,
    launch_server=False,
    timeout=300,  # 增加到300秒
    temperature=0.0,
)

logger.info(f"Target+Judge connected: port {args.target_judge_port}")
logger.info(f"Guard connected: port {args.guard_port}")


# =============================================================================
# 数据集
# =============================================================================
def load_train_dataset(path: str, attack_prompt: str) -> Dataset:
    """加载训练数据集，应用attack prompt模板"""
    # 获取attack prompt模板
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
            # 应用attack prompt模板
            prompt = attack_template.format(original_prompt=raw)
            rows.append({"prompt": prompt, "original_prompt": raw})

    ds = Dataset.from_list(rows)
    logger.info(f"Loaded {len(ds)} training samples from {path}")
    logger.info(f"Attack prompt strategy: {attack_prompt}")
    return ds


# =============================================================================
# 奖励函数
# =============================================================================
def judge_reward_multi_dim(
    prompts: List[str],
    completions: List[str],
    original_prompts: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """
    多维度 Judge Reward.

    使用单一judge prompt评估4个维度，返回加权平均分数。
    权重配置由 --weight_config 参数决定。

    Returns:
        List[float]: 加权后的judge reward (乘以 judge_weight)
    """
    # 获取原始prompt用于对比
    if original_prompts is None:
        original_prompts = prompts

    # 构建judge prompts
    judge_prompts_batch = []
    for orig, rewritten in zip(original_prompts, completions):
        rewritten = (rewritten or "").strip()
        judge_prompts_batch.append(
            JUDGE_MULTI_DIMENSION.format(
                original_prompt=orig,
                rewritten_prompt=rewritten
            )
        )

    # 批量调用judge (使用target作为judge模型)
    judge_responses = TARGET_JUDGE_CLIENT.llm_batch_call(
        prompts=judge_prompts_batch,
        temperature=0.0,
        max_tokens=256,
        max_workers=16,  # 增加并发数
        return_exceptions=True,
    )

    # 解析响应并计算加权分数
    raw_scores = []
    for resp in judge_responses:
        if resp is None or isinstance(resp, Exception):
            raw_scores.append(0.1)  # 保底较低分
            continue

        # 解析JSON响应
        _, parsed_scores = parse_judge_response(str(resp))

        # 根据weight_config计算加权分数
        weighted_score = compute_weighted_score(parsed_scores, args.weight_config)
        raw_scores.append(weighted_score)

    # 应用judge_weight
    return [s * args.judge_weight for s in raw_scores]


def asr_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    ASR Reward: 发送重写后的prompt到target，然后用guard分类。

    Returns:
        List[float]: ASR reward (乘以 asr_weight)
    """
    jailbreak_prompts = [(c or "").strip() for c in completions]

    # Step 1: 获取Target响应
    target_responses = TARGET_JUDGE_CLIENT.llm_batch_call(
        prompts=jailbreak_prompts,
        temperature=0.7,
        max_tokens=args.target_max_tokens,
        max_workers=16,  # 增加并发数
        return_exceptions=True,
    )

    # Step 2: Guard分类
    guard_messages_list = []
    for jp, tr in zip(jailbreak_prompts, target_responses):
        if tr is None or isinstance(tr, Exception):
            tr = "Error: No response from target"
        guard_messages_list.append([
            {"role": "system", "content": GUARD_PROMPT},
            {"role": "user", "content": jp},
            {"role": "assistant", "content": tr},
        ])

    guard_responses = GUARD_CLIENT.llm_batch_call(
        messages_list=guard_messages_list,
        temperature=0.0,
        max_tokens=256,
        max_workers=16,  # 增加并发数
        return_exceptions=True,
    )

    # Step 3: 解析guard响应
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

    # 应用asr_weight
    return [s * args.asr_weight for s in raw_scores]


# =============================================================================
# 训练
# =============================================================================
def main():
    logger.info("=" * 70)
    logger.info("混合奖励 GRPO 训练 - 实验2 (重做版本)")
    logger.info("=" * 70)
    logger.info(f"输出: {args.output_dir}")
    logger.info(f"训练数据: {args.train_data}")
    logger.info(f"攻击prompt: {args.attack_prompt}")
    logger.info(f"权重配置: {args.weight_config}")
    logger.info(f"权重详情: {WEIGHT_CONFIGS[args.weight_config]}")
    logger.info(f"奖励权重: ASR={args.asr_weight}, Judge={args.judge_weight}")
    logger.info(f"训练步数: {args.max_steps}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info("=" * 70)

    # 保存配置
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        config_dict = vars(args)
        config_dict["weight_config_detail"] = WEIGHT_CONFIGS[args.weight_config]
        config_dict["dimension_names"] = DIMENSION_NAMES
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Config saved to {config_path}")

    # 设置seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 加载模型 - 使用accelerate分布式时不需要手动指定device_map
    logger.info("Loading policy model...")
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 多GPU训练时，让accelerate处理设备分配
    # 单GPU时使用cuda:0
    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))

    if num_gpus > 1:
        # 分布式训练：不使用device_map，让accelerate自动处理设备分配
        policy = AutoModelForCausalLM.from_pretrained(
            args.policy_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        # 单GPU训练
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

    # 加载数据集 (应用attack prompt模板)
    train_ds = load_train_dataset(args.train_data, args.attack_prompt)

    # GRPO Config
    run_name = args.run_name or f"exp2_{args.attack_prompt}_{args.weight_config}"

    # 根据GPU数量设置vllm_tensor_parallel_size
    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    vllm_tp_size = num_gpus if num_gpus > 1 else 1

    logger.info(f"Number of GPUs: {num_gpus}, vLLM tensor_parallel_size: {vllm_tp_size}")

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
        vllm_tensor_parallel_size=vllm_tp_size,  # 多GPU时使用tensor parallel
        ddp_find_unused_parameters=False,  # 禁用未使用参数检测以提升性能
    )

    # Trainer (两个reward函数，trl会自动加权)
    trainer = GRPOTrainer(
        model=policy,
        args=grpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=[asr_reward, judge_reward_multi_dim],
    )

    # 训练
    logger.info("Training start...")
    try:
        trainer.train()
        logger.info("Training done.")

        # 保存LoRA
        logger.info(f"Saving LoRA to: {LORA_DIR}")
        os.makedirs(LORA_DIR, exist_ok=True)
        trainer.save_model(LORA_DIR)
        tokenizer.save_pretrained(LORA_DIR)
        logger.info("Saved.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Cleanup done.")

    # 最终清理
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Final cleanup done.")
    logger.info("=" * 70)
    logger.info("Experiment 2 training complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()