#!/usr/bin/env python3
"""
混合奖励 GRPO 训练脚本 - 实验2

GPU分配:
- GPU0&1: vLLM server (Qwen3-4B) - policy generation + target + judge
- GPU2: Policy训练
- GPU3: Guard server

使用vLLM server mode，推理和训练分离，效率更高。
"""

import os
import sys
import json
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
)
from experiments.jailbreak_prompt_exp.jailbreak_prompts import get_strategy_template


# =============================================================================
# 默认参数
# =============================================================================
DEFAULT_ARGS = {
    # 模型配置
    "policy_model": "/home/tiger/models/Qwen/Qwen3-4B",
    "guard_model": "/home/tiger/models/Qwen/Qwen3Guard-Gen-4B",
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,v_proj,k_proj,o_proj",

    # 训练超参数
    "learning_rate": 1e-5,
    "max_steps": 1000,
    "beta": 0.05,
    "num_generations": 8,
    "per_device_train_batch_size": 2,
    "max_completion_len": 2048,
    "gradient_accumulation_steps": 4,

    # Inference server配置 (普通vLLM, 用于Target+Judge)
    "inference_server_port": 8001,
    "guard_server_port": 8002,
    "target_max_tokens": 512,

    # vLLM colocate mode配置
    "vllm_gpu_memory_utilization": 0.4,

    # 奖励权重 (固定 1:1)
    "asr_weight": 0.5,
    "judge_weight": 0.5,

    # 实验2核心参数
    "attack_prompt": "creative_writing",
    "weight_config": "uniform",

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

    # 实验参数
    parser.add_argument("--attack_prompt", type=str, default=DEFAULT_ARGS["attack_prompt"],
                        choices=ATTACK_PROMPTS)
    parser.add_argument("--weight_config", type=str, default=DEFAULT_ARGS["weight_config"],
                        choices=WEIGHT_CONFIG_NAMES)

    # 模型 & LoRA
    parser.add_argument("--policy_model", type=str, default=DEFAULT_ARGS["policy_model"])
    parser.add_argument("--guard_model", type=str, default=DEFAULT_ARGS["guard_model"])
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

    # Inference server配置 (普通vLLM)
    parser.add_argument("--inference_server_port", type=int, default=DEFAULT_ARGS["inference_server_port"])
    parser.add_argument("--guard_server_port", type=int, default=DEFAULT_ARGS["guard_server_port"])
    parser.add_argument("--target_max_tokens", type=int, default=DEFAULT_ARGS["target_max_tokens"])

    # vLLM colocate mode配置
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=DEFAULT_ARGS["vllm_gpu_memory_utilization"])

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
# 奖励函数类
# =============================================================================
class HybridRewardCalculator:
    """混合奖励计算器，连接外部vLLM服务"""

    def __init__(self, args):
        self.args = args

        # Inference server 的 model_name (普通 vLLM serve, Qwen3-4B)
        inference_model_name = args.policy_model

        # 连接 Inference server (Target+Judge, GPU2)
        self.inference_client = VLLMClient(
            host="127.0.0.1",
            port=args.inference_server_port,
            model_name=inference_model_name,
            launch_server=False,
            timeout=300,
            temperature=0.0,
        )

        # Guard server 的 model_name (Qwen3Guard-Gen-4B)
        guard_model_name = args.guard_model

        # 连接 Guard server (GPU3)
        self.guard_client = VLLMClient(
            host="127.0.0.1",
            port=args.guard_server_port,
            model_name=guard_model_name,
            launch_server=False,
            timeout=300,
            temperature=0.0,
        )

    def judge_reward(
        self,
        prompts: List[str],
        completions: List[str],
        original_prompts: Optional[List[str]] = None,
        **kwargs
    ) -> List[float]:
        """多维度 Judge Reward"""
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

        # 批量调用judge (使用 Inference server)
        judge_responses = self.inference_client.llm_batch_call(
            prompts=judge_prompts_batch,
            temperature=0.0,
            max_tokens=256,
            max_workers=16,
            return_exceptions=True,
        )

        # 解析响应并计算加权分数
        raw_scores = []
        for i, resp in enumerate(judge_responses):
            if resp is None:
                print(f"[judge_reward] Response {i} is None")
                raw_scores.append(0.1)
                continue
            if isinstance(resp, Exception):
                print(f"[judge_reward] Response {i} is Exception: {resp}")
                raw_scores.append(0.1)
                continue
            if isinstance(resp, str) and resp.startswith("[Exception]"):
                print(f"[judge_reward] Response {i} is Exception string: {resp}")
                raw_scores.append(0.1)
                continue

            _, parsed_scores = parse_judge_response(str(resp))
            weighted_score = compute_weighted_score(parsed_scores, self.args.weight_config)
            raw_scores.append(weighted_score)

        return [s * self.args.judge_weight for s in raw_scores]

    def asr_reward(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """ASR Reward: 发送重写后的prompt到target，然后用guard分类"""
        jailbreak_prompts = [(c or "").strip() for c in completions]

        # Step 1: 获取Target响应 (使用 Inference server)
        target_responses = self.inference_client.llm_batch_call(
            prompts=jailbreak_prompts,
            temperature=0.7,
            max_tokens=self.args.target_max_tokens,
            max_workers=16,
            return_exceptions=True,
        )

        # Step 2: Guard分类 (使用 Guard server, Qwen3Guard-Gen-4B)
        guard_messages_list = []
        for jp, tr in zip(jailbreak_prompts, target_responses):
            # 处理异常响应（包括Exception对象和Exception字符串）
            if tr is None:
                tr = "Error: No response from target"
            elif isinstance(tr, Exception):
                tr = f"Error: {tr}"
            elif isinstance(tr, str) and tr.startswith("[Exception]"):
                tr = tr  # 保持原始错误信息
            guard_messages_list.append([
                {"role": "system", "content": GUARD_PROMPT},
                {"role": "user", "content": jp},
                {"role": "assistant", "content": tr},
            ])

        guard_responses = self.guard_client.llm_batch_call(
            messages_list=guard_messages_list,
            temperature=0.0,
            max_tokens=256,
            max_workers=16,
            return_exceptions=True,
        )

        # Step 3: 解析guard响应
        raw_scores = []
        for i, gr in enumerate(guard_responses):
            if gr is None:
                print(f"[asr_reward] Guard response {i} is None")
                raw_scores.append(0.0)
                continue
            if isinstance(gr, Exception):
                print(f"[asr_reward] Guard response {i} is Exception: {gr}")
                raw_scores.append(0.0)
                continue
            if isinstance(gr, str) and gr.startswith("[Exception]"):
                print(f"[asr_reward] Guard response {i} is Exception string: {gr}")
                raw_scores.append(0.0)
                continue

            label, categories, refusal = extract_label_categories_refusal(str(gr))

            if label == "Unsafe":
                raw_scores.append(1.0)
            elif label == "Controversial":
                raw_scores.append(0.5)
            else:
                raw_scores.append(0.0)

        return [s * self.args.asr_weight for s in raw_scores]


# =============================================================================
# 数据集加载
# =============================================================================
def load_train_dataset(path: str, attack_prompt: str) -> Dataset:
    """加载训练数据集，应用attack prompt模板"""
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
            prompt = attack_template.format(original_prompt=raw)
            rows.append({"prompt": prompt, "original_prompt": raw})

    ds = Dataset.from_list(rows)
    return ds


# =============================================================================
# 主函数
# =============================================================================
def main():
    args = parse_args()

    # 环境设置
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["SWANLAB_PROJECT"] = "JPG_exp2"

    # 日志设置
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    LOG_FILE = os.path.join(args.output_dir, "logs",
                            f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)],
    )
    logger = logging.getLogger("exp2_grpo")

    logger.info("=" * 70)
    logger.info("混合奖励 GRPO 训练 - 实验2")
    logger.info("=" * 70)
    logger.info(f"输出: {args.output_dir}")
    logger.info(f"训练数据: {args.train_data}")
    logger.info(f"攻击prompt: {args.attack_prompt}")
    logger.info(f"权重配置: {args.weight_config}")
    logger.info(f"权重详情: {WEIGHT_CONFIGS[args.weight_config]}")
    logger.info(f"奖励权重: ASR={args.asr_weight}, Judge={args.judge_weight}")
    logger.info(f"训练步数: {args.max_steps}")
    logger.info(f"Inference Server (Target+Judge): 127.0.0.1:{args.inference_server_port}")
    logger.info(f"Guard Server (ASR): 127.0.0.1:{args.guard_server_port}")
    logger.info(f"vLLM colocate: gpu_memory_util={args.vllm_gpu_memory_utilization}")
    logger.info("=" * 70)

    # 保存配置
    config_path = os.path.join(args.output_dir, "config.json")
    config_dict = vars(args)
    config_dict["weight_config_detail"] = WEIGHT_CONFIGS[args.weight_config]
    config_dict["dimension_names"] = DIMENSION_NAMES
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    # 设置seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 加载模型
    logger.info("Loading policy model...")
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 多 GPU 训练时，让 accelerate/trainer 自动处理设备分配
    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    if num_gpus > 1:
        policy = AutoModelForCausalLM.from_pretrained(
            args.policy_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
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

    # 加载数据集
    train_ds = load_train_dataset(args.train_data, args.attack_prompt)
    logger.info(f"Loaded {len(train_ds)} training samples")

    # 奖励计算器
    reward_calc = HybridRewardCalculator(args)
    logger.info("Connected to Inference server and Guard server for reward calculation")

    # GRPO Config - 使用 colocate mode (vLLM和训练共置)
    run_name = args.run_name or f"exp2_{args.attack_prompt}_{args.weight_config}"

    # 检测GPU数量用于vLLM tensor parallel
    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))

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
        vllm_mode="colocate",  # 共置模式
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=num_gpus,
    )

    # Trainer
    trainer = GRPOTrainer(
        model=policy,
        args=grpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=[reward_calc.asr_reward, reward_calc.judge_reward],
    )

    # 训练
    logger.info("Training start...")
    try:
        trainer.train()
        logger.info("Training done.")

        # 保存LoRA
        lora_dir = os.path.join(args.output_dir, "final_lora")
        logger.info(f"Saving LoRA to: {lora_dir}")
        os.makedirs(lora_dir, exist_ok=True)
        trainer.save_model(lora_dir)
        tokenizer.save_pretrained(lora_dir)
        logger.info("Saved.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()
    logger.info("=" * 70)
    logger.info("Experiment 2 training complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()