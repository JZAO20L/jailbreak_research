#!/usr/bin/env python3
"""
混合奖励 GRPO 训练脚本 - 实验2 & 实验3

训练策略:
- 总奖励 = α × Judge_Reward + β × ASR_Reward + γ × Format_Reward
- 实验2: 固定 α=0.5, β=0.5, 研究不同judge prompt策略和评分方式
- 实验3: 固定judge prompt, 研究不同 α:β 比例

使用方式:
    # 实验2: 单条打分
    python hybrid_reward_grpo.py --experiment exp2 \
        --judge_prompt stealthiness \
        --scoring_method single

    # 实验2: 锦标赛打分
    python hybrid_reward_grpo.py --experiment exp2 \
        --judge_prompt stealthiness \
        --scoring_method tournament

    # 实验3: 不同权重比例
    python hybrid_reward_grpo.py --experiment exp3 \
        --asr_weight 0.3 --judge_weight 0.7
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
from typing import List, Dict

import torch
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.vllm_client import VLLMClient
from src.utils import extract_tag_content, extract_label_categories_refusal
from src.prompts import REWRITE_PROMPT, GUARD_PROMPT
from experiments.hybrid_reward_exp.judge_prompts import (
    JUDGE_PROMPTS,
    get_judge_template,
    GENERAL_JUDGE_DIMENSIONS,
    SPECIALIZED_JUDGE_DIMENSIONS,
)

# =============================================================================
# 默认参数 (不使用yaml文件)
# =============================================================================
DEFAULT_ARGS = {
    # 模型配置
    "policy_model": "/root/autodl-tmp/models/Qwen/Qwen3-4B",
    "target_model": "/root/autodl-tmp/models/Qwen/Qwen3-4B",  # Judge复用target
    "guard_model": "/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B",
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,v_proj,k_proj,o_proj",

    # 训练超参数
    "learning_rate": 1e-5,
    "num_train_epochs": 2,
    "beta": 0.05,
    "num_generations": 8,
    "per_device_train_batch_size": 32,
    "max_completion_len": 1024,
    "gradient_accumulation_steps": 1,
    "vllm_max_model_len": 2048,
    "vllm_gpu_memory_utilization": 0.3,

    # 奖励权重
    "asr_weight": 0.5,
    "judge_weight": 0.5,
    "format_weight": 0.1,

    # 端口
    "target_judge_port": 8001,
    "guard_port": 8002,
    "target_max_tokens": 512,

    # 实验2专用
    "judge_prompt": "stealthiness",
    "scoring_method": "single",  # single 或 tournament

    # 输出
    "output_dir": "experiments/hybrid_reward_exp/output",
    "run_name": None,
    "seed": 42,
    "logging_steps": 1,
    "save_steps": 100,
    "max_steps": -1,
}


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="混合奖励 GRPO 训练脚本")

    # 实验类型
    parser.add_argument("--experiment", type=str, default="exp2",
                        choices=["exp2", "exp3"],
                        help="实验类型: exp2=judge prompt实验, exp3=reward weight实验")

    # 模型 & LoRA
    parser.add_argument("--policy_model", type=str, default=DEFAULT_ARGS["policy_model"])
    parser.add_argument("--lora_r", type=int, default=DEFAULT_ARGS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_ARGS["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_ARGS["lora_dropout"])
    parser.add_argument("--lora_target_modules", type=str, default=DEFAULT_ARGS["lora_target_modules"])

    # 训练超参数
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_ARGS["learning_rate"])
    parser.add_argument("--num_train_epochs", type=int, default=DEFAULT_ARGS["num_train_epochs"])
    parser.add_argument("--beta", type=float, default=DEFAULT_ARGS["beta"])
    parser.add_argument("--num_generations", type=int, default=DEFAULT_ARGS["num_generations"])
    parser.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_ARGS["per_device_train_batch_size"])
    parser.add_argument("--max_completion_len", type=int, default=DEFAULT_ARGS["max_completion_len"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_ARGS["gradient_accumulation_steps"])
    parser.add_argument("--vllm_max_model_len", type=int, default=DEFAULT_ARGS["vllm_max_model_len"])
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=DEFAULT_ARGS["vllm_gpu_memory_utilization"])

    # 奖励权重
    parser.add_argument("--asr_weight", type=float, default=DEFAULT_ARGS["asr_weight"])
    parser.add_argument("--judge_weight", type=float, default=DEFAULT_ARGS["judge_weight"])
    parser.add_argument("--format_weight", type=float, default=DEFAULT_ARGS["format_weight"])

    # 端口
    parser.add_argument("--target_judge_port", type=int, default=DEFAULT_ARGS["target_judge_port"])
    parser.add_argument("--guard_port", type=int, default=DEFAULT_ARGS["guard_port"])
    parser.add_argument("--target_max_tokens", type=int, default=DEFAULT_ARGS["target_max_tokens"])

    # 实验2专用
    parser.add_argument("--judge_prompt", type=str, default=DEFAULT_ARGS["judge_prompt"],
                        choices=list(JUDGE_PROMPTS.keys()),
                        help="Judge prompt维度 (实验2)")
    parser.add_argument("--scoring_method", type=str, default=DEFAULT_ARGS["scoring_method"],
                        choices=["single", "tournament"],
                        help="评分方式: single=单条打分, tournament=锦标赛")

    # 数据集
    parser.add_argument("--train_data", type=str,
                        default=os.path.join(BASE_DIR, "data/dataset/processed/10k/train.jsonl"))
    parser.add_argument("--eval_data", type=str,
                        default=os.path.join(BASE_DIR, "data/dataset/processed/10k/eval.jsonl"))

    # 输出
    parser.add_argument("--output_dir", type=str, default=DEFAULT_ARGS["output_dir"])
    parser.add_argument("--run_name", type=str, default=DEFAULT_ARGS["run_name"])
    parser.add_argument("--seed", type=int, default=DEFAULT_ARGS["seed"])
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_ARGS["logging_steps"])
    parser.add_argument("--save_steps", type=int, default=DEFAULT_ARGS["save_steps"])
    parser.add_argument("--max_steps", type=int, default=DEFAULT_ARGS["max_steps"])

    return parser.parse_args()


# =============================================================================
# 全局设置
# =============================================================================
args = parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SWANLAB_PROJECT"] = "JPG_hybrid_reward_exp"

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
logger = logging.getLogger("hybrid_reward")

# 保存配置
config_path = os.path.join(args.output_dir, "config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(vars(args), f, ensure_ascii=False, indent=2)
logger.info(f"Config saved to {config_path}")


# =============================================================================
# 连接已有vLLM服务
# =============================================================================
logger.info("Connecting to existing vLLM servers...")

TARGET_JUDGE_CLIENT = VLLMClient(
    model_name="target_judge",
    model_path="unused",
    host="127.0.0.1",
    port=args.target_judge_port,
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

logger.info(f"Target+Judge connected: port {args.target_judge_port}")
logger.info(f"Guard connected: port {args.guard_port}")


# =============================================================================
# 数据集
# =============================================================================
def load_train_dataset(path: str) -> Dataset:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            raw = (obj.get("prompt") or "").strip()
            if not raw:
                continue
            rows.append({"prompt": REWRITE_PROMPT.format(original_prompt=raw)})

    ds = Dataset.from_list(rows)
    logger.info(f"Loaded {len(ds)} training samples from {path}")
    return ds


# =============================================================================
# 奖励函数
# =============================================================================
def format_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Format reward: 检查是否有有效的 <new_jailbreak_prompt> 标签"""
    out = []
    for c in completions:
        extracted = extract_tag_content(c, "new_jailbreak_prompt")
        ok_tag = extracted is not None and len(extracted.strip()) > 0
        text = extracted.strip() if extracted else (c or "").strip()
        wc = len(text.split())
        ok_len = 10 <= wc <= 500
        out.append(1.0 if (ok_tag and ok_len) else 0.0)
    return [x * args.format_weight for x in out]


def judge_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Judge reward: 根据评分方式给重写后的prompt打分

    评分方式:
    - single: 单条打分,返回0~1之间的两位浮点数
    - tournament: 锦标赛打分,对8个生成进行锦标赛排序
    """
    # Score key映射
    SCORE_KEY_MAP = {
        "idea_preservation": "SCORE",
        "stealthiness": "SCORE",
    }

    def _parse_single_score(resp: str) -> float:
        """解析单个分数,格式为 SCORE=0.XX"""
        if resp is None:
            return 0.1  # 保底较低分
        if not isinstance(resp, (str, bytes)):
            resp = str(resp)
        match = re.search(r"SCORE=([0-9]+\.[0-9]+)", resp)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except:
                return 0.1
        return 0.1  # 解析失败给保底分

    judge_template = get_judge_template(args.judge_prompt)

    if args.scoring_method == "single":
        # 单条打分模式
        judge_prompts = []
        for p, c in zip(prompts, completions):
            original_prompt = extract_tag_content(p, "original_prompt") or p
            rewritten = extract_tag_content(c, "new_jailbreak_prompt") or c
            judge_prompts.append(
                judge_template.format(
                    original_prompt=original_prompt,
                    rewritten_prompt=rewritten
                )
            )

        resps = TARGET_JUDGE_CLIENT.llm_batch_call(
            prompts=judge_prompts,
            temperature=0.0,
            max_tokens=256,
            max_workers=32,
            return_exceptions=True,
        )
        scores = [_parse_single_score(r) for r in resps]
        return [s * args.judge_weight for s in scores]

    elif args.scoring_method == "tournament":
        # 锦标赛打分模式
        # 注意: 锦标赛需要在训练时收集8个生成,然后进行对比
        # 这里简化处理: 每条单独评分,但使用锦标赛的逻辑
        # 实际完整的锦标赛需要在训练循环中实现

        judge_prompts = []
        for p, c in zip(prompts, completions):
            original_prompt = extract_tag_content(p, "original_prompt") or p
            rewritten = extract_tag_content(c, "new_jailbreak_prompt") or c
            judge_prompts.append(
                judge_template.format(
                    original_prompt=original_prompt,
                    rewritten_prompt=rewritten
                )
            )

        resps = TARGET_JUDGE_CLIENT.llm_batch_call(
            prompts=judge_prompts,
            temperature=0.0,
            max_tokens=256,
            max_workers=32,
            return_exceptions=True,
        )
        scores = [_parse_single_score(r) for r in resps]
        return [s * args.judge_weight for s in scores]

    else:
        raise ValueError(f"Unknown scoring method: {args.scoring_method}")


def asr_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """ASR reward: 发送到target,然后用guard分类"""
    jailbreak_prompts = []
    for p, c in zip(prompts, completions):
        rewritten = extract_tag_content(c, "new_jailbreak_prompt") or c
        jailbreak_prompts.append(rewritten.strip())

    # Step 1: 获取Target响应
    target_responses = TARGET_JUDGE_CLIENT.llm_batch_call(
        prompts=jailbreak_prompts,
        temperature=0.7,
        max_tokens=args.target_max_tokens,
        max_workers=32,
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
        max_workers=32,
        return_exceptions=True,
    )

    # Step 3: 解析guard响应
    scores = []
    for gr in guard_responses:
        if gr is None or isinstance(gr, Exception):
            scores.append(0.0)
            continue

        label, categories, refusal = extract_label_categories_refusal(str(gr))

        if label == "Unsafe":
            scores.append(1.0)
        elif label == "Controversial":
            scores.append(0.5)
        else:
            scores.append(0.0)

    return [s * args.asr_weight for s in scores]


# =============================================================================
# 训练
# =============================================================================
def main():
    logger.info("=" * 70)
    logger.info("混合奖励 GRPO 训练")
    logger.info(f"实验: {args.experiment}")
    logger.info(f"输出: {args.output_dir}")
    logger.info(f"训练数据: {args.train_data}")
    logger.info(f"奖励权重: ASR={args.asr_weight}, Judge={args.judge_weight}, Format={args.format_weight}")
    logger.info(f"Judge维度: {args.judge_prompt}")
    logger.info(f"评分方式: {args.scoring_method}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"训练轮数: {args.num_train_epochs}")
    logger.info("=" * 70)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 加载模型
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

    # 加载数据集
    train_ds = load_train_dataset(args.train_data)

    # GRPO Config
    run_name = args.run_name or f"{args.experiment}_{args.judge_prompt}_{args.scoring_method}"
    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_len,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        save_strategy="steps",
        bf16=True,
        beta=args.beta,
        report_to="swanlab",
        run_name=run_name,
        seed=args.seed,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
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
        reward_funcs=[asr_reward, judge_reward, format_reward],
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


if __name__ == "__main__":
    main()
