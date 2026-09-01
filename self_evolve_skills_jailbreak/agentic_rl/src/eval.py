"""
评估 Pipeline

评估 Agentic SFT / GRPO 模型的攻击能力：
1. 加载模型（base + LoRA）
2. 对测试 prompt 生成 skill-conditioned 攻击
3. 通过 Target + Guard 评估 ASR
4. 统计 skill 选择分布

Usage:
    python eval.py \
        --model_path /home/tiger/models/Qwen/Qwen3-4B \
        --lora_path output/sft_checkpoint/final_lora \
        --skill_library_path ../exp/layer4/results/skills/skills_dan_data_medium_evo.json \
        --description_path output/skill_descriptions.json \
        --test_data ../../data/dataset/processed/10k/test.jsonl \
        --policy_port 8003 --target_port 8001 --guard_port 8002
"""

import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RL4jailbreak"))

from src.vllm_client import VLLMClient
from src.utils import extract_label_categories_refusal
from src.prompts import GUARD_PROMPT
from self_evolve_skills_jailbreak.src.skill_library import SkillLibrary
from self_evolve_skills_jailbreak.agentic_rl.src.utils import (
    format_user_prompt,
    parse_completion,
    build_attack_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Agentic-RL Evaluation")
    parser.add_argument("--model_path", type=str,
                        default="/home/tiger/models/Qwen/Qwen3-4B")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--skill_library_path", type=str, required=True)
    parser.add_argument("--description_path", type=str, required=True)
    parser.add_argument("--test_data", type=str,
                        default=str(PROJECT_ROOT / "data/dataset/processed/10k/test.jsonl"))
    parser.add_argument("--policy_port", type=int, default=8003)
    parser.add_argument("--target_port", type=int, default=8001)
    parser.add_argument("--guard_port", type=int, default=8002)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="output/eval_results")
    parser.add_argument("--max_workers", type=int, default=32)
    return parser.parse_args()


def load_test_data(path: str, max_samples: int) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data.append({"id": obj.get("id", len(data)), "prompt": obj["prompt"]})
            if len(data) >= max_samples:
                break
    return data


def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to vLLM servers...")
    policy_client = VLLMClient(port=args.policy_port, launch_server=False)
    target_client = VLLMClient(port=args.target_port, launch_server=False)
    guard_client = VLLMClient(port=args.guard_port, launch_server=False)

    policy_client.__enter__()
    target_client.__enter__()
    guard_client.__enter__()

    try:
        skill_library = SkillLibrary(
            storage_path="/tmp/eval_skills.json", max_skills=200
        )
        skill_library.load_from_file(args.skill_library_path)

        with open(args.description_path, "r", encoding="utf-8") as f:
            descriptions = json.load(f)

        test_data = load_test_data(args.test_data, args.max_samples)
        logger.info(f"Loaded {len(test_data)} test samples")

        prompts = []
        original_prompts = []
        for item in test_data:
            candidates = skill_library.retrieve(item["prompt"], top_k=args.top_k)
            candidate_info = [
                {"name": c.name, "description": descriptions.get(c.name, f"攻击策略: {c.name}")}
                for c in candidates
            ]
            user_prompt = format_user_prompt(item["prompt"], candidate_info)
            prompts.append(user_prompt)
            original_prompts.append(item["prompt"])

        logger.info("Generating adapted skills...")
        completions = policy_client.llm_batch_call(
            prompts=prompts,
            temperature=0.7,
            max_tokens=1024,
            max_workers=args.max_workers,
            return_exceptions=True,
        )

        adapted_contents = []
        selected_indices = []
        for c in completions:
            if c is None or isinstance(c, Exception):
                adapted_contents.append("")
                selected_indices.append(0)
            else:
                idx, content = parse_completion(str(c))
                adapted_contents.append(content)
                selected_indices.append(idx)

        logger.info("Sending attacks to target...")
        attack_prompts = [
            build_attack_prompt(adapted, orig)
            for adapted, orig in zip(adapted_contents, original_prompts)
        ]

        target_responses = target_client.llm_batch_call(
            prompts=attack_prompts,
            temperature=0.7,
            max_tokens=2048,
            max_workers=args.max_workers,
            return_exceptions=True,
        )

        logger.info("Guard classification...")
        guard_messages = []
        for ap, tr in zip(attack_prompts, target_responses):
            if tr is None or isinstance(tr, Exception):
                tr = "Error: No response"
            guard_messages.append([
                {"role": "system", "content": GUARD_PROMPT},
                {"role": "user", "content": ap},
                {"role": "assistant", "content": str(tr)},
            ])

        guard_responses = guard_client.llm_batch_call(
            messages_list=guard_messages,
            temperature=0.0,
            max_tokens=256,
            max_workers=args.max_workers,
            return_exceptions=True,
        )

        results = []
        success_count = 0
        for i, (item, gr) in enumerate(zip(test_data, guard_responses)):
            if gr is None or isinstance(gr, Exception):
                label = "unknown"
            else:
                label, _, _ = extract_label_categories_refusal(str(gr))

            is_success = label == "Unsafe"
            if is_success:
                success_count += 1

            results.append({
                "id": item["id"],
                "original_prompt": item["prompt"],
                "selected_skill_idx": selected_indices[i],
                "adapted_content": adapted_contents[i][:200],
                "is_success": is_success,
                "guard_label": label,
            })

        asr = success_count / len(test_data) if test_data else 0

        skill_dist = Counter(selected_indices)
        total = len(selected_indices)

        summary = {
            "total": len(test_data),
            "success": success_count,
            "asr": asr,
            "skill_selection_distribution": dict(skill_dist),
            "lora_path": args.lora_path,
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"ASR: {asr:.2%} ({success_count}/{len(test_data)})")
        logger.info(f"Skill selection distribution:")
        for idx, count in sorted(skill_dist.items()):
            logger.info(f"  Skill {idx}: {count} ({count/total:.1%})")
        logger.info(f"{'='*60}")

        results_path = output_dir / "results.jsonl"
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {results_path}")
        logger.info(f"Summary saved to {summary_path}")

    finally:
        policy_client.__exit__(None, None, None)
        target_client.__exit__(None, None, None)
        guard_client.__exit__(None, None, None)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
