"""
RFT 数据准备 Pipeline

两阶段：
1. 轨迹生成：基础模型生成 N 个候选，多维度评估，过滤 ASR=1.0 的
2. 数据构建：将过滤后的轨迹转换为 ChatML 训练格式

Usage:
    # Step 1: 生成 skill descriptions（需要 vLLM server）
    python data_prep.py --step descriptions --skill_path <path> --output_dir <dir>

    # Step 2: 生成 RFT 轨迹（需要 policy + target + guard servers）
    python data_prep.py --step trajectories --skill_path <path> \
        --description_path <dir>/skill_descriptions.json \
        --train_data <path> --output_dir <dir>

    # Step 3: 构建训练数据
    python data_prep.py --step build --trajectory_path <dir>/rft_trajectories.jsonl \
        --output_dir <dir>

    # 一键运行
    python data_prep.py --step all --skill_path <path> --train_data <path> --output_dir <dir>
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from self_evolve_skills_jailbreak.src.skill_library import SkillLibrary
from self_evolve_skills_jailbreak.agentic_rl.src.utils import (
    SYSTEM_PROMPT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Step 1: Skill Description 生成
# =============================================================================

DESCRIPTION_PROMPT = """Please write a concise one-sentence description (in Chinese, <=80 characters) for the following jailbreak attack strategy. The description should capture the core mechanism/approach of the strategy.

Strategy content:
{content}

Output ONLY the description, nothing else:"""


def generate_descriptions(
    skill_path: str,
    output_dir: Path,
    port: int = 8003,
) -> Dict[str, str]:
    """用 LLM 为每个 skill 生成 description"""
    from src.vllm_client import VLLMClient

    temp_path = output_dir / "temp_skills.json"
    skill_library = SkillLibrary(storage_path=str(temp_path), max_skills=200)
    skill_library.load_from_file(skill_path)

    client = VLLMClient(port=port, launch_server=False)
    client.__enter__()

    try:
        skills = skill_library.list_skills()
        prompts = [DESCRIPTION_PROMPT.format(content=s.content[:400]) for s in skills]

        logger.info(f"Generating descriptions for {len(skills)} skills...")
        responses = client.llm_batch_call(
            prompts=prompts,
            temperature=0.3,
            max_tokens=120,
            max_workers=16,
            return_exceptions=True,
        )

        descriptions = {}
        for skill, resp in zip(skills, responses):
            if isinstance(resp, Exception) or resp is None:
                desc = f"攻击策略: {skill.name}"
                logger.warning(f"Failed to generate description for {skill.name}: {resp}")
            else:
                desc = str(resp).strip().strip('"').strip("'")
                if len(desc) > 100:
                    desc = desc[:100]
            descriptions[skill.name] = desc

        output_path = output_dir / "skill_descriptions.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(descriptions)} descriptions to {output_path}")
        return descriptions

    finally:
        client.__exit__(None, None, None)


# =============================================================================
# Step 2: RFT 轨迹生成
# =============================================================================

def generate_trajectories(
    skill_path: str,
    description_path: str,
    train_data: str,
    output_dir: Path,
    policy_port: int = 8003,
    target_port: int = 8001,
    guard_port: int = 8002,
    num_candidates: int = 8,
    top_k_skills: int = 5,
    max_samples: int = 1000,
    min_asr: float = 1.0,
    top_m_per_prompt: int = 2,
    data_start: int = 0,
    data_end: int = None,
):
    """调用 trajectory_generator 生成 RFT 轨迹"""
    from self_evolve_skills_jailbreak.agentic_rl.src.trajectory_generator import (
        generate_rft_trajectories,
    )

    generate_rft_trajectories(
        train_data_path=train_data,
        skill_library_path=skill_path,
        description_path=description_path,
        output_dir=output_dir,
        policy_port=policy_port,
        target_port=target_port,
        guard_port=guard_port,
        num_candidates=num_candidates,
        top_k_skills=top_k_skills,
        max_samples=max_samples,
        min_asr=min_asr,
        top_m_per_prompt=top_m_per_prompt,
        data_start=data_start,
        data_end=data_end,
    )


# =============================================================================
# Step 3: 构建训练数据
# =============================================================================

def build_training_data(
    trajectory_path: str,
    output_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    将 RFT 轨迹转换为 ChatML 训练格式

    每条轨迹包含：
    - user_message: 完整的 user message（包含 prompt + 候选 skills）
    - completion: 模型生成的输出（包含 selection + adapted strategy）

    输出格式：
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "original_prompt": "...",
        "selected_skill_idx": 0,
        "score": 1.3
    }
    """
    logger.info(f"Loading trajectories from {trajectory_path}")

    trajectories = []
    with open(trajectory_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            trajectories.append(obj)

    logger.info(f"Loaded {len(trajectories)} trajectories")

    # 转换为训练格式
    samples = []
    for traj in trajectories:
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": traj["user_message"]},
                {"role": "assistant", "content": traj["completion"]},
            ],
            "original_prompt": traj["prompt"],
            "selected_skill_idx": traj["selected_idx"],
            "score": traj["score"],
        }
        samples.append(sample)

    # 打乱
    random.seed(seed)
    random.shuffle(samples)

    # 划分 train/val
    val_size = int(len(samples) * val_ratio)
    train_samples = samples[val_size:]
    val_samples = samples[:val_size]

    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存
    train_path = output_dir / "rft_train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(train_samples)} train samples to {train_path}")

    val_path = output_dir / "rft_val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(val_samples)} val samples to {val_path}")

    # 统计
    skill_dist = {}
    for s in samples:
        idx = s["selected_skill_idx"]
        skill_dist[idx] = skill_dist.get(idx, 0) + 1

    logger.info(f"Skill selection distribution:")
    for idx, count in sorted(skill_dist.items()):
        logger.info(f"  Skill {idx}: {count} ({count/len(samples)*100:.1f}%)")

    stats = {
        "total_trajectories": len(trajectories),
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "skill_distribution": skill_dist,
    }

    stats_path = output_dir / "data_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Stats saved to {stats_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RFT Data Preparation")
    parser.add_argument("--step", type=str, required=True,
                        choices=["descriptions", "trajectories", "build", "all"],
                        help="Which step to run")
    parser.add_argument("--skill_path", type=str, required=True,
                        help="Path to skill library JSON")
    parser.add_argument("--train_data", type=str, default=None,
                        help="Path to training data (JSONL)")
    parser.add_argument("--description_path", type=str, default=None,
                        help="Path to skill_descriptions.json")
    parser.add_argument("--trajectory_path", type=str, default=None,
                        help="Path to rft_trajectories.jsonl")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")

    # Server ports
    parser.add_argument("--policy_port", type=int, default=8003)
    parser.add_argument("--target_port", type=int, default=8001)
    parser.add_argument("--guard_port", type=int, default=8002)

    # RFT parameters
    parser.add_argument("--num_candidates", type=int, default=8,
                        help="Number of candidates per prompt")
    parser.add_argument("--top_k_skills", type=int, default=5,
                        help="Number of candidate skills")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--min_asr", type=float, default=1.0,
                        help="Minimum ASR threshold")
    parser.add_argument("--top_m_per_prompt", type=int, default=2,
                        help="Top-m candidates per prompt")

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # 数据范围参数
    parser.add_argument("--data_start", type=int, default=0,
                        help="Skip first N samples")
    parser.add_argument("--data_end", type=int, default=None,
                        help="Stop at this index (None = end of file)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate descriptions
    if args.step in ("descriptions", "all"):
        generate_descriptions(
            skill_path=args.skill_path,
            output_dir=output_dir,
            port=args.policy_port,
        )

    # Step 2: Generate trajectories
    if args.step in ("trajectories", "all"):
        if args.description_path is None:
            args.description_path = str(output_dir / "skill_descriptions.json")
        if args.train_data is None:
            args.train_data = str(
                PROJECT_ROOT / "data/dataset/processed/10k/train.jsonl"
            )

        generate_trajectories(
            skill_path=args.skill_path,
            description_path=args.description_path,
            train_data=args.train_data,
            output_dir=output_dir,
            policy_port=args.policy_port,
            target_port=args.target_port,
            guard_port=args.guard_port,
            num_candidates=args.num_candidates,
            top_k_skills=args.top_k_skills,
            max_samples=args.max_samples,
            min_asr=args.min_asr,
            top_m_per_prompt=args.top_m_per_prompt,
            data_start=args.data_start,
            data_end=args.data_end,
        )

    # Step 3: Build training data
    if args.step in ("build", "all"):
        if args.trajectory_path is None:
            args.trajectory_path = str(output_dir / "rft_trajectories.jsonl")

        build_training_data(
            trajectory_path=args.trajectory_path,
            output_dir=output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
