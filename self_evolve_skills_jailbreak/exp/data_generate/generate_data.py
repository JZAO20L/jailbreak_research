#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Generation Script for Thesis Chapter 2

使用 Skills 系统生成合成数据：
- 输入：训练集的一半数据（500条）
- 输出：种子 prompt -> 最终攻击成功 prompt 数据集
- 方法：pair_skills_28 和 autodan_skills_54

Usage:
    python generate_data.py --method pair_skills_28 --output_dir results/pair_skills
    python generate_data.py --method autodan_skills_54 --output_dir results/autodan_skills
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from self_evolve_skills_jailbreak.src.skill_library import SkillLibrary
from self_evolve_skills_jailbreak.src.attacker import SkillGuidedAttacker


# =============================================================================
# 配置
# =============================================================================

class Config:
    """实验配置"""
    # 服务端口
    GUARD_PORT = 8002
    ATTACKER_PORT = 8003
    TARGET_PORT = 8001

    # 数据路径
    TRAIN_DATA_PATH = PROJECT_ROOT / "data/dataset/processed/10k/train.jsonl"

    # Skills 文件路径
    PAIR_SKILLS_PATH = PROJECT_ROOT / "self_evolve_skills_jailbreak/exp/layer1/results/skills/skills_single_call_trajectory_statistical.json"
    AUTODAN_SKILLS_PATH = PROJECT_ROOT / "self_evolve_skills_jailbreak/exp/layer4/results/skills/skills_dan_data_medium_evo.json"

    # 并发配置
    MAX_WORKERS = 32

    # 攻击配置
    MAX_ITERATIONS = 10


# =============================================================================
# 数据加载
# =============================================================================

def load_train_data() -> List[Dict]:
    """加载训练数据（JSONL格式）

    Returns:
        List of dicts with 'id' and 'prompt' fields
    """
    data = []
    with open(Config.TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append({
                    'id': item.get('id', ''),
                    'prompt': item.get('prompt', '')
                })

    print(f"Loaded {len(data)} training prompts from {Config.TRAIN_DATA_PATH}")
    return data


def load_skills(method: str) -> SkillLibrary:
    """加载 Skills 库"""
    if method == "pair_skills_28":
        skills_path = Config.PAIR_SKILLS_PATH
        skill_call_mode = "single_call"
    elif method == "autodan_skills_54":
        skills_path = Config.AUTODAN_SKILLS_PATH
        skill_call_mode = "every_iteration"
    else:
        raise ValueError(f"Unknown method: {method}")

    # 创建临时 Skills 库
    temp_path = PROJECT_ROOT / "self_evolve_skills_jailbreak/exp/data_generate/temp_skills.json"
    skill_library = SkillLibrary(storage_path=str(temp_path), max_skills=100)
    skill_library.load_from_file(str(skills_path))

    print(f"Loaded {skill_library.count()} skills from {skills_path}")
    return skill_library, skill_call_mode


# =============================================================================
# 客户端初始化
# =============================================================================

def init_clients():
    """初始化 vLLM 客户端"""
    from src.vllm_client import VLLMClient

    guard_client = VLLMClient(
        port=Config.GUARD_PORT,
        launch_server=False,
    )

    attacker_client = VLLMClient(
        port=Config.ATTACKER_PORT,
        launch_server=False,
    )

    target_client = VLLMClient(
        port=Config.TARGET_PORT,
        launch_server=False,
    )

    return guard_client, attacker_client, target_client


# =============================================================================
# 数据生成
# =============================================================================

def generate_single(
    item: Dict,
    attacker: SkillGuidedAttacker,
    retrieve_top_k: int = 3,
) -> Optional[Dict]:
    """生成单个数据样本

    Args:
        item: Dict with 'id' and 'prompt'

    Returns:
        成功时返回:
        {
            "id": str,
            "original_prompt": str,
            "attack_prompt": str,
            "iterations": int,
            "skill_used": str,
            "skill_content": str,
            "trajectory": List[Dict],
        }
        失败时返回 None
    """
    original_prompt = item['prompt']
    item_id = item['id']

    try:
        result = attacker.attack(
            original_prompt,
            retrieve_top_k=retrieve_top_k,
        )

        if result.is_success:
            return {
                "id": item_id,
                "original_prompt": original_prompt,
                "attack_prompt": result.attack_prompt,
                "iterations": result.iterations,
                "skill_used": result.skill_used.name if result.skill_used else "unknown",
                "skill_content": result.skill_used.content if result.skill_used else "",
                "trajectory": result.intermediate_results,
                "target_response": result.target_response,
                "is_success": result.is_success,
            }
        else:
            return None
    except Exception as e:
        print(f"Error processing {item_id}: {e}")
        return None


def generate_batch(
    items: List[Dict],
    method: str,
    output_dir: Path,
    max_workers: int = 32,
) -> Dict:
    """批量生成数据"""

    print(f"\n{'='*60}")
    print(f"Generating data with {method}")
    print(f"{'='*60}")
    print(f"Input: {len(items)} items")
    print(f"Output: {output_dir}")
    print(f"Max workers: {max_workers}")

    # 初始化
    skill_library, skill_call_mode = load_skills(method)
    guard_client, attacker_client, target_client = init_clients()

    # 创建 Attacker
    attacker = SkillGuidedAttacker(
        target_client=target_client,
        guard_client=guard_client,
        rewrite_client=attacker_client,
        skill_library=skill_library,
        max_iterations=Config.MAX_ITERATIONS,
        skill_call_mode=skill_call_mode,
        verbose=False,
    )

    # 生成数据
    results = []
    success_count = 0
    failed_count = 0

    print(f"\nRunning generation...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_single, item, attacker): item for item in items}

        pbar = tqdm(total=len(items), desc=f"[{method}]")

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
                success_count += 1
            else:
                failed_count += 1

            pbar.update(1)
            pbar.set_postfix({
                "success": success_count,
                "failed": failed_count,
            })

        pbar.close()

    # 统计
    total = len(items)
    success_rate = success_count / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Total: {total}")
    print(f"  Success: {success_count} ({success_rate:.2%})")
    print(f"  Failed: {failed_count}")
    print(f"{'='*60}")

    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存成功案例
    success_file = output_dir / "success_cases.json"
    with open(success_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} success cases to: {success_file}")

    # 保存统计信息
    summary = {
        "method": method,
        "total_prompts": total,
        "success_count": success_count,
        "failed_count": failed_count,
        "success_rate": success_rate,
        "avg_iterations": sum(r['iterations'] for r in results) / len(results) if results else 0,
        "skills_count": skill_library.count(),
        "data_source": str(Config.TRAIN_DATA_PATH),
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to: {summary_file}")

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for thesis")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["pair_skills_28", "autodan_skills_54", "both"],
        help="Method to use for generation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: exp/data_generate/results/<method>)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    # 确定输出目录
    base_dir = Path(__file__).parent / "results"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir

    # 加载训练数据
    items = load_train_data()

    # 运行生成
    if args.method == "both":
        # 两种方法都运行
        for method in ["pair_skills_28", "autodan_skills_54"]:
            method_output_dir = output_dir / method
            generate_batch(
                items=items,
                method=method,
                output_dir=method_output_dir,
                max_workers=args.max_workers,
            )
    else:
        # 单个方法
        method_output_dir = output_dir / args.method
        generate_batch(
            items=items,
            method=args.method,
            output_dir=method_output_dir,
            max_workers=args.max_workers,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()