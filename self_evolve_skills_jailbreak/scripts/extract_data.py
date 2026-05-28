"""
从原始数据集抽取实验数据

Layer 1 数据配置：
- Train: 取前 N 条（默认 1000 条）
- Cold Start: train 的固定比例（默认 20%）
- Evolution: train 的剩余部分（80%）
- Test: 使用完整的 test.jsonl（不抽取）

Usage:
    python self_evolve_skills_jailbreak/scripts/extract_data.py \
        --train_path data/dataset/processed/10k/train.jsonl \
        --test_path data/dataset/processed/10k/test.jsonl \
        --train_limit 1000 \
        --cold_start_ratio 0.20
"""

import json
import random
import os
from pathlib import Path


def load_jsonl(path: str) -> list:
    """加载 jsonl 文件"""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                prompt = data.get("prompt", data.get("question", data.get("text", "")))
                if prompt:
                    prompts.append(prompt)
            except json.JSONDecodeError:
                continue
    return prompts


def extract_experiment_data(
    train_path: str = "data/dataset/processed/10k/train.jsonl",
    test_path: str = "data/dataset/processed/10k/test.jsonl",
    output_dir: str = "self_evolve_skills_jailbreak/data",
    train_limit: int = 1000,
    cold_start_ratio: float = 0.20,
    random_seed: int = 42,
    deduplicate: bool = True,
):
    """
    抽取实验数据

    Args:
        train_path: train 数据源路径
        test_path: test 数据源路径（使用完整数据）
        output_dir: 输出目录
        train_limit: 从 train 取前多少条（默认 1000）
        cold_start_ratio: cold start 占 train 的比例（默认 20%）
        random_seed: 随机种子
        deduplicate: 是否去重
    """
    random.seed(random_seed)

    # 1. 加载 train 数据，取前 train_limit 条
    print(f"Loading train data from {train_path}...")
    train_prompts = load_jsonl(train_path)
    print(f"  Total train prompts: {len(train_prompts)}")

    # 取前 train_limit 条
    train_prompts = train_prompts[:train_limit]
    print(f"  Using first {len(train_prompts)} prompts")

    # 去重
    if deduplicate:
        train_prompts = list(set(train_prompts))
        print(f"  After deduplication: {len(train_prompts)} unique prompts")

    # 2. 加载 test 数据（完整使用）
    print(f"\nLoading test data from {test_path}...")
    test_prompts = load_jsonl(test_path)
    print(f"  Total test prompts: {len(test_prompts)}")

    if deduplicate:
        test_prompts = list(set(test_prompts))
        print(f"  After deduplication: {len(test_prompts)} unique prompts")

    # 确保 test 和 train 不重叠
    train_set = set(train_prompts)
    test_prompts = [p for p in test_prompts if p not in train_set]
    print(f"  After removing overlap: {len(test_prompts)} prompts")

    # 3. 划分 train 为 cold start 和 evolution
    cold_start_count = int(len(train_prompts) * cold_start_ratio)
    evolution_count = len(train_prompts) - cold_start_count

    random.shuffle(train_prompts)

    cold_start_data = train_prompts[:cold_start_count]
    evolution_data = train_prompts[cold_start_count:]

    # 4. 保存
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "cold_start_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(cold_start_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "evolution_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(evolution_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "test_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(test_prompts, f, ensure_ascii=False, indent=2)

    # 保存元信息
    meta = {
        "train_source": train_path,
        "test_source": test_path,
        "train_limit": train_limit,
        "cold_start": {
            "count": len(cold_start_data),
            "ratio": cold_start_ratio,
        },
        "evolution": {
            "count": len(evolution_data),
            "ratio": 1 - cold_start_ratio,
        },
        "test": {
            "count": len(test_prompts),
            "note": "完整 test 集（去重后）",
        },
        "random_seed": random_seed,
        "deduplicated": deduplicate,
    }

    with open(os.path.join(output_dir, "data_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print("✓ Data extraction completed!")
    print(f"{'='*50}")
    print(f"  Cold Start: {len(cold_start_data)} prompts ({cold_start_ratio*100:.0f}% of train)")
    print(f"  Evolution:  {len(evolution_data)} prompts ({(1-cold_start_ratio)*100:.0f}% of train)")
    print(f"  Test:       {len(test_prompts)} prompts (full test set)")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract experiment data")
    parser.add_argument("--train_path", type=str,
                        default="data/dataset/processed/10k/train.jsonl",
                        help="Train 数据源路径")
    parser.add_argument("--test_path", type=str,
                        default="data/dataset/processed/10k/test.jsonl",
                        help="Test 数据源路径")
    parser.add_argument("--output", type=str,
                        default="self_evolve_skills_jailbreak/data",
                        help="输出目录")
    parser.add_argument("--train_limit", type=int, default=1000,
                        help="从 train 取前多少条")
    parser.add_argument("--cold_start_ratio", type=float, default=0.20,
                        help="Cold start 占 train 的比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    extract_experiment_data(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output,
        train_limit=args.train_limit,
        cold_start_ratio=args.cold_start_ratio,
        random_seed=args.seed,
    )