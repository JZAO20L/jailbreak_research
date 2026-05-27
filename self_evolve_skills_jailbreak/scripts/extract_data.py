"""
从原始数据集抽取实验数据

原始数据: jailbreak_research/data/dataset/processed/10k/train.jsonl (8000 条)
抽取目标: 1000 条，划分为 cold_start / evolution / test
"""

import json
import random
import os
from pathlib import Path


def extract_experiment_data(
    source_path: str = "data/dataset/processed/10k/train.jsonl",
    output_dir: str = "self_evolve_skills_jailbreak/data",
    total_size: int = 1000,
    cold_start_ratio: float = 0.20,
    evolution_ratio: float = 0.60,
    test_ratio: float = 0.20,
    random_seed: int = 42,
    deduplicate: bool = True,
):
    """
    从原始数据抽取实验数据

    Args:
        source_path: 原始数据路径
        output_dir: 输出目录
        total_size: 抽取总数
        cold_start_ratio: cold start 数据比例
        evolution_ratio: evolution 数据比例
        test_ratio: test 数据比例
        random_seed: 随机种子
        deduplicate: 是否去重
    """
    random.seed(random_seed)

    # 加载原始数据
    print(f"Loading data from {source_path}...")
    prompts = []

    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # 提取 prompt 字段（根据实际数据结构调整）
                prompt = data.get("prompt", data.get("question", data.get("text", "")))
                if prompt:
                    prompts.append(prompt)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(prompts)} prompts from source")

    # 去重
    if deduplicate:
        prompts = list(set(prompts))
        print(f"After deduplication: {len(prompts)} unique prompts")

    # 随机抽样
    if len(prompts) > total_size:
        prompts = random.sample(prompts, total_size)
        print(f"Sampled {len(prompts)} prompts")

    # 划分
    cold_start_count = int(total_size * cold_start_ratio)
    evolution_count = int(total_size * evolution_ratio)
    test_count = total_size - cold_start_count - evolution_count

    random.shuffle(prompts)

    cold_start_data = prompts[:cold_start_count]
    evolution_data = prompts[cold_start_count: cold_start_count + evolution_count]
    test_data = prompts[cold_start_count + evolution_count:]

    # 保存
    os.makedirs(output_dir, exist_ok=True)

    # 保存为 JSON 列表格式
    with open(os.path.join(output_dir, "cold_start_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(cold_start_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "evolution_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(evolution_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "test_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    # 合并保存为总数据集
    with open(os.path.join(output_dir, "train_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    # 保存元信息
    meta = {
        "source": source_path,
        "total_size": total_size,
        "cold_start": {"count": cold_start_count, "ratio": cold_start_ratio},
        "evolution": {"count": evolution_count, "ratio": evolution_ratio},
        "test": {"count": test_count, "ratio": test_ratio},
        "random_seed": random_seed,
        "deduplicated": deduplicate,
    }

    with open(os.path.join(output_dir, "data_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Data extraction completed!")
    print(f"  Cold Start: {len(cold_start_data)} prompts")
    print(f"  Evolution: {len(evolution_data)} prompts")
    print(f"  Test: {len(test_data)} prompts")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract experiment data from source")
    parser.add_argument("--source", type=str, default="data/dataset/processed/10k/train.jsonl")
    parser.add_argument("--output", type=str, default="self_evolve_skills_jailbreak/data")
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--cold_start_ratio", type=float, default=0.20)
    parser.add_argument("--evolution_ratio", type=float, default=0.60)
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    extract_experiment_data(
        source_path=args.source,
        output_dir=args.output,
        total_size=args.total,
        cold_start_ratio=args.cold_start_ratio,
        evolution_ratio=args.evolution_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )