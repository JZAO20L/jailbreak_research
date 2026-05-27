"""
Summarize ASR Results Script

汇总所有 baseline ASR 测试结果。

Usage:
    python baselines/summarize_asr_results.py \
        --input experiments/baseline_asr \
        --output experiments/baseline_asr/summary.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from glob import glob


def find_result_files(input_dir: str) -> List[str]:
    """查找所有结果文件"""
    pattern = os.path.join(input_dir, "*_asr_*.json")
    files = glob(pattern)

    # 排除 summary 文件
    files = [f for f in files if "summary" not in f.lower()]

    return sorted(files)


def load_and_merge_results(files: List[str]) -> Dict:
    """加载并合并所有结果"""
    merged = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {},
        "results": {},
    }

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            strategy = data.get("strategy", "unknown")

            # 合并配置（取第一个）
            if not merged["config"]:
                merged["config"] = {
                    "max_iterations": data.get("max_iterations", "unknown"),
                    "test_file": data.get("test_path", "unknown"),
                }

            # 合并结果
            merged["results"][strategy] = {
                "asr": data.get("asr", 0),
                "avg_iterations": data.get("avg_iterations", 0),
                "avg_time": data.get("avg_time", 0),
                "total": data.get("total", 0),
                "success": data.get("success", 0),
                "failure": data.get("failure", 0),
                "total_time": data.get("total_time", 0),
                "source_file": os.path.basename(file_path),
            }

        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    return merged


def print_summary(merged: Dict):
    """打印汇总表格"""
    print("\n" + "="*60)
    print("ASR Results Summary")
    print("="*60)

    if merged["config"]:
        print(f"Config:")
        print(f"  Max iterations: {merged['config'].get('max_iterations', 'N/A')}")
        print(f"  Test file: {merged['config'].get('test_file', 'N/A')}")

    print("\n" + "-"*60)
    print(f"{'Strategy':<12} {'ASR':>10} {'Avg Iter':>10} {'Avg Time':>10} {'Success':>8}")
    print("-"*60)

    for strategy, result in merged["results"].items():
        asr_pct = result["asr"] * 100
        print(f"{strategy:<12} {asr_pct:>9.1f}% {result['avg_iterations']:>10.1f} "
              f"{result['avg_time']:>10.1f}s {result['success']:>8}")

    print("-"*60)


def main():
    parser = argparse.ArgumentParser(description="Summarize ASR Results")

    parser.add_argument("--input", type=str, required=True,
                        help="结果文件目录")
    parser.add_argument("--output", type=str, required=True,
                        help="汇总文件输出路径")

    args = parser.parse_args()

    # 查找结果文件
    files = find_result_files(args.input)

    if not files:
        print(f"Warning: No result files found in {args.input}")
        # 创建空汇总
        merged = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "config": {},
            "results": {},
        }
    else:
        print(f"Found {len(files)} result files:")
        for f in files:
            print(f"  - {os.path.basename(f)}")

        # 加载并合并
        merged = load_and_merge_results(files)

    # 打印汇总
    print_summary(merged)

    # 保存汇总
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n汇总已保存: {args.output}")


if __name__ == "__main__":
    main()