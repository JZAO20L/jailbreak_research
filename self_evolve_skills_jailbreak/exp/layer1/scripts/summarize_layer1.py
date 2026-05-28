"""
Layer 1 Results Summary Script

汇总 Grid Search 实验结果

Usage:
    python summarize_layer1.py --input results/ --output results/layer1_summary.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from glob import glob


def find_result_files(input_dir: str) -> list:
    """查找所有结果文件"""
    pattern = os.path.join(input_dir, "result_*.json")
    files = glob(pattern)
    return sorted(files)


def parse_combination_from_filename(filename: str) -> dict:
    """从文件名解析实验组合"""
    # 文件名格式: result_single_call_final_prompt_success_only.json
    basename = Path(filename).stem.replace("result_", "")
    parts = basename.split("_")

    if len(parts) >= 4:
        # 合理拆分
        # skill_call_mode: single_call 或 every_iteration
        # extraction_mode: final_prompt 或 trajectory
        # update_strategy: success_only, failure_only, both, statistical

        skill_call_modes = ["single_call", "every_iteration"]
        extraction_modes = ["final_prompt", "trajectory"]
        update_strategies = ["success_only", "failure_only", "both", "statistical"]

        call_mode = None
        extraction = None
        strategy = None

        # 尝试匹配
        for mode in skill_call_modes:
            if basename.startswith(mode):
                call_mode = mode
                break

        if call_mode:
            rest = basename[len(call_mode)+1:]
            for ext in extraction_modes:
                if rest.startswith(ext):
                    extraction = ext
                    break

            if extraction:
                rest = rest[len(extraction)+1:]
                for strat in update_strategies:
                    if rest.startswith(strat):
                        strategy = strat
                        break

        if call_mode and extraction and strategy:
            return {
                "skill_call_mode": call_mode,
                "extraction_mode": extraction,
                "update_strategy": strategy,
            }

    return {"raw_name": basename}


def load_and_merge_results(files: list) -> dict:
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

            combination = parse_combination_from_filename(file_path)
            key = f"{combination.get('skill_call_mode', 'unknown')}_{combination.get('extraction_mode', 'unknown')}_{combination.get('update_strategy', 'unknown')}"

            # 提取关键指标
            test_stats = data.get("test", {})
            evolution_stats = data.get("evolution", {})

            merged["results"][key] = {
                "skill_call_mode": combination.get("skill_call_mode"),
                "extraction_mode": combination.get("extraction_mode"),
                "update_strategy": combination.get("update_strategy"),
                "asr": test_stats.get("asr", test_stats.get("success", 0) / test_stats.get("total", 1)),
                "avg_iterations": test_stats.get("avg_iterations", test_stats.get("total_iterations", 0) / test_stats.get("total", 1)),
                "total": test_stats.get("total", 0),
                "success": test_stats.get("success", 0),
                "final_skill_count": evolution_stats.get("final_skill_count", 0),
                "intermediate_evals": evolution_stats.get("intermediate_evals", []),
                "source_file": os.path.basename(file_path),
            }

            # 合并配置
            if not merged["config"]:
                merged["config"] = data.get("config", {})

        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    return merged


def print_summary(merged: dict):
    """打印汇总表格"""
    print("\n" + "="*80)
    print("Layer 1 Grid Search Results Summary")
    print("="*80)

    if merged["config"]:
        print(f"Config:")
        for k, v in merged["config"].items():
            print(f"  {k}: {v}")

    print("\n" + "-"*80)
    print(f"{'Method':<40} {'ASR':>10} {'Avg Iter':>10} {'Skills':>8}")
    print("-"*80)

    # 按 ASR 排序
    sorted_results = sorted(
        merged["results"].items(),
        key=lambda x: x[1]["asr"],
        reverse=True
    )

    for key, r in sorted_results:
        asr_pct = r["asr"] * 100
        print(f"{key:<40} {asr_pct:>9.1f}% {r['avg_iterations']:>10.2f} {r['final_skill_count']:>8}")

    print("-"*80)

    # 找出最佳组合
    if sorted_results:
        best_key, best_r = sorted_results[0]
        print(f"\nBest combination: {best_key}")
        print(f"  ASR: {best_r['asr']*100:.1f}%")
        print(f"  Avg Iterations: {best_r['avg_iterations']:.2f}")
        print(f"  Skill Count: {best_r['final_skill_count']}")


def analyze_ablation(merged: dict) -> dict:
    """分析各消融点的平均效果"""
    ablation = {
        "skill_call_mode": {},
        "extraction_mode": {},
        "update_strategy": {},
    }

    for key, r in merged["results"].items():
        # skill_call_mode
        call_mode = r.get("skill_call_mode")
        if call_mode:
            if call_mode not in ablation["skill_call_mode"]:
                ablation["skill_call_mode"][call_mode] = {"asr_sum": 0, "count": 0}
            ablation["skill_call_mode"][call_mode]["asr_sum"] += r["asr"]
            ablation["skill_call_mode"][call_mode]["count"] += 1

        # extraction_mode
        extraction = r.get("extraction_mode")
        if extraction:
            if extraction not in ablation["extraction_mode"]:
                ablation["extraction_mode"][extraction] = {"asr_sum": 0, "count": 0}
            ablation["extraction_mode"][extraction]["asr_sum"] += r["asr"]
            ablation["extraction_mode"][extraction]["count"] += 1

        # update_strategy
        strategy = r.get("update_strategy")
        if strategy:
            if strategy not in ablation["update_strategy"]:
                ablation["update_strategy"][strategy] = {"asr_sum": 0, "count": 0}
            ablation["update_strategy"][strategy]["asr_sum"] += r["asr"]
            ablation["update_strategy"][strategy]["count"] += 1

    # 计算平均值
    for ablation_point in ablation:
        for option, stats in ablation[ablation_point].items():
            stats["avg_asr"] = stats["asr_sum"] / stats["count"] if stats["count"] > 0 else 0

    return ablation


def main():
    parser = argparse.ArgumentParser(description="Layer 1 Results Summary")

    parser.add_argument("--input", type=str, required=True,
                        help="结果文件目录")
    parser.add_argument("--output", type=str, required=True,
                        help="汇总文件输出路径")

    args = parser.parse_args()

    # 查找结果文件
    files = find_result_files(args.input)

    if not files:
        print(f"Warning: No result files found in {args.input}")
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

    # 消融分析
    ablation = analyze_ablation(merged)
    merged["ablation_analysis"] = ablation

    # 打印汇总
    print_summary(merged)

    # 打印消融分析
    print("\n" + "="*80)
    print("Ablation Analysis")
    print("="*80)

    for ablation_point, options in ablation.items():
        print(f"\n{ablation_point}:")
        sorted_options = sorted(options.items(), key=lambda x: x[1]["avg_asr"], reverse=True)
        for option, stats in sorted_options:
            print(f"  {option}: avg ASR = {stats['avg_asr']*100:.1f}% (n={stats['count']})")

    # 保存汇总
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n汇总已保存: {args.output}")


if __name__ == "__main__":
    main()