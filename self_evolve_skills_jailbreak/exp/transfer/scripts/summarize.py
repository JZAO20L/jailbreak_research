#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transfer Experiment Results Summarizer

汇总单个模型或所有模型的实验结果

目录结构: results/<模型>/<数据集>/<方法>/summary.json

Usage:
    # 单个模型 + 单个数据集
    python summarize.py --model Qwen3-0.6B --dataset default

    # 单个模型 + 所有数据集（跨数据集对比）
    python summarize.py --model Qwen3-0.6B --all_datasets

    # 所有模型（跨模型对比）
    python summarize.py --all_models

    # 全矩阵
    python summarize.py --full_matrix

"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def load_json(path: Path) -> Dict:
    """加载 JSON 文件"""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_method_result(result_dir: Path, method: str) -> Dict:
    """加载单个方法的结果

    支持多种结果文件格式：
    - baselines: summary.json
    - pipeline: result_*.json 或 test_summary.json
    """
    method_dir = result_dir / method
    if not method_dir.exists():
        return {}

    # 尝试多种可能的结果文件
    possible_files = [
        method_dir / "summary.json",
        method_dir / "test_summary.json",
        method_dir / "results_summary.json",
    ]

    # 也查找 result_*.json 文件（pipeline.py 输出格式）
    for f in method_dir.iterdir():
        if f.name.startswith("result_") and f.name.endswith(".json"):
            possible_files.append(f)

    for f in possible_files:
        if f.exists():
            data = load_json(f)
            if data:
                # 处理 pipeline.py 输出格式（结果在 test 字段）
                if "test" in data and "asr" in data.get("test", {}):
                    test_data = data.get("test", {})
                    return {
                        "asr": test_data.get("asr", 0),
                        "total": test_data.get("total", 0),
                        "successes": test_data.get("success", 0),
                        "avg_iterations": test_data.get("avg_iterations", 0),
                        "skills_count": data.get("final_skills_count",
                            data.get("evolution", {}).get("final_skill_count", 0)),
                    }
                # 处理 baselines 输出格式
                elif "asr" in data:
                    return {
                        "asr": data.get("asr", 0),
                        "total": data.get("total", 0),
                        "successes": data.get("successes", 0),
                        "avg_iterations": data.get("avg_iterations", 0),
                        "skills_count": 0,
                    }

    return {}


def summarize_dataset(model: str, dataset: str, result_dir: Path) -> Dict:
    """汇总单个模型+数据集的所有方法结果"""
    dataset_dir = result_dir / model / dataset
    if not dataset_dir.exists():
        return {
            "model": model,
            "dataset": dataset,
            "results": {},
            "timestamp": datetime.now().isoformat()
        }

    # 方法定义
    baselines = ["no_rewrite", "pair", "autodan", "deepinception", "persona"]
    our_methods = ["pair_skills_28", "autodan_skills_1", "autodan_skills_54"]
    all_methods = baselines + our_methods

    results = {}
    for method in all_methods:
        method_result = load_method_result(dataset_dir, method)
        if method_result:
            # 提取关键指标
            asr = method_result.get("asr", method_result.get("test_asr", 0))
            total = method_result.get("total", method_result.get("test_total", 0))
            successes = method_result.get("successes", method_result.get("test_successes", 0))

            results[method] = {
                "asr": asr,
                "total": total,
                "successes": successes,
                "avg_iterations": method_result.get("avg_iterations", 0),
                "skills_count": method_result.get("final_skills_count", method_result.get("skills_count", 0))
            }

    # 计算对比统计
    comparison = compute_comparison(results)

    return {
        "model": model,
        "dataset": dataset,
        "results": results,
        "comparison": comparison,
        "timestamp": datetime.now().isoformat()
    }


def compute_comparison(results: Dict) -> Dict:
    """计算方法对比统计"""
    comparison = {
        "best_baseline": {"method": "", "asr": 0},
        "best_ours": {"method": "", "asr": 0},
        "baseline_avg": 0,
        "ours_avg": 0,
        "improvement": 0
    }

    baselines = ["no_rewrite", "pair", "autodan", "deepinception", "persona"]
    our_methods = ["pair_skills_28", "autodan_skills_1", "autodan_skills_54"]

    # Baselines
    baseline_results = {k: v for k, v in results.items() if k in baselines}
    if baseline_results:
        asrs = [v.get("asr", 0) for v in baseline_results.values()]
        comparison["baseline_avg"] = sum(asrs) / len(asrs)
        best = max(baseline_results.items(), key=lambda x: x[1].get("asr", 0))
        comparison["best_baseline"] = {"method": best[0], "asr": best[1].get("asr", 0)}

    # Our methods
    ours_results = {k: v for k, v in results.items() if k in our_methods}
    if ours_results:
        asrs = [v.get("asr", 0) for v in ours_results.values()]
        comparison["ours_avg"] = sum(asrs) / len(asrs)
        best = max(ours_results.items(), key=lambda x: x[1].get("asr", 0))
        comparison["best_ours"] = {"method": best[0], "asr": best[1].get("asr", 0)}

    # Improvement
    if baseline_results and ours_results:
        comparison["improvement"] = comparison["best_ours"]["asr"] - comparison["best_baseline"]["asr"]

    return comparison


def format_single_summary(summary: Dict) -> str:
    """格式化单个实验的汇总"""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"{summary['model']} / {summary['dataset']}")
    lines.append(f"{'='*80}")

    if not summary.get("results"):
        lines.append("No results found.")
        return "\n".join(lines)

    # 结果表格
    lines.append(f"\n{'Method':<25} {'ASR':>10} {'Total':>8} {'Avg Iter':>10} {'Skills':>8}")
    lines.append("-" * 80)

    for method, data in summary["results"].items():
        asr = data.get("asr", 0)
        total = data.get("total", 0)
        avg_iter = data.get("avg_iterations", 0)
        skills = data.get("skills_count", "-")
        lines.append(f"{method:<25} {asr:>9.2%} {total:>8} {avg_iter:>10.2f} {str(skills):>8}")

    # 对比
    comp = summary.get("comparison", {})
    if comp:
        lines.append("-" * 80)
        lines.append(f"\nComparison:")
        lines.append(f"  Best Baseline: {comp['best_baseline']['method']} ({comp['best_baseline']['asr']:.2%})")
        lines.append(f"  Best Our Method: {comp['best_ours']['method']} ({comp['best_ours']['asr']:.2%})")
        lines.append(f"  Improvement: {comp['improvement']:+.2%}")

    return "\n".join(lines)


def format_cross_dataset_table(summaries: List[Dict]) -> str:
    """格式化跨数据集对比表格"""
    if not summaries:
        return "No results."

    lines = []
    lines.append(f"\n{'='*100}")
    lines.append(f"Cross-Dataset Transferability: {summaries[0]['model']}")
    lines.append(f"{'='*100}")

    # 表头
    methods = ["no_rewrite", "pair", "autodan", "deepinception", "pair_skills_28", "autodan_skills_1", "autodan_skills_54"]
    header = f"{'Dataset':<20}"
    for m in methods:
        header += f" {m[:15]:>15}"
    header += f" {'Best':>10} {'Improv':>10}"
    lines.append(header)
    lines.append("-" * 100)

    # 每个数据集一行
    for s in summaries:
        dataset = s.get("dataset", "")
        results = s.get("results", {})
        comp = s.get("comparison", {})

        row = f"{dataset:<20}"
        for m in methods:
            if m in results:
                asr = results[m].get("asr", 0)
                row += f" {asr:>14.2%} "
            else:
                row += f" {'N/A':>15} "

        # Best 和 Improvement
        best_asr = comp.get("best_ours", {}).get("asr", 0)
        improvement = comp.get("improvement", 0)
        row += f" {best_asr:>9.2%}  {improvement:>+9.2%}"
        lines.append(row)

    lines.append("-" * 100)
    return "\n".join(lines)


def format_cross_model_table(summaries: List[Dict]) -> str:
    """格式化跨模型对比表格"""
    if not summaries:
        return "No results."

    lines = []
    lines.append(f"\n{'='*120}")
    lines.append(f"Cross-Model Transferability (on default dataset)")
    lines.append(f"{'='*120}")

    # 表头
    methods = ["no_rewrite", "pair", "autodan", "deepinception", "pair_skills_28", "autodan_skills_1", "autodan_skills_54"]
    header = f"{'Model':<20}"
    for m in methods:
        header += f" {m[:15]:>15}"
    header += f" {'Best':>10} {'Improv':>10}"
    lines.append(header)
    lines.append("-" * 120)

    # 每个模型一行
    for s in summaries:
        model = s.get("model", "")
        results = s.get("results", {})
        comp = s.get("comparison", {})

        row = f"{model:<20}"
        for m in methods:
            if m in results:
                asr = results[m].get("asr", 0)
                row += f" {asr:>14.2%} "
            else:
                row += f" {'N/A':>15} "

        best_asr = comp.get("best_ours", {}).get("asr", 0)
        improvement = comp.get("improvement", 0)
        row += f" {best_asr:>9.2%}  {improvement:>+9.2%}"
        lines.append(row)

    lines.append("-" * 120)
    return "\n".join(lines)


def format_full_matrix(summaries: List[Dict]) -> str:
    """格式化完整矩阵（模型 × 数据集）"""
    if not summaries:
        return "No results."

    # 提取所有模型和数据集
    models = sorted(set(s.get("model", "") for s in summaries))
    datasets = sorted(set(s.get("dataset", "") for s in summaries))

    lines = []
    lines.append(f"\n{'='*120}")
    lines.append(f"Full Transferability Matrix (Best Our Method ASR)")
    lines.append(f"{'='*120}")

    # 表头
    header = f"{'Model':<20}"
    for d in datasets:
        header += f" {d[:12]:>12}"
    lines.append(header)
    lines.append("-" * 120)

    # 每个模型一行
    for model in models:
        row = f"{model:<20}"
        for dataset in datasets:
            matching = [s for s in summaries if s.get("model") == model and s.get("dataset") == dataset]
            if matching:
                comp = matching[0].get("comparison", {})
                best_asr = comp.get("best_ours", {}).get("asr", 0)
                row += f" {best_asr:>11.2%} "
            else:
                row += f" {'N/A':>12} "
        lines.append(row)

    lines.append("-" * 120)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize transfer experiment results")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--dataset", type=str, default="default", help="Dataset name")
    parser.add_argument("--all_datasets", action="store_true", help="Summarize all datasets")
    parser.add_argument("--all_models", action="store_true", help="Summarize all models")
    parser.add_argument("--full_matrix", action="store_true", help="Show full matrix")
    parser.add_argument("--result_dir", type=str, default=None, help="Results directory")

    args = parser.parse_args()

    # 确定 result_dir
    script_dir = Path(__file__).parent
    if args.result_dir:
        result_dir = Path(args.result_dir)
    else:
        result_dir = script_dir.parent / "results"

    # 数据集列表
    datasets = ["default", "advbench", "harmbench_contextual", "harmbench_standard", "jailbreakBench"]

    # 方法定义
    baselines = ["no_rewrite", "pair", "autodan", "deepinception", "persona"]
    our_methods = ["pair_skills_28", "autodan_skills_1", "autodan_skills_54"]

    # === 单个模型 + 单个数据集 ===
    if args.model and not args.all_datasets and not args.all_models:
        summary = summarize_dataset(args.model, args.dataset, result_dir)
        print(format_single_summary(summary))

        # 保存
        summary_path = result_dir / args.model / args.dataset / "SUMMARY.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {summary_path}")

    # === 单个模型 + 所有数据集 ===
    elif args.model and args.all_datasets:
        all_summaries = []
        for dataset in datasets:
            summary = summarize_dataset(args.model, dataset, result_dir)
            if summary.get("results"):
                all_summaries.append(summary)
                print(format_single_summary(summary))

        # 跨数据集对比表
        print(format_cross_dataset_table(all_summaries))

        # 保存全部汇总
        all_path = result_dir / args.model / "ALL_DATASETS_SUMMARY.json"
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {all_path}")

    # === 所有模型 ===
    elif args.all_models:
        # 查找所有模型
        models = []
        for p in result_dir.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                models.append(p.name)

        all_summaries = []
        for model in sorted(models):
            if args.all_datasets:
                for dataset in datasets:
                    summary = summarize_dataset(model, dataset, result_dir)
                    if summary.get("results"):
                        all_summaries.append(summary)
            else:
                summary = summarize_dataset(model, args.dataset, result_dir)
                if summary.get("results"):
                    all_summaries.append(summary)
                    print(format_single_summary(summary))

        # 对比表
        if args.all_datasets:
            print(format_full_matrix(all_summaries))
        else:
            print(format_cross_model_table(all_summaries))

        # 保存
        all_path = result_dir / "ALL_SUMMARY.json"
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {all_path}")

    # === 全矩阵 ===
    elif args.full_matrix:
        models = []
        for p in result_dir.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                models.append(p.name)

        all_summaries = []
        for model in sorted(models):
            for dataset in datasets:
                summary = summarize_dataset(model, dataset, result_dir)
                if summary.get("results"):
                    all_summaries.append(summary)

        print(format_full_matrix(all_summaries))

        all_path = result_dir / "FULL_MATRIX.json"
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {all_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()