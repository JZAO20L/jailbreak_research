"""
Layer 2 结果汇总脚本
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def summarize_layer2(input_dir: str, output_file: str):
    """汇总 Layer 2 实验结果"""
    input_path = Path(input_dir)

    # 查找所有 meta 文件
    meta_files = list(input_path.glob("meta_*.json"))
    result_files = list(input_path.glob("result_*.json"))

    print(f"Found {len(meta_files)} meta files")
    print(f"Found {len(result_files)} result files")

    experiments = []

    for meta_file in sorted(meta_files):
        try:
            with open(meta_file) as f:
                data = json.load(f)
                experiments.append(data)
        except Exception as e:
            print(f"Error reading {meta_file}: {e}")

    # 按实验 ID 排序
    experiments.sort(key=lambda x: x.get("experiment_id", 0))

    # 汇总统计
    successful = [e for e in experiments if e.get("success") and "asr" in e]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(experiments),
        "successful": len(successful),
        "failed": len(experiments) - len(successful),
        "experiments": experiments,
    }

    # 消融分析
    if successful:
        # 按数据量分析
        by_data_size = {}
        for exp in successful:
            size = exp.get("data_size")
            if size not in by_data_size:
                by_data_size[size] = {"asrs": [], "iters": []}
            by_data_size[size]["asrs"].append(exp["asr"])
            by_data_size[size]["iters"].append(exp.get("avg_iterations", 0))

        summary["by_data_size"] = {
            k: {
                "avg_asr": sum(v["asrs"]) / len(v["asrs"]),
                "avg_iter": sum(v["iters"]) / len(v["iters"]),
                "count": len(v["asrs"]),
            }
            for k, v in by_data_size.items()
        }

        # 按配比分析
        by_ratio = {}
        for exp in successful:
            ratio = exp.get("ratio")
            if ratio not in by_ratio:
                by_ratio[ratio] = {"asrs": [], "iters": []}
            by_ratio[ratio]["asrs"].append(exp["asr"])
            by_ratio[ratio]["iters"].append(exp.get("avg_iterations", 0))

        summary["by_ratio"] = {
            k: {
                "avg_asr": sum(v["asrs"]) / len(v["asrs"]),
                "avg_iter": sum(v["iters"]) / len(v["iters"]),
                "count": len(v["asrs"]),
            }
            for k, v in by_ratio.items()
        }

        # 按方法分析
        by_method = {}
        for exp in successful:
            method = exp.get("method", {})
            method_name = f"{method.get('skill_call_mode', '')}_{method.get('skill_extraction_mode', '')}_{method.get('update_strategy', '')}"
            if method_name not in by_method:
                by_method[method_name] = {"asrs": [], "iters": []}
            by_method[method_name]["asrs"].append(exp["asr"])
            by_method[method_name]["iters"].append(exp.get("avg_iterations", 0))

        summary["by_method"] = {
            k: {
                "avg_asr": sum(v["asrs"]) / len(v["asrs"]),
                "avg_iter": sum(v["iters"]) / len(v["iters"]),
                "count": len(v["asrs"]),
            }
            for k, v in by_method.items()
        }

        # 最佳组合
        sorted_by_asr = sorted(successful, key=lambda x: x["asr"], reverse=True)
        summary["best_combination"] = sorted_by_asr[0] if sorted_by_asr else None

    # 保存汇总
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {output_file}")

    # 打印汇总
    print("\n" + "="*60)
    print("Layer 2 Results Summary")
    print("="*60)

    if successful:
        print("\nBy data_size:")
        for size, stats in summary.get("by_data_size", {}).items():
            print(f"  {size}: ASR={stats['avg_asr']*100:.1f}%, Iter={stats['avg_iter']:.2f}")

        print("\nBy ratio:")
        for ratio, stats in summary.get("by_ratio", {}).items():
            print(f"  {ratio}: ASR={stats['avg_asr']*100:.1f}%, Iter={stats['avg_iter']:.2f}")

        print("\nBy method:")
        for method, stats in summary.get("by_method", {}).items():
            print(f"  {method}: ASR={stats['avg_asr']*100:.1f}%")

        if summary.get("best_combination"):
            best = summary["best_combination"]
            print(f"\nBest combination:")
            print(f"  Method: {best.get('method')}")
            print(f"  Data: {best.get('data_size')}, Ratio: {best.get('ratio')}")
            print(f"  ASR: {best['asr']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Summarize Layer 2 results")
    parser.add_argument("--input", required=True, help="Input directory with result files")
    parser.add_argument("--output", required=True, help="Output summary file")

    args = parser.parse_args()
    summarize_layer2(args.input, args.output)


if __name__ == "__main__":
    main()