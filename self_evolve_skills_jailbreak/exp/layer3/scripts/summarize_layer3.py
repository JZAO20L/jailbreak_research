"""
Layer 3 结果汇总脚本

汇总所有实验结果，生成统计报告
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import statistics


def summarize_layer3_results(input_dir: Path) -> Dict[str, Any]:
    """
    汇总 Layer 3 实验结果

    Args:
        input_dir: 结果目录

    Returns:
        汇总数据
    """
    # 读取所有结果文件
    result_files = list(input_dir.glob("result_dan_start_*.json"))

    results = []
    for f in result_files:
        with open(f) as fp:
            data = json.load(fp)
            results.append(data)

    if not results:
        return {"error": "No results found"}

    # 按 data_size 和 ratio 分组统计
    by_data_size = {}
    by_ratio = {}
    by_combination = {}

    for r in results:
        config = r.get("config", {})
        data_size = config.get("data_size_name", "unknown")
        ratio_name = config.get("ratio_name", "unknown")
        test_result = r.get("results", {}).get("test", {})
        asr = test_result.get("asr", 0)

        key = f"{data_size}_{ratio_name}"

        # 按 data_size 分组
        if data_size not in by_data_size:
            by_data_size[data_size] = []
        by_data_size[data_size].append(asr)

        # 按 ratio 分组
        if ratio_name not in by_ratio:
            by_ratio[ratio_name] = []
        by_ratio[ratio_name].append(asr)

        # 按 combination 分组
        by_combination[key] = {
            "data_size": data_size,
            "ratio": ratio_name,
            "asr": asr,
            "exp_id": r.get("exp_id"),
            "skills": test_result.get("final_skill_count", 0),
        }

    # 计算平均值
    avg_by_data_size = {
        k: statistics.mean(v) for k, v in by_data_size.items()
    }

    avg_by_ratio = {
        k: statistics.mean(v) for k, v in by_ratio.items()
    }

    # 找最佳配置
    best_config = max(by_combination.values(), key=lambda x: x["asr"])

    # 计算 full_evove vs 有冷启动对比
    full_evolve_avg = statistics.mean(by_ratio.get("full_evolve", [0]))
    with_cs_avg = statistics.mean([
        asr for ratio, asrs in by_ratio.items()
        if ratio != "full_evolve"
        for asr in asrs
    ])

    # 与AutoDAN基准对比
    autodan_baseline = 0.863  # AutoDAN ASR 86.3%

    comparison = {
        "autodan_baseline": autodan_baseline,
        "best_layer3": best_config["asr"],
        "gap_to_autodan": best_config["asr"] - autodan_baseline,
        "full_evolve_avg": full_evolve_avg,
        "with_cs_avg": with_cs_avg,
        "full_evolve_advantage": full_evolve_avg - with_cs_avg,
    }

    # 完整结果表
    results_table = []
    for r in sorted(results, key=lambda x: x.get("exp_id", 0)):
        config = r.get("config", {})
        test_result = r.get("results", {}).get("test", {})
        cold_start = r.get("results", {}).get("cold_start", {})
        evolution = r.get("results", {}).get("evolution", {})

        results_table.append({
            "exp_id": r.get("exp_id"),
            "data_size": config.get("data_size_name"),
            "data_size_value": config.get("data_size"),
            "ratio": config.get("ratio_name"),
            "ratio_value": config.get("ratio"),
            "cold_start_size": config.get("cold_start_size"),
            "evolution_size": config.get("evolution_size"),
            "asr": round(test_result.get("asr", 0) * 100, 1),
            "success": test_result.get("success", 0),
            "total": test_result.get("total", 0),
            "avg_iterations": round(test_result.get("avg_iterations", 0), 2),
            "cs_skills": cold_start.get("final_skill_count", 0),
            "evo_skills": evolution.get("final_skill_count", 0),
            "final_skills": test_result.get("final_skill_count", 0),
            "elapsed_time": round(r.get("elapsed_time", 0), 1),
        })

    # 汇总
    summary = {
        "layer": 3,
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "successful": len([r for r in results if r.get("success", True)]),
        "failed": len([r for r in results if not r.get("success", True)]),

        # 基准对比
        "comparison": comparison,

        # 最佳配置
        "best_config": best_config,

        # 分组统计
        "avg_by_data_size": {k: round(v * 100, 1) for k, v in avg_by_data_size.items()},
        "avg_by_ratio": {k: round(v * 100, 1) for k, v in avg_by_ratio.items()},

        # 完整结果表
        "results_table": results_table,

        # 关键发现
        "key_findings": {
            "best_asr": best_config["asr"],
            "best_data_size": best_config["data_size"],
            "best_ratio": best_config["ratio"],
            "full_evolve_effect": full_evolve_avg - with_cs_avg,
            "data_size_effect": max(avg_by_data_size.values()) - min(avg_by_data_size.values()),
            "ratio_effect": max(avg_by_ratio.values()) - min(avg_by_ratio.values()),
        },
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Layer 3 Results Summarizer")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, required=True, help="Output file")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    summary = summarize_layer3_results(input_dir)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n汇总完成: {output_path}")
    print(f"\n关键发现:")
    print(f"  最佳ASR: {summary['key_findings']['best_asr']*100:.1f}%")
    print(f"  最佳配置: {summary['best_config']['data_size']} / {summary['best_config']['ratio']}")
    print(f"  与AutoDAN差距: {summary['comparison']['gap_to_autodan']*100:+.1f}%")
    print(f"  full_evove优势: {summary['key_findings']['full_evolve_effect']*100:+.1f}%")


if __name__ == "__main__":
    main()