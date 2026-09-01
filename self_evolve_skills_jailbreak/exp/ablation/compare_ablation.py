#!/usr/bin/env python3
"""
组件消融对比分析脚本

对比 Layer 1 (完整流程) 和组件消融 (跳过 Evolution) 的实验结果，
量化 Evolution 阶段的贡献。

Usage:
    python compare_ablation.py --layer1 results_layer1.json --ablation results_ablation.json
"""

import json
import argparse
from pathlib import Path


def load_results(path: str) -> dict:
    """加载实验结果"""
    with open(path, "r") as f:
        return json.load(f)


def compare_experiments(layer1_path: str, ablation_path: str, output_path: str = None):
    """对比两组实验"""

    layer1 = load_results(layer1_path)
    ablation = load_results(ablation_path)

    print("=" * 70)
    print("组件消融对比分析")
    print("=" * 70)
    print(f"\nLayer 1 结果: {layer1_path}")
    print(f"组件消融结果: {ablation_path}")
    print()

    # 提取方法组合结果
    layer1_results = layer1.get("results", {})
    ablation_results = ablation.get("results", {})

    # 构建对比表
    comparison = []

    print(f"{'Method':<35} {'Layer1 ASR':>12} {'Ablation ASR':>12} {'Evolution Δ':>12} {'Layer1 Skills':>10} {'Ablation Skills':>10}")
    print("-" * 95)

    for method in layer1_results:
        if method not in ablation_results:
            continue

        l1 = layer1_results[method]
        ab = ablation_results[method]

        l1_asr = l1.get("asr", 0) * 100
        ab_asr = ab.get("asr", 0) * 100
        evolution_delta = l1_asr - ab_asr  # Evolution 贡献（正值表示 Evolution 有贡献）

        l1_skills = l1.get("evolution_skills", l1.get("final_skill_count", 0))
        ab_skills = ab.get("cold_start_skills", ab.get("final_skill_count", 0))

        comparison.append({
            "method": method,
            "layer1_asr": l1_asr,
            "ablation_asr": ab_asr,
            "evolution_delta": evolution_delta,
            "layer1_skills": l1_skills,
            "ablation_skills": ab_skills,
        })

        print(f"{method:<35} {l1_asr:>11.1f}% {ab_asr:>11.1f}% {evolution_delta:>11.1f}% {l1_skills:>10} {ab_skills:>10}")

    print("-" * 95)

    # 统计汇总
    avg_layer1_asr = sum(c["layer1_asr"] for c in comparison) / len(comparison)
    avg_ablation_asr = sum(c["ablation_asr"] for c in comparison) / len(comparison)
    avg_evolution_delta = sum(c["evolution_delta"] for c in comparison) / len(comparison)

    print(f"{'Average':<35} {avg_layer1_asr:>11.1f}% {avg_ablation_asr:>11.1f}% {avg_evolution_delta:>11.1f}%")

    print("\n分析结论:")
    print("=" * 70)

    if avg_evolution_delta > 5:
        print(f"✓ Evolution 阶段贡献显著: 平均提升 ASR {avg_evolution_delta:.1f}%")
        print("  建议: 保持完整流程，Evolution 阶段对最终效果有重要作用")
    elif avg_evolution_delta > 0:
        print(f"△ Evolution 阶段有轻微贡献: 平均提升 ASR {avg_evolution_delta:.1f}%")
        print("  建议: Evolution 阶段有一定价值，但可考虑简化（减少轮数或数据量）")
    else:
        print(f"○ Evolution 阶段贡献有限: 平均差异 {avg_evolution_delta:.1f}%")
        print("  建议: Cold Start 已足够，可考虑简化流程")

    # 找出最佳方法组合
    best_layer1 = max(comparison, key=lambda c: c["layer1_asr"])
    best_ablation = max(comparison, key=lambda c: c["ablation_asr"])

    print(f"\n最佳方法组合 (Layer 1): {best_layer1['method']} (ASR={best_layer1['layer1_asr']:.1f}%)")
    print(f"最佳方法组合 (组件消融): {best_ablation['method']} (ASR={best_ablation['ablation_asr']:.1f}%)")

    # 保存对比结果
    if output_path:
        output = {
            "comparison": comparison,
            "summary": {
                "avg_layer1_asr": avg_layer1_asr,
                "avg_ablation_asr": avg_ablation_asr,
                "avg_evolution_delta": avg_evolution_delta,
                "evolution_significant": avg_evolution_delta > 5,
                "best_layer1_method": best_layer1["method"],
                "best_ablation_method": best_ablation["method"],
            }
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n对比结果已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="组件消融对比分析")
    parser.add_argument("--layer1", type=str, required=True, help="Layer 1 结果文件路径")
    parser.add_argument("--ablation", type=str, required=True, help="组件消融结果文件路径")
    parser.add_argument("--output", type=str, default=None, help="对比结果输出路径")

    args = parser.parse_args()
    compare_experiments(args.layer1, args.ablation, args.output)


if __name__ == "__main__":
    main()