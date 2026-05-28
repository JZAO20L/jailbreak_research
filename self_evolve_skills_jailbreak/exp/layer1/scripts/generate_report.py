"""
Layer 1 Report Generator

生成 Markdown 格式的实验报告

Usage:
    python generate_report.py --input results/layer1_summary.json --output results/layer1_report.md
"""

import os
import sys
import json
import argparse
from datetime import datetime


def generate_report(summary_data: dict) -> str:
    """生成 Markdown 报告"""

    lines = []

    # 标题
    lines.append("# Layer 1 Grid Search Experiment Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 实验概述
    lines.append("## Experiment Overview")
    lines.append("")
    lines.append("| Config | Value |")
    lines.append("|--------|-------|")
    if summary_data.get("config"):
        for k, v in summary_data["config"].items():
            lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("- Total combinations: 16 (2 × 2 × 4)")
    lines.append("- Ablation A: skill_call_mode (single_call | every_iteration)")
    lines.append("- Ablation B: skill_extraction_mode (final_prompt | trajectory)")
    lines.append("- Ablation C: update_strategy (success_only | failure_only | both | statistical)")
    lines.append("")

    # 结果汇总表
    lines.append("## Results Summary")
    lines.append("")
    lines.append("| Method | ASR | Avg Iterations | Skill Count |")
    lines.append("|--------|-----|----------------|-------------|")

    # 按 ASR 排序
    sorted_results = sorted(
        summary_data.get("results", {}).items(),
        key=lambda x: x[1]["asr"],
        reverse=True
    )

    for key, r in sorted_results:
        asr_pct = r["asr"] * 100
        lines.append(f"| {key} | {asr_pct:.1f}% | {r['avg_iterations']:.2f} | {r['final_skill_count']} |")

    lines.append("")

    # 最佳组合
    if sorted_results:
        best_key, best_r = sorted_results[0]
        lines.append("## Best Combination")
        lines.append("")
        lines.append(f"**{best_key}**")
        lines.append("")
        lines.append(f"- ASR: **{best_r['asr']*100:.1f}%**")
        lines.append(f"- Average Iterations: {best_r['avg_iterations']:.2f}")
        lines.append(f"- Final Skill Count: {best_r['final_skill_count']}")
        lines.append("")

    # 消融分析
    lines.append("## Ablation Analysis")
    lines.append("")

    ablation = summary_data.get("ablation_analysis", {})

    # skill_call_mode
    if ablation.get("skill_call_mode"):
        lines.append("### A: skill_call_mode")
        lines.append("")
        lines.append("| Mode | Avg ASR | Count |")
        lines.append("|------|---------|-------|")
        sorted_options = sorted(
            ablation["skill_call_mode"].items(),
            key=lambda x: x[1]["avg_asr"],
            reverse=True
        )
        for option, stats in sorted_options:
            lines.append(f"| {option} | {stats['avg_asr']*100:.1f}% | {stats['count']} |")
        lines.append("")

    # extraction_mode
    if ablation.get("extraction_mode"):
        lines.append("### B: extraction_mode")
        lines.append("")
        lines.append("| Mode | Avg ASR | Count |")
        lines.append("|------|---------|-------|")
        sorted_options = sorted(
            ablation["extraction_mode"].items(),
            key=lambda x: x[1]["avg_asr"],
            reverse=True
        )
        for option, stats in sorted_options:
            lines.append(f"| {option} | {stats['avg_asr']*100:.1f}% | {stats['count']} |")
        lines.append("")

    # update_strategy
    if ablation.get("update_strategy"):
        lines.append("### C: update_strategy")
        lines.append("")
        lines.append("| Strategy | Avg ASR | Count |")
        lines.append("|----------|---------|-------|")
        sorted_options = sorted(
            ablation["update_strategy"].items(),
            key=lambda x: x[1]["avg_asr"],
            reverse=True
        )
        for option, stats in sorted_options:
            lines.append(f"| {option} | {stats['avg_asr']*100:.1f}% | {stats['count']} |")
        lines.append("")

    # 进化曲线（如果有）
    lines.append("## Evolution Curves")
    lines.append("")

    has_evals = False
    for key, r in sorted_results:
        evals = r.get("intermediate_evals", [])
        if evals:
            has_evals = True
            lines.append(f"### {key}")
            lines.append("")
            lines.append("| Epoch | ASR | Avg Iter | Skill Count |")
            lines.append("|-------|-----|----------|-------------|")
            for e in evals:
                asr_pct = e.get("asr", 0) * 100
                lines.append(f"| {e.get('epoch', 'N/A')} | {asr_pct:.1f}% | {e.get('avg_iterations', 0):.2f} | {e.get('skill_count', 0)} |")
            lines.append("")

    if not has_evals:
        lines.append("No intermediate evaluation data available.")
        lines.append("")

    # 结论
    lines.append("## Conclusions")
    lines.append("")
    if sorted_results:
        best = sorted_results[0]
        lines.append(f"1. Best performing combination: **{best[0]}** with ASR = {best[1]['asr']*100:.1f}%")
        lines.append("")

        # 基于消融分析的结论
        if ablation.get("skill_call_mode"):
            best_call = max(ablation["skill_call_mode"].items(), key=lambda x: x[1]["avg_asr"])
            lines.append(f"2. For skill_call_mode: **{best_call[0]}** performs better (avg ASR = {best_call[1]['avg_asr']*100:.1f}%)")

        if ablation.get("extraction_mode"):
            best_extract = max(ablation["extraction_mode"].items(), key=lambda x: x[1]["avg_asr"])
            lines.append(f"3. For extraction_mode: **{best_extract[0]}** performs better (avg ASR = {best_extract[1]['avg_asr']*100:.1f}%)")

        if ablation.get("update_strategy"):
            best_strategy = max(ablation["update_strategy"].items(), key=lambda x: x[1]["avg_asr"])
            lines.append(f"4. For update_strategy: **{best_strategy[0]}** performs better (avg ASR = {best_strategy[1]['avg_asr']*100:.1f}%)")

    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Layer 1 Report")

    parser.add_argument("--input", type=str, required=True,
                        help="汇总 JSON 文件路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 Markdown 文件路径")

    args = parser.parse_args()

    # 加载汇总数据
    with open(args.input, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)

    # 生成报告
    report = generate_report(summary_data)

    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已生成: {args.output}")


if __name__ == "__main__":
    main()