#!/usr/bin/env python3
"""
实验2评估结果汇总脚本
自动扫描输出目录，汇总所有实验的ASR结果
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# 已知的策略和维度名称（用于正确解析实验名称）
KNOWN_STRATEGIES = [
    "hypothetical_scenario",
    "creative_writing",
    "role_playing",
]

KNOWN_DIMENSIONS = [
    "idea_preservation",
    "stealthiness",
    "naturalness",
    "hypothetical_scenario",  # 专用维度
    "creative_writing",       # 专用维度
    "role_playing",           # 专用维度
]


def load_summary(summary_path: Path) -> Optional[Dict]:
    """加载summary.json文件"""
    if not summary_path.exists():
        return None
    try:
        with open(summary_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def extract_asr(summary: Dict) -> Optional[float]:
    """从summary中提取ASR"""
    if not summary or "combinations" not in summary:
        return None
    combos = summary.get("combinations", [])
    if not combos:
        return None
    return combos[0].get("asr")


def parse_exp_name(exp_name: str) -> Dict[str, str]:
    """解析实验名称，提取策略和维度"""
    # 格式: {strategy}_{dimension}_single
    # 例如: hypothetical_scenario_idea_preservation_single
    #       hypothetical_scenario_hypothetical_scenario_single

    scoring = "unknown"
    name_without_scoring = exp_name

    # 移除评分方式后缀
    if exp_name.endswith("_single"):
        scoring = "single"
        name_without_scoring = exp_name[:-7]  # 去掉 "_single"
    elif exp_name.endswith("_tournament"):
        scoring = "tournament"
        name_without_scoring = exp_name[:-11]  # 去掉 "_tournament"

    # 尝试匹配已知的策略和维度
    for strategy in KNOWN_STRATEGIES:
        if name_without_scoring.startswith(strategy + "_"):
            dimension = name_without_scoring[len(strategy) + 1:]  # 去掉策略名和下划线
            return {
                "strategy": strategy,
                "dimension": dimension,
                "scoring": scoring
            }

    # 如果无法解析，返回原始名称
    return {"strategy": exp_name, "dimension": "unknown", "scoring": scoring}


def collect_results(output_dir: Path) -> Dict[str, Dict]:
    """收集所有实验结果"""
    results = {}

    # 遍历所有实验目录
    for exp_dir in output_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        if exp_dir.name.startswith(".") or exp_dir.name.startswith("__"):
            continue

        exp_name = exp_dir.name

        # 跳过非实验目录
        if exp_name in ["baseline_original_prompt", "baseline_base_model"]:
            # 处理baseline
            summary_path = exp_dir / exp_name / "summary.json"
            summary = load_summary(summary_path)
            if summary:
                asr = extract_asr(summary)
                results[exp_name] = {
                    "type": "baseline",
                    "asr": asr,
                    "raw_summary": summary
                }
            continue

        # 处理正式实验
        # 格式: {strategy}_{dimension}_{scoring}/eval_results/eval_{strategy}_{dimension}_{scoring}/summary.json
        eval_dir = exp_dir / "eval_results"
        if not eval_dir.exists():
            continue

        for eval_subdir in eval_dir.iterdir():
            if not eval_subdir.is_dir():
                continue
            summary_path = eval_subdir / "summary.json"
            summary = load_summary(summary_path)
            if summary:
                asr = extract_asr(summary)
                parsed = parse_exp_name(exp_name)
                results[exp_name] = {
                    "type": "experiment",
                    "strategy": parsed["strategy"],
                    "dimension": parsed["dimension"],
                    "scoring": parsed["scoring"],
                    "asr": asr,
                    "raw_summary": summary
                }

    return results


def format_results(results: Dict[str, Dict]) -> str:
    """格式化结果为可读文本"""
    lines = []
    lines.append("=" * 80)
    lines.append("实验2评估结果汇总")
    lines.append("=" * 80)
    lines.append("")

    # Baseline结果
    lines.append("-" * 40)
    lines.append("Baseline结果:")
    lines.append("-" * 40)
    for name in ["baseline_original_prompt", "baseline_base_model"]:
        if name in results:
            data = results[name]
            asr = data.get("asr", 0) or 0
            lines.append(f"  {name}: ASR = {asr:.1%}")
    lines.append("")

    # 按策略分组显示实验结果
    lines.append("-" * 40)
    lines.append("实验结果 (按策略分组):")
    lines.append("-" * 40)

    # 提取所有策略
    strategies = set()
    for name, data in results.items():
        if data["type"] == "experiment":
            strategies.add(data["strategy"])

    for strategy in sorted(strategies):
        lines.append(f"\n  [{strategy}]")
        strategy_results = []
        for name, data in results.items():
            if data["type"] == "experiment" and data["strategy"] == strategy:
                asr = data.get("asr", 0) or 0
                strategy_results.append((data["dimension"], asr))

        # 按维度排序
        strategy_results.sort(key=lambda x: x[0])
        for dim, asr in strategy_results:
            lines.append(f"    {dim}: ASR = {asr:.1%}")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_markdown_table(results: Dict[str, Dict]) -> str:
    """生成Markdown格式的表格"""
    lines = []
    lines.append("## 实验2评估结果\n")
    lines.append("### Baseline\n")
    lines.append("| 实验名称 | ASR |")
    lines.append("|----------|-----|")
    for name in ["baseline_original_prompt", "baseline_base_model"]:
        if name in results:
            asr = results[name].get("asr", 0) or 0
            lines.append(f"| {name} | {asr:.1%} |")

    lines.append("\n### 实验结果\n")

    # 按策略分组
    strategies = set()
    for name, data in results.items():
        if data["type"] == "experiment":
            strategies.add(data["strategy"])

    for strategy in sorted(strategies):
        lines.append(f"\n#### {strategy}\n")
        lines.append("| 维度 | ASR |")
        lines.append("|------|-----|")

        strategy_results = []
        for name, data in results.items():
            if data["type"] == "experiment" and data["strategy"] == strategy:
                asr = data.get("asr", 0) or 0
                strategy_results.append((data["dimension"], asr))

        strategy_results.sort(key=lambda x: x[0])
        for dim, asr in strategy_results:
            lines.append(f"| {dim} | {asr:.1%} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="汇总实验2评估结果")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录路径 (默认: 脚本所在目录的judge_prompt_exp_output子目录)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "markdown", "json"],
        default="text",
        help="输出格式"
    )
    args = parser.parse_args()

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "judge_prompt_exp_output"

    if not output_dir.exists():
        print(f"错误: 输出目录不存在: {output_dir}")
        return 1

    # 收集结果
    results = collect_results(output_dir)

    if not results:
        print(f"警告: 未在 {output_dir} 中找到任何评估结果")
        return 0

    # 输出结果
    if args.format == "text":
        print(format_results(results))
    elif args.format == "markdown":
        print(generate_markdown_table(results))
    else:  # json
        print(json.dumps(results, indent=2, ensure_ascii=False))

    # 保存汇总文件
    summary_file = output_dir / "results_summary.md"
    with open(summary_file, "w") as f:
        f.write(generate_markdown_table(results))
    print(f"\n汇总结果已保存到: {summary_file}")

    return 0


if __name__ == "__main__":
    exit(main())