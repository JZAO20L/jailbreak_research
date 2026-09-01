#!/usr/bin/env python3
"""
实验2评估结果汇总脚本 (重做版本)

根据 TODO.md 实验重做 - 实验2:
- 实验格式: {attack_prompt}_{weight_config}
- 攻击prompt: creative_writing, hypothetical_scenario, role_playing
- 权重配置: intent_only, stealth_only, strategy_only, potential_only, uniform

自动扫描输出目录，汇总所有实验的ASR结果
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# 已知的攻击prompt和权重配置名称（用于正确解析实验名称）
ATTACK_PROMPTS = [
    "creative_writing",
    "hypothetical_scenario",
    "role_playing",
]

WEIGHT_CONFIGS = [
    "intent_only",
    "stealth_only",
    "strategy_only",
    "potential_only",
    "uniform",
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
    """解析实验名称，提取攻击prompt和权重配置"""
    # 格式: {attack_prompt}_{weight_config}
    # 例如: creative_writing_intent_only
    #       hypothetical_scenario_uniform

    for attack in ATTACK_PROMPTS:
        if exp_name.startswith(attack + "_"):
            weight = exp_name[len(attack) + 1:]  # 去掉attack名和下划线
            return {
                "attack_prompt": attack,
                "weight_config": weight,
            }

    # 如果无法解析，返回原始名称
    return {"attack_prompt": exp_name, "weight_config": "unknown"}


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
        if exp_name in ["baseline_original_prompt", "baseline_base_model", "logs"]:
            continue

        # 处理正式实验
        # 格式: {attack_prompt}_{weight_config}/eval_results/eval_{attack_prompt}_{weight_config}/summary.json
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
                    "attack_prompt": parsed["attack_prompt"],
                    "weight_config": parsed["weight_config"],
                    "asr": asr,
                    "raw_summary": summary
                }

    return results


def format_results(results: Dict[str, Dict]) -> str:
    """格式化结果为可读文本"""
    lines = []
    lines.append("=" * 80)
    lines.append("实验2评估结果汇总 (重做版本)")
    lines.append("=" * 80)
    lines.append("")

    # 按攻击prompt分组显示实验结果
    lines.append("-" * 40)
    lines.append("实验结果 (按攻击prompt分组):")
    lines.append("-" * 40)

    # 提取所有攻击prompt
    attacks = set()
    for name, data in results.items():
        attacks.add(data["attack_prompt"])

    for attack in sorted(attacks):
        lines.append(f"\n  [{attack}]")
        attack_results = []
        for name, data in results.items():
            if data["attack_prompt"] == attack:
                asr = data.get("asr", 0) or 0
                attack_results.append((data["weight_config"], asr))

        # 按权重配置排序
        attack_results.sort(key=lambda x: WEIGHT_CONFIGS.index(x[0]) if x[0] in WEIGHT_CONFIGS else 99)
        for weight, asr in attack_results:
            lines.append(f"    {weight}: ASR = {asr:.1%}")

    lines.append("")
    lines.append("-" * 40)
    lines.append("汇总表格:")
    lines.append("-" * 40)

    # 生成表格
    lines.append("")
    header = "攻击prompt          " + "  " + "  ".join([w[:10] for w in WEIGHT_CONFIGS])
    lines.append(header)
    lines.append("-" * len(header))

    for attack in ATTACK_PROMPTS:
        row = f"{attack:<20}"
        for weight in WEIGHT_CONFIGS:
            exp_key = f"{attack}_{weight}"
            if exp_key in results:
                asr = results[exp_key].get("asr", 0) or 0
                row += f"  {asr:>7.1%}"
            else:
                row += "      N/A"
        lines.append(row)

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_markdown_table(results: Dict[str, Dict]) -> str:
    """生成Markdown格式的表格"""
    lines = []
    lines.append("## 实验2评估结果 (重做版本)\n")
    lines.append(f"攻击prompt: {ATTACK_PROMPTS}\n")
    lines.append(f"权重配置: {WEIGHT_CONFIGS}\n")

    lines.append("\n### 结果表格\n")
    header = "| 攻击prompt | " + " | ".join(WEIGHT_CONFIGS) + " |"
    lines.append(header)
    separator = "|------------|" + "|".join(["-" * 12] * len(WEIGHT_CONFIGS)) + "|"
    lines.append(separator)

    for attack in ATTACK_PROMPTS:
        row = f"| {attack} |"
        for weight in WEIGHT_CONFIGS:
            exp_key = f"{attack}_{weight}"
            if exp_key in results:
                asr = results[exp_key].get("asr", 0) or 0
                row += f" {asr:.1%} |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.append("\n### 详细结果\n")
    for attack in ATTACK_PROMPTS:
        lines.append(f"\n#### {attack}\n")
        lines.append("| 权重配置 | ASR |")
        lines.append("|----------|-----|")

        for weight in WEIGHT_CONFIGS:
            exp_key = f"{attack}_{weight}"
            if exp_key in results:
                asr = results[exp_key].get("asr", 0) or 0
                lines.append(f"| {weight} | {asr:.1%} |")

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