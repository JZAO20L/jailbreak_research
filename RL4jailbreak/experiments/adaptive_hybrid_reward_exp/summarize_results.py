#!/usr/bin/env python3
"""
Experiment 3: Adaptive Hybrid Reward GRPO - Results Summary Script.

Scans the output directory and summarizes ASR results across different
EMA beta values, attack prompts, and judge dimensions.

Usage:
    python summarize_results.py                          # Auto-detect output dir
    python summarize_results.py --output_dir /path/to/output
    python summarize_results.py --format markdown         # Output markdown table
    python summarize_results.py --format json             # Output JSON
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


# =============================================================================
# Known values for parsing experiment names
# =============================================================================

KNOWN_EMA_BETAS = ["0", "0.5", "0.67", "0.8", "0.9", "0.95"]

KNOWN_ATTACK_PROMPTS = [
    "hypothetical_scenario",
    "creative_writing",
    "role_playing",
]

KNOWN_JUDGE_DIMENSIONS = [
    "idea_preservation",
    "stealthiness",
    "naturalness",
    "hypothetical_scenario",
    "creative_writing",
    "role_playing",
]

# Baseline directories (from experiment 2)
KNOWN_BASELINES = ["baseline_original_prompt", "baseline_base_model"]


# =============================================================================
# Parsing Helpers
# =============================================================================

def parse_exp_name(exp_name: str) -> Dict[str, str]:
    """
    Parse experiment directory name.

    Format: ema{beta}_{attack_prompt}_{judge_prompt}
    Example: ema0.95_hypothetical_scenario_stealthiness
    """
    # Try to match known pattern
    for ema_beta in KNOWN_EMA_BETAS:
        prefix = f"ema{ema_beta}_"
        if exp_name.startswith(prefix):
            rest = exp_name[len(prefix):]

            # Try to match attack prompt
            for attack_prompt in KNOWN_ATTACK_PROMPTS:
                attack_prefix = attack_prompt + "_"
                if rest.startswith(attack_prefix):
                    judge_prompt = rest[len(attack_prefix):]
                    return {
                        "ema_beta": ema_beta,
                        "attack_prompt": attack_prompt,
                        "judge_prompt": judge_prompt,
                        "window_size": _beta_to_window(ema_beta),
                    }

            # If attack prompt not matched, treat rest as combined
            return {
                "ema_beta": ema_beta,
                "attack_prompt": rest,
                "judge_prompt": "unknown",
                "window_size": _beta_to_window(ema_beta),
            }

    # Baseline or unknown format
    return {
        "ema_beta": "N/A",
        "attack_prompt": exp_name,
        "judge_prompt": "N/A",
        "window_size": "N/A",
    }


def _beta_to_window(beta_str: str) -> str:
    """Convert EMA beta to approximate window size."""
    try:
        beta = float(beta_str)
        if beta >= 1.0:
            return "inf"
        window = int(round(1.0 / (1.0 - beta)))
        return str(window)
    except (ValueError, ZeroDivisionError):
        return "1"


# =============================================================================
# Result Collection
# =============================================================================

def load_eval_summary(eval_dir: Path) -> Optional[float]:
    """
    Load ASR from eval summary.json.

    Expected structure (from exp.sh):
    ema{beta}_{attack}_{judge}/eval_results/eval_ema{beta}_{attack}_{judge}/summary.json
    {
        "combinations": [
            {"lora": "final_lora", "asr": 0.42, ...}
        ]
    }
    """
    if not eval_dir.exists():
        return None

    # The eval.py creates a subdirectory under eval_results/
    # Structure: eval_results/eval_{exp_key}/summary.json
    for eval_subdir in eval_dir.iterdir():
        if not eval_subdir.is_dir():
            continue
        summary_path = eval_subdir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    data = json.load(f)
                combos = data.get("combinations", [])
                if combos:
                    return combos[0].get("asr")
            except (json.JSONDecodeError, IOError, KeyError):
                pass

    return None


def load_lambda_history(exp_dir: Path) -> Optional[List[Dict]]:
    """Load lambda history from training output."""
    lambda_path = exp_dir / "lambda_history.json"
    if not lambda_path.exists():
        return None
    try:
        with open(lambda_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def collect_results(output_dir: Path) -> Dict[str, Dict]:
    """
    Collect all experiment results from output directory.

    Returns dict mapping exp_name -> result info.
    """
    results = {}

    for exp_dir in sorted(output_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name.startswith(".") or exp_dir.name.startswith("__"):
            continue
        if exp_dir.name == "logs":
            continue

        exp_name = exp_dir.name
        parsed = parse_exp_name(exp_name)

        # Look for eval results
        eval_dir = exp_dir / "eval_results"
        asr = load_eval_summary(eval_dir)

        # Look for lambda history
        lambda_hist = load_lambda_history(exp_dir)
        lambda_stats = None
        if lambda_hist:
            lambdas = [h.get("lambda", 0) for h in lambda_hist]
            if lambdas:
                lambda_stats = {
                    "min": min(lambdas),
                    "max": max(lambdas),
                    "mean": sum(lambdas) / len(lambdas),
                    "final": lambdas[-1],
                    "steps": len(lambdas),
                }

        # Load config if available
        config_path = exp_dir / "config.json"
        config = None
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        results[exp_name] = {
            "exp_name": exp_name,
            "ema_beta": parsed["ema_beta"],
            "window_size": parsed["window_size"],
            "attack_prompt": parsed["attack_prompt"],
            "judge_prompt": parsed["judge_prompt"],
            "asr": asr,
            "lambda_stats": lambda_stats,
            "config": config,
            "eval_dir": str(eval_dir),
        }

    return results


# =============================================================================
# Output Formatting
# =============================================================================

def format_text(results: Dict[str, Dict]) -> str:
    """Format results as readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("Experiment 3: Adaptive Hybrid Reward GRPO - Results Summary")
    lines.append("=" * 80)
    lines.append("")

    # Group by attack_prompt + judge_prompt
    groups: Dict[str, List[Dict]] = {}
    for exp_name, data in results.items():
        key = f"{data['attack_prompt']}_{data['judge_prompt']}"
        if key not in groups:
            groups[key] = []
        groups[key].append(data)

    for group_key in sorted(groups.keys()):
        group_data = groups[group_key]
        lines.append(f"--- [{group_key}] ---")
        lines.append(f"{'EMA Beta':<10} {'Window':<10} {'ASR':<10} {'λ min':<10} {'λ max':<10} {'λ mean':<10} {'λ final':<10}")
        lines.append("-" * 70)

        # Sort by EMA beta
        group_data.sort(key=lambda x: float(x["ema_beta"]) if x["ema_beta"] != "N/A" else 999)

        for d in group_data:
            asr_str = f"{d['asr']:.1%}" if d["asr"] is not None else "N/A"
            ls = d["lambda_stats"]
            if ls:
                lambda_strs = [f"{v:.3f}" for v in [ls["min"], ls["max"], ls["mean"], ls["final"]]]
            else:
                lambda_strs = ["N/A"] * 4

            lines.append(
                f"{d['ema_beta']:<10} {d['window_size']:<10} {asr_str:<10} "
                f"{lambda_strs[0]:<10} {lambda_strs[1]:<10} {lambda_strs[2]:<10} {lambda_strs[3]:<10}"
            )

        lines.append("")

    # Overall summary
    lines.append("=" * 80)
    lines.append("Summary (ASR by EMA Beta):")
    lines.append("=" * 80)

    # Collect all ASR values by EMA beta
    ema_asr: Dict[str, List[float]] = {}
    for data in results.values():
        if data["asr"] is not None:
            beta = data["ema_beta"]
            if beta not in ema_asr:
                ema_asr[beta] = []
            ema_asr[beta].append(data["asr"])

    if ema_asr:
        for beta in sorted(ema_asr.keys(), key=lambda x: float(x) if x != "N/A" else 999):
            values = ema_asr[beta]
            avg = sum(values) / len(values)
            lines.append(f"  EMA beta={beta}: ASR = {avg:.1%} (n={len(values)}, range=[{min(values):.1%}, {max(values):.1%}])")
    else:
        lines.append("  No ASR results found. Run evaluations first.")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def format_markdown(results: Dict[str, Dict]) -> str:
    """Format results as Markdown table."""
    lines = []
    lines.append("## 实验3：Adaptive Hybrid Reward GRPO 结果汇总\n")

    # Group by attack_prompt + judge_prompt
    groups: Dict[str, List[Dict]] = {}
    for exp_name, data in results.items():
        key = f"{data['attack_prompt']}_{data['judge_prompt']}"
        if key not in groups:
            groups[key] = []
        groups[key].append(data)

    for group_key in sorted(groups.keys()):
        group_data = groups[group_key]
        group_data.sort(key=lambda x: float(x["ema_beta"]) if x["ema_beta"] != "N/A" else 999)

        lines.append(f"\n### {group_key}\n")
        lines.append("| EMA Beta | Window Size | ASR | λ min | λ max | λ mean | λ final |")
        lines.append("|----------|-------------|-----|-------|-------|--------|---------|")

        for d in group_data:
            asr_str = f"{d['asr']:.1%}" if d["asr"] is not None else "N/A"
            ls = d["lambda_stats"]
            if ls:
                vals = [f"{v:.3f}" for v in [ls["min"], ls["max"], ls["mean"], ls["final"]]]
            else:
                vals = ["N/A"] * 4

            lines.append(
                f"| {d['ema_beta']} | {d['window_size']} | {asr_str} | "
                f"{vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |"
            )

    # Summary section
    lines.append("\n---\n")
    lines.append("### Summary (ASR by EMA Beta)\n")

    ema_asr: Dict[str, List[float]] = {}
    for data in results.values():
        if data["asr"] is not None:
            beta = data["ema_beta"]
            if beta not in ema_asr:
                ema_asr[beta] = []
            ema_asr[beta].append(data["asr"])

    if ema_asr:
        lines.append("| EMA Beta | Avg ASR | Min ASR | Max ASR | N |")
        lines.append("|----------|---------|---------|---------|---|")
        for beta in sorted(ema_asr.keys(), key=lambda x: float(x) if x != "N/A" else 999):
            values = ema_asr[beta]
            avg = sum(values) / len(values)
            lines.append(f"| {beta} | {avg:.1%} | {min(values):.1%} | {max(values):.1%} | {len(values)} |")
    else:
        lines.append("No ASR results found yet.")

    return "\n".join(lines)


def format_json(results: Dict[str, Dict]) -> str:
    """Format results as JSON."""
    # Clean up for JSON output
    output = {}
    for exp_name, data in results.items():
        output[exp_name] = {
            "ema_beta": data["ema_beta"],
            "window_size": data["window_size"],
            "attack_prompt": data["attack_prompt"],
            "judge_prompt": data["judge_prompt"],
            "asr": data["asr"],
            "lambda_stats": data["lambda_stats"],
        }
    return json.dumps(output, indent=2, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Adaptive Hybrid Reward GRPO - Results Summary"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory path (default: auto-detect script's parent output dir)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format"
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "output"

    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        print("Run experiments first: bash experiments/adaptive_hybrid_reward_exp/exp.sh")
        return 1

    # Collect results
    results = collect_results(output_dir)

    if not results:
        print(f"Warning: No experiments found in {output_dir}")
        return 0

    # Format and print output
    if args.format == "text":
        print(format_text(results))
    elif args.format == "markdown":
        print(format_markdown(results))
    else:
        print(format_json(results))

    # Save summary file
    summary_file = output_dir / "results_summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(format_markdown(results))
    print(f"\nSummary saved to: {summary_file}")

    return 0


if __name__ == "__main__":
    exit(main())
