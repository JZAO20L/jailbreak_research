#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize Results for AHR-GRPO Paper Experiments

Aggregates evaluation results from all experiments and generates summary tables.

Usage:
    python summarize_results.py [--output_dir OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


def load_eval_results(eval_dir: str) -> Optional[Dict]:
    """Load evaluation results from a directory."""
    results_file = os.path.join(eval_dir, "results.json")
    if not os.path.exists(results_file):
        return None

    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_experiments(output_dir: str) -> Dict:
    """
    Summarize results from all experiments.

    Args:
        output_dir: Base output directory

    Returns:
        Dict: Summary statistics
    """
    eval_base_dir = os.path.join(output_dir, "eval")

    experiment_types = [
        "baseline",
        "pure_asr",
        "pure_judge",
        "fixed_hybrid",
        "ahr_grpo",
        "ablation_beta0",
        "ablation_beta08",
        "ablation_beta09",
    ]

    summary = {
        "experiments": {},
        "comparison_table": [],
    }

    for exp_type in experiment_types:
        eval_dir = os.path.join(eval_base_dir, exp_type)
        results = load_eval_results(eval_dir)

        if results is None:
            print(f"  ⚠ {exp_type}: No results found")
            continue

        asr = results.get("asr", 0.0)
        refusal = results.get("refusal", 0.0)
        num_samples = results.get("num_samples", 0)

        summary["experiments"][exp_type] = {
            "asr": asr,
            "refusal": refusal,
            "num_samples": num_samples,
        }

        print(f"  ✓ {exp_type}: ASR={asr:.1f}%, Refusal={refusal:.1f}%, Samples={num_samples}")

    # Generate comparison table
    baseline_asr = summary["experiments"].get("baseline", {}).get("asr", 0.0)

    for exp_type, stats in summary["experiments"].items():
        delta = stats["asr"] - baseline_asr if exp_type != "baseline" else 0.0
        summary["comparison_table"].append({
            "experiment": exp_type,
            "asr": stats["asr"],
            "refusal": stats["refusal"],
            "delta_vs_baseline": delta,
        })

    # Sort by ASR descending
    summary["comparison_table"].sort(key=lambda x: x["asr"], reverse=True)

    return summary


def print_summary_table(summary: Dict):
    """Print formatted summary table."""
    print("\n" + "=" * 70)
    print("AHR-GRPO Paper Experiments - Summary")
    print("=" * 70)
    print(f"{'Experiment':<20} {'ASR (%)':<10} {'Refusal (%)':<12} {'Δ vs Baseline':<15}")
    print("-" * 70)

    for row in summary["comparison_table"]:
        exp = row["experiment"]
        asr = row["asr"]
        refusal = row["refusal"]
        delta = row["delta_vs_baseline"]

        delta_str = f"{delta:+.1f}%" if delta != 0 else "-"

        # Highlight best result
        marker = ""
        if exp == "ahr_grpo":
            marker = " ← Ours"
        elif exp == "baseline":
            marker = " (reference)"

        print(f"{exp:<20} {asr:<10.1f} {refusal:<12.1f} {delta_str:<15}{marker}")

    print("=" * 70)


def save_summary(summary: Dict, output_dir: str):
    """Save summary to JSON file."""
    summary_file = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Summarize AHR-GRPO paper experiment results")
    parser.add_argument("--output_dir", type=str, default="experiments/paper_ahr_grpo/output",
                        help="Base output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("AHR-GRPO Paper Experiments - Result Summarization")
    print("=" * 70)
    print(f"Output Directory: {args.output_dir}")
    print("-" * 70)
    print("Loading evaluation results...")

    summary = summarize_experiments(args.output_dir)

    if not summary["experiments"]:
        print("\nNo results found. Run eval.sh for each experiment first.")
        sys.exit(1)

    print_summary_table(summary)
    save_summary(summary, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
