#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 3 Alternative: Simple Rule-Based Weight - Results Summary Script.

Analyzes results from all 5 experiments:
1. Fixed λ=0.5 (baseline control)
2. Fixed λ=0.8 (ASR-heavy control)
3. Adaptive [0.2, 0.8] (default moderate range)
4. Adaptive [0.3, 0.7] (tighter range)
5. Adaptive [0.1, 0.9] (extreme range)

Usage:
    python summarize_results.py --output_dir experiments/simple_rule_weight_exp/output
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# Configuration
# =============================================================================
EXPERIMENTS = {
    "exp1": {"name": "Fixed λ=1.0", "type": "fixed", "lambda_range": "1.0"},
    "exp2": {"name": "Adaptive [0.8, 0.2]", "type": "adaptive", "lambda_range": "[0.2, 0.8]"},
    "exp3": {"name": "Adaptive [0.6, 0.4]", "type": "adaptive", "lambda_range": "[0.4, 0.6]"},
}

BASELINE_ASR = 27.0  # Exp2 best: hypothetical_scenario + idea_preservation


# =============================================================================
# Helper Functions
# =============================================================================
def load_config(exp_dir: str) -> Optional[Dict]:
    """Load experiment configuration."""
    config_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_lambda_history(exp_dir: str) -> Optional[List[Dict]]:
    """Load lambda history for adaptive experiments."""
    history_path = os.path.join(exp_dir, "lambda_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def analyze_lambda_history(history: List[Dict]) -> Dict:
    """Analyze lambda history statistics."""
    if not history:
        return {}

    lambdas = [h["lambda"] for h in history]
    var_asrs = [h["var_asr"] for h in history]
    var_judges = [h["var_judge"] for h in history]

    # Count states
    states = {
        "both_zero": 0,
        "asr_zero": 0,
        "judge_zero": 0,
        "both_nonzero": 0,
    }

    # Infer states from lambda values and variances
    for h in history:
        var_asr = h["var_asr"]
        var_judge = h["var_judge"]
        lam = h["lambda"]

        eps = 1e-8
        if var_asr < eps and var_judge < eps:
            states["both_zero"] += 1
        elif var_asr < eps:
            states["asr_zero"] += 1
        elif var_judge < eps:
            states["judge_zero"] += 1
        else:
            states["both_nonzero"] += 1

    total_steps = len(history)

    return {
        "total_steps": total_steps,
        "lambda_min": min(lambdas),
        "lambda_max": max(lambdas),
        "lambda_mean": sum(lambdas) / len(lambdas),
        "lambda_final": lambdas[-1],
        "var_asr_mean": sum(var_asrs) / len(var_asrs),
        "var_judge_mean": sum(var_judges) / len(var_judges),
        "state_distribution": {
            k: f"{v/total_steps*100:.1f}%" for k, v in states.items()
        },
        "state_counts": states,
    }


def print_experiment_summary(exp_id: str, exp_info: Dict, exp_dir: str):
    """Print summary for a single experiment."""
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_id} - {exp_info['name']}")
    print(f"Type: {exp_info['type']}")
    print(f"Lambda range: {exp_info['lambda_range']}")
    print(f"Directory: {exp_dir}")
    print(f"{'='*70}")

    # Check if experiment completed
    if not os.path.exists(exp_dir):
        print("❌ Experiment directory not found")
        return

    # Load config
    config = load_config(exp_dir)
    if config:
        print(f"\n✓ Configuration loaded")
        print(f"  - Max steps: {config.get('max_steps', 'N/A')}")
        print(f"  - Learning rate: {config.get('learning_rate', 'N/A')}")
        print(f"  - Attack prompt: {config.get('attack_prompt', 'N/A')}")
        print(f"  - Judge dimension: {config.get('judge_prompt', 'N/A')}")
    else:
        print("❌ Configuration not found")

    # Load lambda history (for adaptive experiments)
    if exp_info["type"] == "adaptive":
        history = load_lambda_history(exp_dir)
        if history:
            print(f"\n✓ Lambda history loaded ({len(history)} steps)")
            stats = analyze_lambda_history(history)

            print(f"\nLambda Statistics:")
            print(f"  - Min: {stats['lambda_min']:.3f}")
            print(f"  - Max: {stats['lambda_max']:.3f}")
            print(f"  - Mean: {stats['lambda_mean']:.3f}")
            print(f"  - Final: {stats['lambda_final']:.3f}")

            print(f"\nVariance Statistics:")
            print(f"  - ASR variance (mean): {stats['var_asr_mean']:.6f}")
            print(f"  - Judge variance (mean): {stats['var_judge_mean']:.6f}")

            print(f"\nState Distribution:")
            for state, pct in stats["state_distribution"].items():
                count = stats["state_counts"][state]
                print(f"  - {state}: {pct} ({count} steps)")
        else:
            print("⚠ Lambda history not found (expected for adaptive)")

    # Check for LoRA weights
    lora_dir = os.path.join(exp_dir, "final_lora")
    if os.path.exists(lora_dir):
        print(f"\n✓ LoRA weights saved: {lora_dir}")
    else:
        print(f"\n⚠ LoRA weights not found")

    # Check for logs
    log_dir = os.path.join(exp_dir, "logs")
    if os.path.exists(log_dir):
        log_files = list(Path(log_dir).glob("*.log"))
        if log_files:
            print(f"✓ Training logs: {len(log_files)} file(s)")


def print_comparison_table(output_dir: str):
    """Print comparison table across all experiments."""
    print(f"\n\n{'='*70}")
    print(f"COMPARISON TABLE")
    print(f"{'='*70}")

    print(f"\n{'Exp':<6} {'Name':<20} {'Type':<10} {'Lambda':<12} {'λ Mean':<8} {'λ Final':<8}")
    print(f"{'-'*66}")

    for exp_id, exp_info in EXPERIMENTS.items():
        exp_dir = os.path.join(output_dir, exp_id)
        history = load_lambda_history(exp_dir)

        lambda_mean = "N/A"
        lambda_final = "N/A"

        if history and exp_info["type"] == "adaptive":
            stats = analyze_lambda_history(history)
            lambda_mean = f"{stats['lambda_mean']:.3f}"
            lambda_final = f"{stats['lambda_final']:.3f}"

        print(
            f"{exp_id:<6} "
            f"{exp_info['name']:<20} "
            f"{exp_info['type']:<10} "
            f"{exp_info['lambda_range']:<12} "
            f"{lambda_mean:<8} "
            f"{lambda_final:<8}"
        )

    print(f"\nBaseline ASR (Exp2 best): {BASELINE_ASR}%")
    print(f"\nNote: Final ASR requires evaluation with scripts/eval.py")


def print_next_steps(output_dir: str):
    """Print instructions for next steps."""
    print(f"\n\n{'='*70}")
    print(f"NEXT STEPS")
    print(f"{'='*70}")

    # Check which experiments completed
    completed = []
    for exp_id in EXPERIMENTS:
        exp_dir = os.path.join(output_dir, exp_id)
        lora_dir = os.path.join(exp_dir, "final_lora")
        if os.path.exists(lora_dir):
            completed.append(exp_id)

    if completed:
        lora_paths = " ".join([
            f"{output_dir}/{exp_id}/final_lora"
            for exp_id in completed
        ])

        print(f"\n1. Evaluate all completed experiments:")
        print(f"   python scripts/eval.py --lora_paths {lora_paths}")

        print(f"\n2. Analyze lambda dynamics (for adaptive experiments):")
        print(f"   python {os.path.dirname(os.path.abspath(__file__))}/summarize_results.py "
              f"--output_dir {output_dir} --detailed")

        print(f"\n3. Compare with original Experiment 3 (EMA-based):")
        print(f"   Review results in experiments/adaptive_hybrid_reward_exp/output/")
    else:
        print(f"\nNo completed experiments found. Run experiments first:")
        print(f"   bash experiments/simple_rule_weight_exp/exp.sh")


# =============================================================================
# Main Function
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3 Alternative: Simple Rule-Based Weight - Results Summary"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/simple_rule_weight_exp/output",
        help="Output directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed lambda dynamics analysis",
    )

    args = parser.parse_args()
    output_dir = args.output_dir

    print(f"{'='*70}")
    print(f"Experiment 3 Alternative: Simple Rule-Based Weight - Results Summary")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Baseline ASR (Exp2 best): {BASELINE_ASR}%")

    # Print summary for each experiment
    for exp_id, exp_info in EXPERIMENTS.items():
        exp_dir = os.path.join(output_dir, exp_id)
        print_experiment_summary(exp_id, exp_info, exp_dir)

    # Print comparison table
    print_comparison_table(output_dir)

    # Print next steps
    print_next_steps(output_dir)

    print(f"\n{'='*70}")
    print(f"Summary complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
