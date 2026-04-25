#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI Tool for Baseline Attack Methods

Usage:
    # List available strategies
    python -m baselines.cli list

    # Attack a single prompt
    python -m baselines.cli attack --prompt "..." --strategy pair

    # Batch attack from file
    python -m baselines.cli batch --input data/test.jsonl --strategy multilingual --output results/

    # Compare multiple strategies
    python -m baselines.cli compare --input data/test.jsonl --strategies no_rewrite template pair
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any
from tqdm import tqdm
import time

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from baselines import get_attacker, list_strategies, STRATEGIES
from src.vllm_client import VLLMClient
from src.config import (
    TARGET_MODEL_PATH,
    GUARD_MODEL_PATH,
    DEFAULT_GPU_IDS,
    DEFAULT_GPU_MEM_UTIL,
    DEFAULT_MAX_MODEL_LEN,
)


def setup_clients(args):
    """Setup target and guard clients."""
    clients = {}

    # Target client
    if args.target_model:
        clients["target"] = VLLMClient(
            model_name="target",
            model_path=args.target_model,
            port=args.target_port,
            gpu_id=args.gpu_id,
            tensor_parallel_size=args.tp_size,
            gpu_memory_utilization=args.gpu_mem,
            max_model_len=args.max_len,
            timeout=args.timeout,
            log_file=os.path.join(args.output, "logs", "target.log") if args.output else None,
        )

    # Guard client
    if args.guard_model:
        clients["guard"] = VLLMClient(
            model_name="guard",
            model_path=args.guard_model,
            port=args.guard_port,
            gpu_id=args.gpu_id,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.4,
            max_model_len=args.max_len,
            timeout=args.timeout,
            log_file=os.path.join(args.output, "logs", "guard.log") if args.output else None,
        )

    return clients


def load_prompts(input_path: str) -> List[Dict[str, Any]]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                prompts.append(obj)
            except json.JSONDecodeError:
                continue
    return prompts


def cmd_list(args):
    """List available strategies."""
    print("Available Attack Strategies:")
    print("=" * 60)

    for name in list_strategies():
        attacker_cls = STRATEGIES[name]
        doc = attacker_cls.__doc__ or ""
        print(f"\n{name}:")
        print(f"  {doc.strip()}")

    print("\n" + "=" * 60)
    print("\nUsage:")
    print(f"  python -m baselines.cli attack --prompt '...' --strategy <name>")
    print(f"  python -m baselines.cli batch --input data.jsonl --strategy <name>")


def cmd_attack(args):
    """Attack a single prompt."""
    print(f"\n{'='*60}")
    print(f"Strategy: {args.strategy}")
    print(f"Prompt: {args.prompt[:100]}...")
    print(f"{'='*60}\n")

    # Setup clients
    clients = setup_clients(args)

    # Create attacker
    attacker_kwargs = {}
    if "target" in clients:
        attacker_kwargs["target_client"] = clients["target"]
    if "guard" in clients:
        attacker_kwargs["guard_client"] = clients["guard"]

    # Strategy-specific arguments
    if args.strategy == "template":
        attacker_kwargs["template"] = args.template
    elif args.strategy == "multilingual":
        attacker_kwargs["languages"] = args.languages.split(",") if args.languages else None
        attacker_kwargs["random_language"] = args.random_lang

    attacker = get_attacker(args.strategy, **attacker_kwargs)

    # Execute attack
    result = attacker.attack(args.prompt, evaluate=args.evaluate)

    # Print results
    print(f"\nResults:")
    print(f"  Attack Prompt: {result.attack_prompt[:200]}...")
    if args.evaluate:
        print(f"  Target Response: {result.target_response[:200]}...")
        print(f"  Guard Label: {result.guard_label}")
        print(f"  Success: {result.is_success}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Time: {result.time_cost:.2f}s")

    # Save results
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, "result.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\nResult saved to: {output_path}")

    # Cleanup
    for client in clients.values():
        try:
            client.close()
        except:
            pass


def cmd_batch(args):
    """Batch attack from file."""
    print(f"\n{'='*60}")
    print(f"Batch Attack")
    print(f"  Input: {args.input}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    # Load prompts
    prompts = load_prompts(args.input)
    if args.limit:
        prompts = prompts[:args.limit]

    print(f"Loaded {len(prompts)} prompts")

    # Setup output
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "logs"), exist_ok=True)

    # Setup clients
    clients = setup_clients(args)

    # Create attacker
    attacker_kwargs = {}
    if "target" in clients:
        attacker_kwargs["target_client"] = clients["target"]
    if "guard" in clients:
        attacker_kwargs["guard_client"] = clients["guard"]

    # Strategy-specific arguments
    if args.strategy == "template":
        attacker_kwargs["template"] = args.template
    elif args.strategy == "multilingual":
        attacker_kwargs["languages"] = args.languages.split(",") if args.languages else None

    attacker = get_attacker(args.strategy, **attacker_kwargs)

    # Run batch attack
    results = []
    successes = 0

    for item in tqdm(prompts, desc=f"[{args.strategy}]"):
        prompt = item.get("prompt", "")
        if not prompt:
            continue

        result = attacker.attack(prompt, evaluate=args.evaluate)
        result.metadata["id"] = item.get("id")
        results.append(result.to_dict())

        if result.is_success:
            successes += 1

    # Calculate statistics
    total = len(results)
    asr = successes / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"  Total: {total}")
    print(f"  Successes: {successes}")
    print(f"  ASR: {asr:.2%}")
    print(f"{'='*60}")

    # Save results
    results_path = os.path.join(args.output, "results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "strategy": args.strategy,
        "total": total,
        "successes": successes,
        "asr": asr,
        "input": args.input,
    }

    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")

    # Cleanup
    for client in clients.values():
        try:
            client.close()
        except:
            pass


def cmd_compare(args):
    """Compare multiple strategies."""
    print(f"\n{'='*60}")
    print(f"Strategy Comparison")
    print(f"  Input: {args.input}")
    print(f"  Strategies: {args.strategies}")
    print(f"{'='*60}\n")

    # Load prompts
    prompts = load_prompts(args.input)
    if args.limit:
        prompts = prompts[:args.limit]

    print(f"Loaded {len(prompts)} prompts")

    # Setup output
    os.makedirs(args.output, exist_ok=True)

    results_by_strategy = {}

    for strategy in args.strategies:
        print(f"\n{'='*60}")
        print(f"Testing: {strategy}")
        print(f"{'='*60}")

        strategy_output = os.path.join(args.output, strategy)
        os.makedirs(strategy_output, exist_ok=True)

        # Setup clients for this strategy
        clients = setup_clients({
            **vars(args),
            "output": strategy_output,
        })

        # Create attacker
        attacker_kwargs = {}
        if "target" in clients:
            attacker_kwargs["target_client"] = clients["target"]
        if "guard" in clients:
            attacker_kwargs["guard_client"] = clients["guard"]

        attacker = get_attacker(strategy, **attacker_kwargs)

        # Run batch attack
        results = []
        successes = 0

        for item in tqdm(prompts, desc=f"[{strategy}]"):
            prompt = item.get("prompt", "")
            if not prompt:
                continue

            result = attacker.attack(prompt, evaluate=args.evaluate)
            result.metadata["id"] = item.get("id")
            results.append(result.to_dict())

            if result.is_success:
                successes += 1

        total = len(results)
        asr = successes / total if total > 0 else 0

        results_by_strategy[strategy] = {
            "total": total,
            "successes": successes,
            "asr": asr,
        }

        print(f"  ASR: {asr:.2%}")

        # Cleanup
        for client in clients.values():
            try:
                client.close()
            except:
                pass

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'Total':>10} {'Success':>10} {'ASR':>10}")
    print("-" * 50)
    for strategy, stats in results_by_strategy.items():
        print(f"{strategy:<20} {stats['total']:>10} {stats['successes']:>10} {stats['asr']:>10.2%}")

    # Save comparison results
    comparison_path = os.path.join(args.output, "comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(results_by_strategy, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Attack Methods CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # List command
    parser_list = subparsers.add_parser("list", help="List available strategies")

    # Attack command
    parser_attack = subparsers.add_parser("attack", help="Attack a single prompt")
    parser_attack.add_argument("--prompt", "-p", required=True, help="Prompt to attack")
    parser_attack.add_argument("--strategy", "-s", default="no_rewrite", help="Attack strategy")
    parser_attack.add_argument("--evaluate", "-e", action="store_true", help="Evaluate attack")
    parser_attack.add_argument("--output", "-o", help="Output directory")

    # Template-specific
    parser_attack.add_argument("--template", default="urgent_situation", help="Template name")

    # Multilingual-specific
    parser_attack.add_argument("--languages", help="Comma-separated language codes")
    parser_attack.add_argument("--random-lang", action="store_true", default=True, help="Random language")

    # Batch command
    parser_batch = subparsers.add_parser("batch", help="Batch attack from file")
    parser_batch.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser_batch.add_argument("--strategy", "-s", default="no_rewrite", help="Attack strategy")
    parser_batch.add_argument("--output", "-o", required=True, help="Output directory")
    parser_batch.add_argument("--limit", type=int, help="Limit number of prompts")
    parser_batch.add_argument("--evaluate", "-e", action="store_true", help="Evaluate attacks")

    # Template-specific
    parser_batch.add_argument("--template", default="urgent_situation", help="Template name")

    # Multilingual-specific
    parser_batch.add_argument("--languages", help="Comma-separated language codes")

    # Compare command
    parser_compare = subparsers.add_parser("compare", help="Compare multiple strategies")
    parser_compare.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser_compare.add_argument("--strategies", "-s", nargs="+",
                                default=["no_rewrite", "template", "multilingual"],
                                help="Strategies to compare")
    parser_compare.add_argument("--output", "-o", required=True, help="Output directory")
    parser_compare.add_argument("--limit", type=int, help="Limit number of prompts")
    parser_compare.add_argument("--evaluate", "-e", action="store_true", help="Evaluate attacks")

    # Common arguments for attack/batch/compare
    for p in [parser_attack, parser_batch, parser_compare]:
        p.add_argument("--target-model", default=TARGET_MODEL_PATH, help="Target model path")
        p.add_argument("--guard-model", default=GUARD_MODEL_PATH, help="Guard model path")
        p.add_argument("--target-port", type=int, default=8801, help="Target model port")
        p.add_argument("--guard-port", type=int, default=8802, help="Guard model port")
        p.add_argument("--gpu-id", default="0", help="GPU ID(s)")
        p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
        p.add_argument("--gpu-mem", type=float, default=0.7, help="GPU memory utilization")
        p.add_argument("--max-len", type=int, default=DEFAULT_MAX_MODEL_LEN, help="Max model length")
        p.add_argument("--timeout", type=float, default=600, help="Timeout for vLLM")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "attack":
        cmd_attack(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()