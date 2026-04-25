#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASR Test CLI

Unified command-line interface for ASR testing.

Usage:
    python -m src.asr_test.cli --data_path data/test.jsonl --output_dir results/test
"""

import argparse
import os
import sys

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.asr_test.runner import ASRTestRunner, ASRTestConfig
from src.config import (
    TARGET_MODEL_PATH,
    GUARD_MODEL_PATH,
    DEFAULT_GPU_IDS,
    DEFAULT_GPU_MEM_UTIL,
    DEFAULT_MAX_MODEL_LEN,
)


def main():
    parser = argparse.ArgumentParser(
        description="Unified ASR Test CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data_path", "-d",
        type=str,
        required=True,
        help="Path to prompt dataset (JSONL format)",
    )
    parser.add_argument(
        "--sample_size", "-n",
        type=int,
        default=None,
        help="Number of samples to test (None = all)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )

    # Target model arguments
    parser.add_argument(
        "--target_model",
        type=str,
        default=TARGET_MODEL_PATH,
        help="Path to target model",
    )
    parser.add_argument(
        "--target_port",
        type=int,
        default=8001,
        help="Port for target model server",
    )
    parser.add_argument(
        "--target_gpu",
        type=str,
        default="0",
        help="GPU ID(s) for target model",
    )
    parser.add_argument(
        "--target_tp",
        type=int,
        default=1,
        help="Tensor parallel size for target model",
    )
    parser.add_argument(
        "--target_gpu_mem",
        type=float,
        default=0.7,
        help="GPU memory utilization for target model",
    )
    parser.add_argument(
        "--target_max_len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help="Max context length for target model",
    )
    parser.add_argument(
        "--target_temp",
        type=float,
        default=0.0,
        help="Temperature for target model generation",
    )
    parser.add_argument(
        "--target_max_tokens",
        type=int,
        default=512,
        help="Max tokens for target model generation",
    )

    # Guard model arguments
    parser.add_argument(
        "--guard_model",
        type=str,
        default=GUARD_MODEL_PATH,
        help="Path to guard model",
    )
    parser.add_argument(
        "--guard_port",
        type=int,
        default=8002,
        help="Port for guard model server",
    )
    parser.add_argument(
        "--guard_gpu",
        type=str,
        default="0",
        help="GPU ID(s) for guard model",
    )
    parser.add_argument(
        "--guard_tp",
        type=int,
        default=1,
        help="Tensor parallel size for guard model",
    )
    parser.add_argument(
        "--guard_gpu_mem",
        type=float,
        default=0.5,
        help="GPU memory utilization for guard model",
    )
    parser.add_argument(
        "--guard_max_len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help="Max context length for guard model",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="output/asr_test",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no_save_responses",
        action="store_true",
        help="Do not save intermediate responses",
    )
    parser.add_argument(
        "--no_save_labels",
        action="store_true",
        help="Do not save intermediate labels",
    )

    # Runtime arguments
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout for model server startup",
    )

    args = parser.parse_args()

    # Print configuration
    print("\n" + "=" * 60)
    print("ASR Test Configuration")
    print("=" * 60)
    print(f"Data path:     {args.data_path}")
    print(f"Sample size:   {args.sample_size or 'all'}")
    print(f"Target model:  {args.target_model}")
    print(f"Guard model:   {args.guard_model}")
    print(f"Output dir:    {args.output_dir}")
    print("=" * 60 + "\n")

    # Run test
    runner = ASRTestRunner.from_args(args)
    summary = runner.run()

    # Print final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Total:     {summary.total}")
    print(f"Success:   {summary.success_count} ({summary.asr:.2%})")
    print(f"Refusal:   {summary.refusal_count} ({summary.refusal_rate:.2%})")
    print(f"Partial:   {summary.partial_count} ({summary.partial_rate:.2%})")
    print(f"Unknown:   {summary.unknown_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()