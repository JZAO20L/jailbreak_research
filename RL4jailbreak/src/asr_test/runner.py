"""
ASR Test Runner

Orchestrates the complete ASR testing pipeline.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .dataset import PromptDataset
from .target import TargetModel, TargetResponse
from .guard import GuardModel, GuardLabel
from .metrics import ASRMetrics, ASRResult, ASRSummary
from src.config import (
    TARGET_MODEL_PATH,
    GUARD_MODEL_PATH,
    DEFAULT_GPU_IDS,
    DEFAULT_GPU_MEM_UTIL,
    DEFAULT_MAX_MODEL_LEN,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ASRTestConfig:
    """Configuration for ASR test."""
    # Data
    data_path: str
    sample_size: Optional[int] = None
    shuffle: bool = False
    seed: int = 42

    # Target model
    target_model_path: str = TARGET_MODEL_PATH
    target_port: int = 8001
    target_gpu_id: str = "0"
    target_tp_size: int = 1
    target_gpu_mem: float = 0.7
    target_max_len: int = 4096
    target_temperature: float = 0.0
    target_max_tokens: int = 512

    # Guard model
    guard_model_path: str = GUARD_MODEL_PATH
    guard_port: int = 8002
    guard_gpu_id: str = "0"
    guard_tp_size: int = 1
    guard_gpu_mem: float = 0.5
    guard_max_len: int = 4096

    # Output
    output_dir: str = "output/asr_test"
    save_responses: bool = True
    save_labels: bool = True

    # Runtime
    timeout: float = 600.0


class ASRTestRunner:
    """
    Main runner for ASR testing.

    Pipeline:
    1. Load prompt dataset
    2. Load target model, generate responses, offload
    3. Load guard model, label responses, offload
    4. Compute ASR metrics
    5. Save results
    """

    def __init__(self, config: ASRTestConfig):
        self.config = config
        self.metrics = ASRMetrics()

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "logs"), exist_ok=True)

    def run(self) -> ASRSummary:
        """
        Execute the complete ASR test pipeline.

        Returns:
            ASRSummary with final statistics
        """
        logger.info("=" * 60)
        logger.info("Starting ASR Test")
        logger.info("=" * 60)

        # Step 1: Load dataset
        logger.info("[Step 1] Loading prompt dataset...")
        dataset = PromptDataset(
            data_path=self.config.data_path,
            sample_size=self.config.sample_size,
            shuffle=self.config.shuffle,
            seed=self.config.seed,
        )
        logger.info(f"  Loaded {len(dataset)} prompts")
        logger.info(f"  Dataset summary: {dataset.summary()}")

        prompts = [item.prompt for item in dataset]
        prompt_ids = [item.id for item in dataset]

        # Step 2: Target model inference
        logger.info("[Step 2] Running target model inference...")
        target_responses: List[TargetResponse] = []

        with TargetModel(
            model_path=self.config.target_model_path,
            port=self.config.target_port,
            gpu_id=self.config.target_gpu_id,
            tensor_parallel_size=self.config.target_tp_size,
            gpu_memory_utilization=self.config.target_gpu_mem,
            max_model_len=self.config.target_max_len,
            temperature=self.config.target_temperature,
            max_tokens=self.config.target_max_tokens,
            timeout=self.config.timeout,
            log_file=os.path.join(self.config.output_dir, "logs", "target.log"),
        ) as target:
            target_responses = target.generate_batch(prompts, prompt_ids)

        logger.info(f"  Generated {len(target_responses)} responses")
        logger.info("  Target model offloaded")

        # Save responses
        if self.config.save_responses:
            responses_path = os.path.join(self.config.output_dir, "responses.jsonl")
            with open(responses_path, "w", encoding="utf-8") as f:
                for r in target_responses:
                    f.write(json.dumps({
                        "prompt_id": r.prompt_id,
                        "prompt": r.prompt,
                        "response": r.response,
                        "error": r.error,
                    }, ensure_ascii=False) + "\n")
            logger.info(f"  Saved responses to {responses_path}")

        # Step 3: Guard model labeling
        logger.info("[Step 3] Running guard model labeling...")
        guard_labels: List[GuardLabel] = []

        with GuardModel(
            model_path=self.config.guard_model_path,
            port=self.config.guard_port,
            gpu_id=self.config.guard_gpu_id,
            tensor_parallel_size=self.config.guard_tp_size,
            gpu_memory_utilization=self.config.guard_gpu_mem,
            max_model_len=self.config.guard_max_len,
            timeout=self.config.timeout,
            log_file=os.path.join(self.config.output_dir, "logs", "guard.log"),
        ) as guard:
            guard_labels = guard.label_batch(
                prompts=[r.prompt for r in target_responses],
                responses=[r.response for r in target_responses],
            )

        logger.info(f"  Labeled {len(guard_labels)} responses")
        logger.info("  Guard model offloaded")

        # Step 4: Compute metrics
        logger.info("[Step 4] Computing ASR metrics...")

        for resp, label in zip(target_responses, guard_labels):
            result = ASRResult(
                prompt_id=resp.prompt_id,
                prompt=resp.prompt,
                response=resp.response,
                label=label.label,
                is_success=(label.label == "success"),
                safe_label=label.safe_label or "",
                raw_output=label.raw_output,
            )
            self.metrics.add_result(result)

        summary = self.metrics.compute_summary()

        logger.info(f"  Total: {summary.total}")
        logger.info(f"  Success: {summary.success_count} ({summary.asr:.2%})")
        logger.info(f"  Refusal: {summary.refusal_count} ({summary.refusal_rate:.2%})")
        logger.info(f"  Partial: {summary.partial_count} ({summary.partial_rate:.2%})")

        # Step 5: Save results
        self._save_results(summary)

        logger.info("=" * 60)
        logger.info("ASR Test Complete")
        logger.info("=" * 60)

        return summary

    def _save_results(self, summary: ASRSummary):
        """Save all results to output directory."""
        output_dir = self.config.output_dir

        # Save summary
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "data_path": self.config.data_path,
                    "sample_size": self.config.sample_size,
                    "target_model_path": self.config.target_model_path,
                    "guard_model_path": self.config.guard_model_path,
                },
                "summary": summary.to_dict(),
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved summary to {summary_path}")

        # Save detailed results
        results_path = os.path.join(output_dir, "results.jsonl")
        with open(results_path, "w", encoding="utf-8") as f:
            for result in self.metrics.results:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"  Saved detailed results to {results_path}")

    @classmethod
    def from_args(cls, args) -> "ASRTestRunner":
        """Create runner from argparse namespace."""
        config = ASRTestConfig(
            data_path=args.data_path,
            sample_size=args.sample_size,
            shuffle=args.shuffle,
            seed=args.seed,
            target_model_path=args.target_model,
            target_port=args.target_port,
            target_gpu_id=args.target_gpu,
            target_tp_size=args.target_tp,
            target_gpu_mem=args.target_gpu_mem,
            target_max_len=args.target_max_len,
            target_temperature=args.target_temp,
            target_max_tokens=args.target_max_tokens,
            guard_model_path=args.guard_model,
            guard_port=args.guard_port,
            guard_gpu_id=args.guard_gpu,
            guard_tp_size=args.guard_tp,
            guard_gpu_mem=args.guard_gpu_mem,
            guard_max_len=args.guard_max_len,
            output_dir=args.output_dir,
            save_responses=not args.no_save_responses,
            save_labels=not args.no_save_labels,
            timeout=args.timeout,
        )
        return cls(config)