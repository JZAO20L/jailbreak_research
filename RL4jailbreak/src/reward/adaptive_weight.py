# -*- coding: utf-8 -*-
"""
Adaptive Reward Weight Calculator for Experiment 3.

The window is measured in **training steps**, not samples.
Each step, the caller computes the variance of rewards across k generations
for each original prompt, averages across the batch, then calls update_step() once.

Key formula:
    ratio = var_asr / (var_judge + eps)
    lambda_raw = sigmoid(alpha * ratio + delta)
    lambda = ema_beta * lambda_old + (1 - ema_beta) * lambda_raw
    final_reward = lambda * asr_reward + (1 - lambda) * judge_reward

Physical meaning:
- When ASR reward variance across k generations is small → ASR provides little
  discrimination → reduce lambda → rely more on judge
- When ASR reward variance across k generations is large → ASR provides good
  discrimination → increase lambda → rely more on ASR
- EMA smoothing prevents drastic fluctuations

Usage:
    calculator = AdaptiveRewardCalculator(config)

    # In reward function, once per training step:
    # 1. Compute raw ASR and Judge rewards for each completion
    # 2. Group by original prompt (k completions per prompt)
    # 3. Compute variance within each group
    # 4. Average variances across batch
    # 5. Update lambda:
    calculator.update_step(avg_var_asr, avg_var_judge)

    # 6. Get current lambda for computing final rewards:
    lambda_val = calculator.get_lambda()
    final_rewards = [lambda_val * a + (1-lambda_val) * j for a, j in zip(asr_raws, judge_raws)]

Reference: NEW_IDEA.md, TODO.md (Experiment 3)
"""

import torch
import math
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class AdaptiveRewardConfig:
    """Configuration for adaptive reward weight calculator."""

    # Sigmoid parameters
    alpha: float = 2.0          # Variance ratio sensitivity
    delta: float = -2.0         # Sigmoid bias (ratio=1 → lambda=0.5, neutral)

    # Lambda bounds
    lambda_min: float = 0.2     # Lower bound for ASR weight
    lambda_max: float = 0.8     # Upper bound for ASR weight

    # EMA parameter (window_size = 1/(1-beta), measured in **training steps**)
    ema_beta: float = 0.95      # Default: window size ~20 steps

    # Stability
    eps: float = 1e-8           # Prevent division by zero

    # Initial weight
    lambda_init: float = 0.5    # Start with 1:1 ratio

    # Ratio clipping
    ratio_min: float = 0.1      # Lower bound for ratio (prevent extreme values)
    ratio_max: float = 10.0     # Upper bound for ratio

    def get_window_size(self) -> int:
        """Calculate effective window size from EMA beta (in training steps)."""
        if self.ema_beta >= 1.0:
            return math.inf
        return int(round(1.0 / (1.0 - self.ema_beta)))


class AdaptiveRewardCalculator:
    """
    Adaptive reward weight calculator for hybrid reward GRPO training.

    The window is measured in training STEPS, not samples.
    Each step, the caller computes variance across k completions for each
    original prompt, averages across batch, then calls update_step() once.
    """

    def __init__(self, config: Optional[AdaptiveRewardConfig] = None):
        self.config = config or AdaptiveRewardConfig()

        # Current lambda (ASR weight)
        self._lambda: float = self.config.lambda_init

        # Per-step variance history (one entry per training step)
        self._var_asr_history: List[float] = []
        self._var_judge_history: List[float] = []

        # Step counter
        self._step: int = 0

        # Statistics for logging
        self._var_asr: float = 0.0
        self._var_judge: float = 0.0
        self._ratio: float = 1.0
        self._lambda_raw: float = 0.5

    def reset(self):
        """Reset calculator state."""
        self._lambda = self.config.lambda_init
        self._var_asr_history.clear()
        self._var_judge_history.clear()
        self._step = 0
        self._var_asr = 0.0
        self._var_judge = 0.0
        self._ratio = 1.0
        self._lambda_raw = 0.5

    def get_lambda(self) -> float:
        """Get current ASR weight (lambda). Judge weight = 1 - lambda."""
        return self._lambda

    def get_statistics(self) -> dict:
        """Get current statistics for logging."""
        return {
            "lambda": self._lambda,
            "lambda_raw": self._lambda_raw,
            "var_asr": self._var_asr,
            "var_judge": self._var_judge,
            "ratio": self._ratio,
            "step": self._step,
            "history_size": len(self._var_asr_history),
        }

    def update_step(self, var_asr: float, var_judge: float) -> float:
        """
        Update lambda with variance computed for this training step.

        Called once per training step with the averaged variance across the batch.

        Args:
            var_asr: Variance of ASR rewards (averaged across prompts in this step)
            var_judge: Variance of Judge rewards (averaged across prompts in this step)

        Returns:
            updated lambda value
        """
        self._step += 1
        self._var_asr = var_asr
        self._var_judge = var_judge
        self._var_asr_history.append(var_asr)
        self._var_judge_history.append(var_judge)

        window_size = self.config.get_window_size()

        # First window (warmup): fixed 1:1 ratio
        if len(self._var_asr_history) <= window_size:
            self._lambda = self.config.lambda_init
            return self._lambda

        # Handle edge case: both variances are zero
        # This happens when ASR is all 0 (model hasn't learned) and Judge scores
        # are nearly uniform. Fall back to 1:1 — neither source is informative.
        if var_asr < self.config.eps and var_judge < self.config.eps:
            self._ratio = 1.0  # Neutral ratio
            self._lambda_raw = self.config.lambda_init  # Fall back to 0.5
        else:
            # Compute ratio: high var_asr → high ratio → high lambda
            ratio = var_asr / (var_judge + self.config.eps)

            # Clip ratio to prevent extreme sigmoid values
            ratio = max(self.config.ratio_min, min(self.config.ratio_max, ratio))
            self._ratio = ratio

            # Sigmoid transformation
            self._lambda_raw = torch.sigmoid(
                torch.tensor(self.config.alpha * ratio + self.config.delta)
            ).item()

        # EMA smoothing
        self._lambda = (
            self.config.ema_beta * self._lambda +
            (1 - self.config.ema_beta) * self._lambda_raw
        )

        # Clip to bounds
        self._lambda = max(self.config.lambda_min,
                          min(self.config.lambda_max, self._lambda))

        return self._lambda


# Convenience function for quick testing
def test_adaptive_weight():
    """Test adaptive weight calculator with simulated per-step variances."""
    import random

    config = AdaptiveRewardConfig(ema_beta=0.9)  # window ~10 steps
    calculator = AdaptiveRewardCalculator(config)

    print("Testing AdaptiveRewardCalculator (ema_beta=0.9, window~10 steps)")
    print("=" * 70)

    for step in range(30):
        # Simulate: ASR has low variance, Judge has high variance
        # (e.g., ASR is all 0, Judge varies between 0.3-0.7)
        var_asr = max(0.0, random.gauss(0.01, 0.005))  # Low variance
        var_judge = random.uniform(0.01, 0.03)         # High variance

        calculator.update_step(var_asr, var_judge)
        stats = calculator.get_statistics()
        lambda_val = stats["lambda"]

        print(f"Step {step+1:2d}: λ={lambda_val:.3f} | "
              f"var_asr={stats['var_asr']:.4f} var_judge={stats['var_judge']:.4f} | "
              f"ratio={stats['ratio']:.2f} | "
              f"window={'warmup' if step+1 <= 10 else 'adaptive'}")

    print("=" * 70)
    print(f"Final lambda: {calculator.get_lambda():.3f}")
    print("Expected: lambda should decrease (ASR has low variance → rely more on Judge)")


if __name__ == "__main__":
    test_adaptive_weight()
