# -*- coding: utf-8 -*-
"""
Simple Rule-Based Reward Weight Calculator for Experiment 3 (Alternative).

This is a simpler alternative to the EMA-based adaptive weight calculator.
Instead of using windowed EMA and sigmoid transformation, it uses direct rules:

- If ASR variance = 0 AND Judge variance = 0 → lambda = 0.5 (neutral, neither is informative)
- If ASR variance = 0 AND Judge variance > 0 → lambda = lambda_min (trust Judge)
- If ASR variance > 0 AND Judge variance = 0 → lambda = lambda_max (trust ASR)
- If both variances > 0 → lambda based on normalized ratio

Key insight:
- Variance = 0 means all k generations got the same reward → no discrimination power
- But variance = 0 doesn't mean no gradient contribution!
  - If all rewards are 0 → negative gradient (model learns "these are bad")
  - If all rewards are 1 → positive gradient (model learns "these are good")
  - Variance > 0 → strongest learning signal (model learns "which are better")

Physical meaning:
- When ASR can't distinguish between good/bad generations → reduce ASR weight
- When ASR provides discrimination signal → increase ASR weight
- No EMA/window → immediate response to variance changes

Usage:
    calculator = SimpleRuleCalculator(config)

    # In reward function, once per training step:
    calculator.update_step(avg_var_asr, avg_var_judge)
    lambda_val = calculator.get_lambda()
    final_rewards = [lambda_val * a + (1-lambda_val) * j for a, j in zip(asr_raws, judge_raws)]
"""

import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class SimpleRuleConfig:
    """Configuration for simple rule-based weight calculator."""

    # Lambda bounds
    lambda_min: float = 0.2     # Lower bound for ASR weight (when ASR has no variance)
    lambda_max: float = 0.8     # Upper bound for ASR weight (when Judge has no variance)

    # Stability
    eps: float = 1e-8           # Threshold for "zero" variance

    # Initial weight
    lambda_init: float = 0.5    # Start with 1:1 ratio


class SimpleRuleCalculator:
    """
    Simple rule-based reward weight calculator.

    No EMA, no window, no sigmoid — just direct rules based on variance states.
    """

    def __init__(self, config: Optional[SimpleRuleConfig] = None):
        self.config = config or SimpleRuleConfig()

        # Current lambda (ASR weight)
        self._lambda: float = self.config.lambda_init

        # Current variances (for logging)
        self._var_asr: float = 0.0
        self._var_judge: float = 0.0

        # Step counter
        self._step: int = 0

        # State tracking (for analysis)
        self._state_history: list = []

    def reset(self):
        """Reset calculator state."""
        self._lambda = self.config.lambda_init
        self._step = 0
        self._var_asr = 0.0
        self._var_judge = 0.0
        self._state_history.clear()

    def get_lambda(self) -> float:
        """Get current ASR weight (lambda). Judge weight = 1 - lambda."""
        return self._lambda

    def get_statistics(self) -> dict:
        """Get current statistics for logging."""
        return {
            "lambda": self._lambda,
            "var_asr": self._var_asr,
            "var_judge": self._var_judge,
            "step": self._step,
        }

    def update_step(self, var_asr: float, var_judge: float) -> float:
        """
        Update lambda based on simple rules using this step's variances.

        Called once per training step with the averaged variance across the batch.

        Rules:
        1. Both zero → neutral (lambda = 0.5)
        2. ASR zero, Judge nonzero → trust Judge (lambda = lambda_min)
        3. ASR nonzero, Judge zero → trust ASR (lambda = lambda_max)
        4. Both nonzero → proportional to variance ratio

        Args:
            var_asr: Variance of ASR rewards (averaged across prompts in this step)
            var_judge: Variance of Judge rewards (averaged across prompts in this step)

        Returns:
            updated lambda value
        """
        self._step += 1
        self._var_asr = var_asr
        self._var_judge = var_judge

        asr_is_zero = var_asr < self.config.eps
        judge_is_zero = var_judge < self.config.eps

        if asr_is_zero and judge_is_zero:
            # Case 1: Neither source provides discrimination → neutral
            self._lambda = 0.5
            state = "both_zero"

        elif asr_is_zero and not judge_is_zero:
            # Case 2: ASR has no discrimination → trust Judge
            self._lambda = self.config.lambda_min
            state = "asr_zero"

        elif not asr_is_zero and judge_is_zero:
            # Case 3: Judge has no discrimination → trust ASR
            self._lambda = self.config.lambda_max
            state = "judge_zero"

        else:
            # Case 4: Both have discrimination → proportional weighting
            # ratio = var_asr / (var_asr + var_judge) → [0, 1]
            # When var_asr >> var_judge → ratio → 1 → lambda → lambda_max
            # When var_asr << var_judge → ratio → 0 → lambda → lambda_min
            ratio = var_asr / (var_asr + var_judge)
            self._lambda = self.config.lambda_min + ratio * (
                self.config.lambda_max - self.config.lambda_min
            )
            state = "both_nonzero"

        # Log state for analysis
        self._state_history.append({
            "step": self._step,
            "state": state,
            "lambda": self._lambda,
            "var_asr": var_asr,
            "var_judge": var_judge,
        })

        return self._lambda


# Convenience function for quick testing
def test_simple_rule():
    """Test simple rule calculator with simulated per-step variances."""
    import random

    config = SimpleRuleConfig()
    calculator = SimpleRuleCalculator(config)

    print("Testing SimpleRuleCalculator")
    print("=" * 70)

    # Simulate different scenarios
    scenarios = [
        ("Both zero (early training)", 0.0, 0.0),
        ("ASR zero, Judge high", 0.0, 0.02),
        ("ASR high, Judge zero", 0.02, 0.0),
        ("Both high, ASR dominant", 0.03, 0.01),
        ("Both high, Judge dominant", 0.01, 0.03),
        ("Both equal", 0.02, 0.02),
    ]

    for step, (name, var_asr, var_judge) in enumerate(scenarios, 1):
        calculator.update_step(var_asr, var_judge)
        stats = calculator.get_statistics()
        lambda_val = stats["lambda"]

        print(f"Step {step}: {name:30s} | λ={lambda_val:.3f} | "
              f"var_asr={var_asr:.4f} var_judge={var_judge:.4f}")

    print("=" * 70)
    print("Expected behavior:")
    print("  - Both zero → λ=0.5 (neutral)")
    print("  - ASR zero → λ=0.2 (min, trust Judge)")
    print("  - Judge zero → λ=0.8 (max, trust ASR)")
    print("  - Both nonzero → λ proportional to variance ratio")


if __name__ == "__main__":
    test_simple_rule()
