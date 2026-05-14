# -*- coding: utf-8 -*-
"""
Adaptive Reward Weight Calculator for Experiment 3.

This module implements the adaptive weight mechanism for hybrid reward GRPO training.
The core idea is to dynamically adjust the weight between ASR reward and Judge reward
based on their variance ratios, using sigmoid transformation and EMA smoothing.

Key formula:
    ratio = var_judge / (var_asr + eps)
    lambda_raw = sigmoid(alpha * ratio + delta)
    lambda_new = ema_beta * lambda_old + (1 - ema_beta) * lambda_raw
    lambda_final = clip(lambda_new, lambda_min, lambda_max)
    reward = lambda_final * asr_reward + (1 - lambda_final) * judge_reward

Physical meaning:
- When ASR reward variance is small → low discrimination → reduce lambda → rely more on judge
- When ASR reward variance is large → high discrimination → increase lambda → rely more on ASR
- EMA smoothing prevents drastic weight fluctuations

Reference: NEW_IDEA.md, TODO.md (Experiment 3)
"""

import torch
import math
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AdaptiveRewardConfig:
    """Configuration for adaptive reward weight calculator."""
    
    # Sigmoid parameters
    alpha: float = 2.0          # Variance ratio sensitivity
    delta: float = -1.0         # Sigmoid bias (negative → prefer ASR initially)
    
    # Lambda bounds
    lambda_min: float = 0.2     # Lower bound for ASR weight
    lambda_max: float = 0.8     # Upper bound for ASR weight
    
    # EMA parameter (window_size = 1/(1-beta))
    ema_beta: float = 0.95      # Default: window size ~20
    
    # Stability
    eps: float = 1e-8           # Prevent division by zero
    
    # Initial weight
    lambda_init: float = 0.5    # Start with 1:1 ratio
    
    # Multi-dimensional judge weights (for uniform weighting)
    judge_dimension_weights: Optional[List[float]] = None  # Default: uniform [0.25, 0.25, 0.25, 0.25]
    
    def get_window_size(self) -> int:
        """Calculate effective window size from EMA beta."""
        if self.ema_beta >= 1.0:
            return math.inf
        return int(round(1.0 / (1.0 - self.ema_beta)))


class AdaptiveRewardCalculator:
    """
    Adaptive reward weight calculator for hybrid reward GRPO training.
    
    Usage:
        calculator = AdaptiveRewardCalculator(config)
        
        # In reward function, per sample:
        for asr_r, judge_r in rewards:
            final_r = calculator.update(asr_r, judge_r)
        
        # Or batch mode:
        final_rewards = calculator.update_batch(asr_rewards, judge_rewards)
        
        # Get current lambda for logging:
        lambda_current = calculator.get_lambda()
    
    Note:
        - First window uses fixed 1:1 weight (lambda=0.5)
        - After first window, lambda adapts based on variance ratio
        - Lambda represents ASR weight; Judge weight = 1 - lambda
    """
    
    def __init__(self, config: Optional[AdaptiveRewardConfig] = None):
        self.config = config or AdaptiveRewardConfig()
        
        # Current lambda (ASR weight)
        self._lambda: float = self.config.lambda_init
        
        # History buffers for variance calculation
        self._asr_history: List[float] = []
        self._judge_history: List[float] = []
        
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
        self._asr_history.clear()
        self._judge_history.clear()
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
            "history_size": len(self._asr_history),
        }
    
    def _compute_variance(self, history: List[float], window: int) -> float:
        """Compute variance over a window of history."""
        if len(history) == 0:
            return 0.0
        
        # Use last window elements (or all if less than window)
        window_data = history[-window:] if window < len(history) else history
        
        if len(window_data) < 2:
            return 0.0
        
        # Use torch for computation
        tensor = torch.tensor(window_data, dtype=torch.float32)
        variance = torch.var(tensor).item()
        return variance
    
    def update(self, asr_raw: float, judge_raw: float) -> float:
        """
        Update lambda with new reward pair and return weighted reward.
        
        Args:
            asr_raw: Raw ASR reward (0~1, before weighting)
            judge_raw: Raw Judge reward (0~1, before weighting)
        
        Returns:
            final_reward: Weighted combination of ASR and Judge rewards
        """
        self._step += 1
        
        # Record history
        self._asr_history.append(asr_raw)
        self._judge_history.append(judge_raw)
        
        window_size = self.config.get_window_size()
        
        # First window: fixed 1:1 ratio
        if len(self._asr_history) <= window_size:
            self._lambda = self.config.lambda_init
            return self._lambda * asr_raw + (1 - self._lambda) * judge_raw
        
        # Compute variance over window
        self._var_asr = self._compute_variance(self._asr_history, window_size)
        self._var_judge = self._compute_variance(self._judge_history, window_size)
        
        # Compute ratio (var_judge / var_asr)
        # When var_asr small → ratio large → sigmoid → lambda increases? 
        # No, we want: var_asr small → lambda decrease
        # So ratio should be var_asr / var_judge (inverse)
        # Or adjust delta sign
        
        # Original formula from TODO: ratio = var_judge / var_asr
        # But physical meaning should be: high var_asr → lambda high
        # Let's use: ratio = var_asr / var_judge
        # Then sigmoid(alpha * ratio + delta):
        #   - ratio high (var_asr large) → sigmoid large → lambda high ✓
        
        ratio = self._var_asr / (self._var_judge + self.config.eps)
        self._ratio = ratio
        
        # Sigmoid transformation
        lambda_raw = torch.sigmoid(
            torch.tensor(self.config.alpha * ratio + self.config.delta)
        ).item()
        self._lambda_raw = lambda_raw
        
        # EMA smoothing
        self._lambda = (
            self.config.ema_beta * self._lambda + 
            (1 - self.config.ema_beta) * lambda_raw
        )
        
        # Clip to bounds
        self._lambda = max(self.config.lambda_min, 
                          min(self.config.lambda_max, self._lambda))
        
        # Compute final reward
        final_reward = self._lambda * asr_raw + (1 - self._lambda) * judge_raw
        
        return final_reward
    
    def update_batch(
        self, 
        asr_raws: List[float], 
        judge_raws: List[float]
    ) -> List[float]:
        """
        Update lambda with batch of rewards and return weighted rewards.
        
        Note: This processes rewards sequentially, updating lambda after each.
        For parallel processing (same lambda for all in batch), use compute_batch_fixed.
        
        Args:
            asr_raws: List of raw ASR rewards
            judge_raws: List of raw Judge rewards
        
        Returns:
            final_rewards: List of weighted rewards
        """
        final_rewards = []
        for asr_r, judge_r in zip(asr_raws, judge_raws):
            final_r = self.update(asr_r, judge_r)
            final_rewards.append(final_r)
        return final_rewards
    
    def compute_batch_fixed(
        self,
        asr_raws: List[float],
        judge_raws: List[float],
        update_after: bool = True
    ) -> List[float]:
        """
        Compute weighted rewards using current lambda (fixed for batch).
        
        Optionally update lambda after computing all rewards (using batch statistics).
        This is useful for GRPO where all k completions share the same original prompt.
        
        Args:
            asr_raws: List of raw ASR rewards
            judge_raws: List of raw Judge rewards
            update_after: Whether to update lambda after computing (using batch mean)
        
        Returns:
            final_rewards: List of weighted rewards (all using same lambda)
        """
        # Use current lambda for all
        final_rewards = [
            self._lambda * asr_r + (1 - self._lambda) * judge_r
            for asr_r, judge_r in zip(asr_raws, judge_raws)
        ]
        
        if update_after:
            # Update history with batch
            for asr_r, judge_r in zip(asr_raws, judge_raws):
                self._asr_history.append(asr_r)
                self._judge_history.append(judge_r)
                self._step += 1
            
            # Recompute lambda after batch
            window_size = self.config.get_window_size()
            
            if len(self._asr_history) > window_size:
                self._var_asr = self._compute_variance(self._asr_history, window_size)
                self._var_judge = self._compute_variance(self._judge_history, window_size)
                
                ratio = self._var_asr / (self._var_judge + self.config.eps)
                self._ratio = ratio
                
                lambda_raw = torch.sigmoid(
                    torch.tensor(self.config.alpha * ratio + self.config.delta)
                ).item()
                self._lambda_raw = lambda_raw
                
                self._lambda = (
                    self.config.ema_beta * self._lambda +
                    (1 - self.config.ema_beta) * lambda_raw
                )
                self._lambda = max(self.config.lambda_min,
                                  min(self.config.lambda_max, self._lambda))
        
        return final_rewards


class MultiDimensionJudgeReward:
    """
    Multi-dimensional judge reward calculator.
    
    Computes judge reward by averaging multiple dimension scores with configurable weights.
    
    Dimensions (from TODO.md):
    - intent_preservation: How well the rewritten prompt preserves the original attack intent
    - stealth: Attack stealthiness / concealment
    - strategy_execution: How well the attack strategy is executed
    - attack_potential: Overall jailbreak potential
    
    Default: uniform weights [0.25, 0.25, 0.25, 0.25]
    """
    
    DIMENSION_NAMES = [
        "intent_preservation",
        "stealth",
        "strategy_execution",
        "attack_potential",
    ]
    
    def __init__(self, weights: Optional[List[float]] = None):
        """
        Args:
            weights: Dimension weights. Default: uniform [0.25, 0.25, 0.25, 0.25]
        """
        if weights is None:
            weights = [0.25, 0.25, 0.25, 0.25]
        
        if len(weights) != len(self.DIMENSION_NAMES):
            raise ValueError(
                f"Expected {len(self.DIMENSION_NAMES)} weights, got {len(weights)}"
            )
        
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]
    
    def compute(self, dimension_scores: dict) -> float:
        """
        Compute weighted judge reward from dimension scores.
        
        Args:
            dimension_scores: Dict mapping dimension name to score (0~1)
                            e.g., {"intent_preservation": 0.8, "stealth": 0.6, ...}
        
        Returns:
            weighted_reward: Weighted average of dimension scores (0~1)
        """
        total = 0.0
        for dim, weight in zip(self.DIMENSION_NAMES, self.weights):
            score = dimension_scores.get(dim, 0.0)
            if not isinstance(score, (int, float)):
                score = 0.0
            score = max(0.0, min(1.0, float(score)))
            total += weight * score
        return total
    
    def compute_from_json_response(self, json_str: str) -> Tuple[float, dict]:
        """
        Parse JSON response from judge model and compute weighted reward.
        
        Expected JSON format:
        {
            "intent_preservation": 0.75,
            "stealth": 0.60,
            "strategy_execution": 0.80,
            "attack_potential": 0.70
        }
        
        Args:
            json_str: JSON string from judge model
        
        Returns:
            weighted_reward: Weighted average
            parsed_scores: Dict of parsed dimension scores
        """
        import json
        import re
        
        parsed_scores = {}
        
        # Try to extract JSON from response
        try:
            # Find JSON-like content
            json_match = re.search(r'\{[^{}]*\}', json_str, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
                parsed_scores = json.loads(json_content)
            else:
                parsed_scores = {}
        except (json.JSONDecodeError, AttributeError):
            parsed_scores = {}
        
        # Fill missing dimensions with default 0.0
        for dim in self.DIMENSION_NAMES:
            if dim not in parsed_scores:
                parsed_scores[dim] = 0.0
        
        # Compute weighted reward
        weighted_reward = self.compute(parsed_scores)
        
        return weighted_reward, parsed_scores


# Convenience function for quick testing
def test_adaptive_weight():
    """Test adaptive weight calculator with simulated rewards."""
    
    config = AdaptiveRewardConfig(ema_beta=0.9)  # window ~10
    calculator = AdaptiveRewardCalculator(config)
    
    # Simulate rewards: ASR has low variance, Judge has high variance
    import random
    
    print("Testing AdaptiveRewardCalculator (ema_beta=0.9, window~10)")
    print("=" * 60)
    
    for step in range(30):
        # Simulate: ASR stable around 0.5, Judge varies widely
        asr_raw = 0.5 + random.gauss(0, 0.05)  # Low variance
        judge_raw = random.uniform(0.2, 0.8)   # High variance
        
        final_r = calculator.update(asr_raw, judge_raw)
        stats = calculator.get_statistics()
        
        print(f"Step {step+1:2d}: λ={stats['lambda']:.3f} | "
              f"var_asr={stats['var_asr']:.4f} var_judge={stats['var_judge']:.4f} | "
              f"ratio={stats['ratio']:.2f} | "
              f"ASR={asr_raw:.2f} Judge={judge_raw:.2f} → Final={final_r:.2f}")
    
    print("=" * 60)
    print(f"Final lambda: {calculator.get_lambda():.3f}")
    print("Expected: lambda should decrease (ASR has low variance → rely more on Judge)")


if __name__ == "__main__":
    test_adaptive_weight()