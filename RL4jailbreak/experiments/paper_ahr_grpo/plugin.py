# -*- coding: utf-8 -*-
"""
Plugin for ms-swift GRPO training with custom reward functions.

Architecture:
    GPU 0:    Guard server (port 8001) + Target&Judge server (port 8002)
    GPU 1-2:  Policy training (colocate mode, vLLM TP=2)

The reward functions connect to GPU 0 servers via HTTP.

Usage:
    swift rlhf --rlhf_type grpo \
        --external_plugins experiments/paper_ahr_grpo/plugin.py \
        --reward_funcs adaptive_hybrid_reward
"""

import os
import sys
import logging

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# Import reward functions
from experiments.paper_ahr_grpo.reward_functions import (
    asr_reward,
    judge_reward,
    fixed_hybrid_reward,
    adaptive_hybrid_reward,
    init_clients,
)

# Import reward_functions module to set global config
import experiments.paper_ahr_grpo.reward_functions as rf_module

# Import ms-swift reward registry
try:
    from swift.rewards import ORM, orms
except ImportError:
    from swift.plugin import ORM, orms


logger = logging.getLogger(__name__)


# =============================================================================
# Reward Function Wrappers
# =============================================================================
class ASRReward(ORM):
    def __call__(self, completions, **kwargs):
        return asr_reward(completions, **kwargs)


class JudgeReward(ORM):
    def __call__(self, completions, **kwargs):
        return judge_reward(completions, **kwargs)


class FixedHybridReward(ORM):
    def __call__(self, completions, **kwargs):
        return fixed_hybrid_reward(completions, **kwargs)


class AdaptiveHybridReward(ORM):
    def __call__(self, completions, **kwargs):
        return adaptive_hybrid_reward(completions, **kwargs)


# =============================================================================
# Register Reward Functions
# =============================================================================
orms['asr_reward'] = ASRReward
orms['judge_reward'] = JudgeReward
orms['fixed_hybrid_reward'] = FixedHybridReward
orms['adaptive_hybrid_reward'] = AdaptiveHybridReward

logger.info("Registered reward functions: asr_reward, judge_reward, fixed_hybrid_reward, adaptive_hybrid_reward")


# =============================================================================
# Initialize from Environment Variables (set by train.sh)
# =============================================================================
def init_from_env():
    """Read configuration from environment variables and initialize clients."""
    guard_port = int(os.environ.get("GUARD_PORT", "8001"))
    target_port = int(os.environ.get("TARGET_JUDGE_PORT", "8002"))
    ema_beta = float(os.environ.get("EMA_BETA", "0.9"))
    alpha = float(os.environ.get("ALPHA", "2.0"))
    delta = float(os.environ.get("DELTA", "-2.0"))
    lambda_min = float(os.environ.get("LAMBDA_MIN", "0.2"))
    lambda_max = float(os.environ.get("LAMBDA_MAX", "0.8"))
    num_generations = int(os.environ.get("NUM_GENERATIONS", "8"))
    judge_prompt = os.environ.get("JUDGE_PROMPT", "idea_preservation")
    target_max_tokens = int(os.environ.get("TARGET_MAX_TOKENS", "2048"))

    # Set global config
    rf_module.EMA_BETA = ema_beta
    rf_module.ALPHA = alpha
    rf_module.DELTA = delta
    rf_module.LAMBDA_MIN = lambda_min
    rf_module.LAMBDA_MAX = lambda_max
    rf_module.NUM_GENERATIONS = num_generations
    rf_module.JUDGE_PROMPT_NAME = judge_prompt
    rf_module.TARGET_MAX_TOKENS = target_max_tokens

    logger.info(f"Plugin config: guard_port={guard_port}, target_port={target_port}")
    logger.info(f"Adaptive: ema_beta={ema_beta}, alpha={alpha}, delta={delta}, lambda=[{lambda_min},{lambda_max}]")
    logger.info(f"Judge: {judge_prompt}, Target max_tokens: {target_max_tokens}")

    # Initialize clients
    init_clients(guard_port=guard_port, target_port=target_port)
    logger.info("Plugin initialized successfully")


# Auto-initialize when plugin is loaded
init_from_env()
