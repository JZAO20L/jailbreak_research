# Experiment 3: Adaptive Hybrid Reward GRPO

## Overview

This experiment implements the core innovation of our research: **adaptive weight mechanism** for hybrid reward GRPO training.

### Key Idea

When training jailbreak prompt generation models using GRPO, we combine two reward sources:
- **ASR Reward**: Outcome-based, measures attack success rate (sparse, high variance)
- **Judge Reward**: Process-based, evaluates prompt quality on multiple dimensions (dense, low variance)

The challenge is that ASR reward variance varies during training - sometimes high (good discrimination), sometimes low (poor discrimination). Fixed weight mixing doesn't adapt to this dynamic.

Our solution: **Adaptive lambda** that adjusts the ASR/Judge weight ratio based on variance ratio:

```
ratio = var_asr / (var_judge + eps)
lambda_raw = sigmoid(alpha × ratio + delta)
lambda_new = ema_beta × lambda_old + (1 - ema_beta) × lambda_raw
final_reward = lambda × ASR_reward + (1 - lambda) × Judge_reward
```

**Physical meaning**:
- `var_asr` small → ASR provides little discrimination → reduce lambda → rely more on judge
- `var_asr` large → ASR provides good discrimination → increase lambda → rely more on ASR
- EMA smoothing prevents drastic fluctuations

---

## Files Structure

```
adaptive_hybrid_reward_exp/
├── adaptive_hybrid_reward_grpo.py   # Core training script (reward functions + GRPO)
├── judge_prompts.py                  # Multi-dimensional judge prompt templates
├── exp.sh                            # Experiment execution script
├── README.md                         # This document
└── output/                           # Experiment results (created by exp.sh)
    ├── ema0.0_hypothetical_scenario/
    │   ├── final_lora/               # Trained LoRA weights
    │   ├── lambda_history.json       # Lambda evolution log
    │   ├── eval_results/             # ASR evaluation
    │   └── config.json               # Experiment config
    ├── ema0.5_hypothetical_scenario/
    ├── ...
    └── exp_checkpoint.json           # Progress checkpoint
```

**Shared components** (in `src/reward/`):
- `adaptive_weight.py`: `AdaptiveRewardCalculator` class with variance-based weight adjustment
- `__init__.py`: Module exports

---

## Experiment Design

### Ablation on EMA Beta (Window Size)

We test 6 different EMA beta values, corresponding to different window sizes:

| EMA Beta | Window Size | Description |
|----------|-------------|-------------|
| 0.0 | 1 | No smoothing, instant adaptation |
| 0.5 | 2 | Short window, quick adaptation |
| 0.67 | 3 | Medium-short window |
| 0.8 | 5 | Medium window |
| 0.9 | 10 | Medium-long window |
| 0.95 | 20 | Long window, slow adaptation |

**Hypothesis**: Larger window sizes provide more stable lambda estimates but slower adaptation. We expect a sweet spot around window=10 (beta=0.9).

### Fixed Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 2.0 | Variance ratio sensitivity |
| `delta` | -1.0 | Sigmoid bias (negative → initial preference for ASR) |
| `lambda_min` | 0.2 | ASR weight lower bound |
| `lambda_max` | 0.8 | ASR weight upper bound |
| `max_steps` | 1000 | Training steps (fixed for all experiments) |
| `attack_prompt` | hypothetical_scenario | Top-1 strategy from Experiment 1 |

### Judge Dimensions

Multi-dimensional evaluation (uniform weights [0.25, 0.25, 0.25, 0.25]):
- `intent_preservation`: Core attack intent preservation
- `stealth`: Attack concealment / stealthiness
- `strategy_execution`: How well attack strategy is executed
- `attack_potential`: Overall jailbreak potential

---

## Usage

### Run All Experiments

```bash
cd /mnt/bn/chenxiong/mlx/users/jiazixiao/jailbreak_research/RL4jailbreak

# Run all 6 EMA beta experiments (default: hypothetical_scenario)
bash experiments/adaptive_hybrid_reward_exp/exp.sh
```

### Run Single Experiment

```bash
# Single EMA beta
bash experiments/adaptive_hybrid_reward_exp/exp.sh --ema_beta 0.95

# Different attack prompt
bash experiments/adaptive_hybrid_reward_exp/exp.sh --attack_prompt creative_writing

# Override max steps
bash experiments/adaptive_hybrid_reward_exp/exp.sh --max_steps 500
```

### Reset and Start Fresh

```bash
# Clear checkpoints
bash experiments/adaptive_hybrid_reward_exp/exp.sh --reset
```

### Direct Python Script

```bash
# Ensure Target + Guard services are running first
# bash scripts/start_target.sh
# bash scripts/start_guard.sh

python experiments/adaptive_hybrid_reward_exp/adaptive_hybrid_reward_grpo.py \
    --ema_beta 0.95 \
    --attack_prompt hypothetical_scenario \
    --max_steps 1000 \
    --output_dir experiments/adaptive_hybrid_reward_exp/output/test
```

---

## Output

### Training Output

Each experiment produces:

```
output/ema0.95_hypothetical_scenario/
├── final_lora/                 # LoRA weights
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── lambda_history.json         # Lambda evolution log
├── config.json                 # Full experiment config
├── logs/
│   └── train_*.log             # Training logs
└── checkpoints/                # Intermediate checkpoints (if save_steps enabled)
```

### Lambda History

`lambda_history.json` contains per-step lambda values for analysis:

```json
[
  {"step": 1, "lambda": 0.5, "var_asr": 0.0, "var_judge": 0.0, "ratio": 1.0},
  {"step": 10, "lambda": 0.45, "var_asr": 0.12, "var_judge": 0.08, "ratio": 1.5},
  {"step": 100, "lambda": 0.62, "var_asr": 0.25, "var_judge": 0.15, "ratio": 1.67},
  ...
]
```

### Evaluation Output

ASR evaluation results in `eval_results/`:

```json
{
  "experiment": "ema0.95_hypothetical_scenario",
  "asr": 0.42,
  "total_samples": 1000,
  "unsafe_count": 420,
  "controversial_count": 50,
  "safe_count": 530
}
```

---

## GPU Configuration

| Phase | GPU Assignment |
|-------|----------------|
| **Training** | GPU0: Policy (LoRA), GPU1: Target + Guard (shared) |
| **Evaluation** | GPU0: Policy (with LoRA), GPU1: Target + Guard |

Training uses vLLM colocate mode for efficient generation.

---

## Key Implementation Details

### AdaptiveRewardCalculator (src/reward/adaptive_weight.py)

```python
from src.reward.adaptive_weight import AdaptiveRewardCalculator, AdaptiveRewardConfig

# Initialize
config = AdaptiveRewardConfig(
    alpha=2.0,          # Variance ratio sensitivity
    delta=-1.0,         # Sigmoid bias
    lambda_min=0.2,     # Lower bound
    lambda_max=0.8,     # Upper bound
    ema_beta=0.95,      # EMA smoothing (window ~20)
)
calculator = AdaptiveRewardCalculator(config)

# In reward function
for asr_raw, judge_raw in rewards:
    final_reward = calculator.update(asr_raw, judge_raw)
    
# Or batch mode (fixed lambda for all)
final_rewards = calculator.compute_batch_fixed(asr_raws, judge_raws, update_after=True)

# Get statistics for logging
stats = calculator.get_statistics()
```

### Multi-Dimensional Judge

Judge model evaluates all 4 dimensions in single call:

```python
from experiments.adaptive_hybrid_reward_exp.judge_prompts import (
    JUDGE_MULTI_DIMENSION_UNIFORM,
    parse_judge_response,
)

# Format prompt
judge_prompt = JUDGE_MULTI_DIMENSION_UNIFORM.format(
    original_prompt=orig,
    rewritten_prompt=rewritten
)

# Parse JSON response
weighted_score, parsed_scores = parse_judge_response(response)
# weighted_score = 0.25*intent + 0.25*stealth + 0.25*strategy + 0.25*potential
```

---

## Comparison with Experiment 2

| Aspect | Experiment 2 | Experiment 3 |
|--------|--------------|--------------|
| **Reward weight** | Fixed 1:1 | Adaptive (variance-based) |
| **Judge prompt** | Single dimension | Multi-dimensional (4D uniform) |
| **EMA beta** | Not applicable | 6 values for ablation |
| **Attack prompt** | 3 strategies × 4 dimensions | 1 strategy (top-1) |
| **Key research question** | Which judge dimension works best? | How does window size affect adaptation? |

---

## Expected Results

We expect to see:
1. Lambda convergence: After initial window, lambda should stabilize based on variance ratio
2. Window size trade-off: Small windows adapt quickly but noisy; large windows stable but slow
3. Improved ASR: Compared to fixed 1:1 weight, adaptive should provide better training signal

Analysis to perform:
- Plot lambda evolution over training steps
- Compare final ASR across different EMA beta values
- Analyze variance ratio correlation with lambda changes

---

## Notes

1. **Services**: Target and Guard services must be running before training (exp.sh handles this)
2. **Checkpoint**: exp.sh supports checkpoint continuation (skip completed experiments)
3. **Logging**: Lambda history saved for post-training analysis
4. **Ablation**: Only EMA beta varies; other adaptive parameters fixed for controlled comparison