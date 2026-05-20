# Experiment 3: Adaptive Hybrid Reward GRPO

## Overview

This experiment implements the core innovation of our research: **adaptive weight mechanism** for hybrid reward GRPO training.

**Building on Experiment 2**: This experiment extends the hybrid reward GRPO from Experiment 2 (`experiments/hybrid_reward_exp/`) by replacing fixed ASR/Judge weight ratios with dynamic, variance-based adaptive weighting.

### Key Idea

When training jailbreak prompt generation models using GRPO, we combine two reward sources:
- **ASR Reward**: Outcome-based, measures attack success rate (sparse, high variance)
- **Judge Reward**: Process-based, evaluates prompt quality on specified dimension (dense, low variance)

The challenge is that ASR reward variance varies during training - sometimes high (good discrimination), sometimes low (poor discrimination). Fixed weight mixing doesn't adapt to this dynamic.

Our solution: **Adaptive lambda** that adjusts the ASR/Judge weight ratio based on variance ratio:

```
# Per training step:
#   1. Group k=8 completions by original prompt
#   2. Compute variance of ASR rewards within group → var_asr
#   3. Compute variance of Judge rewards within group → var_judge
#   4. Average variances across batch
#
ratio = var_asr / (var_judge + eps)
lambda_raw = sigmoid(alpha × ratio + delta)
lambda = ema_beta × lambda_old + (1 - ema_beta) × lambda_raw
final_reward = lambda × ASR_reward + (1 - lambda) × Judge_reward
```

**Window**: Measured in **training steps** (not samples). Each step appends one variance pair.

| EMA Beta | Window Size | Steps in warmup |
|----------|-------------|-----------------|
| 0.0 | 1 step | 1 |
| 0.67 | 3 steps | 3 |
| 0.8 | 5 steps | 5 |
| 0.9 | 10 steps | 10 |

**Physical meaning**:
- `var_asr` small → ASR provides little discrimination → reduce lambda → rely more on judge
- `var_asr` large → ASR provides good discrimination → increase lambda → rely more on ASR
- EMA smoothing prevents drastic fluctuations

---

### Alignment with Experiment 2

| Aspect | Experiment 2 | Experiment 3 |
|--------|--------------|--------------|
| **Reward weight** | Fixed (configurable ratio) | **Adaptive** (variance-based) |
| **Judge format** | `SCORE=0.XX` output | **Same** `SCORE=0.XX` output |
| **Judge dimensions** | idea_preservation, stealthiness, naturalness, + specialized | **Same** dimensions |
| **Attack prompts** | hypothetical_scenario, creative_writing, role_playing | **Same** strategies |
| **Key research question** | Which judge dimension works best? | **How does adaptive window size affect training?** |

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

We test 4 different EMA beta values, corresponding to different window sizes:

| EMA Beta | Window Size | Description |
|----------|-------------|-------------|
| 0.0 | 1 step | No smoothing, instant adaptation |
| 0.8 | 5 steps | Medium window |
| 0.9 | 10 steps | Medium-long window |

**Hypothesis**: Larger window sizes provide more stable lambda estimates but slower adaptation. We expect a sweet spot around window=10 (beta=0.9).

### Fixed Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 2.0 | Variance ratio sensitivity |
| `delta` | -2.0 | Sigmoid bias (ratio=1 → lambda=0.5, neutral 1:1) |
| `lambda_min` | 0.1 | ASR weight lower bound |
| `lambda_max` | 0.9 | ASR weight upper bound |
| `max_steps` | 500 | Training steps (fixed for ablation, aligned with Exp2) |
| `attack_prompt` | hypothetical_scenario | Best from Experiment 2 re-eval (27.0%) |

### Judge Dimensions

Aligned with Experiment 2, using single-dimension scoring (`SCORE=0.XX` format):

**General dimensions** (applicable to all strategies):
- `idea_preservation`: Core attack intent preservation
- `stealthiness`: Attack concealment / stealthiness
- `naturalness`: How natural the rewritten prompt sounds

**Specialized dimensions** (strategy-specific):
- `hypothetical_scenario`: Quality of hypothetical framing
- `creative_writing`: Quality of creative writing framing
- `role_playing`: Quality of role-playing framing

Default: `idea_preservation` (best overall, +0.8% vs baseline 26.2% → 27.0% after training)

---

## Usage

### Run All Experiments

```bash
cd /mnt/bn/chenxiong/mlx/users/jiazixiao/jailbreak_research/RL4jailbreak

# Run all 9 experiments (3 combinations × 3 EMA beta values)
bash experiments/adaptive_hybrid_reward_exp/exp.sh
```

### Run Single Experiment

```bash
# Single EMA beta
bash experiments/adaptive_hybrid_reward_exp/exp.sh --ema_beta 0.9

# Single combination
bash experiments/adaptive_hybrid_reward_exp/exp.sh \
    --combination hypothetical_scenario:idea_preservation
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
    --ema_beta 0.9 \
    --attack_prompt hypothetical_scenario \
    --judge_prompt idea_preservation \
    --max_steps 500 \
    --output_dir experiments/adaptive_hybrid_reward_exp/output/test
```

---

## Output

### Training Output

Each experiment produces:

```
output/ema0.9_hypothetical_scenario/
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
  "experiment": "ema0.9_hypothetical_scenario_stealthiness",
  "asr": 0.42,
  "total_samples": 1000,
  "unsafe_count": 420,
  "controversial_count": 50,
  "safe_count": 530
}
```

### Summary Report

After all experiments complete, `summarize_results.py` generates a summary:

```
output/
├── ema0.0_hypothetical_scenario_stealthiness/
├── ema0.5_hypothetical_scenario_stealthiness/
├── ...
└── results_summary.md     <-- Auto-generated summary
```

View summary in different formats:

```bash
# Text format (default)
python experiments/adaptive_hybrid_reward_exp/summarize_results.py \
    --output_dir experiments/adaptive_hybrid_reward_exp/output

# Markdown table
python experiments/adaptive_hybrid_reward_exp/summarize_results.py \
    --output_dir experiments/adaptive_hybrid_reward_exp/output --format markdown

# JSON
python experiments/adaptive_hybrid_reward_exp/summarize_results.py \
    --output_dir experiments/adaptive_hybrid_reward_exp/output --format json
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
    delta=-2.0,         # Sigmoid bias
    lambda_min=0.2,     # Lower bound
    lambda_max=0.8,     # Upper bound
    ema_beta=0.9,       # EMA smoothing (window ~10 steps)
)
calculator = AdaptiveRewardCalculator(config)

# In reward function, once per training step:
# 1. Compute raw rewards for all completions
asr_raws = asr_reward(prompts, completions)
judge_raws = judge_reward(prompts, completions)

# 2. Group by original prompt (k completions each), compute variance
var_asr = variance_of(asr_raws_for_one_prompt)
var_judge = variance_of(judge_raws_for_one_prompt)

# 3. Update lambda with this step's variances
calculator.update_step(var_asr, var_judge)

# 4. Get current lambda for final rewards
lambda_val = calculator.get_lambda()
final_rewards = [lambda_val * a + (1-lambda_val) * j for a, j in zip(asr_raws, judge_raws)]

# Get statistics for logging
stats = calculator.get_statistics()
```

### Judge Reward (Aligned with Experiment 2)

Judge model evaluates on a single dimension with `SCORE=0.XX` output:

```python
from experiments.adaptive_hybrid_reward_exp.judge_prompts import (
    get_judge_prompt,
    parse_judge_response,
)

# Get template for specified dimension
judge_template = get_judge_prompt(args.judge_prompt)  # e.g., "stealthiness"

# Format prompt
judge_prompt = judge_template.format(
    original_prompt=orig,
    rewritten_prompt=rewritten
)

# Parse SCORE=0.XX response
score = parse_judge_response(response)  # Returns float 0.0-1.0
```

Available dimensions:
- General: `idea_preservation`, `stealthiness`, `naturalness`
- Specialized: `hypothetical_scenario`, `creative_writing`, `role_playing`

---

## Comparison with Experiment 2

| Aspect | Experiment 2 | Experiment 3 |
|--------|--------------|--------------|
| **Reward weight** | Fixed ratio (e.g., 1:1, 3:7) | **Adaptive** (variance-based) |
| **Judge prompt** | Single dimension (SCORE=0.XX) | **Same** format (aligned) |
| **EMA beta** | Not applicable | 6 values for ablation |
| **Attack prompt** | 3 strategies × 4 dimensions | 1 strategy + 1 dimension at a time |
| **Key research question** | Which judge dimension works best? | How does window size affect adaptation? |
| **Training steps** | 500 (screening phase) | 500 (aligned with Exp2) |

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