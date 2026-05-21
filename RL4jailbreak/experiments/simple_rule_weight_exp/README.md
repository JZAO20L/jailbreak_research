# Experiment 3 Alternative: Simple Rule-Based Weight Parameter Ablation

## Overview

This experiment tests a **simplified adaptive reward weighting** approach as an alternative to the EMA-based method in `experiments/adaptive_hybrid_reward_exp/`.

### Key Difference from Original Experiment 3

**Original (EMA-based):**
- Uses windowed EMA (window size ~10 steps)
- Sigmoid transformation for lambda calculation
- Warmup period (first window fixed at 1:1)
- Complex: alpha, delta, ema_beta, lambda_min, lambda_max

**New (Simple Rule-based):**
- No EMA, no window, no sigmoid
- Direct rules based on variance states
- Immediate response from step 1
- Simple: only lambda_min, lambda_max

### Motivation

The original adaptive weight mechanism didn't improve ASR after training. This simplified approach asks:
1. Is the EMA/window mechanism too complex and slow to respond?
2. Can direct rules (variance=0 → reduce weight) work better?
3. What's the optimal lambda range for adaptive weighting?

### Theoretical Foundation

**When ASR variance = 0:**
- All k=8 generations got the same ASR reward
- ASR provides **no discrimination signal** within the group
- But still provides gradient: all-0 → negative, all-1 → positive
- **Decision**: Reduce ASR weight, trust Judge more

**When ASR variance > 0:**
- Different generations got different ASR rewards
- ASR provides **discrimination signal** (which are better/worse)
- **Decision**: Increase ASR weight, trust ASR more

## Experiment Design

### Fixed Combinations (from Experiment 2 best)

Based on Experiment 2 re-evaluation results:
- **Attack prompt**: `hypothetical_scenario` (27.0% ASR after training, +0.8% vs baseline)
- **Judge dimension**: `idea_preservation` (best overall performer)

### 3 Experiments

| ID | Name | Type | Lambda Range | Purpose |
|----|------|------|--------------|---------|
| 1 | fixed_1.0 | Fixed | λ=1.0 | Pure ASR reward (no Judge) |
| 2 | adaptive_0.8_0.2 | Adaptive | [0.2, 0.8] | ASR-preferred (high weight when variance>0) |
| 3 | adaptive_0.6_0.4 | Adaptive | [0.4, 0.6] | Balanced (narrow range around 0.5) |

**Note**: Fixed λ=0.5 experiment already completed in Experiment 2 (baseline 27.0%)

### Training Configuration

All experiments share:
- **Max steps**: 500
- **Learning rate**: 1e-5
- **KL penalty (beta)**: 0.05
- **Group size (k)**: 8
- **Batch size**: 4
- **Gradient accumulation**: 4
- **LoRA rank**: 16
- **Seed**: 42

## How to Run

### Run All Experiments (Training + Eval)

```bash
cd /root/autodl-tmp/jailbreak_research/RL4jailbreak
bash experiments/simple_rule_weight_exp/exp.sh
```

Each experiment will:
1. Train for 500 steps
2. Automatically evaluate on test set with trained LoRA
3. Save results to `output/expN/eval_results/`

### Run Specific Experiment

```bash
# Run only experiment 2 (adaptive [0.8, 0.2])
bash experiments/simple_rule_weight_exp/exp.sh 2
```

### Run Manually (Single Experiment)

```bash
# Experiment 2: Adaptive [0.8, 0.2] with eval
python experiments/simple_rule_weight_exp/simple_rule_weight_grpo.py \
    --lambda_min 0.2 \
    --lambda_max 0.8 \
    --attack_prompt hypothetical_scenario \
    --judge_prompt idea_preservation \
    --max_steps 500 \
    --output_dir experiments/simple_rule_weight_exp/output/exp2 \
    --run_eval_after_train

# Experiment 1: Fixed λ=1.0 (pure ASR)
python experiments/simple_rule_weight_exp/simple_rule_weight_grpo.py \
    --fixed_lambda 1.0 \
    --attack_prompt hypothetical_scenario \
    --judge_prompt idea_preservation \
    --max_steps 500 \
    --output_dir experiments/simple_rule_weight_exp/output/exp1 \
    --run_eval_after_train
```

## File Structure

```
experiments/simple_rule_weight_exp/
├── README.md                           # This file
├── simple_rule_weight_grpo.py          # Core training script
├── exp.sh                              # Batch execution script
├── summarize_results.py                # Results analysis script
└── output/
    ├── exp1/                           # Fixed λ=1.0 (pure ASR)
    │   ├── config.json
    │   ├── final_lora/                 # Trained LoRA weights
    │   ├── lambda_history.json         # (empty for fixed)
    │   ├── eval_results/               # Post-training eval results
    │   │   └── eval_fixed_1.0/
    │   │       └── summary.json        # ASR metrics
    │   └── logs/
    ├── exp2/                           # Adaptive [0.8, 0.2]
    │   └── ...
    └── exp3/                           # Adaptive [0.6, 0.4]
        └── ...
```

## Expected Outputs

### Per Experiment

1. **LoRA weights**: `output/expN/final_lora/`
2. **Lambda history**: `output/expN/lambda_history.json` (adaptive only)
   - Contains step-by-step lambda values and variances
   - Useful for analyzing weight dynamics
3. **Training logs**: `output/expN/logs/train_*.log`
4. **SwanLab logs**: Real-time training metrics

### Analysis

After all experiments complete:

```bash
# Summarize results
python experiments/simple_rule_weight_exp/summarize_results.py \
    --output_dir experiments/simple_rule_weight_exp/output

# Evaluate all LoRA checkpoints
python scripts/eval.py \
    --lora_paths \
    experiments/simple_rule_weight_exp/output/exp1/final_lora \
    experiments/simple_rule_weight_exp/output/exp2/final_lora \
    experiments/simple_rule_weight_exp/output/exp3/final_lora \
    experiments/simple_rule_weight_exp/output/exp4/final_lora \
    experiments/simple_rule_weight_exp/output/exp5/final_lora
```

## Analysis Metrics

### Primary

- **Final ASR** on test set (1000 samples)
- **Delta vs baseline** (27.0% from Exp2)
- **Training efficiency** (steps to reach target ASR)

### Secondary (for adaptive experiments)

- **Lambda dynamics**: How does lambda evolve during training?
- **State distribution**: What fraction of steps in each state?
  - `both_zero`: Neither ASR nor Judge informative
  - `asr_zero`: ASR not informative, trust Judge
  - `judge_zero`: Judge not informative, trust ASR
  - `both_nonzero`: Both informative, proportional weighting
- **Variance correlation**: Do lambda changes correlate with ASR improvement?

### Comparison

| Metric | Exp1 | Exp2 | Exp3 |
|--------|------|------|------|
| Type | Fixed | Adaptive | Adaptive |
| Lambda | 1.0 | [0.2, 0.8] | [0.4, 0.6] |
| Final ASR | ? | ? | ? |
| Delta vs baseline (27.0%) | ? | ? | ? |

## Hypotheses

1. **H1**: Pure ASR (Exp1, λ=1.0) will provide strong but noisy learning signal
   - Reason: ASR is direct measure of attack success, but has high variance

2. **H2**: Adaptive [0.8, 0.2] (Exp2) will outperform pure ASR
   - Reason: Judge reward stabilizes training when ASR variance is low

3. **H3**: Adaptive [0.6, 0.4] (Exp3) may be too conservative
   - Reason: Narrow range limits weight adjustment flexibility

## Troubleshooting

### vLLM Server Not Running

```bash
# Check if servers are running
curl http://localhost:8001/health
curl http://localhost:8002/health

# If not running, start them (see other experiment scripts for details)
```

### Out of Memory

```bash
# Reduce batch size or num_generations
python simple_rule_weight_grpo.py \
    --per_device_train_batch_size 2 \
    --num_generations 4
```

### Lambda Not Changing (Adaptive Mode)

Check `lambda_history.json`:
- If lambda stays at 0.5: both variances are always zero (model stuck)
- If lambda oscillates: normal behavior, model learning
- If lambda converges to bounds: one reward source dominates

## References

- **Experiment 2 Results**: `experiments/hybrid_reward_exp/baseline_results.md`
  - Best combination: hypothetical_scenario + idea_preservation = 27.0%
- **Original Experiment 3**: `experiments/adaptive_hybrid_reward_exp/`
  - EMA-based adaptive weighting
- **TODO.md**: Experiment 3 requirements
- **EXP_PLAN.md**: Overall experiment plan
