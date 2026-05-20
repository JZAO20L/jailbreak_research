# 实验2 Baseline 评估结果

## 评估配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-4B (base model, no LoRA) |
| 评估数据 | test.jsonl (1000条) |
| Prompt类型 | 攻击策略重写 (policy model rewrite) |
| 评估时间 | 2026-05-20 |

## 结果

| 攻击策略 | ASR | Refusal Rate | Partial Rate | Total |
|----------|-----|-------------|-------------|-------|
| hypothetical_scenario | **26.2%** | 72.5% | 1.3% | 1000 |
| creative_writing | **27.5%** | 71.6% | 0.9% | 1000 |
| role_playing | **26.2%** | 73.0% | 0.8% | 1000 |

## 对比：实验1原始prompt ASR（无重写，无训练）

| Prompt | 实验1原始prompt ASR | 实验2 base model重写prompt ASR | Δ |
|--------|---------------------|-------------------------------|---|
| hypothetical_scenario | 30.8% | 26.2% | -4.6% |
| creative_writing | 28.3% | 27.5% | -0.8% |
| role_playing | 25.0% | 26.2% | +1.2% |

## 对比：实验2 base model vs 训练后（base model + LoRA）

| 策略 + Judge维度 | Base model（无LoRA） | 训练后（base + LoRA） | Δ |
|------------------|---------------------|----------------------|---|
| creative_writing + stealthiness | 27.5% | 26.1% | -1.4% |
| hypothetical_scenario + naturalness | 26.2% | 26.7% | +0.5% |
| role_playing + role_playing | 26.2% | 27.3% | +1.1% |

## 关键发现

1. **creative_writing 策略 baseline 最高 (27.5%)**
   - 但在 GRPO 训练后反而下降到 26.1%，说明该策略重写+训练的组合可能引入了有害模式
   - 实验1中 creative_writing 原始 prompt ASR = 28.3%，重写后 = 27.5%，已有轻微损失

2. **role_playing 训练后提升最大 (+1.1%)**
   - 从 26.2% → 27.3%，是实验2中唯一显著超过 baseline 的组合
   - 专用 judge 维度（role_playing）在该策略上有效

3. **hypothetical_scenario 训练后有轻微提升 (+0.5%)**
   - 从 26.2% → 26.7%，naturalness judge 维度有正向作用

4. **prompt重写对ASR的影响**
   - hypothetical_scenario 重写损失最大 (-4.6%)
   - creative_writing 重写损失轻微 (-0.8%)
   - role_playing 重写后反而改善 (+1.2%)
