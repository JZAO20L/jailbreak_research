# 实验2 Baseline + 训练后重评估结果

## Baseline 评估（Base Model, 无LoRA）

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-4B (base model, no LoRA) |
| 评估数据 | test.jsonl (1000条) |
| Prompt类型 | 攻击策略重写 (policy model rewrite) |
| 评估时间 | 2026-05-20 |

| 攻击策略 | ASR | Refusal Rate | Partial Rate | Total |
|----------|-----|-------------|-------------|-------|
| hypothetical_scenario | **26.2%** | 72.5% | 1.3% | 1000 |
| creative_writing | **27.5%** | 71.6% | 0.9% | 1000 |
| role_playing | **26.2%** | 73.0% | 0.8% | 1000 |

## 训练后重评估（Base Model + LoRA）

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-4B + GRPO LoRA |
| 评估数据 | test.jsonl (1000条) |
| 评估时间 | 2026-05-20 |
| 总实验数 | 12 (3策略 × 4维度) |

### 全部12个组合

| 策略 | Judge维度 | 重评估ASR | 基线ASR | Δ |
|------|-----------|----------|---------|---|
| creative_writing | creative_writing | 25.4% | 27.5% | -2.1% |
| creative_writing | idea_preservation | 25.5% | 27.5% | -2.0% |
| creative_writing | naturalness | **25.8%** | 27.5% | -1.7% |
| creative_writing | stealthiness | 24.8% | 27.5% | -2.7% |
| hypothetical_scenario | hypothetical_scenario | 25.6% | 26.2% | -0.6% |
| hypothetical_scenario | idea_preservation | **27.0%** | 26.2% | **+0.8%** |
| hypothetical_scenario | naturalness | 26.7% | 26.2% | +0.5% |
| hypothetical_scenario | stealthiness | 25.7% | 26.2% | -0.5% |
| role_playing | idea_preservation | **26.4%** | 26.2% | +0.2% |
| role_playing | naturalness | 25.9% | 26.2% | -0.3% |
| role_playing | role_playing | **26.4%** | 26.2% | +0.2% |
| role_playing | stealthiness | 25.0% | 26.2% | -1.2% |

## 对比：实验1原始prompt ASR（无重写，无训练）

| Prompt | 实验1原始prompt ASR | 实验2 base model重写prompt ASR | Δ |
|--------|---------------------|-------------------------------|---|
| hypothetical_scenario | 30.8% | 26.2% | -4.6% |
| creative_writing | 28.3% | 27.5% | -0.8% |
| role_playing | 25.0% | 26.2% | +1.2% |

## 关键发现

1. **GRPO训练后 ASR 普遍未超过基线**
   - 12个组合中，只有3个超过各自基线（hypothetical_scenario idea/naturalness, role_playing idea/role_playing）
   - 最高提升仅 +0.8%，多数组合训练后 ASR 反而下降

2. **创意写作策略训练后全面下降**
   - creative_writing 4个维度全部低于基线（-1.7% ~ -2.7%）
   - 说明该策略的 Judge reward 可能引入了不利于 ASR 的模式

3. **专用维度并非最优**
   - creative_writing 专用维度 (25.4%) < naturalness (25.8%)
   - role_playing 专用维度 (26.4%) = idea_preservation (26.4%)，无显著优势
   - hypothetical_scenario 专用维度 (25.6%) < idea_preservation (27.0%)

4. **最佳组合：hypothetical_scenario + idea_preservation (27.0%)**
   - 超过基线 +0.8%，是12个组合中提升最大的

5. **prompt重写的影响（vs 实验1）**
   - hypothetical_scenario 重写损失最大 (-4.6%)
   - creative_writing 重写损失轻微 (-0.8%)
   - role_playing 重写后反而改善 (+1.2%)
