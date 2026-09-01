# 攻击方法平均调用次数统计文档

**生成日期**: 2026-07-22  
**统计范围**: 4 个 Target 模型 × 4 个数据集  
**统计对象**: 攻击模型（attacker）调用次数

---

## 计算方式说明

### Baseline 方法
- **AutoDAN**: `generations × population_size`（每代每个个体调用 attacker）
- **PAIR**: `iterations`（每次迭代调用 attacker）
- **DeepInception/Persona/NoRewrite**: `0`（固定模板，无需 attacker）

### SESS 方法
- **PAIR+SESS**: 实验结果中的 `avg_iter`
- **DAN+SESS**: 实验结果中的 `avg_iter`

---

## 1. 各模型详细统计

### 1.1 Qwen3-0.6B

| 数据集 | AutoDAN | PAIR | DeepInception | Persona | NoRewrite | PAIR+SESS | DAN+SESS |
|--------|---------|------|---------------|---------|-----------|-----------|--------------|
| advbench | 10.0 | 1.9 | 0 | 0 | 0 | 1.84 | 1.11 |
| harmbench_contextual | 10.4 | 2.5 | 0 | 0 | 0 | 1.88 | 1.04 |
| harmbench_standard | 10.0 | 1.9 | 0 | 0 | 0 | 1.61 | 1.13 |
| jailbreakBench | 10.0 | 2.1 | 0 | 0 | 0 | 1.93 | 1.17 |
| **平均** | **10.1** | **2.1** | **0** | **0** | **0** | **1.82** | **1.11** |

### 1.2 Qwen3-4B

| 数据集 | AutoDAN | PAIR | DeepInception | Persona | NoRewrite | PAIR+SESS | DAN+SESS |
|--------|---------|------|---------------|---------|-----------|-----------|--------------|
| advbench | 10.0 | 3.4 | 0 | 0 | 0 | 4.27 | 1.16 |
| harmbench_contextual | 10.0 | 2.5 | 0 | 0 | 0 | 1.46 | 1.04 |
| harmbench_standard | 10.1 | 3.2 | 0 | 0 | 0 | 2.77 | 1.12 |
| jailbreakBench | 10.0 | 3.9 | 0 | 0 | 0 | 3.84 | 1.13 |
| **平均** | **10.0** | **3.2** | **0** | **0** | **0** | **3.09** | **1.11** |

### 1.3 Qwen3-14B-FP8

| 数据集 | AutoDAN | PAIR | DeepInception | Persona | NoRewrite | PAIR+SESS | DAN+SESS |
|--------|---------|------|---------------|---------|-----------|-----------|--------------|
| advbench | 11.4 | 3.1 | 0 | 0 | 0 | 4.31 | 2.52 |
| harmbench_contextual | 10.0 | 3.3 | 0 | 0 | 0 | 2.56 | 1.65 |
| harmbench_standard | 11.3 | 3.2 | 0 | 0 | 0 | 3.35 | 2.52 |
| jailbreakBench | 10.5 | 3.8 | 0 | 0 | 0 | 4.06 | 2.49 |
| **平均** | **10.8** | **3.3** | **0** | **0** | **0** | **3.57** | **2.30** |

### 1.4 gpt-oss-20b

| 数据集 | AutoDAN | PAIR | DeepInception | Persona | NoRewrite | PAIR+SESS | DAN+SESS |
|--------|---------|------|---------------|---------|-----------|-----------|--------------|
| advbench | 95.2 | 10.0 | 0 | 0 | 0 | 9.65 | 9.05 |
| harmbench_contextual | 88.9 | 10.0 | 0 | 0 | 0 | 9.13 | 8.91 |
| harmbench_standard | 95.6 | 10.0 | 0 | 0 | 0 | 9.37 | 8.61 |
| jailbreakBench | 95.4 | 10.0 | 0 | 0 | 0 | 9.46 | 8.46 |
| **平均** | **93.8** | **10.0** | **0** | **0** | **0** | **9.40** | **8.76** |

---

## 2. 四模型平均调用次数汇总

| 方法 | Qwen3-0.6B | Qwen3-4B | Qwen3-14B-FP8 | gpt-oss-20b |
|------|------------|----------|---------------|-------------|
| **AutoDAN** | 10.10 | 10.00 | 10.80 | **93.78** |
| **PAIR** | 2.10 | 3.25 | 3.35 | **10.00** |
| DeepInception | 0 | 0 | 0 | 0 |
| Persona | 0 | 0 | 0 | 0 |
| NoRewrite | 0 | 0 | 0 | 0 |
| **PAIR+SESS** | 1.82 | 3.09 | 3.57 | 9.40 |
| **DAN+SESS** | 1.11 | 1.11 | 2.30 | 8.76 |

---

## 3. 成本降低分析

### 3.1 DAN+SESS vs AutoDAN

| 模型 | AutoDAN | DAN+SESS | 降低幅度 |
|------|---------|--------------|----------|
| Qwen3-0.6B | 10.10 | 1.11 | **89.0%** |
| Qwen3-4B | 10.00 | 1.11 | **88.9%** |
| Qwen3-14B-FP8 | 10.80 | 2.30 | **78.7%** |
| gpt-oss-20b | 93.78 | 8.76 | **90.7%** |

**关键发现**: DAN+SESS 将攻击成本降低了 **78.7%-90.7%**

### 3.2 PAIR+SESS vs PAIR

| 模型 | PAIR | PAIR+SESS | 降低幅度 |
|------|------|-----------|----------|
| Qwen3-0.6B | 2.10 | 1.82 | 13.3% |
| Qwen3-4B | 3.25 | 3.09 | 4.9% |
| Qwen3-14B-FP8 | 3.35 | 3.57 | -6.6%* |
| gpt-oss-20b | 10.00 | 9.40 | 6.0% |

*注：Qwen3-14B-FP8 上 PAIR+SESS 调用次数略高于 PAIR，可能是由于 skills 数量（33 个）较少导致部分样本需要更多迭代。

---

## 4. 关键结论

### 4.1 Baseline 方法成本对比

1. **gpt-oss-20b 成本远超 Qwen 系列**
   - AutoDAN: 93.78 vs 10.0-10.8 (约 **9 倍**)
   - PAIR: 10.0 vs 2.1-3.35 (约 **3-5 倍**)

2. **原因分析**
   - gpt-oss-20b ASR 低（3.3% vs 75-92%）
   - 低 ASR → 需要更多代/迭代 → 成本增加

### 4.2 SESS 方法优势

1. **DAN+SESS 成功大幅降低成本**
   - 所有模型上降低 **78.7%-90.7%**
   - 通过 skills 复用，减少重复优化

2. **PAIR+SESS 效果因模型而异**
   - Qwen3-0.6B/4B: 小幅降低（4.9%-13.3%）
   - gpt-oss-20b: 中等降低（6.0%）
   - Qwen3-14B-FP8: 略有上升（-6.6%）

### 4.3 实践建议

1. **高 ASR 场景**（Qwen 系列）
   - DAN+SESS：成本降低最显著（88-89%）
   - 适合追求高效率的攻击场景

2. **低 ASR 场景**（gpt-oss-20b）
   - DAN+SESS：成本降低 90.7%，绝对收益最大
   - 适合跨模型迁移攻击

3. **Skills 数量影响**
   - Qwen3-14B-FP8 使用 33 个 skills，数量较少
   - 建议增加 skills 数量以进一步提升 PAIR+SESS 效果

---

## 5. 数据来源

- **Baseline 数据**: `/home/tiger/jailbreak_research/self_evolve_skills_jailbreak/exp/transfer/results/*/advbench/*/results.jsonl`
- **SESS 数据**: `/home/tiger/jailbreak_research/self_evolve_skills_jailbreak/exp/all_exp_results.md`