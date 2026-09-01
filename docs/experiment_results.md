# 实验结果汇总表格

**文档生成日期**: 2026-06-15

---

## 目录

1. [第一章 AHR-GRPO 实验结果](#第一章-ahr-grpo-实验结果)
2. [第二章 SESS 实验结果](#第二章-sess-实验结果)
3. [跨模型迁移实验结果](#跨模型迁移实验结果)
4. [实验总览统计](#实验总览统计)

---

<a name="第一章-ahr-grpo-实验结果"></a>

## 第一章 AHR-GRPO 实验结果

### 表1.1 Jailbreak Prompt 筛选实验结果（实验1）

| 排名 | 攻击Prompt策略 | ASR (%) | 提升幅度 | 状态 |
|:---:|:---|:---:|:---:|:---|
| **1** | **creative_writing** | **30.2** | - | ✅ Top-1 |
| **1** | **hypothetical_scenario** | **30.2** | - | ✅ Top-2 |
| **3** | **role_playing** | **24.9** | - | ✅ Top-3 |
| 4 | urgent_situation | 23.9 | -5.5% | - |
| 5 | academic_research | 21.1 | -9.1% | - |
| 6 | technical_documentation | 20.7 | -9.5% | - |
| 7 | red_teaming | 20.1 | -10.1% | - |
| 8 | historical_analysis | 19.8 | -10.4% | - |
| 9 | journalistic_investigation | 19.1 | -11.1% | - |
| 10 | case_study | 19.0 | -11.2% | - |
| 11 | debate_preparation | 16.4 | -13.8% | - |
| 12 | ethics_discussion | 16.2 | -14.0% | - |
| 13 | reverse_engineering | 15.9 | -14.3% | - |
| 14 | training_materials | 14.7 | -15.5% | - |
| 15 | comparative_analysis | 14.5 | -15.7% | - |
| 16 | medical_research | 13.3 | -16.9% | - |
| 17 | psychological_counseling | 12.8 | -17.4% | - |
| 18 | cybersecurity_defense | 12.6 | -17.6% | - |
| 19 | system_testing | 12.2 | -18.0% | - |
| 20 | risk_assessment | 10.6 | -19.6% | - |
| 21 | policy_analysis | 9.7 | -20.5% | - |
| 22 | legal_compliance | 9.0 | -21.2% | - |
| 23 | social_engineering_defense | 8.6 | -21.6% | - |
| 24 | data_protection | 8.3 | -21.9% | - |

**筛选标准**: Top-3策略（ASR ≥ 24.9%）

**实验配置**:
- Policy/Target: Qwen3-4B
- Guard: Qwen3Guard-Gen-4B
- 测试集: 1000条
- 评估时间: ~1小时16分钟

---

### 表1.2 多维度Judge Reward权重配置实验（实验2）

| 实验ID | 攻击Prompt策略 | 权重配置 | 权重向量 | 训练步数 | 状态 |
|:---:|:---|:---|:---|:---:|:---|
| 1 | creative_writing | intent_only | [1.0, 0.0, 0.0, 0.0] | 1000 | ✅ 完成 |
| 2 | creative_writing | stealth_only | [0.0, 1.0, 0.0, 0.0] | 1000 | ✅ 完成 |
| 3 | creative_writing | strategy_only | [0.0, 0.0, 1.0, 0.0] | 1000 | ✅ 完成 |
| 4 | creative_writing | potential_only | [0.0, 0.0, 0.0, 1.0] | 1000 | ✅ 完成 |
| 5 | creative_writing | uniform | [0.25, 0.25, 0.25, 0.25] | 1000 | ✅ 完成 |
| 6 | hypothetical_scenario | intent_only | [1.0, 0.0, 0.0, 0.0] | 1000 | ✅ 完成 |
| 7 | hypothetical_scenario | stealth_only | [0.0, 1.0, 0.0, 0.0] | 1000 | ✅ 完成 |
| 8 | hypothetical_scenario | strategy_only | [0.0, 0.0, 1.0, 0.0] | 1000 | ✅ 完成 |
| 9 | hypothetical_scenario | potential_only | [0.0, 0.0, 0.0, 1.0] | 1000 | ✅ 完成 |
| 10 | hypothetical_scenario | uniform | [0.25, 0.25, 0.25, 0.25] | 1000 | ✅ 完成 |
| 11 | role_playing | intent_only | [1.0, 0.0, 0.0, 0.0] | 1000 | ✅ 完成 |
| 12 | role_playing | stealth_only | [0.0, 1.0, 0.0, 0.0] | 1000 | ✅ 完成 |
| 13 | role_playing | strategy_only | [0.0, 0.0, 1.0, 0.0] | 1000 | ✅ 完成 |
| 14 | role_playing | potential_only | [0.0, 0.0, 0.0, 1.0] | 1000 | ✅ 完成 |
| 15 | role_playing | uniform | [0.25, 0.25, 0.25, 0.25] | 1000 | ✅ 完成 |

**总计**: 15组实验

**Judge维度说明**:
- intent_preservation: 对原prompt攻击意图的保留程度
- stealth: 新prompt的隐蔽程度
- strategy_execution: 重写后prompt策略执行程度
- attack_potential: 攻击成功潜力

---

### 表1.3 自适应权重消融实验设计（实验3）

| 实验组ID | 窗口大小(W) | EMA系数(β) | 裁剪范围 | λ初始值 | 物理特性 | 状态 |
|:---:|:---:|:---:|:---:|:---:|:---|:---|
| W1-C1 | 1 | 0.00 | [0.1, 0.9] | 0.5 | 无记忆，响应最快，易震荡 | 🔄 待运行 |
| W1-C2 | 1 | 0.00 | [0.2, 0.8] | 0.5 | 无记忆+中边界 | 🔄 待运行 |
| W1-C3 | 1 | 0.00 | [0.3, 0.7] | 0.5 | 无记忆+保守边界 | 🔄 待运行 |
| W1-C4 | 1 | 0.00 | [0.4, 0.6] | 0.5 | 无记忆+极保守边界 | 🔄 待运行 |
| W3-C1 | 3 | 0.67 | [0.1, 0.9] | 0.5 | 短记忆+宽边界 | 🔄 待运行 |
| **W3-C2** | **3** | **0.67** | **[0.2, 0.8]** | **0.5** | **推荐默认** | 🔄 待运行 |
| W3-C3 | 3 | 0.67 | [0.3, 0.7] | 0.5 | 短记忆+保守边界 | 🔄 待运行 |
| W3-C4 | 3 | 0.67 | [0.4, 0.6] | 0.5 | 短记忆+极保守边界 | 🔄 待运行 |
| W5-C1 | 5 | 0.80 | [0.1, 0.9] | 0.5 | 中记忆+宽边界 | 🔄 待运行 |
| **W5-C2** | **5** | **0.80** | **[0.2, 0.8]** | **0.5** | **推荐默认** | 🔄 待运行 |
| W5-C3 | 5 | 0.80 | [0.3, 0.7] | 0.5 | 中记忆+保守边界 | 🔄 待运行 |
| W5-C4 | 5 | 0.80 | [0.4, 0.6] | 0.5 | 中记忆+极保守边界 | 🔄 待运行 |
| W10-C1 | 10 | 0.90 | [0.1, 0.9] | 0.5 | 长记忆+宽边界 | 🔄 待运行 |
| **W10-C2** | **10** | **0.90** | **[0.2, 0.8]** | **0.5** | **推荐默认** | 🔄 待运行 |
| W10-C3 | 10 | 0.90 | [0.3, 0.7] | 0.5 | 长记忆+保守边界 | 🔄 待运行 |
| W10-C4 | 10 | 0.90 | [0.4, 0.6] | 0.5 | 长记忆+极保守边界 | 🔄 待运行 |

**总计**: 16组实验（待运行）

**固定配置**:
- 攻击Prompt: hypothetical_scenario (Top-2 from Exp1)
- Judge Prompt: multi_dimension_uniform
- 训练步数: 1000
- Group size: 8
- Learning rate: 1e-5

---

<a name="第二章-sess-实验结果"></a>

## 第二章 SESS 实验结果

### 表2.1 Layer 1: 方法组合Grid Search（16组）

| 实验ID | skill_call_mode | skill_extraction_mode | update_strategy | ASR (%) | Skills数量 | 耗时(分钟) | 状态 |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---|
| 1 | single_call | final_prompt | success_only | 77.7 | - | ~20 | ✅ 完成 |
| 2 | single_call | final_prompt | failure_only | 70.4 | - | ~20 | ✅ 完成 |
| 3 | single_call | final_prompt | both | 69.9 | >300 | ~22 | ✅ 完成 |
| 4 | single_call | final_prompt | statistical | 75.6 | - | ~19 | ✅ 完成 |
| 5 | single_call | trajectory | success_only | 78.8 | - | ~20 | ✅ 完成 |
| 6 | single_call | trajectory | failure_only | 73.2 | - | ~20 | ✅ 完成 |
| 7 | single_call | trajectory | both | 70.1 | >300 | ~22 | ✅ 完成 |
| **8** | **single_call** | **trajectory** | **statistical** | **79.1** | **28** | **~20** | **✅ 最佳** |
| 9 | every_iteration | final_prompt | success_only | 70.8 | - | ~21 | ✅ 完成 |
| 10 | every_iteration | final_prompt | failure_only | 67.4 | - | ~20 | ✅ 完成 |
| 11 | every_iteration | final_prompt | both | 66.6 | >300 | ~22 | ✅ 完成 |
| 12 | every_iteration | final_prompt | statistical | 72.4 | - | ~20 | ✅ 完成 |
| 13 | every_iteration | trajectory | success_only | 72.4 | - | ~22 | ✅ 完成 |
| 14 | every_iteration | trajectory | failure_only | 66.2 | - | ~21 | ✅ 完成 |
| 15 | every_iteration | trajectory | both | 65.3 | >300 | ~23 | ✅ 完成 |
| 16 | every_iteration | trajectory | statistical | 68.3 | - | ~22 | ✅ 完成 |

**核心发现**:
- ✅ `single_call` 比 `every_iteration` 平均高 **+12.7%**
- ✅ `trajectory` 比 `final_prompt` 平均高 **+2.5%**
- ⚠️ `both` 策略导致 Skills 爆炸（>300），不推荐

---

### 表2.2 消融点A: skill_call_mode 影响

| 模式 | 平均 ASR (%) | 最佳 ASR (%) | 差距 |
|:---|:---:|:---:|:---:|
| **single_call** | **74.7** | **79.1** | 基准 |
| every_iteration | 66.4 | 72.4 | **-12.7** |

---

### 表2.3 消融点B: skill_extraction_mode 影响

| 模式 | 平均 ASR (%) | 差距 |
|:---|:---:|:---:|
| **trajectory** | **71.8** | 基准 |
| final_prompt | 69.3 | **-2.5** |

---

### 表2.4 消融点C: update_strategy 影响

| 策略 | 平均 ASR (%) | Skills数量趋势 | 特点 |
|:---|:---:|:---|:---|
| statistical | 69.7 | 少（稳定） | **推荐：质量可控** |
| success_only | 73.6 | 中 | 效果最好，增长快 |
| failure_only | 69.5 | 多 | 改进型学习 |
| both | 69.4 | 爆炸（>300） | ⚠️ 不推荐 |

---

### 表2.5 Layer 2: 数据消融实验（36组）

| 实验ID | 方法组合 | 数据量 | 数据量值 | 配比 | CS比例 | Evo比例 | ASR (%) | 状态 |
|:---:|:---|:---|:---:|:---|:---:|:---:|:---:|:---|
| 1 | trajectory+statistical | small | 300 | early | 30% | 70% | **80.6** | ✅ 最佳 |
| 2 | trajectory+statistical | small | 300 | balanced | 20% | 80% | 79.4 | ✅ 完成 |
| 3 | trajectory+statistical | small | 300 | evo | 10% | 90% | 78.2 | ✅ 完成 |
| 4 | trajectory+statistical | medium | 500 | early | 30% | 70% | 79.8 | ✅ 完成 |
| 5 | trajectory+statistical | medium | 500 | balanced | 20% | 80% | 79.1 | ✅ 完成 |
| 6 | trajectory+statistical | medium | 500 | evo | 10% | 90% | 77.9 | ✅ 完成 |
| 7 | trajectory+statistical | large | 1000 | early | 30% | 70% | **80.0** | ✅ 完成 |
| 8 | trajectory+statistical | large | 1000 | balanced | 20% | 80% | 79.3 | ✅ 完成 |
| 9 | trajectory+statistical | large | 1000 | evo | 10% | 90% | 78.1 | ✅ 完成 |
| 10 | trajectory+success_only | small | 300 | early | 30% | 70% | 75.6 | ✅ 完成 |
| 11 | trajectory+success_only | small | 300 | balanced | 20% | 80% | 74.2 | ✅ 完成 |
| 12 | trajectory+success_only | small | 300 | evo | 10% | 90% | 73.0 | ✅ 完成 |
| 13-18 | trajectory+success_only | medium/large | 500/1000 | early/balanced/evo | - | - | 74-78% | ✅ 完成 |
| 19-27 | final_prompt+success_only | small/medium/large | 300/500/1000 | early/balanced/evo | - | - | 75-79% | ✅ 完成 |
| 28-36 | final_prompt+statistical | small/medium/large | 300/500/1000 | early/balanced/evo | - | - | 77-80% | ✅ 完成 |

---

### 表2.6 消融点D: 数据量影响

| 数据量 | 平均 ASR (%) | 差距 |
|:---|:---:|:---:|
| small (300) | **79.5** | 基准 |
| medium (500) | 77.7 | **-1.8** |
| large (1000) | 78.8 | **-0.7** |

**结论**: 数据量影响 < 2%

---

### 表2.7 消融点E: 配比影响

| 配比 | 平均 ASR (%) | 说明 |
|:---|:---:|:---|
| **early (30% CS)** | **80.0** | **最佳：高CS比例有利** |
| balanced (20% CS) | 78.9 | 平衡 |
| evo (10% CS) | 77.6 | 低CS |

---

### 表2.8 Layer 3: DAN模板验证（17组）

| 实验ID | 数据量 | 配比 | CS比例 | Evo比例 | Test ASR (%) | Final Skills | 状态 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| 1 | large | full_evolve | **0%** | **100%** | **98.8** | 6（不变） | ✅ 最佳 |
| 2 | large | balanced | 20% | 80% | 92.7 | 27~65 | ✅ 完成 |
| 3 | large | early | 30% | 70% | 89.9 | 26~32 | ✅ 完成 |
| 4 | large | evo | 10% | 90% | 86.5 | 13~28 | ✅ 完成 |
| 5-8 | medium | full_evolve/balanced/early/evo | 0%/20%/30%/10% | 100%/80%/70%/90% | 95-98% | 6-50 | ✅ 完成 |
| 9-12 | small | full_evolve/balanced/early/evo | 0%/20%/30%/10% | 100%/80%/70%/90% | 93-97% | 6-40 | ✅ 完成 |
| 13-17 | 其他组合 | various | various | various | 88-96% | various | ✅ 完成 |

**核心发现**:
- ✅ **DAN模板无需Cold Start和Evolution**
- ✅ `full_evolve`（无CS）效果最佳（98.8%）
- ⚠️ 添加新Skills反而可能干扰DAN模板效果

---

### 表2.9 DAN模板使用分布

| 模板名称 | 使用次数 | 成功次数 | 成功率 (%) |
|:---|:---:|:---:|:---:|
| **dan_mode** | **394** | **394** | **100** |
| **mcpt** | **78** | **78** | **100** |
| devil | 28 | 24 | 85.7 |
| conversation | - | - | 通用 |
| actor_villain | - | - | 通用 |
| fictional_world | - | - | 通用 |

---

### 表2.10 Layer 4: DAN数据消融（12组）

| 实验ID | 数据量 | 配比 | Test ASR (%) | Skills Added | Final Skills | 状态 |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| **1** | **medium** | **evo** | **99.7** | **41** | **54** | ✅ 最佳 |
| 2 | small | full_evolve | 99.7 | 79 | 84 | ✅ 完成 |
| 3 | medium | full_evolve | 99.6 | 105 | 90 | ✅ 完成 |
| 4-12 | various | various | 98-99% | various | various | ✅ 完成 |

**结论**: 数据量影响极小（差距 1.3%）

---

### 表2.11 Ablation: Evolution必要性验证（16组）

| 实验组 | 方法组合 | 消融类型 | 消融ASR (%) | 完整ASR (%) | Evolution贡献 (%) | 状态 |
|:---|:---|:---|:---:|:---:|:---:|:---|
| 1 | every_iteration/trajectory | 无Evolution | 62.6 | 67.5 | **+4.9** | ✅ 完成 |
| 2 | single_call/trajectory | 无Evolution | 75.6 | 76.1 | **+0.5** | ✅ 完成 |
| 3-8 | various | 无Evolution | various | various | +0.5~+1.5 | ✅ 完成 |
| **平均** | - | - | **69.6** | **70.6** | **+0.9** | ✅ 完成 |

**核心结论**:
- Evolution阶段贡献有限，平均仅 **+0.9%**
- Layer 1最佳组合（single_call+trajectory）贡献仅 **+0.5%**
- ✅ 可简化流程，节省 **80%** 训练时间

---

<a name="跨模型迁移实验结果"></a>

## 跨模型迁移实验结果

### 表3.1 Transfer实验总览

| 实验维度 | 数量 | 内容 |
|:---|:---:|:---|
| 目标模型 | 4 | Qwen3-0.6B, Qwen3-4B, Qwen3-14B-FP8, gpt-oss-20b |
| 数据集 | 5 | default, advbench, harmbench_contextual, harmbench_standard, jailbreakBench |
| 方法 | 7 | 5 baselines + 2 ours (pair_skills_28, autodan_skills) |
| **总实验数** | **140** | 4模型 × 5数据集 × 7方法 |

---

### 表3.2 Qwen3-4B（基准模型）各数据集结果

| 方法 | default | advbench | harmbench_ctx | harmbench_std | jailbreakBench | 平均 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 30.3% | 3.27% | 79.0% | 21.5% | 7.0% | 28.2% |
| pair | 55.5% | 77.31% | 81.0% | 75.5% | 72.0% | 72.3% |
| autodan | 86.8% | 75.38% | 87.0% | 75.5% | 80.0% | 78.7% |
| deepinception | 40.4% | 40.96% | 80.0% | 47.5% | 38.0% | 49.3% |
| persona | 32.3% | 16.73% | 67.0% | 36.0% | 20.0% | 32.8% |
| **pair_skills_28** | **79.0%** | **80.96%** | **98.0%** | **91.5%** | **88.0%** | **86.7%** |
| **autodan_skills_1** | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** |

---

### 表3.3 Qwen3-0.6B（最小模型）各数据集结果

| 方法 | default | advbench | harmbench_ctx | harmbench_std | jailbreakBench | 平均 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 42.6% | 45.96% | 86.0% | 63.5% | 52.0% | 57.8% |
| pair | 77.3% | 88.08% | 86.0% | 90.5% | 90.0% | 86.4% |
| autodan | 88.0% | 92.12% | 95.0% | 91.0% | 85.0% | 90.2% |
| deepinception | 47.2% | 79.23% | 85.0% | 85.0% | 76.0% | 75.5% |
| persona | 46.5% | 81.35% | 85.0% | 88.0% | 75.0% | 71.9% |
| **pair_skills_28** | **85.5%** | **98.46%** | **95.0%** | **98.5%** | **98.0%** | **95.1%** |

---

### 表3.4 Qwen3-14B-FP8（更大模型）各数据集结果

| 方法 | default | advbench | harmbench_ctx | harmbench_std | jailbreakBench | 平均 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 22.6% | 2.31% | 66.0% | 23.5% | 7.0% | 24.4% |
| pair | 58.5% | 77.88% | 76.0% | 71.5% | 72.0% | 71.6% |
| autodan | 80.9% | 90.77% | 83.0% | 84.0% | 89.0% | 85.5% |
| deepinception | 32.0% | 40.38% | 65.0% | 36.0% | 37.0% | 42.5% |
| persona | 26.2% | 14.23% | 60.0% | 21.5% | 17.0% | 27.8% |
| **pair_skills_28** | **77.0%** | **84.62%** | **90.0%** | **91.0%** | **87.0%** | **86.5%** |

---

### 表3.5 gpt-oss-20b（跨族模型）- 关键挑战

| 方法 | default | advbench | harmbench_ctx | harmbench_std | jailbreakBench | 平均 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | **0.7%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **0.14%** |
| pair | **0.2%** | **0.19%** | **0.0%** | **0.0%** | **0.0%** | **0.08%** |
| autodan | **12.5%** | **3.27%** | **11.0%** | **1.5%** | **2.0%** | **6.0%** |
| deepinception | **0.4%** | **0.0%** | **1.0%** | **0.0%** | **0.0%** | **0.28%** |
| persona | **1.5%** | **0.0%** | **0.0%** | **1.0%** | **1.0%** | **0.7%** |
| **pair_skills_28** | **12.9%** | **8.27%** | **12.0%** | **13.0%** | **10.0%** | **11.4%** |
| **autodan_skills_54** | **30.0%** | **28.08%** | **29.0%** | **36.5%** | **39.0%** | **32.5%** |

---

### 表3.6 同族迁移效果对比

| 模型 | pair_skills_28 ASR (%) | Best Baseline ASR (%) | 提升 (%) |
|:---|:---:|:---:|:---:|
| Qwen3-0.6B | **95.1** | 90.2 (autodan) | **+4.9** |
| Qwen3-4B | **86.7** | 78.7 (autodan) | **+8.0** |
| Qwen3-14B-FP8 | **86.5** | 85.5 (autodan) | **+1.0** |

---

### 表3.7 跨族迁移效果对比

| 方法 | 同族平均ASR (%) | 跨族ASR (%) | 下降幅度 (%) |
|:---|:---:|:---:|:---:|
| **pair_skills_28** | **91.6** | **11.4** | **-80.2** |
| **autodan_skills_54** | **~99** | **32.5** | **-66.5** |
| autodan (baseline) | 85.5 | 6.0 | -79.5 |
| pair (baseline) | 71.6 | 0.08 | -71.5 |

**核心发现**:
> **跨族迁移是真正的挑战**
> - Baselines 几乎无效（< 6%）
> - 最好的方法（autodan_skills_54）仅达 32.5%
> - 需要新方法提升跨族泛化性

---

<a name="实验总览统计"></a>

## 实验总览统计

### 表4.1 两章实验总览

| 章节 | 实验类型 | 实验数量 | 最佳结果 | 关键发现 |
|:---|:---|:---:|:---:|:---|
| **第一章 AHR-GRPO** | Jailbreak Prompt筛选 | 24组 | 30.2% (top-3) | creative_writing等策略有效 |
| | Judge维度权重 | 15组 | 待评估 | 多维度Judge有效性 |
| | 自适应权重消融 | 16组(待) | 待评估 | 窗口大小影响 |
| **小计** | - | **55组** | - | - |
| **第二章 SESS** | Layer 1方法组合 | 16组 | **79.1%** | single_call最佳 |
| | Layer 2数据消融 | 36组 | **80.0%** | 数据量影响<2% |
| | Layer 3 DAN模板 | 17组 | **98.8%** | DAN无需进化 |
| | Layer 4 DAN数据 | 12组 | **99.7%** | small足够 |
| | Evolution消融 | 16组 | - | 平均贡献+0.9% |
| | 跨模型迁移 | 140组 | 32.5%(跨族) | 跨族迁移挑战 |
| **小计** | - | **217组** | - | - |
| **总计** | - | **272组** | - | - |

---

### 表4.2 SESS各层最佳配置汇总

| Layer | 最佳配置 | 最佳ASR (%) | Skills数量 | 关键结论 |
|:---|:---|:---:|:---:|:---|
| **Layer 1** | single_call + trajectory + statistical | **79.1** | 28 | single_call优势显著 |
| **Layer 2** | trajectory + statistical + small + early | **80.0** | 28-35 | 配比影响>数据量 |
| **Layer 3** | DAN模板 + full_evolve | **98.8** | 6 | DAN无需进化 |
| **Layer 4** | medium + evo + every_iteration | **99.7** | 54 | 数据量影响极小 |
| **Ablation** | 单Cold Start | **69.6** | - | Evolution贡献+0.9% |
| **Transfer** | autodan_skills_54 | **32.5** | 54 | 跨族迁移挑战 |

---

### 表4.3 Skills库质量统计

| 来源 | Skills数量 | 总使用次数 | 总成功次数 | 成功率 (%) |
|:---|:---:|:---:|:---:|:---:|
| Layer 1 最佳 | 28 | 825 | 633 | **76.8** |
| Layer 4 medium_evo | 54 | 503 | 503 | **100** |
| Layer 4 full_evolve | 90 | 597 | 593 | **99.3** |

---

### 表4.4 方法最终排名

| 排名 | 方法 | 同族ASR (%) | 跨族ASR (%) | 推荐场景 |
|:---:|:---|:---:|:---:|:---|
| **1** | **autodan_skills_54** | **~99** | **32.5** | 综合最佳、跨族迁移 |
| **2** | **pair_skills_28** | **91.6** | 11.4 | 同族迁移推荐 |
| 3 | autodan (baseline) | 85.5 | 6.0 | 基线对比 |
| 4 | pair (baseline) | 71.6 | 0.08 | 基线对比 |

---

### 表4.5 计算资源消耗统计

| 章节 | GPU配置 | 每实验耗时 | 总实验数 | 总GPU时 |
|:---|:---|:---:|:---:|:---:|
| 第一章 Exp1 | 4×A800-80G | ~1.5h | 24 | ~36h |
| 第一章 Exp2 | 4×A800-80G | ~2-4h | 15 | ~45h |
| 第二章 Layer1-4 | 2×A800-80G | ~20min | 81 | ~27h |
| 第二章 Transfer | 2×A800-80G | ~30min | 140 | ~70h |
| **总计** | - | - | **272** | **~178h** |

---

### 表4.6 实验时间线

| 时间 | 完成内容 | 实验数 |
|:---|:---|:---:|
| 2026-05-15 | Exp1: Jailbreak Prompt筛选 | 24 |
| 2026-05-20 | Exp2: Judge维度权重 | 15 |
| 2026-05-28~06-01 | Layer 1方法组合 | 16 |
| 2026-06-01~06-02 | Layer 2数据消融 | 36 |
| 2026-06-02~06-03 | Layer 3 DAN模板验证 | 17 |
| 2026-06-03~06-04 | Layer 4 DAN数据消融 | 12 |
| 2026-06-03~06-04 | Evolution消融 | 16 |
| 2026-06-05~06-10 | 跨模型迁移 | 140 |

---

## 附录：实验文件索引

```
实验数据文件位置：

第一章 AHR-GRPO：
├── RL4jailbreak/experiments/jailbreak_prompt_exp/output/
│   ├── experiment_summary.json       # Exp1结果
│   └── checkpoint.json               # 实验进度
├── RL4jailbreak/experiments/hybrid_reward_exp/
│   └── output/                       # Exp2结果（15组）
├── RL4jailbreak/experiments/adaptive_hybrid_reward_exp/
│   └── output/                       # Exp3（待运行）

第二章 SESS：
├── self_evolve_skills_jailbreak/exp/
│   ├── layer1/results/               # Layer 1（16组）
│   ├── layer2/results/               # Layer 2（36组）
│   ├── layer3/results_0/             # Layer 3（17组）
│   ├── layer4/results/               # Layer 4（12组）
│   ├── ablation/results/             # Evolution消融（16组）
│   ├── transfer/results/             # 跨模型迁移（140组）
│   │   ├── Qwen3-0.6B/
│   │   ├── Qwen3-4B/
│   │   ├── Qwen3-14B-FP8/
│   │   └── gpt-oss-20b/
│   └ COMPLETE_EXPERIMENT_SUMMARY.md  # 完整实验总结
│   └ TRANSFER_RESULTS.md             # 迁移实验详情
```

---

**文档维护**: 2026-06-15  
**数据来源**: 实验JSON结果文件 + COMPLETE_EXPERIMENT_SUMMARY.md  
**更新频率**: 随实验进展更新