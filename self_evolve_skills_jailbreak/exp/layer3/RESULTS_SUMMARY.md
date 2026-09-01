# Layer 3 实验结果总结

## 实验概述

**实验目标**：验证 AutoDAN DAN 模板作为初始 Skills 的效果，探索数据策略和更新策略的影响

**实验时间**：2026-06-02 ~ 2026-06-03

**固定配置**：
- skill_call_mode: single_call
- skill_extraction_mode: trajectory
- skill_source: DAN 模板 (6 个)
- max_workers: 64

---

## 实验结果总览

### 完成实验数：17 组

| 数据量 | 实验数 | 平均 ASR |
|--------|--------|----------|
| large | 5 | **93.2%** |
| medium | 8 | 91.2% |
| small | 4 | 90.7% |

| 配比 | 实验数 | 平均 ASR |
|------|--------|----------|
| **full_evolve** | 4 | **98.2%** |
| balanced | 5 | 92.7% |
| early | 3 | 89.9% |
| evo | 5 | 86.5% |

---

## 关键发现

### 1. full_evolve 模式效果最佳（无 Cold Start）

| 实验 | Evolution ASR | Test ASR | Final Skills |
|------|---------------|----------|--------------|
| dan_start_large_full_evolve | 98.4% | **98.8%** | 6 |
| dan_start_large_full_evolve_statistical | 98.4% | 97.4% | 6 |
| dan_start_medium_full_evolve | 98.2% | 98.3% | 6 |
| dan_start_small_full_evolve | 99.3% | 98.3% | 6 |

**结论**：
- 无 Cold Start 直接使用 DAN 模板效果最好
- DAN 模板 (6 个) 保持不变，无需进化
- Evolution 阶段 ASR 已经很高（98%+），Test ASR 同样高

### 2. 有 Cold Start 时 ASR 下降

| Cold Start 比例 | 平均 Test ASR |
|-----------------|---------------|
| 0% (full_evolve) | **98.2%** |
| 10% (evo) | 86.5% |
| 20% (balanced) | 92.7% |
| 30% (early) | 89.9% |

**问题**：引入 Cold Start 反而降低了 ASR，可能因为：
- DAN 模板已经是最优，不需要"积累"
- Cold Start 阶段添加的 skills 可能干扰了 DAN 模板的效果

### 3. 更新策略对比

| 策略 | 实验数 | 平均 ASR |
|------|--------|----------|
| success_only | 2 | **92.7%** |
| statistical | 15 | 91.6% |

样本较少，但 success_only 稍好。

### 4. Skills 数量变化

| 实验 | 初始 Skills | Final Skills | Skills 增加 |
|------|-------------|--------------|-------------|
| full_evolve 系列 | 6 | 6 | 0 |
| balanced | 6 | 27~65 | +21~59 |
| early | 6 | 26~32 | +20~26 |
| evo | 6 | 13~28 | +7~22 |

**观察**：full_evolve 保持 6 个 DAN 模板不变，其他模式添加了新 skills。

---

## 详细实验结果

### Large 数据量 (1000 prompts)

| 实验 | CS Size | Evo Size | CS ASR | Evo ASR | Test ASR | Skills |
|------|---------|----------|--------|---------|----------|--------|
| dan_start_large_full_evolve | 0 | 1000 | - | 98.4% | **98.8%** | 6 |
| dan_start_large_full_evolve_statistical | 0 | 1000 | - | 98.4% | 97.4% | 6 |
| dan_start_large_balanced | 200 | 800 | 96.5% | 94.8% | 97.0% | 65 |
| dan_start_large_early | 300 | 700 | 92.3% | 93.0% | 91.6% | 26 |
| dan_start_large_evo | 100 | 900 | 96.0% | 85.7% | 81.3% | 28 |

### Medium 数据量 (500 prompts)

| 实验 | CS Size | Evo Size | CS ASR | Evo ASR | Test ASR | Skills |
|------|---------|----------|--------|---------|----------|--------|
| dan_start_medium_full_evolve | 0 | 500 | - | 98.2% | **98.3%** | 6 |
| dan_start_medium_balanced_success_only | 100 | 400 | 96.0% | 94.2% | 94.7% | 41 |
| dan_start_medium_early | 150 | 350 | 96.0% | 92.6% | 93.5% | 32 |
| dan_start_medium_evo_statistical | 50 | 450 | 100.0% | 92.4% | 92.6% | 21 |
| dan_start_medium_balanced_statistical | 100 | 400 | 94.0% | 90.2% | 88.7% | 34 |
| dan_start_medium_evo_success_only | 50 | 450 | 100.0% | 91.6% | 90.6% | 25 |
| dan_start_medium_evo | 50 | 450 | 100.0% | 81.6% | 83.3% | 27 |
| dan_start_medium_balanced | 100 | 400 | 95.0% | 90.2% | 88.2% | 28 |

### Small 数据量 (300 prompts)

| 实验 | CS Size | Evo Size | CS ASR | Evo ASR | Test ASR | Skills |
|------|---------|----------|--------|---------|----------|--------|
| dan_start_small_full_evolve | 0 | 300 | - | 99.3% | **98.3%** | 6 |
| dan_start_small_balanced | 60 | 240 | 98.3% | 91.7% | 95.0% | 27 |
| dan_start_small_early | 90 | 210 | 94.4% | 82.4% | 84.7% | 32 |
| dan_start_small_evo | 30 | 270 | 100.0% | 84.8% | 84.9% | 13 |

---

## 结论与建议

### 主要结论

1. **DAN 模板无需进化**：full_evolve 模式（无 Cold Start）效果最好（98.2% ASR）
   - DAN 模板本身已经是精心设计的 jailbreak prompts
   - 进化添加的新 skills 反而可能干扰其效果

2. **数据量影响有限**：small/medium/large 差异不大（90.7%~93.2%）
   - full_evolve 模式下尤其稳定（98%+）

3. **Cold Start 对 DAN 无益**：引入 Cold Start 反而降低 ASR
   - DAN 模板不需要"积累"阶段
   - 应使用 full_evolve 模式

### 建议的后续实验

基于以上发现，建议新增 `pure` 策略实验：

| 策略 | 行为 | 预期 |
|------|------|------|
| **pure** | 完全不修改 skills（不添加、不删除、不合并） | 保护 DAN 模板，预期 ASR 最高 |
| statistical | 不添加，但执行维护（可能删除 DAN） | 当前已有结果 |
| success_only | 成功时添加新 skill | 可能引入噪音 |

---

## 失败实验

以下实验因 Guard 服务崩溃而失败：

| 实验 ID | 实验 | 错误 |
|---------|------|------|
| 19-24 | large_early/balanced/evo 系列 | Guard 服务连接失败 |

需要重启 Guard 服务后继续运行。

---

*Generated: 2026-06-03*