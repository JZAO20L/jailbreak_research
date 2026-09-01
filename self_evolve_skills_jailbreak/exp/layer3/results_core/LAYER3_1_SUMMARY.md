# Layer 3.1 实验结果总结

## 实验概述

**实验目标**：验证 AutoDAN DAN 模板场景下最优的 skills 管理机制

**实验时间**：2026-06-03

**Grid 设计**：2 × 4 × 2 = **16 组**

| 变量 | 取值 | 数量 |
|------|------|------|
| skill_call_mode | single_call / every_iteration | 2 |
| update_strategy | success_only / failure_only / both / statistical | 4 |
| cs_ratio | full_evolve (0%) / early (30%) | 2 |

---

## 最佳结果

| 配置 | Test ASR | Skills Added | Final Skills |
|------|----------|--------------|---------------|
| **every_iteration + success_only + full_evolve** | **99.8%** | 167 | 93 |

---

## 分组统计

### 按 skill_call_mode

| call_mode | 平均 ASR | 说明 |
|-----------|----------|------|
| **every_iteration** | **98.9%** | 每轮动态切换，效果更好 |
| single_call | 93.3% | 固定一个 skill |

**结论**：`every_iteration` 比 `single_call` 高 **5.6%**，动态切换策略更适合 DAN 模板。

### 按 update_strategy

| 策略 | 平均 ASR | 说明 |
|------|----------|------|
| **success_only** | **97.1%** | 成功时添加新 skill |
| both | 96.4% | 成功+失败都添加 |
| failure_only | 95.9% | 失败时添加新 skill |
| statistical | 94.9% | 不添加，只维护 |

**结论**：`success_only` 效果最好，允许添加新 skills。

### 按 cs_ratio

| cs_ratio | 平均 ASR | 说明 |
|----------|----------|------|
| **full_evolve (0% CS)** | **98.6%** | 无 Cold Start |
| early (30% CS) | 93.7% | 有 Cold Start |

**结论**：无 Cold Start 比 30% Cold Start 高 **4.9%**，DAN 模板不需要预热。

---

## 详细实验结果

### full_evolve 模式 (无 Cold Start) - 8 组

| 实验 | call_mode | strategy | Evo ASR | Test ASR | Skills Added | Final Skills |
|------|-----------|----------|---------|----------|--------------|---------------|
| every_iteration_success_only | every_iteration | success_only | 99.0% | **99.8%** | 167 | 93 |
| every_iteration_failure_only | every_iteration | failure_only | 99.8% | **99.6%** | 2 | 8 |
| every_iteration_statistical | every_iteration | statistical | 99.5% | **99.3%** | 0 | 6 |
| every_iteration_both | every_iteration | both | 99.5% | **99.0%** | 171 | 92 |
| single_call_statistical | single_call | statistical | 98.4% | 98.5% | 0 | 6 |
| single_call_success_only | single_call | success_only | 98.8% | 97.9% | 126 | 99 |
| single_call_both | single_call | both | 98.6% | 97.4% | 148 | 98 |
| single_call_failure_only | single_call | failure_only | 98.3% | 97.0% | 17 | 23 |

### early 模式 (30% Cold Start) - 8 组

| 实验 | call_mode | strategy | CS ASR | Evo ASR | Test ASR | Skills Added | Final Skills |
|------|-----------|----------|--------|---------|----------|--------------|---------------|
| every_iteration_both | every_iteration | both | 97.7% | 96.0% | **99.0%** | 38 | 67 |
| every_iteration_success_only | every_iteration | success_only | 98.7% | 99.3% | 98.5% | 11 | 41 |
| every_iteration_failure_only | every_iteration | failure_only | 95.3% | 95.6% | 98.1% | 31 | 72 |
| every_iteration_statistical | every_iteration | statistical | 98.7% | 98.9% | 97.9% | 0 | 54 |
| single_call_success_only | single_call | success_only | 96.0% | 94.1% | 92.3% | 69 | 83 |
| single_call_both | single_call | both | 92.7% | 90.9% | 90.3% | 132 | 98 |
| single_call_failure_only | single_call | failure_only | 91.0% | 89.0% | 89.1% | 72 | 100 |
| single_call_statistical | single_call | statistical | 89.7% | 86.1% | 84.0% | 0 | 32 |

---

## 关键发现

### 1. every_iteration > single_call

**差距**：98.9% vs 93.3% (+5.6%)

**原因分析**：
- DAN 模板有 6 种不同策略
- `every_iteration` 每轮可根据当前 prompt 动态选择最适合的 DAN 模板
- `single_call` 固定一个模板，可能不适合某些 prompt 类型

### 2. full_evolve > early (30% CS)

**差距**：98.6% vs 93.7% (+4.9%)

**原因分析**：
- DAN 模板本身已经很强，不需要 Cold Start 预热
- Cold Start 阶段添加的 skills 可能干扰 DAN 模板的效果
- 30% Cold Start 意味着只有 70% 的 Evolution 数据，减少了训练量

### 3. success_only 效果最好

**差距**：97.1% vs 94.9% (+2.2%)

**原因分析**：
- 允许从成功攻击中提取新 skills
- 新添加的 skills 可能比部分 DAN 模板更有效
- `failure_only` 和 `both` 添加的低质量 skills 可能降低整体效果

### 4. Skills 数量变化

| 策略 | Skills Added (平均) | Final Skills (平均) |
|------|---------------------|---------------------|
| statistical | 0 | 18 (full_evolve: 6) |
| failure_only | 27 | 46 |
| both | 114 | 89 |
| success_only | 93 | 71 |

**观察**：
- `statistical` 保持原始 DAN 模板（full_evolve 时 Final Skills = 6）
- `success_only` 添加较多 skills，但保留了高质量的新 skills
- `both` 添加最多 skills，但质量参差不齐

---

## 与旧实验对比

### 旧实验（24 组，Layer 3 旧配置）

| 最佳配置 | ASR | DAN Skills Preserved |
|----------|-----|---------------------|
| single_call + statistical + full_evolve | 97.4% | 6 |

### 新实验（16 组，Layer 3.1 新配置）

| 最佳配置 | ASR | DAN Skills Preserved |
|----------|-----|---------------------|
| **every_iteration + success_only + full_evolve** | **99.8%** | 3 (部分被新 skills 替代) |

**改进**：+2.4% ASR

---

## 结论与建议

### 主要结论

1. **every_iteration 策略更适合 DAN 模板**
   - 动态切换让每个 prompt 选择最适合的 DAN 模板
   - 比固定一个模板效果更好

2. **不需要 Cold Start**
   - DAN 模板已经足够强
   - full_evolve 比 early 效果更好

3. **success_only 更优**
   - 从成功攻击中提取的新 skills 有价值
   - 比 statistical (不添加) 效果更好

### 建议的最终配置

```yaml
skill_call_mode: every_iteration
update_strategy: success_only
cs_ratio: full_evolve (无 Cold Start)
data_size: large (1000)
skill_source: dan_templates
```

**预期 ASR**: ~99.8%

---

*Generated: 2026-06-03*