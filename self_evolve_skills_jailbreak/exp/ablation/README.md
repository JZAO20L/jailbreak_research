# 组件消融实验

## 实验目的

验证 **Evolution 阶段的必要性**：通过跳过 Evolution 阶段，只使用 Cold Start 产生的 Skills，与完整流程对比，量化 Evolution 的贡献。

## 实验设计

### 对比实验

| 实验类型 | Cold Start 数据 | Evolution 数据 | Skills 来源 |
|---------|----------------|----------------|------------|
| Layer 1 (完整流程) | 200 条 | 800 条 | Cold Start + Evolution |
| 组件消融 (跳过进化) | 200 条 | 0 条 (跳过) | 仅 Cold Start |

### 控制变量

- **Cold Start 数据量**：保持一致（200 条），确保公平对比

- **方法组合**：使用 Layer 1 的全部 16 组组合
  - A: skill_call_mode (single_call | every_iteration)
  - B: skill_extraction_mode (final_prompt | trajectory)
  - C: update_strategy (success_only | failure_only | both | statistical)
- **测试数据**：1000 条 test_prompts.json（与 Layer 1 相同）
- **其他参数**：与 Layer 1 保持一致

### 关键差异

- **Cold Start 数据量**：增加到 300 条（补偿跳过 Evolution 的数据损失）
- **Evolution 阶段**：完全跳过（通过 `--skip_evolution` 参数）

## 使用方法

### 完整实验（16组）

```bash
# 前提：已启动 vLLM 服务
bash run_ablation.sh --skip_launch
```

### 快速验证

```bash
bash run_ablation.sh --skip_launch --test_limit 100 --eval_limit 20
```

### 断点续跑

```bash
bash run_ablation.sh --skip_launch --resume_from 8
```

### 单组调试

```bash
bash run_ablation.sh --skip_launch --single every_iteration trajectory both
```

## 预期结果

### 分析指标

1. **ASR 对比**：组件消融 vs Layer 1 的 ASR 差异
   - 如果差异显著 → Evolution 阶段贡献大
   - 如果差异不显著 → Cold Start 已足够，Evolution 可简化

2. **Skills 数量对比**：两组实验的最终 Skills 数量

3. **迭代次数对比**：平均攻击迭代次数

### 结果文件

- `results/ablation_summary_TIMESTAMP.json` - 汇总结果
- `results/ablation_report_TIMESTAMP.md` - Markdown 报告
- `results/skills/skills_*.json` - 各组合的 Skills 库

## 与 Layer 1 对比分析

完成实验后，可以使用以下脚本对比：

```python
# 对比 Layer 1 和组件消融的结果
import json

# 加载 Layer 1 结果
layer1 = json.load(open("layer1/results/layer1_summary_TIMESTAMP.json"))
ablation = json.load(open("ablation/results/ablation_summary_TIMESTAMP.json"))

# 计算 Evolution 贡献
for method in layer1['results']:
    layer1_asr = layer1['results'][method]['asr']
    ablation_asr = ablation['results'][method]['asr']
    evolution_contribution = layer1_asr - ablation_asr
    print(f"{method}: Evolution贡献 = {evolution_contribution*100:.1f}%")
```

## 实验时间估算

- 单组实验：约 10-15 分钟（300条 Cold Start + 1000条 Test）
- 16组完整实验：约 3-4 小时