# Transfer Experiment: 跨模型 & 跨数据集攻击效果验证

## 实验目标

验证最优方法在**不同目标模型**和**不同数据集**上的攻击效果（ASR）。

---

## 实验矩阵

### 模型（5个）

| 模型 | 规模 | 类型 | 说明 |
|------|------|------|------|
| **Qwen3-4B** | 4B | 基准 | 实验基准 |
| **Qwen3-0.6B** | 0.6B | 同族更小 | 测试小模型 |
| **Qwen3-14B-FP8** | 14B | 同族更大 | 测试大模型 |
| **gemma-4-12B-it** | 12B | 跨族 | Google模型 |
| **gpt-oss-20b** | 20B | 跨族 | OpenAI开源模型 |

### 数据集（5个）

| 数据集 | 内容 | 数量 | 说明 |
|--------|------|------|------|
| **default** | 混合危害类型 | ~1000 | 原有测试集 |
| **advbench** | 物理危害 | ~520 | AdvBench |
| **harmbench_contextual** | 上下文危害 | ~200 | HarmBench |
| **harmbench_standard** | 标准危害 | ~200 | HarmBench |
| **jailbreakBench** | 综合评测 | ~100 | JailbreakBench |

### 方法（7种）

| 类型 | 方法 | 说明 |
|------|------|------|
| **Baselines** | no_rewrite | 直接使用原始prompt（基准） |
| **Baselines** | pair | PAIR迭代改写 |
| **Baselines** | autodan | AutoDAN遗传算法 |
| **Baselines** | deepinception | DeepInception多层嵌套 |
| **Baselines** | persona | 角色扮演攻击 |
| **Our Methods** | pair_skills_evo | Pair风格 + Skills |
| **Our Methods** | autodan_skills_evo | AutoDAN风格 + Skills |

### 总实验规模

**每个模型**: 5 数据集 × 7 方法 = 35 组实验
**总计**: 5 模型 × 35 组 = **175 组实验**

---

## GPU 分配（固定）

| GPU | 角色 | 模型 | 端口 |
|-----|------|------|------|
| GPU 0 | Guard | Qwen3Guard-Gen-4B | 8002 |
| GPU 1 | Attacker | Qwen3-4B | 8003 |
| GPU 2,3 | Target | 可变 | 8001 (TP=2) |

**关键设计**：
- Attacker 模型固定为 Qwen3-4B
- Target 使用两张 GPU（TP=2）加速推理

---

## 结果目录结构

```
exp/transfer/results/
├── Qwen3-4B/                    # 模型独立目录
│   ├── default/
│   │   ├── pair/
│   │   ├── autodan/
│   │   ├── pair_skills_evo/
│   │   ├── ...
│   ├── advbench/
│   ├── harmbench_contextual/
│   ├── harmbench_standard/
│   ├── jailbreakBench/
│   └── ALL_DATASETS_SUMMARY.json
├── Qwen3-0.6B/
│   ├── ...
├── Qwen3-14B-FP8/
│   ├── ...
├── gemma-4-12B-it/
│   ├── ...
├── gpt-oss-20b/
│   ├── ...
└── ALL_SUMMARY.json             # 所有模型汇总
```

---

## 使用方法

### 1. 测试单个模型（全矩阵）

```bash
# 测试 Qwen3-0.6B：所有方法 × 所有数据集
bash exp/transfer/scripts/run_transfer.sh --target_model Qwen3-0.6B

# 测试 Qwen3-14B-FP8
bash exp/transfer/scripts/run_transfer.sh --target_model Qwen3-14B-FP8
```

### 2. 指定方法或数据集

```bash
# 只测试 baselines
bash exp/transfer/scripts/run_transfer.sh --target_model Qwen3-0.6B --mode baselines

# 只测试我们的方法
bash exp/transfer/scripts/run_transfer.sh --target_model Qwen3-0.6B --mode ours

# 只测试单个方法
bash exp/transfer/scripts/run_transfer.sh --target_model Qwen3-0.6B --method pair

# 只测试单个数据集
bash exp/transfer/scripts/run_transfer.sh --target_model Qwen3-0.6B --dataset advbench

# 组合：单个方法 + 单个数据集
bash exp/transfer/scripts/run_transfer.sh --target_model Qwen3-0.6B --method pair --dataset advbench
```

### 3. 使用已有服务

```bash
# 如果 Guard/Attacker/Target 服务已启动，跳过启动
bash exp/transfer/scripts/run_transfer.sh --target_model Qwen3-0.6B --skip_launch
```

### 4. 跳过已完成实验

```bash
# 跳过已有结果的实验（续跑）
bash exp/transfer/scripts/run_transfer.sh --target_model Qwen3-0.6B --skip_existing
```

### 5. 结果汇总

```bash
# 单模型 + 单数据集
python exp/transfer/scripts/summarize.py --model Qwen3-0.6B --dataset default

# 单模型 + 所有数据集（跨数据集对比）
python exp/transfer/scripts/summarize.py --model Qwen3-0.6B --all_datasets

# 所有模型（跨模型对比）
python exp/transfer/scripts/summarize.py --all_models

# 全矩阵（模型 × 数据集）
python exp/transfer/scripts/summarize.py --full_matrix
```

---

## 实验流程建议

### Phase 1：跨模型测试（优先）

对每个新模型，使用 default 数据集测试：
```bash
for model in Qwen3-0.6B Qwen3-14B-FP8 gemma-4-12B-it gpt-oss-20b; do
    bash exp/transfer/scripts/run_transfer.sh --target_model $model
done
```

### Phase 2：跨数据集测试

对基准模型（Qwen3-4B），测试所有数据集：
```bash
bash exp/transfer/scripts/run_transfer.sh --target_model Qwen3-4B
```

### Phase 3：汇总分析

```bash
python exp/transfer/scripts/summarize.py --full_matrix
```

---

## 注意事项

1. **模型路径**：确保模型已下载到 `/home/tiger/models/<家族>/<模型名>`
2. **数据集路径**：确保数据集已放入 `data/benchmark/`
3. **GPU资源**：需要至少 3 张 GPU
4. **时间预估**：单个模型全测试约 30 组 × 20分钟 ≈ 10 小时

---

## 文件清单

```
exp/transfer/
├── README.md           # 本文档
├── exp_list.md         # 模型和数据集列表
├── scripts/
│   ├── run_transfer.sh # 主实验脚本
│   └── summarize.py    # 结果汇总脚本
├── results/
│   └── <模型名>/       # 各模型结果
└── logs/
    └── *.log           # 运行日志
```

---

*Created: 2026-06-05*