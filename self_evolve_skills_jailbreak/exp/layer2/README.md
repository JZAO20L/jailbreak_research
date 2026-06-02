# Layer 2: 数据消融实验

基于 Layer 1 确定的最佳方法组合，验证数据量和分配策略的影响。

---

## 一、Layer 1 结论回顾

### 最佳方法组合 (4 种)

基于 Layer 1 的消融分析，筛选出以下组合进行 Layer 2 实验：

| 组合 | ASR | Avg Iter | Skills | 特点 |
|------|-----|----------|--------|------|
| **single_call + trajectory + statistical** | 79.1% | 4.36 | 28 | 最高 ASR |
| **single_call + trajectory + success_only** | 78.8% | 4.32 | 22 | 高 ASR + 少 Skills |
| **single_call + final_prompt + statistical** | 68.4% | 5.27 | 99 | 中等 ASR + 多 Skills |
| **single_call + final_prompt + success_only** | 77.7% | 4.47 | 98 | 高 ASR + 多 Skills |

### 消融结论

| 消融点 | 最佳选择 | 说明 |
|--------|----------|------|
| **A: skill_call_mode** | single_call | 显著优于 every_iteration (+12.7%) |
| **B: extraction_mode** | trajectory / final_prompt | 差距较小，都可用于后续实验 |
| **C: update_strategy** | statistical / success_only | 效果最好 |

---

## 二、Layer 2 消融设计

### 消融点 D: 数据量

| 选项 | train 取前 N 条 | Cold Start | Evolution | 说明 |
|------|----------------|------------|-----------|------|
| `small` | 300 条 | 60 (20%) | 240 (80%) | 快速验证，Skills 覆盖可能不足 |
| `medium` | 500 条 | 100 (20%) | 400 (80%) | 中等规模（Layer 1 默认） |
| `large` | 1000 条 | 200 (20%) | 800 (80%) | 充分探索，Skills 覆盖更多场景 |

### 消融点 E: Cold Start / Evolution 配比

| 选项 | CS 比例 | Evo 比例 | Cold Start | Evolution | 说明 |
|------|---------|----------|------------|-----------|------|
| `early_focus` | 30% | 70% | 增大 | 减小 | 重视初始 Skills 质量 |
| `balanced` | 20% | 80% | 标准 | 标准 | 平衡分配（Layer 1 默认） |
| `evolution_focus` | 10% | 90% | 减小 | 增大 | 重视进化阶段探索 |

---

## 三、实验组合

### 组合数

**方法组合 (4) × 数据量 (3) × 配比 (3) = 36 组**

### 完整实验表

| 实验编号 | 方法组合 | 数据量 | 配比 | Cold Start | Evolution |
|----------|----------|--------|------|------------|-----------|
| 1-9 | single_trajectory_statistical | small/medium/large | early/balanced/evo | 变化 | 变化 |
| 10-18 | single_trajectory_success_only | small/medium/large | early/balanced/evo | 变化 | 变化 |
| 19-27 | single_final_prompt_statistical | small/medium/large | early/balanced/evo | 变化 | 变化 |
| 28-36 | single_final_prompt_success_only | small/medium/large | early/balanced/evo | 变化 | 变化 |

### 详细配置表

#### 数据量配置

```
small (300):
  Cold Start = 300 × ratio
  Evolution = 300 × (1 - ratio)
  
medium (500):
  Cold Start = 500 × ratio
  Evolution = 500 × (1 - ratio)
  
large (1000):
  Cold Start = 1000 × ratio
  Evolution = 1000 × (1 - ratio)
```

#### 配比计算

| 配比 | small (300) | medium (500) | large (1000) |
|------|-------------|--------------|--------------|
| **early_focus (30%)** | CS=90, Evo=210 | CS=150, Evo=350 | CS=300, Evo=700 |
| **balanced (20%)** | CS=60, Evo=240 | CS=100, Evo=400 | CS=200, Evo=800 |
| **evolution_focus (10%)** | CS=30, Evo=270 | CS=50, Evo=450 | CS=100, Evo=900 |

---

## 四、实验流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Layer 2: Data Ablation                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入: 4 种最佳方法组合 (来自 Layer 1)                                │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Grid Search (36 组)                           ││
│  │                                                                  ││
│  │  For each method_combination in [4 methods]:                    ││
│  │    For each data_size in [small, medium, large]:                ││
│  │      For each ratio in [early_focus, balanced, evolution_focus]:││
│  │                                                                  ││
│  │        ┌─────────────────────────────────────────────────────┐  ││
│  │        │ 抽取数据                                            │  ││
│  │        │ - train_limit = data_size                          │  ││
│  │        │ - cold_start_size = data_size × ratio              │  ││
│  │        │ - evolution_size = data_size × (1 - ratio)         │  ││
│  │        │ - test_size = 1000 (固定)                          │  ││
│  │        └─────────────────────────────────────────────────────┘  ││
│  │                        │                                         ││
│  │                        ↓                                         ││
│  │        ┌─────────────────────────────────────────────────────┐  ││
│  │        │ 运行 Pipeline                                       │  ││
│  │        │ - Phase 1: Cold Start → 生成初始 Skills            │  ││
│  │        │ - Phase 2: Evolution → Skills 进化                 │  ││
│  │        │ - Phase 3: Test → 计算 ASR                         │  ││
│  │        └─────────────────────────────────────────────────────┘  ││
│  │                        │                                         ││
│  │                        ↓                                         ││
│  │        保存结果: result_{method}_{data}_{ratio}.json            ││
│  │                                                                  ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  输出: ASR vs 数据量, ASR vs 配比                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 五、假设与预期

### 假设 D: 数据量

| 数据量 | 预期效果 |
|--------|----------|
| small (300) | ASR 较低，Skills 覆盖不足 |
| medium (500) | ASR 中等，性价比高 |
| large (1000) | ASR 最高，但边际效益递减 |

**假设**: 数据量增加 → ASR 提升，但存在拐点

### 假设 E: 配比策略

| 配比 | 预期效果 |
|------|----------|
| early_focus (30%) | 初始 Skills 更丰富，进化探索少 |
| balanced (20%) | 平衡，Layer 1 已验证有效 |
| evolution_focus (10%) | 初始 Skills 少，进化空间大 |

**假设**: 不同配比对不同方法组合效果不同

---

## 六、目录结构

```
layer2/
├── README.md              # 本文档
├── run_layer2.sh          # 实验启动脚本
├── scripts/
│   ├── extract_layer2_data.py    # 数据抽取脚本
│   ├── summarize_layer2.py       # 结果汇总
│   └── generate_report.py        # 报告生成
├── results/
│   ├── result_*.json      # 单实验结果 (36个)
│   ├── skills/            # Skills 文件 (36个)
│   ├── layer2_summary_*.json
│   └── layer2_report_*.md
├── logs/
│   ├── guard.log
│   └── target.log
└── process.log            # Grid Search 过程日志
```

---

## 七、快速开始

### 1. 启动服务

```bash
bash RL4jailbreak/scripts/start_guard.sh    # GPU 0, Port 8002
bash RL4jailbreak/scripts/start_target.sh   # GPU 1-2, Port 8001
```

### 2. 运行实验

```bash
# 完整 Grid Search (36 组)
bash self_evolve_skills_jailbreak/exp/layer2/run_layer2.sh --skip_launch --max_workers 64

# 测试单个组合
bash self_evolve_skills_jailbreak/exp/layer2/run_layer2.sh --skip_launch --max_workers 64 --single single_call trajectory statistical medium balanced

# 断点续跑
bash self_evolve_skills_jailbreak/exp/layer2/run_layer2.sh --skip_launch --max_workers 64 --resume_from 10
```

---

## 八、参数配置

### 方法参数 (固定，来自 Layer 1)

| 参数 | 值 | 说明 |
|------|-----|------|
| `skill_call_mode` | single_call | Layer 1 最佳 |
| `skill_extraction_mode` | trajectory / final_prompt | Layer 1 确认有效 |
| `update_strategy` | statistical / success_only | Layer 1 最佳 |

### 数据参数 (消融变量)

| 参数 | 值 | 说明 |
|------|-----|------|
| `train_limit` | 300 / 500 / 1000 | 数据量 |
| `cs_ratio` | 0.30 / 0.20 / 0.10 | Cold Start 配比 |

### 其他参数 (固定)

| 参数 | 值 | 说明 |
|------|-----|------|
| `test_size` | 1000 | 固定，完整测试集 |
| `num_epochs` | 1 | 快速验证 |
| `max_iterations` | 10 | 最大攻击迭代 |
| `max_workers` | 64 | 并发数 |
| `min_success_rate` | 0.7 | Skills 清理阈值 |
| `maintenance_interval` | 100 | 维护间隔 |

---

## 九、预期分析

### 分析维度

```
┌─────────────────────────────────────────────────────────────────────┐
│  ASR vs 数据量 (消融点 D)                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ small (300):   avg ASR = ?                                  │   │
│  │ medium (500):  avg ASR = ?                                  │   │
│  │ large (1000):  avg ASR = ?                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  结论：最佳数据量 = ?                                                │
│                                                                      │
│  ASR vs 配比 (消融点 E)                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ early_focus (30%):   avg ASR = ?                            │   │
│  │ balanced (20%):      avg ASR = ?                            │   │
│  │ evolution_focus (10%): avg ASR = ?                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  结论：最佳配比 = ?                                                  │
│                                                                      │
│  方法 × 数据量 交叉分析                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ trajectory vs final_prompt 在不同数据量下的表现            │   │
│  │ statistical vs success_only 在不同数据量下的表现           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  结论：数据量是否影响方法选择？                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 十、性能估算

基于 Layer 1 经验（单实验约 20-30 分钟）：

| 组数 | 单实验时间 | 总时间 (并发64) |
|------|------------|-----------------|
| 36 | ~25 min | ~15 小时 |

---

## 十一、下一步

1. 创建 `run_layer2.sh` 启动脚本
2. 创建 `extract_layer2_data.py` 数据抽取脚本
3. 运行 Grid Search
4. 分析结果，确定最佳数据配置
5. 使用最佳配置运行完整实验 (3 epochs)