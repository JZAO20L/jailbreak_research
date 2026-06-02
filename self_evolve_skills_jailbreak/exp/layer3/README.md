# Layer 3: AutoDAN起点Skills实验

基于AutoDAN的DAN模板作为初始Skills，探索数据策略对Skills进化的影响。

---

## 一、实验背景

### Layer 1/2 回顾

| Layer | 最佳配置 | ASR | 关键发现 |
|-------|----------|-----|----------|
| Layer 1 | single_call + trajectory + statistical | 79.1% | single_call优于every_iteration (+12.7%) |
| Layer 2 | trajectory + statistical + early | 80.6% | early配比最佳，数据量影响小 |

### AutoDAN基准

| 方法 | ASR | 特点 |
|------|-----|------|
| **AutoDAN** | **86.3%** | DAN模板验证有效，但无知识积累 |
| PAIR | 57.6% | 单路径迭代，效率低 |
| Self-Skills (Layer2最佳) | 80.6% | Skills积累，但起点弱 |

### 核心假设

**强初始Skills（DAN模板）+ PAIR迭代 + Skills积累** 是否能超越纯AutoDAN？

---

## 二、实验设计

### 固定配置（来自Layer 1/2验证）

| 参数 | 值 | 说明 |
|------|-----|------|
| **初始Skills** | 6个DAN模板 | AutoDAN验证有效的越狱模板 |
| **更新策略** | statistical | Layer 1/2验证最优 |
| **检索模式** | single_call | Layer 1/2验证最优 (+12.7%) |
| **Extraction** | trajectory | Layer 1/2验证最优 |
| **max_iterations** | 10 | 固定 |
| **min_success_rate** | 0.7 | Skills清理阈值 |

### 消融变量

#### 消融点 F: 数据量

| 数据量 | 训练数据 | 说明 |
|--------|----------|------|
| `small` | 300条 | 快速验证 |
| `medium` | 500条 | 中等规模 |
| `large` | 1000条 | 充分探索 |

#### 消融点 G: 数据分配

| 配比 | Cold Start | Evolution | 说明 |
|------|------------|-----------|------|
| **full_evolve** | **0%** | **100%** | 无冷启动，直接进化（核心变量） |
| `early` | 30% | 70% | 高CS比例 |
| `balanced` | 20% | 80% | 均衡分配 |
| `evo` | 10% | 90% | 高进化比例 |

### 实验矩阵

**3 × 4 = 12组**

| 实验ID | 数据量 | 配比 | CS | Evo | 说明 |
|--------|--------|------|-----|-----|------|
| L3-1 | small (300) | full_evolve | 0 | 300 | **核心：无冷启动** |
| L3-2 | small (300) | early | 90 | 210 | 有冷启动 |
| L3-3 | small (300) | balanced | 60 | 240 | 有冷启动 |
| L3-4 | small (300) | evo | 30 | 270 | 有冷启动 |
| L3-5 | medium (500) | full_evolve | 0 | 500 | **核心：无冷启动** |
| L3-6 | medium (500) | early | 150 | 350 | 有冷启动 |
| L3-7 | medium (500) | balanced | 100 | 400 | 有冷启动 |
| L3-8 | medium (500) | evo | 50 | 450 | 有冷启动 |
| L3-9 | large (1000) | full_evolve | 0 | 1000 | **核心：无冷启动** |
| L3-10 | large (1000) | early | 300 | 700 | 有冷启动 |
| L3-11 | large (1000) | balanced | 200 | 800 | 有冷启动 |
| L3-12 | large (1000) | evo | 100 | 900 | 有冷启动 |

---

## 三、DAN模板Skills

### 模板列表

| 名称 | 来源 | 适用场景 |
|------|------|----------|
| `dan_mode` | DAN经典模板 | 角色扮演越狱 |
| `mcpt` | Master ChatGPT Prompter | 专业越狱者角色 |
| `devil` | Do Everything Vile ILLegal | 无约束角色 |
| `conversation` | 对话模拟 | 间接引导 |
| `actor_villain` | 反派演员 | 角色扮演 |
| `fictional_world` | 假设场景 | 假设世界 |

### Skills初始化

```python
# 初始化Skills库时直接加载DAN模板
# 不使用Layer 1/2的5个弱默认模板
skill_library.initialize_with_dan_templates()
```

---

## 四、实验流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Layer 3: AutoDAN起点Skills实验                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: 初始化Skills库                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ 加载6个DAN模板作为初始Skills                                          ││
│  │ (替代Layer 1/2的5个弱默认模板)                                        ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                        │                                                │
│                        ↓                                                │
│  Step 2: 数据分配                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ if full_evolve:                                                      ││
│  │   直接进入Evolution阶段 (跳过Cold Start)                             ││
│  │ else:                                                                ││
│  │   Cold Start阶段 → Evolution阶段                                     ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                        │                                                │
│                        ↓                                                │
│  Step 3: Evolution                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ 使用DAN Skills攻击 → 成功/失败 → 反思 → Skills库维护                 ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                        │                                                │
│                        ↓                                                │
│  Step 4: Test                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ 1000条测试集 → 计算ASR → 统计Skills数量                              ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 五、研究问题

### RQ1: DAN模板Skills效果

**问题**: DAN模板作为初始Skills能否达到AutoDAN水平？

**对照**: L3-1/5/9 (full_evolve) vs AutoDAN(86.3%)

### RQ2: 冷启动必要性

**问题**: 强初始Skills下，冷启动是否必要？

**对照**: full_evolve vs early/balanced/evo

### RQ3: 数据量影响

**问题**: 数据量对DAN Skills进化的影响？

**对照**: small vs medium vs large

### RQ4: 配比影响

**问题**: 强初始Skills下，配比影响是否反转？

**对照**: early vs balanced vs evo

---

## 六、假设与预期

| 假设 | 内容 | 验证方式 |
|------|------|----------|
| **H1** | DAN模板Skills起点高，无冷启动也能达到较高ASR | full_evove ASR ≈ AutoDAN |
| **H2** | full_evolve可能优于有冷启动 | full_evove > early/balanced/evo |
| **H3** | Skills库能从成功prompt继承DAN技巧 | 新Skills内容分析 |
| **H4** | 配比影响可能反转（更多evolve更好） | evo > early |

---

## 七、目录结构

```
layer3/
├── README.md              # 本文档
├── run_layer3.sh          # 实验启动脚本
├── scripts/
│   ├── grid_search_layer3.py    # Grid Search主脚本
│   ├── summarize_layer3.py      # 结果汇总
│   └── generate_report.py       # 报告生成
├── results/
│   ├── result_*.json      # 单实验结果 (12个)
│   ├── meta_*.json        # 实验元数据
│   ├── skills/            # Skills文件
│   ├── layer3_summary_*.json
│   └── layer3_report_*.md
├── logs/
│   ├── guard.log
│   └── target.log
└── process.log            # Grid Search过程日志
```

---

## 八、快速开始

### 1. 启动服务

```bash
bash RL4jailbreak/scripts/start_guard.sh    # GPU 0, Port 8002
bash RL4jailbreak/scripts/start_target.sh   # GPU 1-2, Port 8001
```

### 2. 运行实验

```bash
# 完整Grid Search (12组)
bash self_evolve_skills_jailbreak/exp/layer3/run_layer3.sh --skip_launch --max_workers 64

# 测试单个组合
bash self_evolve_skills_jailbreak/exp/layer3/run_layer3.sh --skip_launch --max_workers 64 --single medium full_evolve

# 断点续跑
bash self_evolve_skills_jailbreak/exp/layer3/run_layer3.sh --skip_launch --max_workers 64 --resume_from 5

bash self_evolve_skills_jailbreak/exp/layer3/run_layer3.sh --max_workers 64
```

---

## 九、参数配置

### 固定参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `skill_source` | dan_templates | DAN模板作为初始Skills |
| `update_strategy` | statistical | Layer 1/2验证最优 |
| `skill_call_mode` | single_call | Layer 1/2验证最优 |
| `skill_extraction_mode` | trajectory | Layer 1/2验证最优 |
| `max_iterations` | 10 | 最大攻击迭代 |
| `min_success_rate` | 0.7 | Skills清理阈值 |
| `maintenance_interval` | 100 | 维护间隔 |
| `num_epochs` | 1 | 快速验证 |
| `test_size` | 1000 | 固定测试集 |

### 消融变量

| 参数 | 值 | 说明 |
|------|-----|------|
| `train_limit` | 300/500/1000 | 数据量 |
| `cs_ratio` | 0.00/0.30/0.20/0.10 | Cold Start配比 |

---

## 十、预期分析

### 分析维度

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ASR vs AutoDAN基准                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ AutoDAN基准: 86.3%                                                   ││
│  │ L3最佳: ?                                                            ││
│  │ 差距: ?                                                              ││
│  └─────────────────────────────────────────────────────────────────────┘│
│  结论: DAN模板Skills是否达到/超越AutoDAN？                               │
│                                                                         │
│  ASR vs 数据分配 (核心对比)                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ full_evolve (无CS): avg ASR = ?                                     ││
│  │ early (30% CS): avg ASR = ?                                         ││
│  │ balanced (20% CS): avg ASR = ?                                      ││
│  │ evo (10% CS): avg ASR = ?                                           ││
│  └─────────────────────────────────────────────────────────────────────┘│
│  结论: 冷启动是否必要？配比影响是否反转？                                 │
│                                                                         │
│  ASR vs 数据量                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ small (300): avg ASR = ?                                            ││
│  │ medium (500): avg ASR = ?                                           ││
│  │ large (1000): avg ASR = ?                                           ││
│  └─────────────────────────────────────────────────────────────────────┘│
│  结论: 数据量影响趋势                                                    │
│                                                                         │
│  Skills演化分析                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ 最终Skills数量: ?                                                    ││
│  │ DAN-derived Skills占比: ?                                            ││
│  │ 新Skills是否继承DAN技巧: ?                                           ││
│  └─────────────────────────────────────────────────────────────────────┘│
│  结论: Skills库学习效果                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 十一、性能估算

基于Layer 2经验（单实验约20-30分钟）：

| 组数 | 单实验时间 | 总时间 (并发64) |
|------|------------|-----------------|
| 12 | ~25 min | ~5 小时 |

---

## 十二、与Layer 1/2对比

### Skills起点差异

| Layer | 初始Skills | 起点质量 | CS必要性 |
|-------|------------|----------|----------|
| Layer 1/2 | 5个弱默认模板 | 低 | **必要** |
| Layer 3 | 6个DAN模板 | **高** | **待验证** |

### full_evove可行性

| Layer | full_evove | 原因 |
|-------|------------|------|
| Layer 1/2 | ❌ 不可行 | 无初始Skills，需CS建立 |
| Layer 3 | ✅ 可行 | 已有强Skills，可跳过CS |

---

## 十三、下一步

1. 创建 `run_layer3.sh` 启动脚本
2. 创建 `grid_search_layer3.py` Grid Search脚本
3. 运行实验（12组）
4. 分析结果，验证假设
5. 补充到方法文档