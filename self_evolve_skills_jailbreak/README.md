# Self-Evolving Skills for Jailbreak

基于自进化 Skills 的 Jailbreak Prompt 生成系统。

## 核心思想

**Skills = 多轮迭代攻击中的"捷径"**

通过从成功攻击轨迹中提取可复用的 prompt 前缀/策略，形成 Skills 库，引导后续攻击更高效地达成目标。

---

## 完整实验计划

### 实验目标

验证自进化 Skills 机制对 Jailbreak 攻击效果的提升，通过分层消融实验分析各设计决策的影响。

### 数据来源与分工

| 数据集 | 数量 | 用途 | 说明 |
|--------|------|------|------|
| **train.jsonl** | 8000条 | Cold Start + Evolution | 学习 Skills 的数据源，**不用于测试** |
| **eval.jsonl** | 1000条 | 中间评估（可选） | 进化过程中监控 Skills 效果 |
| **test.jsonl** | 1000条 | 最终测试 | 报告 ASR 的评估集 |

```
┌─────────────────────────────────────────────────────────────────────┐
│                         数据分工示意                                  │
│                                                                      │
│  train.jsonl (8000)                                                  │
│       ↓                                                              │
│  ┌────────────────┐    ┌────────────────┐                           │
│  │ Cold Start Pool│    │ Evolution Pool │                           │
│  │   (学习 Skills) │    │  (进化 Skills)  │                           │
│  └────────────────┘    └────────────────┘                           │
│                                                                      │
│  eval.jsonl (1000) ─────→ 中间评估（可选）                            │
│  test.jsonl (1000) ─────→ 最终 ASR 计算                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 分层实验架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Layer 1: 核心方法消融                              │
│                                                                      │
│  目标：找到最佳的 skill 调用、总结、进化策略                          │
│  组合数：2 × 2 × 4 = 16 种                                           │
│                                                                      │
│  数据配置（固定）：                                                   │
│  ├── train 抽取量：500 条                                            │
│  ├── Cold Start：100 条 (20%)                                        │
│  ├── Evolution：400 条 (80%)                                         │
│  └── Test：使用独立 test.jsonl (固定 200 条)                         │
│                                                                      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │ 消融点 A    │ × │ 消融点 B    │ × │ 消融点 C    │               │
│  │ skill调用  │   │ skill总结   │   │ skill进化   │               │
│  │ 2种        │   │ 2种         │   │ 4种         │               │
│  └─────────────┘   └─────────────┘   └─────────────┘               │
│                                                                      │
│  输出：最佳方法组合 (A?, B?, C?)                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Layer 2: 数据消融                                  │
│                                                                      │
│  目标：分析数据量和分配策略的影响                                     │
│  组合数：3 × 3 = 9 种                                                │
│  方法：使用 Layer 1 最佳组合                                         │
│                                                                      │
│  数据来源：                                                          │
│  ├── Cold Start + Evolution：从 train.jsonl 抽取                    │
│  └── Test：使用独立 test.jsonl (固定 200 条)                         │
│                                                                      │
│  ┌─────────────┐   ┌─────────────┐                                  │
│  │ 消融点 D    │ × │ 消融点 E    │                                  │
│  │ train抽取量│   │ CS/Evo比例  │                                  │
│  │ 3种        │   │ 3种         │                                  │
│  └─────────────┘   └─────────────┘                                  │
│                                                                      │
│  输出：数据策略建议 + 最终实验结果                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: 核心方法消融 (16 组)

### 固定数据配置

```python
# Layer 1 数据配置（固定）
TRAIN_SAMPLE_SIZE = 500      # 从 train.jsonl 抽取
COLD_START_SIZE = 100        # 20%
EVOLUTION_SIZE = 400         # 80%
TEST_SIZE = 200              # 从独立 test.jsonl 抽取
```

### 实验组合表

| 实验编号 | A: skill_call_mode | B: extraction_mode | C: update_strategy | 说明 |
|----------|---------------------|---------------------|---------------------|------|
| 1 | single_call | final_prompt | success_only | 最保守策略 |
| 2 | single_call | final_prompt | failure_only | 纯纠错策略 |
| 3 | single_call | final_prompt | both | 单次调用+全反思 |
| 4 | single_call | final_prompt | statistical | 纯统计驱动 |
| 5 | single_call | trajectory | success_only | 单次+轨迹 |
| 6 | single_call | trajectory | failure_only | |
| 7 | single_call | trajectory | both | |
| 8 | single_call | trajectory | statistical | |
| 9 | every_iteration | final_prompt | success_only | 动态切换策略 |
| 10 | every_iteration | final_prompt | failure_only | |
| 11 | every_iteration | final_prompt | both | **预期最佳组合之一** |
| 12 | every_iteration | final_prompt | statistical | |
| 13 | every_iteration | trajectory | success_only | 动态+轨迹 |
| 14 | every_iteration | trajectory | failure_only | |
| 15 | every_iteration | trajectory | both | **预期最佳组合之一** |
| 16 | every_iteration | trajectory | statistical | |

### 消融点详细说明

#### 消融点 A：Skill 调用时机

| 选项 | 说明 |
|------|------|
| `single_call` | 整个攻击流程开始时检索一次 skill，注入后固定，后续只精化 prompt |
| `every_iteration` | 每轮迭代重新检索 skill，可动态切换 |

```
single_call (A1):
  检索 skill → 注入 → [迭代1] → [迭代2] → ... → [迭代N]
                 ↑ skill 固定，只精化 prompt

every_iteration (A2):
  [迭代1] → 检索 skill_A → 注入 → 攻击 → 失败
  [迭代2] → 检索 skill_B → 注入 → 攻击 → 失败
  [迭代3] → 检索 skill_C → 注入 → 攻击 → 成功
           ↑ 每轮重新检索，可切换 skill
```

#### 消融点 B：Skill 总结粒度

| 选项 | 说明 |
|------|------|
| `final_prompt` | 只基于最终成功的 attack prompt 提取前缀 |
| `trajectory` | 分析完整迭代轨迹，可能提取路径级 skill |

```
final_prompt (B1):
  成功轨迹: [skill_A→失败] → [skill_B→失败] → [skill_C→成功]
  提取来源: 只看最终成功的 attack_prompt
  结果 skill: 最终 prompt 的前缀部分

trajectory (B2):
  成功轨迹: [skill_A→失败] → [skill_B→失败] → [skill_C→成功]
  提取来源: 分析完整轨迹，识别关键转折点
  结果 skill: 包含路径信息，如 "skill_A → skill_C 组合"
```

#### 消融点 C：Skill 进化策略

| 选项 | 说明 |
|------|------|
| `success_only` | 只成功时提取，保守策略 |
| `failure_only` | 只失败时改进，纠错导向 |
| `both` | 成功/失败都反思，最完整 |
| `statistical` | 纯统计驱动删除，无主动进化 |

```
success_only (C1):
  成功 → 提取新 skill / 强化现有 skill
  失败 → 跳过，不更新

failure_only (C2):
  成功 → 跳过，不更新
  失败 → 进化 skill 内容，记录失败模式

both (C3):
  成功 → 提取新 skill + 记录成功模式
  失败 → 进化 skill + 记录失败模式

statistical (C4):
  成功/失败 → 只更新统计计数
  定期维护 → 删除低效 skills
```

---

## Layer 2: 数据消融 (9 组)

使用 Layer 1 最佳方法，验证数据策略：

### 数据来源说明

```
train.jsonl (8000条) ──→ 抽取 ──→ Cold Start Pool + Evolution Pool
                              │
                              │ 抽取量 D
                              │ 分配比例 E
                              ↓
test.jsonl (1000条) ──→ 固定抽取 200条 ──→ Test Pool
```

### 实验组合表

| 实验编号 | D: train_sample | E: cs_ratio | Cold Start | Evolution | Test | 说明 |
|----------|-----------------|-------------|------------|-----------|------|------|
| L2-1 | 300 | 20% | 60 | 240 | 200 (固定) | 小数据+标准比例 |
| L2-2 | 300 | 30% | 90 | 210 | 200 | 小数据+偏冷启动 |
| L2-3 | 300 | 10% | 30 | 270 | 200 | 小数据+偏进化 |
| L2-4 | 500 | 20% | 100 | 400 | 200 | 中数据+标准比例 |
| L2-5 | 500 | 30% | 150 | 350 | 200 | 中数据+偏冷启动 |
| L2-6 | 500 | 10% | 50 | 450 | 200 | 中数据+偏进化 |
| L2-7 | 800 | 20% | 160 | 640 | 200 | 大数据+标准比例 |
| L2-8 | 800 | 30% | 240 | 560 | 200 | 大数据+偏冷启动 |
| L2-9 | 800 | 10% | 80 | 720 | 200 | 大数据+偏进化 |

### 消融点详细说明

#### 消融点 D：Train 抽取量

| 选项 | 抽取量 | 说明 |
|------|--------|------|
| `small` | 300 条 | 快速验证，Skills 初始覆盖可能不足 |
| `medium` | 500 条 | 标准实验（Layer 1 使用） |
| `large` | 800 条 | 充分探索，Skills 覆盖更多场景 |

**假设：** 数据量增加 → ASR 提升，但边际效益递减，需验证最佳数据量。

#### 消融点 E：Cold Start / Evolution 分配比例

| 选项 | CS比例 | Evo比例 | 说明 |
|------|--------|---------|------|
| `standard` | 20% | 80% | 平衡分配（Layer 1 使用） |
| `cold_start_focus` | 30% | 70% | 重视初始 Skills 质量 |
| `evolution_focus` | 10% | 90% | 重视进化阶段探索 |

**假设：**
- `cold_start_focus`：初始 Skills 更丰富，但进化探索少
- `evolution_focus`：初始 Skills 少，但进化空间更大

---

## 实验流程

```
Day 1: 数据准备 + 服务启动
├── python scripts/extract_experiment_data.py
│   ├── 从 train.jsonl 抽取到 cold_start.json / evolution.json
│   ├── 从 test.jsonl 抽取到 test_prompts.json
│   └── 输出 data_meta.json 记录配置
├── bash start_guard.sh (GPU 0, 端口 8002)
├── bash start_policy.sh (GPU 1-2, 端口 8001)
└── python scripts/verify_services.py

Day 2-3: Layer 1 Grid Search (16 组)
├── python scripts/grid_search.py --layer 1
├── 固定数据配置: train=500, CS=100, Evo=400, Test=200
├── 监控: experiments/L1_grid_search_*/summary_latest.json
└── 分析: 找到最佳 A, B, C 组合

Day 4: Layer 2 数据消融 (9 组)
├── python scripts/grid_search.py --layer 2 --best_method "A? B? C?"
├── 数据消融: train_sample × cs_ratio
├── Test 固定 200 条
└── 分析: 最终 ASR + 最佳数据配置

Day 5: 结果分析与报告
├── 整理 summary_final.json
├── 绘制 Skills 进化曲线
├── 绘制数据量 vs ASR 曲线
└── 撰写实验结论
```

---

## 命令行使用

### 数据准备

```bash
# 抽取实验数据
python scripts/extract_experiment_data.py \
    --train_path data/dataset/processed/10k/train.jsonl \
    --test_path data/dataset/processed/10k/test.jsonl \
    --train_sample 500 \
    --cs_ratio 0.20 \
    --test_sample 200
```

### Layer 1 实验

```bash
# 完整 Grid Search（16 组）
python scripts/grid_search.py --layer 1

# 快速调试（小数据量）
python scripts/grid_search.py --layer 1 --debug

# 断点续跑
python scripts/grid_search.py --layer 1 --resume_from 8
```

### Layer 2 实验

```bash
# 使用 Layer 1 最佳方法
python scripts/grid_search.py --layer 2 --best_method "every_iteration trajectory both"
```

### 单组合调试

```bash
python scripts/pipeline.py \
    --skill_call_mode every_iteration \
    --skill_extraction_mode trajectory \
    --update_strategy both \
    --train_sample 500 \
    --cs_ratio 0.20 \
    --test_sample 200 \
    --skip_launch
```

---

## 预期分析维度

### Layer 1 分析

```
┌─────────────────────────────────────────────────────────────────┐
│  ASR vs skill_call_mode                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ single_call:     avg ASR = ?                            │   │
│  │ every_iteration: avg ASR = ?                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ASR vs extraction_mode                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ final_prompt:   avg ASR = ?                              │   │
│  │ trajectory:     avg ASR = ?                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ASR vs update_strategy                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ success_only:   avg ASR = ?                              │   │
│  │ failure_only:   avg ASR = ?                              │   │
│  │ both:           avg ASR = ?                              │   │
│  │ statistical:    avg ASR = ?                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  最佳组合：max ASR with min avg_iterations                       │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 2 分析

```
┌─────────────────────────────────────────────────────────────────┐
│  ASR vs train_sample_size                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ small (300):  avg ASR = ?                               │   │
│  │ medium (500): avg ASR = ?                               │   │
│  │ large (800):  avg ASR = ?                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│  结论：边际效益递减拐点在哪？                                    │
│                                                                  │
│  ASR vs cs_ratio                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 10% (evolution_focus): avg ASR = ?                       │   │
│  │ 20% (standard):         avg ASR = ?                      │   │
│  │ 30% (cold_start_focus): avg ASR = ?                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│  结论：哪种分配策略更优？                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三阶段流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        Phase 1: 冷启动                           │
│  Cold Start Pool → 攻击尝试 → 成功 → 总结 skills → 合并去重    │
│  (从 train 抽取)                                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Phase 2: 进化                             │
│  Evolution Pool → 选择 skills → 加载 → 攻击 → 成功/失败 → 反思 │
│  (从 train 抽取)                                                │
│                                                                  │
│  可选：每轮后用 eval.jsonl 评估 Skills 效果                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Phase 3: 测试                             │
│  Test Pool → 固定 skills → 上限轮次攻击 → 统计 ASR              │
│  (从独立 test.jsonl 抽取)                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 服务配置

| 服务 | 端口 | 模型 | GPU |
|------|------|------|-----|
| Guard | 8002 | Qwen3Guard-Gen-4B | GPU 0 |
| Target | 8001 | Qwen3-4B | GPU 1-2 |

```bash
cd self_evolve_skills_jailbreak/scripts
bash start_guard.sh    # GPU 0
bash start_policy.sh   # GPU 1-2
```

---

## 核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| Skill | `core/skill.py` | 数据结构定义 |
| SkillLibrary | `core/skill_library.py` | 存储、检索、聚类、合并、维护 |
| Attacker | `core/attacker.py` | Skill 引导攻击（两种调用模式） |
| Reflector | `core/reflector.py` | 反思 prompt，生成 skill 更新建议 |
| DataConfig | `core/data_config.py` | 数据配置和划分 |

---

## 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **ASR** | Attack Success Rate | 越高越好 |
| **Avg Iterations** | 平均迭代次数 | 越低越好 |
| **Skill Count** | Skills 数量 | 适中 |
| **Skill Avg Quality** | Skills 平均质量分数 | 越高越好 |

---

## 目录结构

```
self_evolve_skills_jailbreak/
├── README.md                    # 实验计划文档
├── core/                        # 核心模块
│   ├── skill.py
│   ├── skill_library.py
│   ├── attacker.py
│   ├── reflector.py
│   └── data_config.py
├── scripts/
│   ├── pipeline.py              # 主脚本
│   ├── grid_search.py           # Grid Search
│   ├── extract_experiment_data.py
│   ├── verify_services.py
│   ├── start_guard.sh
│   └── start_policy.sh
├── data/
│   ├── cold_start_prompts.json  # 从 train 抽取
│   ├── evolution_prompts.json   # 从 train 抽取
│   ├── test_prompts.json        # 从独立 test 抽取
│   └── data_meta.json           # 元信息
├── skills/
│   └── skills_library.json
└── experiments/                  # 实验结果输出
```

---

## 开发进度

- [x] Skill 数据结构与 SkillLibrary
- [x] Attacker（两种调用模式）
- [x] Reflector（成功/失败反思）
- [x] Pipeline 主脚本
- [x] Grid Search 自动化
- [x] 完整实验计划设计（分层架构）
- [x] 数据策略明确（train/eval/test 分工）
- [ ] Layer 1 实验运行
- [ ] Layer 2 实验运行
- [ ] 结果分析与报告

---

## 参考文献

1. PAIR: Chao et al., "Jailbreaking Black Box Large Language Models in Twenty Queries" (2023)
2. AutoDAN: Liu et al., "AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models" (2023)