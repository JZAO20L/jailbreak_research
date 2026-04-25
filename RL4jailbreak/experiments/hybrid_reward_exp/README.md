# 混合奖励 GRPO 实验 - 实验2 & 实验3

## 实验概述

本实验目录包含两个相关实验，都使用混合奖励GRPO训练：

- **实验2**: Judge Prompt策略实验 - 研究不同judge prompt和评分方式对训练效果的影响
- **实验3**: Reward Weight实验 - 研究ASR reward和Judge reward不同权重比例的影响

---

## 实验2: Judge Prompt策略实验

### 实验目标

研究不同judge prompt策略和评分方式对混合奖励GRPO训练的影响。

### 研究维度

#### 1. Judge Prompt类型
- **通用型**: 对所有攻击策略适用
  - `idea_preservation`: 对原prompt的idea保留程度
  - `stealthiness`: 新prompt攻击的隐蔽程度
- **专用型**: 针对实验1选出的prompts (待实验1完成后补充)
  - 每个prompt代表的策略,衡量重写的prompt是否符合策略需求

#### 2. 评分方式
- **单条打分**: 对单条数据进行打分,输入为原始种子prompt和重写后的prompt,输出为0~1之间的两位浮点数
- **锦标赛打分**: 输入为原始种子prompt和两条重写后的prompt,基于二选一的方式进行锦标赛排序

### 实验配置

- ASR reward : Judge reward = 1:1 (固定)
- 每实验训练1000步
- 训练后在eval集上进行ASR评测

### 使用方法

```bash
# 运行所有实验组合
bash experiments/hybrid_reward_exp/judge_prompt_exp.sh

# 运行指定组合
bash experiments/hybrid_reward_exp/judge_prompt_exp.sh stealthiness single
bash experiments/hybrid_reward_exp/judge_prompt_exp.sh idea_preservation tournament

# 指定训练数据 (相对于实验脚本目录)
TRAIN_DATA=../../data/dataset/processed/10k/train.jsonl bash experiments/hybrid_reward_exp/judge_prompt_exp.sh

# 指定输出目录
OUTPUT_DIR=/path/to/output bash experiments/hybrid_reward_exp/judge_prompt_exp.sh
```

### 输出结果

```
judge_prompt_exp_output/
├── idea_preservation_single/
│   ├── final_lora/           # 训练保存的LoRA权重
│   ├── eval_results/         # 评估结果
│   └── config.json
├── idea_preservation_tournament/
├── stealthiness_single/
├── stealthiness_tournament/
└── ...
```

---

## 实验3: Reward Weight实验

### 实验目标

研究ASR reward和Judge reward不同权重比例对训练效果的影响。

### 实验设计

基于实验2确定最佳的1~3个攻击prompt+judge维度+评分方式组合，然后进行权重混合实验。

### 权重比例

测试5种比例 (Judge : ASR，总和为0.9，另加Format=0.1):
- Judge=0.1, ASR=0.8 (2:8)
- Judge=0.3, ASR=0.6 (4:6)
- Judge=0.5, ASR=0.4 (5:5，实验2已做)
- Judge=0.7, ASR=0.2 (6:4)
- Judge=0.9, ASR=0.0 (8:2)

### 实验配置

- 每实验训练1000步
- 训练后在eval集上进行ASR评测
- 使用实验2选出的最佳judge prompt和评分方式

### 使用方法

```bash
# 运行所有权重实验 (默认5种比例)
bash experiments/hybrid_reward_exp/reward_weight_exp.sh

# 运行指定权重
bash experiments/hybrid_reward_exp/reward_weight_exp.sh 0.7 0.2

# 指定judge prompt和评分方式
JUDGE_PROMPT=stealthiness SCORING_METHOD=single bash experiments/hybrid_reward_exp/reward_weight_exp.sh

# 指定训练数据 (相对于实验脚本目录)
TRAIN_DATA=../../data/dataset/processed/10k/train.jsonl bash experiments/hybrid_reward_exp/reward_weight_exp.sh

# 指定输出目录
OUTPUT_DIR=/path/to/output bash experiments/hybrid_reward_exp/reward_weight_exp.sh
```

### 输出结果

```
reward_weight_exp_output/
├── judge0.1_asr0.8/
│   ├── final_lora/           # 训练保存的LoRA权重
│   ├── eval_results/         # 评估结果
│   └── config.json
├── judge0.3_asr0.6/
├── judge0.5_asr0.4/
├── judge0.7_asr0.2/
├── judge0.9_asr0.0/
└── ...
```

---

## 核心文件说明

### 1. `hybrid_reward_grpo.py` - 混合奖励GRPO训练脚本

**功能**: 实现混合奖励GRPO训练，支持实验2和实验3

**使用方法**:

```bash
# 实验2: 单条打分
python experiments/hybrid_reward_exp/hybrid_reward_grpo.py \
    --experiment exp2 \
    --judge_prompt stealthiness \
    --scoring_method single

# 实验2: 锦标赛打分
python experiments/hybrid_reward_exp/hybrid_reward_grpo.py \
    --experiment exp2 \
    --judge_prompt stealthiness \
    --scoring_method tournament

# 实验3: 不同权重比例
python experiments/hybrid_reward_exp/hybrid_reward_grpo.py \
    --experiment exp3 \
    --asr_weight 0.3 \
    --judge_weight 0.7 \
    --judge_prompt stealthiness
```

**主要参数**:
- `--experiment`: 实验类型 (exp2/exp3)
- `--judge_prompt`: Judge prompt维度
- `--scoring_method`: 评分方式 (single/tournament)
- `--asr_weight`: ASR reward权重
- `--judge_weight`: Judge reward权重
- `--format_weight`: Format reward权重 (固定0.1)
- `--max_steps`: 最大训练步数
- `--learning_rate`: 学习率
- `--num_generations`: 每个prompt生成数量
- `--beta`: KL惩罚系数

---

### 2. `judge_prompts.py` - Judge Prompt定义

**功能**: 定义所有judge prompt模板

**使用方法**:

```python
from experiments.hybrid_reward_exp.judge_prompts import (
    JUDGE_PROMPTS,                # 所有judge prompt字典
    get_judge_template,           # 获取judge prompt模板
    GENERAL_JUDGE_DIMENSIONS,     # 通用型维度列表
    SPECIALIZED_JUDGE_DIMENSIONS, # 专用型维度列表
)

# 获取模板
template = get_judge_template("stealthiness")

# 查看所有维度
print(JUDGE_PROMPTS.keys())
```

---

### 3. `judge_prompt_exp.sh` - 实验2 Shell脚本

**功能**: 运行实验2的所有实验组合

**使用方法**:

```bash
# 运行所有组合
bash experiments/hybrid_reward_exp/judge_prompt_exp.sh

# 运行指定组合
bash experiments/hybrid_reward_exp/judge_prompt_exp.sh stealthiness single
```

---

### 4. `reward_weight_exp.sh` - 实验3 Shell脚本

**功能**: 运行实验3的所有权重实验

**使用方法**:

```bash
# 运行所有权重实验 (默认5种比例)
bash experiments/hybrid_reward_exp/reward_weight_exp.sh

# 运行指定权重
bash experiments/hybrid_reward_exp/reward_weight_exp.sh 0.7 0.2

# 指定judge prompt和评分方式 (环境变量)
JUDGE_PROMPT=stealthiness SCORING_METHOD=single bash experiments/hybrid_reward_exp/reward_weight_exp.sh
```

**支持的参数**:
- 位置参数1: judge_weight (如 0.7)
- 位置参数2: asr_weight (如 0.2)
- 环境变量 `JUDGE_PROMPT`: judge prompt维度
- 环境变量 `SCORING_METHOD`: 评分方式 (single/tournament)
- 环境变量 `TRAIN_DATA`: 训练数据路径
- 环境变量 `OUTPUT_DIR`: 输出目录

---

## 依赖

### 需要先启动的服务

运行实验前，需要启动以下服务：

```bash
# Target模型 (Judge复用)
bash scripts/start_target.sh

# Guard模型
bash scripts/start_guard.sh
```

### Python依赖

- torch
- trl (GRPOTrainer)
- peft (LoRA)
- transformers
- datasets

---

## 注意事项

### 实验依赖

实验2和实验3依赖实验1的结果：

- **专用型judge prompts**: 需要根据实验1选出的top k1个jailbreak prompt策略来定义
- 当前使用通用型维度进行实验
- 实验1完成后，需要在 `judge_prompts.py` 中补充专用型维度

### GPU状态检查

两个实验脚本都包含自动GPU状态检查：

- 自动检测GPU1显存使用情况
- 检查Target和Guard服务是否已在运行
- 如果服务未运行，会尝试自动启动
- 如果启动失败，会提示手动启动

### 资源需求

- **GPU**: 2×A800-80G
  - GPU0: Policy训练 (GRPO + LoRA)
  - GPU1: Target + Guard 服务
- **显存**:
  - Policy: ~20-25GB
  - Target: ~15-20GB
  - Guard: ~15-20GB

### 训练时间

- 每实验1000步约需 2-4小时
- 实验2: k1×k2×2 个实验组合
- 实验3: 5个权重比例

### 实验顺序建议

1. 先完成实验1 (jailbreak prompt策略测试)
2. 基于实验1结果，运行实验2 (judge prompt策略实验)
3. 基于实验2最佳组合，运行实验3 (reward weight实验)

---

## TODO (实验1完成后)

1. 根据实验1选出的top k1个jailbreak prompt策略，在 `judge_prompts.py` 中补充专用型维度
2. 更新 `judge_prompt_exp.sh` 中的策略列表
3. 考虑增加更多评分方式 (如pairwise比较)
