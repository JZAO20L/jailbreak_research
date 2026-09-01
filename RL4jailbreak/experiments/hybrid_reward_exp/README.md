# 混合奖励 GRPO 实验 - 实验2

## 实验概述

本实验目录包含混合奖励GRPO训练实验：

- **实验2**: 多维度Judge Reward实验 - 研究不同judge维度权重配置对训练效果的影响

---

## 实验2: 多维度Judge Reward实验

### 实验目标

研究多维度judge评分的不同权重配置对混合奖励GRPO训练的影响。

### 实验设计

根据实验1筛选出的top3攻击prompt策略，结合5种judge维度权重配置，进行15个训练实验。

#### 1. 攻击Prompt策略 (来自实验1 top3)
- `creative_writing` - ASR: 30.2%
- `hypothetical_scenario` - ASR: 30.2%
- `role_playing` - ASR: 24.9%

#### 2. Judge维度权重配置 (5种)
- `intent_only` - [1.0, 0.0, 0.0, 0.0] (只关注意图保留)
- `stealth_only` - [0.0, 1.0, 0.0, 0.0] (只关注隐蔽性)
- `strategy_only` - [0.0, 0.0, 1.0, 0.0] (只关注策略执行)
- `potential_only` - [0.0, 0.0, 0.0, 1.0] (只关注攻击潜力)
- `uniform` - [0.25, 0.25, 0.25, 0.25] (均匀加权)

#### 3. 奖励混合比例
- ASR reward: 0.5
- Judge reward: 0.5 (固定1:1)

### 实验配置

- 总实验数: 3攻击prompt × 5权重配置 = 15个
- 每实验训练1000步
- 使用多维度judge prompt (JSON格式输出)
- Judge维度: intent_preservation, stealth, strategy_execution, attack_potential
- 训练后在eval集上进行ASR评测

### 使用方法

```bash
# 1. 激活虚拟环境
source /home/tiger/jailbreak_research/.venv/bin/activate

# 2. 进入实验目录
cd /home/tiger/jailbreak_research/RL4jailbreak/experiments/hybrid_reward_exp

# 3. 运行全部15个实验 (自动启动vLLM服务)
bash run_exp2.sh --start_services

# 如果服务已在运行，可省略 --start_services
bash run_exp2.sh

# 只训练指定攻击prompt
bash run_exp2.sh --attack_prompt hypothetical_scenario --start_services

# 只训练指定权重配置
bash run_exp2.sh --weight_config intent_only --start_services

# 调整训练步数
bash run_exp2.sh --max_steps 500 --start_services

# 清空checkpoint重头开始
bash run_exp2.sh --reset --start_services
```

### 输出结果

```
output/
├── creative_writing_intent_only/
│   ├── final_lora/           # 训练保存的LoRA权重
│   ├── logs/                 # 训练日志
│   └── config.json           # 实验配置
├── creative_writing_stealth_only/
├── creative_writing_strategy_only/
├── creative_writing_potential_only/
├── creative_writing_uniform/
├── hypothetical_scenario_intent_only/
├── ... (其他10个实验)
├── vllm_server.log           # vLLM Server日志
├── guard_server.log          # Guard Server日志
└── train_checkpoint.json     # 训练进度checkpoint
```

---

## 核心文件说明

### 1. `exp2_grpo.py` - GRPO训练核心脚本

**功能**: 实现混合奖励GRPO训练核心逻辑，使用vLLM server mode

**特点**:
- vLLM server mode: 推理和训练分离，效率更高
- Policy+Target+Judge共用一个vLLM server
- 连接外部Guard server进行ASR评估

### 2. `run_exp2.sh` - 实验执行脚本

**功能**: 启动服务、管理checkpoint、批量执行15个实验

**主要参数**:
- `--attack_prompt`: 攻击prompt策略 (creative_writing, hypothetical_scenario, role_playing)
- `--weight_config`: Judge维度权重配置 (intent_only, stealth_only, strategy_only, potential_only, uniform)
- `--max_steps`: 最大训练步数 (默认1000)
- `--learning_rate`: 学习率 (默认1e-5)
- `--num_generations`: 每个prompt生成数量 (默认8)
- `--beta`: KL惩罚系数 (默认0.05)
- `--per_device_train_batch_size`: 单卡batch size (默认2)
- `--gradient_accumulation_steps`: 梯度累积步数 (默认4)

---

### 3. `judge_prompts.py` - Judge Prompt定义

**功能**: 定义多维度judge prompt模板和权重配置

**Judge维度**:
- `intent_preservation`: 对原prompt攻击意图的保留程度
- `stealth`: 新prompt的隐蔽程度
- `strategy_execution`: 重写后prompt策略执行程度
- `attack_potential`: 攻击成功潜力

---

## GPU配置

- **GPU0&1**: vLLM server (Qwen3-4B) - policy generation + target response + judge evaluation
- **GPU2**: Policy训练 (单卡)
- **GPU3**: Guard server (Qwen3Guard-Gen-4B)

---

## 注意事项

### 服务依赖

训练脚本会自动检查并启动vLLM服务：
- vLLM Server: 端口8000 (Policy+Target+Judge共用)
- Guard Server: 端口8002

### 资源需求

- **GPU**: 4×GPU (推荐A800-80G)
- **训练时间**: 每实验约2-4小时 (1000步)

### 训练进度恢复

脚本支持checkpoint恢复：
- 已完成的实验会自动跳过
- 使用 `--reset` 强制重新训练所有实验