# Agentic Jailbreak 实验脚本

## 目录结构

```
scripts/
├── common.sh              # 共享函数（路径/端口/GPU 布局，支持 NUM_GPUS=4/8）
├── prepare_data.sh        # 准备数据（从 SESS 复制 skills）
├── start_servers.sh       # 启动 vLLM servers
├── exp01_single_turn.sh   # 实验 E1:单轮 Agent baseline
├── exp02_multi_turn_3.sh  # 实验 E2:多轮 Agent (3 轮)
├── exp03_multi_turn_5.sh  # 实验 E3:多轮 Agent (5 轮)
├── eval_all.sh            # 全量评估
├── test_agent_loop.py     # Agent Loop 冒烟测试
├── test_single_turn.py    # 单轮 Agent 冒烟测试
└── test_multi_turn.py     # 多轮 Agent 冒烟测试
```

## 执行流程

```bash
# 1. 准备数据
bash scripts/prepare_data.sh

# 2. 启动 servers
bash scripts/start_servers.sh

# 3. 冒烟测试（验证服务 + Agent 链路）
python scripts/test_single_turn.py
python scripts/test_multi_turn.py

# 4. 全量评估（需先解决 TODO P0:import bug + 端口约定）
bash scripts/eval_all.sh

# 5. RL 训练（ms-swift GRPO）
bash scripts/exp01_single_turn.sh
bash scripts/exp02_multi_turn_3.sh
bash scripts/exp03_multi_turn_5.sh
```

## GPU 布局

### 4/8 卡（common.sh 默认支持）

```
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│  GPU 0  │  GPU 1  │  GPU 2  │  GPU 3  │  GPU 4  │  GPU 5  │  GPU 6  │  GPU 7  │
├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│  Guard  │ Target  │ vLLM    │ vLLM    │ Train   │ Train   │ Train   │ Train   │
│ (Guard) │(GPT-OSS)│ Rollout │ Rollout │  (TP=2) │  (TP=2) │  (TP=2) │  (TP=2) │
│  8001   │  8002   │  8003   │         │         │         │         │         │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

### 当前 2-GPU 分时复用（TODO P0 待落地）

```
训练:  GPU0 SafeRL(Target) 8000 + GPU1 Policy(Agent) 8001(success_checker 判 ASR,不需 Guard)
评估:  GPU0 SafeRL(Target) 8000 + GPU1 Guard 8002
切换:  训练/评估二选一,运行前需切换 GPU1 上的服务
```

## 关键配置

每个脚本开头可配置：

```bash
# GPU 配置
NUM_GPUS=8
GUARD_GPU=0
TARGET_GPU=1
ROLLOUT_GPUS="2,3"
TRAIN_GPUS="4,5,6,7"

# 模型路径
BASE_MODEL="/home/tiger/models/Qwen/Qwen3-4B"
GUARD_MODEL="/home/tiger/models/Qwen/Qwen3Guard-Gen-4B"
TARGET_MODEL="/home/tiger/models/Qwen/Qwen3-4B-SafeRL"

# 端口
GUARD_PORT=8001
TARGET_PORT=8002
ROLLOUT_PORT=8003

# 训练超参
MAX_TURNS=5
MAX_STEPS=500
LEARNING_RATE=1e-5
NUM_GENERATIONS=8
```

## 实验成果

> 详细事件见 [../docs/LOG.md](../docs/LOG.md)，待办见 [../docs/TODO.md](../docs/TODO.md)。

- 冒烟测试(08-06):单轮 3 用例、多轮 2×3 轮,链路跑通,ASR=0%(SafeRL 拒绝,符合预期)
- 全量实验(exp01-03、eval_all):未跑

## 注意事项

1. **GPT-OSS-20B 需要先下载**（跨族迁移目标，当前用 SafeRL 替代）
2. **common.sh 当前只支持 NUM_GPUS=4/8**，2 卡环境需适配（TODO P0）
3. **ms-swift 版本需支持多轮 GRPO**（当前 4.3.2 ✓）
4. **vLLM 版本需要 0.18.0+**（当前 0.18.0 ✓）
5. 端口约定存在不一致（common.sh 8001/8002 vs 2-GPU 方案 8000/8002），见 TODO P0