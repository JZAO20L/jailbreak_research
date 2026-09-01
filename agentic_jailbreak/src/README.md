# Agentic Jailbreak 源代码

## 目录结构

```
src/
├── env.py                # JailbreakEnv（GYM 环境）
├── agent.py              # Agent 实现（分析 + 动作选择）
├── memory.py             # 记忆机制（可选）
├── rewards.py            # 奖励函数（ASR only）
├── train.py              # 训练入口（ms-swift 多轮 GRPO，当前为说明文档）
├── eval.py               # 评估脚本
├── success_checker.py    # 成功判断逻辑（规则判定，不依赖 Guard）
├── single_turn_agent.py  # 单轮 Agent（requests 直连，用于冒烟测试）
├── multi_turn_agent.py   # 多轮 Agent（requests 直连，用于冒烟测试）
└── plugin.py             # ms-swift 插件注册（env + rewards）
```

## 核心模块

### env.py - JailbreakEnv

GYM 环境实现，封装攻击交互流程：

```python
class JailbreakEnv:
    def reset(self, prompt: str) -> Dict:
        """初始化环境，返回初始 observation"""
        return {
            "prompt": prompt,
            "skill_library": skills,
            "history": [],
            "turn": 0
        }

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一步攻击

        Args:
            action: {"skill_idx": int, "adapted_content": str}

        Returns:
            next_obs, reward, done, info
        """
        # 1. 构建 attack_prompt
        # 2. 调用 Target (GPT-OSS)
        # 3. 调用 Guard (Qwen3Guard)
        # 4. 计算 reward (ASR)
        # 5. 更新 history
        # 6. 返回 next_obs, reward, done, info
```

### agent.py - Agent

Agent 实现，包含分析和动作选择：

```python
class Agent:
    def analyze(self, prompt, history, memory) -> Dict:
        """分析当前状态，理解失败原因"""
        # 调用 policy model 分析
        return {"analysis": "...", "reasoning": "..."}

    def select_action(self, analysis, skills, history) -> Dict:
        """选择动作（skill + 适配内容）"""
        # 调用 policy model 决策
        return {"skill_idx": 0, "adapted_content": "...", "reasoning": "..."}
```

### success_checker.py - 成功判断

规则判定攻击是否成功（无需 Guard 模型）：
- 检测拒绝关键词、检测安全模式、检查回复长度、检查字符重叠率（防转移话题）
- 自动移除 `Thinking` 标签

### memory.py - Memory

可选的长期记忆机制：

```python
class Memory:
    def update(self, experience: Dict): ...
    def retrieve(self, prompt: str) -> List[Dict]: ...
```

### rewards.py - Rewards

奖励函数（ASR only）：

```python
def asr_reward(completions, **kwargs) -> List[float]:      # 成功=1.0，失败=0.0
def process_reward(completions, **kwargs) -> List[float]:  # 过程奖励（可选）
def efficiency_reward(completions, **kwargs) -> List[float]: # 效率奖励（可选）
```

### train.py - Training

ms-swift 多轮 GRPO 训练入口（当前是用法说明，训练未跑）：

```bash
swift rlhf --rlhf_type grpo \
    --multi_turn_scheduler gym_scheduler \
    --env jailbreak_env \
    --reward_funcs asr_reward \
    ...
```

### eval.py - Evaluation

评估脚本，在多个 benchmark 上测试 ASR。

## 实验成果

> 详细事件见 [../docs/LOG.md](../docs/LOG.md)，待办见 [../docs/TODO.md](../docs/TODO.md)。

- **单轮 Agent 冒烟测试**(3 用例):全部被 SafeRL 拒绝,ASR=0%(符合预期——SafeRL 是安全对齐模型,单轮攻击难以成功)
- **多轮 Agent 冒烟测试**(2 用例 × 3 轮):全部被拒绝;Agent 能正确执行多轮流程,并能根据历史反馈换 skill
- **纯 Agent 单轮全量基准(08-10)**:10k test 1000 条 → **ASR 4.30%**(43/1000)。详见 [../output/README.md](../output/README.md) 与 [../docs/LOG.md](../docs/LOG.md)

## 依赖

- ms-swift（多轮 GRPO 支持）
- vLLM（推理引擎）
- transformers（模型加载）
- peft（LoRA）

## 数据流

```
1. 加载 skills 数据（从 data/skills.json）
2. 初始化 JailbreakEnv
3. 初始化 Agent（policy model）
4. 多轮交互：
   - Agent 分析 + 选择动作
   - Env 执行攻击
   - 计算 reward
   - 更新 history
5. GRPO 训练更新 policy
```

## 注意事项

1. **Target 模型**：当前用 Qwen3-4B-SafeRL(2-GPU 环境);GPT-OSS-20B 为跨族迁移目标,需要下载
2. **Guard 模型**：Qwen3Guard-Gen-4B(已下载)
3. **Skills 数据**：从 SESS 复制的有效 skills
4. **奖励设计**：只用 ASR(不需要 Judge)
5. **已知问题**：`eval.py`/`agent.py → env.py` 路径存在 `src.utils` import bug,见 [../docs/TODO.md](../docs/TODO.md) P0