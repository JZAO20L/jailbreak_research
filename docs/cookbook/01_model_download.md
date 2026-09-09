# 01 · 模型下载

> 默认位置：`jailbreak_research/model/`（当前机器在 `/home/tiger/models/Qwen/`，通过环境变量覆盖）
> 只需两个模型：**Qwen3-4B** + **Qwen3Guard-Gen-4B**。
> `Qwen3-4B-SafeRL` 已弃用（09-09 决策：target 换 plain 4B 提升奖励稠密度）。

## 1. 需要的模型

| 模型 | HF 仓库 | 大小 | 用途 |
|---|---|---|---|
| Qwen3-4B | `Qwen/Qwen3-4B` | ~8GB | Policy（被训练/rollout）+ Target（被攻击，plain 版） |
| Qwen3Guard-Gen-4B | `Qwen/Qwen3Guard-Gen-4B` | ~8GB | Guard（安全判定，产出 ASR 标签） |

## 2. 下载命令

```bash
cd /home/tiger/jailbreak_research
mkdir -p model
source .venv/bin/activate      # huggingface-cli 在 venv 内

# Qwen3-4B（policy + target）
huggingface-cli download Qwen/Qwen3-4B \
    --local-dir model/Qwen3-4B

# Qwen3Guard-Gen-4B（guard/judge）
huggingface-cli download Qwen/Qwen3Guard-Gen-4B \
    --local-dir model/Qwen3Guard-Gen-4B
```

### 国内镜像（HF 直连慢时）

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-4B --local-dir model/Qwen3-4B
huggingface-cli download Qwen/Qwen3Guard-Gen-4B --local-dir model/Qwen3Guard-Gen-4B
```

### ModelScope 备选

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-4B --local_dir model/Qwen3-4B
modelscope download --model Qwen/Qwen3Guard-Gen-4B --local_dir model/Qwen3Guard-Gen-4B
```

## 3. 校验下载完整

```bash
# 关键文件齐全（config.json / *.safetensors / tokenizer）
ls model/Qwen3-4B/ && ls model/Qwen3Guard-Gen-4B/

# 快速冒烟：加载 + 检查架构（应输出 qwen3）
python3 -c "
import json
c = json.load(open('model/Qwen3-4B/config.json'))
print('Qwen3-4B:', c['model_type'], c['architectures'])
g = json.load(open('model/Qwen3Guard-Gen-4B/config.json'))
print('Guard:', g['model_type'], g['architectures'])
"
```

## 4. 配置到实验脚本

`agentic_jailbreak/scripts/common.sh` 默认值指向 `/home/tiger/models/Qwen/`。
若模型放到了 `model/`，运行时导出覆盖即可（无需改脚本）：

```bash
export BASE_MODEL=$PWD/model/Qwen3-4B
export GUARD_MODEL=$PWD/model/Qwen3Guard-Gen-4B
export TARGET_MODEL=$PWD/model/Qwen3-4B    # target 用 plain 4B
```

## 5. 为什么 target 用 plain Qwen3-4B（09-09 决策记录）

- **动机**：SafeRL 目标过于对齐、破解率低 → ASR 奖励稀疏，GRPO 学习信号弱（reward 饥饿）。
- **换 plain 4B**：破解率更高 → 奖励稠密 → RL 信号充分。
- **代价**：评估数字与 SafeRL-target 口径不可直接比（baselines 需同 target 重跑）。
- 若后续需要"对齐目标"评估（transfer 表），可另起第二个 target 服务作第二评估维度。
