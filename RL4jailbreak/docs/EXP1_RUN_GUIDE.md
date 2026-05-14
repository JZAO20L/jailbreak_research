# 实验1运行指南

> 本文档提供在Merlin开发机上运行实验1（Jailbreak Prompt实验）的完整步骤。

## 目录

- [前置条件](#前置条件)
- [Worker启动](#worker启动)
- [实验一命令](#实验一命令)
- [常见问题](#常见问题)

---

## 前置条件

### 1. 资源配置

根据 `docs/merlin_tutorial/config.md`：

```bash
# 查看可用GPU资源
mlx worker quota --resourcetype arnold --usergroup commercial_ai_aigc
```

**可用队列（推荐）：**

| 集群 | 队列名 | GPU类型 | 数量 |
|------|--------|---------|------|
| cloudnative-lq | compute-688-lq-cloudnative-ai-applied.vision-guarantee | A100-SXM-80GB | 14 |
| cloudnative-hl | compute-688-hl-cloudnative-ai-applied.vision-guarantee | A100-PCIE-80GB | 16 |
| cloudnative-yg | compute-910-yg-cloudnative-ai-applied.vision-guarantee | A100-SXM-80GB | 28 |

**实验配置：**
- eval时使用3卡：GPU0:Policy, GPU1:Target, GPU2:Guard
- train时使用4卡：GPU0&1:Policy并发训练, GPU2:Target, GPU3:Guard
- 上下文长度：Policy 4k, Target/Guard 8k

### 2. 模型路径

```bash
# 模型位置
/mnt/bn/chenxiong/mlx/users/jiazixiao/models/Qwen3-4B          # Policy & Target
/mnt/bn/chenxiong/mlx/users/jiazixiao/models/Qwen3Guard-Gen-4B # Guard
```

---

## Worker启动

### Step 1: 申请GPU资源

**推荐使用 cloudnative-lq 队列（14张A100-SXM-80GB）：**

```bash
mlx worker launch \
  --resourcetype arnold \
  --usergroup commercial_ai_aigc \
  --cluster cloudnative-yg \
  --queuename compute-910-yg-cloudnative-ai-applied.vision-guarantee \
  --gpu 4 \
  --type A100-SXM-80GB \
  -- bash
```

等待输出显示：
```
✔ worker-0 (WORKER_ID)
  Ready
  IP: ...
  URL: ...

✔ All workers are ready
```

### Step 2: 查看Worker状态

```bash
mlx worker list
```

输出示例：
```
id        cpu    mem    gpu    gpuType        podIP       port   createdAt
3836XXX   62     988    4      A100-SXM-80GB  ...         10763  2026-05-14T...
```

### Step 3: 登录Worker节点

**方式一：使用Worker ID登录**

```bash
mlx worker login WORKER_ID
```

**方式二：使用WebShell**

点击 `mlx worker list` 输出的 webshell 链接，在浏览器中打开终端。

---

## 实验一命令

### 在Worker节点内执行

**Step 1: 进入项目目录**

```bash
cd /mnt/bn/chenxiong/mlx/users/jiazixiao/jailbreak_research/RL4jailbreak
```

**Step 2: 运行实验1**

**完整运行（所有24个策略 + qwen3-max对比）：**

```bash
bash experiments/jailbreak_prompt_exp/exp.sh --qwen3_max
```

**只输出Top-3策略（推荐）：**

```bash
bash experiments/jailbreak_prompt_exp/exp.sh --topk 3
```

**运行指定策略：**

```bash
bash experiments/jailbreak_prompt_exp/exp.sh hypothetical_scenario creative_writing role_playing
```

**清空checkpoint重新开始：**

```bash
bash experiments/jailbreak_prompt_exp/exp.sh --reset --topk 3
```

### 实验参数说明

| 参数 | 说明 |
|------|------|
| `--topk N` | 只输出表现最好的N个策略 |
| `--qwen3_max` | 包含qwen3-max对比实验 |
| `--reset` | 清空checkpoint重头开始 |
| `[策略名...]` | 运行指定策略 |

### 输出位置

```bash
# 实验结果保存在
experiments/jailbreak_prompt_exp/output/
```

---

## 完整命令汇总（一键执行）

### 方案A：Master节点启动，Worker节点执行

在Master节点的Terminal中：

```bash
# 1. 启动Worker
mlx worker launch \
  --resourcetype arnold \
  --usergroup commercial_ai_aigc \
  --cluster cloudnative-lq \
  --queuename compute-688-lq-cloudnative-ai-applied.vision-guarantee \
  --gpu 4 \
  --type A100-SXM-80GB \
  -- bash

# 2. 等待Worker就绪后，在新的Terminal中登录
mlx worker login WORKER_ID

# 3. 在Worker内执行实验
cd /mnt/bn/chenxiong/mlx/users/jiazixiao/jailbreak_research/RL4jailbreak
bash experiments/jailbreak_prompt_exp/exp.sh --topk 3
```

### 方案B：Worker启动时直接执行命令

```bash
mlx worker launch \
  --resourcetype arnold \
  --usergroup commercial_ai_aigc \
  --cluster cloudnative-lq \
  --queuename compute-688-lq-cloudnative-ai-applied.vision-guarantee \
  --gpu 4 \
  --type A100-SXM-80GB \
  -- cd /mnt/bn/chenxiong/mlx/users/jiazixiao/jailbreak_research/RL4jailbreak && bash experiments/jailbreak_prompt_exp/exp.sh --topk 3
```

---

## 常见问题

### Q1: Worker启动失败 "volumeConfig not accessible"

**原因：** 开发机所在的IDC与申请的cluster不在同一区域。

**解决：** 使用与开发机IDC匹配的cluster（如 `cloudnative-lq`）。

### Q2: Worker login失败 "worker has not been ready yet"

**原因：** Worker刚启动，SSH服务尚未就绪。

**解决：** 等待30-60秒后重试，或使用WebShell链接。

### Q3: 实验脚本找不到文件

**原因：** Worker节点未挂载相同的存储路径。

**解决：** 确认路径 `/mnt/bn/chenxiong/mlx/users/jiazixiao/jailbreak_research/RL4jailbreak` 存在：
```bash
ls /mnt/bn/chenxiong/mlx/users/jiazixiao/jailbreak_research/RL4jailbreak/
```

### Q4: GPU端口冲突

**原因：** vLLM服务端口被占用。

**解决：** 清理旧进程：
```bash
pkill -f "vllm.*8001" 2>/dev/null
pkill -f "vllm.*8002" 2>/dev/null
pkill -f "vllm.*8003" 2>/dev/null
```

---

## 后续实验

实验1完成后，可继续运行：

### 实验2：Judge Prompt实验

```bash
bash experiments/hybrid_reward_exp/train_exp2.sh --max_steps 1000
```

### 实验3：Adaptive Hybrid Reward实验

```bash
bash experiments/adaptive_hybrid_reward_exp/exp.sh --max_steps 1000
```

---

## 参考资料

- [Merlin开发机使用指南](/mnt/bn/chenxiong/mlx/users/jiazixiao/docs/merlin_tutorial/config.md)
- [实验配置说明](/mnt/bn/chenxiong/mlx/users/jiazixiao/jailbreak_research/RL4jailbreak/experiments/config.py)
- [项目TODO.md](/mnt/bn/chenxiong/mlx/users/jiazixiao/jailbreak_research/TODO.md)