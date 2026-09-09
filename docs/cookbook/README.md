# Cookbook —— 在 GPU 环境从零跑通第三章实验

> 定位：本手册是"在 GPU 环境直接运行实验"的操作指南（模型下载、起服务、训练、评估全命令）。
> 本仓库位置（`/home/tiger/jailbreak_research`）同时承载论文撰写 / 绘图 / 代码开发。
> 最后更新：2026-09-09

---

## 1. 环境总览

| 项 | 值 |
|---|---|
| GPU | 4×V100-SXM2-32GB（SM 7.0，**无 bf16 计算、无 FA2、无 SDPA mem-efficient**） |
| 虚拟环境 | `.venv`（`/home/tiger/jailbreak_research/.venv`） |
| 训练框架 | **ms-swift 4.3.2**（`swift rlhf --rlhf_type grpo`，内部包 TRL 0.29.1 GRPOTrainer） |
| 推理服务 | vLLM 0.18.0（guard / target / rollout 三服务） |
| 关键依赖 | torch 2.10.0 · transformers 4.57.6 · trl 0.29.1 · xformers 0.0.35 · liger-kernel 0.8.2 |

### 4 卡布局（训练时）

| GPU | 角色 | 模型 | 端口 |
|---|---|---|---|
| GPU0 | Guard（安全判定 Judge） | Qwen3Guard-Gen-4B | 8001 |
| GPU1 | Target（被攻击模型） | **Qwen3-4B**（plain，09-09 定） | 8002 |
| GPU2 | Policy / swift rollout | Qwen3-4B | 8003（eval）/ 8004（GRPO） |
| GPU3 | 训练卡（SFT / GRPO） | — | — |

> ⚠️ V100 不支持 bf16 计算与 FA2：vLLM 服务一律显式 `--dtype float16`；
> GRPO 训练默认 `bf16 原生`（无 AMP，见 `02_run_experiments.md` §6 的 DTYPE 说明）。

## 2. 模型默认位置

**新约定：模型统一放 `jailbreak_research/model/` 目录**（当前机器在 `/home/tiger/models/Qwen/`）。

**只需两个模型**（target 已定 plain Qwen3-4B，`Qwen3-4B-SafeRL` 已弃用）：

| 模型 | HF 仓库 | 本地路径 | 用途 |
|---|---|---|---|
| Qwen3-4B | `Qwen/Qwen3-4B` | `model/Qwen3-4B` | Policy + Target |
| Qwen3Guard-Gen-4B | `Qwen/Qwen3Guard-Gen-4B` | `model/Qwen3Guard-Gen-4B` | Guard / Judge |

下载命令见 [`01_model_download.md`](01_model_download.md)。
若模型不在默认路径，通过环境变量覆盖（`common.sh` 支持）：

```bash
export BASE_MODEL=/path/to/model/Qwen3-4B
export GUARD_MODEL=/path/to/model/Qwen3Guard-Gen-4B
export TARGET_MODEL=/path/to/model/Qwen3-4B   # 09-09 起 target 用 plain 4B
```

## 3. 快速开始（第三章主链路）

```bash
cd /home/tiger/jailbreak_research
source .venv/bin/activate

# ① 下载模型（一次性，见 01）
bash docs/cookbook/download_models.sh          # 或手动 huggingface-cli

# ② 起三服务（每卡一模型）
bash agentic_jailbreak/scripts/start_servers_v100.sh
curl -s http://127.0.0.1:8001/health && curl -s http://127.0.0.1:8002/health && echo OK

# ③ GRPO 训练需要 swift rollout 服务（GPU2 8004，命令见 02 §2）
CUDA_VISIBLE_DEVICES=2 setsid swift rollout --model "$BASE_MODEL" \
    --vllm_tensor_parallel_size 1 --port 8004 --vllm_max_model_len 24576 \
    --vllm_gpu_memory_utilization 0.8 --torch_dtype float16 \
    > agentic_jailbreak/output/logs/rollout_exp04.log 2>&1 &

# ④ GRPO 训练（M1，5 轮恢复臂 / 10 轮协议见 02 §6）
MAX_TURNS=5 bash agentic_jailbreak/scripts/exp04_grpo_10turn.sh

# ⑤ 评估（主协议 skill_decide，评估用 8003 标准 policy，见 02 §6）
bash agentic_jailbreak/scripts/eval_conv.sh skill_decide 300 5
```

## 4. 目录导航

| 文档 | 内容 |
|---|---|
| [`01_model_download.md`](01_model_download.md) | 模型下载（HF / ModelScope / 国内镜像），校验命令 |
| [`02_run_experiments.md`](02_run_experiments.md) | 全链路实验命令：服务 / RFT 采集 / SFT / merge / GRPO / 评估 / 监控 / 排障 |
| [`03_three_chapters.md`](03_three_chapters.md) | 毕业论文三章内容介绍（AHR-GRPO / SESS / Agentic-RL） |
| 第三章详细记录 | `agentic_jailbreak/docs/{TODO,LOG,REPORT}.md` |

## 5. 关键协议（口径，勿随意改动）

- **主配置 = `skill_decide@10skills`**（PAIR 骨架 + 10-skill 精选库，LLM 每轮自选 skill 或自由改写）
  - 精选库：`agentic_jailbreak/exp/skill_asr_sweep/seed_skills_top10.json`
- **数据隔离**：A = RFT 训练 `train[1000:2000]` / B = GRPO 训练 `train[0:1000]` / C = 评估 `test`（A/B/C 两两不重叠）
- **实验矩阵 M0-M3**（10 轮协议）：M0 直接 agent / M1 只 GRPO / M2 RFT / M3 RFT+GRPO
- **训练超参**：max_turns=10（恢复臂 5）/ num_generations=16 / lr 1e-5 / β=0.05 / 300 步
- **三轴**：A 后训练配方（M0-M3）→ B 上下文维护（推理侧消融）→ C Reward 组合（ASR-only → 过程/效率/AHR λ）

## 6. 常见坑（必读）

1. **V100 无 bf16 计算**：vLLM 必须 `--dtype float16`；GRPO 训练默认 `bf16 原生`（09-09 发散修复，见 02 §6）
2. **GRPO 训练发散（09-09 案例）**：fp16 下 `exp(ratio)`/KL 项数值溢出，首个非零奖励批 grad 270→NaN。**用 `DTYPE=bf16` 原生训练**
3. **长序列 OOM**：10 轮序列 ~16k token，attention 显存 O(seq²)。已用 xformers SDPA patch（`src/v100_attn_patch.py`，trainer 自动加载）+ batch1/accum8/grad-checkpointing 解决
4. **TRL 0.29 默认 `loss_type=dapo`**：vanilla GRPO 对照必须显式 `--loss_type grpo`
