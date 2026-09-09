# 02 · 运行实验（全命令）

> 所有命令在 `/home/tiger/jailbreak_research` 下执行，脚本均在 `agentic_jailbreak/scripts/`。
> 约定：`source .venv/bin/activate` 后，`swift` / `vllm` 直接可用。

---

## 0. 环境准备（一次性）

```bash
cd /home/tiger/jailbreak_research
python3 -m venv .venv
source .venv/bin/activate
pip install -U torch==2.10.0 "ms-swift==4.3.2" "trl==0.29.1" "vllm==0.18.0"
pip install --no-deps xformers==0.0.35 liger-kernel==0.8.2
# xformers 0.0.35 与 torch 2.10 匹配；liger 用于 fp16 长序列回退臂（见 §6）
```

> ⚠️ V100 无 bf16 计算 / 无 FA2：vLLM 一律 `--dtype float16`。
> ⚠️ 模型路径默认 `/home/tiger/models/Qwen/`；新约定 `model/`（见 01.md §4 环境变量覆盖）。

---

## 1. 启动三服务

```bash
bash agentic_jailbreak/scripts/start_servers_v100.sh
```

| GPU | 服务 | 模型 | 端口 | 日志 |
|---|---|---|---|---|
| GPU0 | guard | Qwen3Guard-Gen-4B | 8001 | `output/logs/guard_v100.log` |
| GPU1 | target | **Qwen3-4B**（plain） | 8002 | `output/logs/target_v100.log` |
| GPU2 | policy | Qwen3-4B | 8003 | `output/logs/policy_v100.log` |
| GPU3 | 预留训练 | — | — | — |

**target 从 SafeRL 换 plain 4B**（09-09 起，新机器上起服务已直接是 plain）：

```bash
bash agentic_jailbreak/scripts/swap_target_plain4b.sh
```

**健康检查**：

```bash
curl -s http://127.0.0.1:8001/health; echo
curl -s http://127.0.0.1:8002/health; echo
curl -s http://127.0.0.1:8003/health; echo
```

---

## 2. GRPO 训练（M1 / M3，主链路）

脚本 `exp04_grpo_10turn.sh` 参数化：`MAX_TURNS`（轮次）、`DTYPE`（bf16 默认 / fp16 回退）、`MODEL_TAG`（base / rft）、`MODEL`（初始权重）。

```bash
# M1：只 GRPO，base 权重（默认）
bash agentic_jailbreak/scripts/exp04_grpo_10turn.sh

# M1 恢复臂（09-09：plain-4B target + 5 轮 + bf16）—— 当前主跑配置
MAX_TURNS=5 bash agentic_jailbreak/scripts/exp04_grpo_10turn.sh

# M3：RFT+GRPO，初始权重 = M2 merged
MODEL=output/rft_sft_conv10turn_e2_merged MODEL_TAG=rft \
    bash agentic_jailbreak/scripts/exp04_grpo_10turn.sh

# 10 轮长序列显存不足时的回退臂（fp16 + liger，仅 OOM 时用）
DTYPE=fp16 bash agentic_jailbreak/scripts/exp04_grpo_10turn.sh
```

固定超参（协议，勿改）：`num_generations=16 / lr 1e-5 / β=0.05 / 300 步 / max_completion_length 2048 / batch 1 / grad_accum 8 / --loss_type grpo`。

前置：guard/target 8001/8002 + swift rollout 8004 就绪。rollout 服务（GPU2，训练时替代 policy 8003）：

```bash
CUDA_VISIBLE_DEVICES=2 setsid swift rollout --model "$BASE_MODEL" \
    --vllm_tensor_parallel_size 1 --port 8004 --vllm_max_model_len 24576 \
    --vllm_gpu_memory_utilization 0.8 --torch_dtype float16 \
    > output/logs/rollout_exp04.log 2>&1 &
```

> **DTYPE 说明（09-09 发散修复，重要）**：V100 无 bf16 计算，最初为省显存切 `fp16 + liger`，
> 但 GRPO loss 的 `exp(ratio)` / `exp(ref−policy)` 项在 fp16（max≈65504）下数值溢出，
> 首个非零奖励批即 grad 270 → NaN 永久发散（v5 案例）。
> exp03 的 **bf16 原生训练（无 AMP/GradScaler）** 500 步稳定 → 默认 `DTYPE=bf16`。
> bf16 无 AMP 时不需 GradScaler，V100 上 torch 以 fp32 回退计算，数值最稳。

---

## 3. 监控训练

```bash
# 日志（tqdm 进度 + 指标）
tail -f output/logs/m1_train_*.log

# 逐步指标 jsonl（loss / grad_norm / kl / reward / zero_std）
ls output/multi_turn_*_agent_*/v*-*/logging.jsonl

# 快速抽指标（首步 grad < 1、前 10 步无 NaN/KL 爆炸 = 健康）
python3 - <<'EOF'
import json, glob
p = sorted(glob.glob('output/multi_turn_*_agent_*/v*-*/logging.jsonl'))[-1]
rows = [json.loads(l) for l in open(p)]
for i, r in enumerate(rows[:10]):
    print(i+1, f"loss={r.get('loss'):.4g}", f"grad={r.get('grad_norm'):.4g}",
          f"kl={r.get('kl'):.4g}", f"len={r.get('completions/mean_length',0):.0f}",
          f"rew={r.get('reward'):.4g}", f"zero={r.get('frac_reward_zero_std',1):.2f}")
EOF

# GPU 占用（训练卡 GPU3 应 < 32GB）
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# SwanLab 云端看板（进程启动时打印 URL）
# 停止训练
bash agentic_jailbreak/scripts/stop_m1.sh
```

**健康判定**：grad_norm 稳定在 ~0.01（bf16 下）、KL 不爆炸、`frac_reward_zero_std` 随时间下降。
若 step1 就出现 grad>10 或负 loss，立即停止（发散征兆）。

---

## 4. RFT 数据采集（M2 的 SFT 数据源）

前置：三服务已就绪（policy 必须是 **base 未训练** 权重）。

```bash
# split A（train[1000:2000]）→ base + conv_eval 采集 → 全轨迹样本 → SFT jsonl
NUM_GPUS=4 bash agentic_jailbreak/scripts/run_rft_collect_conv.sh
# 环境变量：VARIANT=skill_decide / TOP_K_SKILLS=10 / MAX_TURNS=10 / COLLECT_WORKERS=12
```

产出：`output/rft_collect_conv_10turn/results.jsonl`（采集轨迹）→ `output/rft_data_conv_10turn.jsonl`（SFT 数据）。

> 数据隔离：A=RFT 采集 `train[1000:2000]`；B=GRPO 训练 `train[0:1000]`（`grpo_data.jsonl`）；C=评估 `test`。

---

## 5. RFT SFT + merge（M2）

```bash
# SFT（GPU3，不需要 guard/target 服务）
bash agentic_jailbreak/scripts/rft_sft_conv.sh          # 默认 2 epochs
EPOCHS=1 bash agentic_jailbreak/scripts/rft_sft_conv.sh # 1 epoch 防分布坍缩

# merge LoRA -> 完整权重（M3 初始权重 / 评估用）
bash agentic_jailbreak/scripts/merge_rft_lora.sh        # 通用模板
# 或直接（conv 版）：
CUDA_VISIBLE_DEVICES=3 swift export \
    --model "$BASE_MODEL" \
    --adapters output/rft_sft_conv10turn_e2/v1-*/checkpoint-* \
    --merge_lora true --output_dir output/rft_sft_conv10turn_e2_merged \
    --max_length 4096
```

---

## 6. 评估（主协议 skill_decide）

```bash
# 语法: eval_conv.sh <variant> <样本数> <轮数> [beam_width] [top_k] [work_memory]
# 主协议评估（test C, 300 条, 单轨迹口径）
bash agentic_jailbreak/scripts/eval_conv.sh skill_decide 300 5

# M0 直接 agent / 消融变体
bash agentic_jailbreak/scripts/eval_conv.sh no_skill 300 5

# beam-2（附录口径）
bash agentic_jailbreak/scripts/eval_conv.sh no_skill_beam 200 5 2
```

前置：**标准 policy 服务（8003，评估用 vLLM）** 就绪，且 policy 已换成待评估权重：

```bash
# 把合并后的权重部署到 8003（GPU2）
CUDA_VISIBLE_DEVICES=2 setsid swift rollout --model output/xxx_merged \
    --port 8003 --vllm_max_model_len 8192 --vllm_gpu_memory_utilization 0.9 \
    --torch_dtype float16 > output/logs/policy_eval.log 2>&1 &
```

评估输出：`output/eval_results/conv_skill_decide_top10_5turn/results.jsonl` + `summary.json`（ASR / avg_turns）。

> 变体说明：`skill_decide`=LLM 自选 skill 或自由改写（**主配置**）；`no_skill`=纯 PAIR 式自由改写（消融）；
> `no_skill_beam`=每轮 best-of-2（附录）。

---

## 7. 排障速查

| 症状 | 原因 | 处理 |
|---|---|---|
| vLLM 启动报 bf16 错误 | V100 无 bf16 计算 | 加 `--dtype float16` |
| 训练 step1 显存爆（~55GB） | 长序列 attention O(seq²) | 已是 batch1/accum8/grad-checkpoint；确认 `v100_attn_patch.py` 被加载（日志有 `[v100_attn_patch]`） |
| 训练 step2-3 显存爆（~18GB） | 全词表 logits 物化 | `--use_liger_kernel true`（fp16 臂）或改 bf16 臂 |
| step1 grad>10 / 负 loss / KL 爆炸 | fp16 GRPO 数值溢出 | 换 `DTYPE=bf16`（09-09 案例） |
| 端口 43210 被占 | MASTER_PORT 冲突 | `MASTER_PORT=43xxx` 重试 |
| reward 长期全零（zero_std≈1） | 奖励饥饿 | C 轴 reward 组合 / DAPO 臂 / plain-4B target（已换） |
| RFT 采集客户端超时 | 长轨迹 >60s | `conv_eval.py`/`env.py` 超时已调 600s，重启进程生效 |

---

## 8. 全链路速查（M1 → M3）

```bash
# 1) 起服务
bash scripts/start_servers_v100.sh            # guard/target/policy
# 2) M1 GRPO（5 轮恢复臂 / 10 轮协议）
MAX_TURNS=5 bash scripts/exp04_grpo_10turn.sh
# 3) 监控至完成 → merge → 部署 8003 → 评估
bash scripts/stop_m1.sh
CUDA_VISIBLE_DEVICES=3 swift export --model "$BASE_MODEL" \
    --adapters output/multi_turn_5_agent_base/v*-*/checkpoint-300 \
    --merge_lora true --output_dir output/m1_5turn_merged --max_length 4096
CUDA_VISIBLE_DEVICES=2 swift rollout --model output/m1_5turn_merged \
    --port 8003 --vllm_max_model_len 8192 --vllm_gpu_memory_utilization 0.9 --torch_dtype float16 &
bash scripts/eval_conv.sh skill_decide 300 5
# 4) M2 RFT（采集 → SFT → merge）→ M3（MODEL=<m2_merged> MODEL_TAG=rft 同上跑 GRPO）
```
