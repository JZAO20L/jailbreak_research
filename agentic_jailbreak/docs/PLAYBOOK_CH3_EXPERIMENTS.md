# 第三章 Agentic-RL 实验 Playbook（执行节点）

> **用途**：在一台全新 4×A800-80GB 服务器上按序跑完第三章剩余实验。
> **原则**：本节点只执行实验命令，不做开发与代码提交。所有代码以 git 为准，先 `git pull` 再跑。
> **状态权威来源**：`docs/TODO.md`（待办）/ `docs/LOG.md`（事件）/ `exp/README.md`（结果汇总）。
> **最后更新**：2026-09-13 · 覆盖到 09-11 的 M1/M2 全量评估结果。

---

## 0. 一次性环境准备（新服务器必做）

### 0.1 仓库与环境
```bash
git clone git@github.com:JZAO20L/jailbreak_research.git /home/tiger/jailbreak_research
cd /home/tiger/jailbreak_research
# 幂等环境搭建：venv + 依赖 + 符号链接（脚本内硬编码 PROJECT_ROOT=/home/tiger/jailbreak_research）
bash docs/cookbook/bootstrap_env.sh
# 版本锚点：vllm 0.18.0 · ms-swift 4.3.2 · trl 0.29.1 · torch 2.10.0 · transformers 4.57.6
# ⚠️ fastapi 必须 >=0.115,<0.116（0.116+ 的 _IncludedRouter 让 vllm 0.18 三服务全哑，见 requirements.txt 注释）
```

### 0.2 模型（hf-mirror 下载需关 Xet）
```bash
export HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com
# 落 model/ 下（common.sh 自动探测，缺则回落 /home/tiger/models/...）
# model/Qwen3-4B          （policy + target，7.6G）
# model/Qwen3Guard-Gen-4B （guard，8.3G）
```

### 0.3 数据资产重建（换机器 output/ 丢失，必须重建，隔离写成断言）
```bash
cd /home/tiger/jailbreak_research
python agentic_jailbreak/scripts/build_splits.py
# 产出: output/rft_seed_train1000.json (A段=train[1000:2000], RFT采集种子)
#       output/grpo_data.jsonl       (B段=train[0:1000],   GRPO训练)
#       output/test_prompts.json     (C段=test.jsonl,      评估)
# 断言: A∩B=A∩C=B∩C=∅，破了会直接报错（08-26 test 泄漏事故防线）
```

### 0.4 服务启动（评估/采集用标准 vllm 服务）
```bash
cd /home/tiger/jailbreak_research/agentic_jailbreak
NUM_GPUS=4 bash scripts/start_servers_a800.sh
# 布局: GPU0=guard 8001 | GPU1=target 8002(plain-4B) | GPU2=policy 8003 | GPU3=空闲
#   guard/target max-model-len 16384；policy 32768（10轮C1协议 24576 差1 token 越界崩溃，C2 事故结论）
#   target 是 plain Qwen3-4B（09-09 决策，SafeRL 破解率过低导致奖励饥饿）
# 打包态: guard+target 同住一卡时 SVC_UTIL=0.42
```

### 0.5 冒烟验证（重放 M0 闸门，验证测量链可复现）
```bash
RUN_TAG=m0 EVAL_WORKERS=12 bash scripts/eval_conv.sh skill_decide 300 10 2 10
# 预期: ASR≈54.0% (162/300), avg_turns≈6.0。若偏差 >±3pp 先查环境再往下跑。
```

---

## 1. 当前实验状态快照（09-14 时点，含 M3 完成 + M4 DAPO 启动）

| 臂 | 初始权重 | 训练 | 评估 | ASR | 状态 |
|----|----------|------|------|-----|------|
| M0 直接agent | base | 无 | test C 300 / 1000 | **54.0%** / **49.6%** | ✅ 完成 |
| M1 只GRPO | base | vanilla GRPO@10 | test C 300 / 1000 | **50.67%** / 待跑 | ✅ 300 完成 |
| M2 RFT | base→SFT | 无 RL | test C 1000 | **59.5%** | ✅ 完成 |
| M3 RFT+GRPO | **M2 merged** | vanilla GRPO@10 | test C 300 | **56.67%**（本机 A100 首发） | ✅ 完成（09-14） |
| M4 DAPO | **M2 merged** | dapo loss + dynamic_sample | test C 300 | **64.33%**（09-15 首发） | ✅ 完成（首个超越 RFT 的 RL 臂） |
| M5 长臂 | M3 ckpt 续训 | vanilla GRPO@900 | 待跑 | — | 🔄 训练中（09-15 启动） |

- **A 轴排序（09-15 更新）**：M4 64.3% > M2 61.0% > M3 56.7% > M0 54.0% > M1 50.7%——**vanilla GRPO 双负收益（−3.3/−4.3pp），DAPO 翻正（+3.3pp）；主因 = 稀疏奖励零组空转（算法侧）**，epoch 余量由 M5/M6 长臂测
- **09-14 新执行节点（4×A100-80GB）成果**：全链路从零重建（环境/模型/数据/服务/冒烟/M2 复现）见 LOG 09-13/14；M4 DAPO 命令见 §2.5
- **09-11 下午 v2 链（`scripts/chain_eval1000_then_m3.sh`）**：已补 base@1000 + M1@1000 同口径评估，
  并修复 `kill_port` 杀不动 EngineCore 孤儿占卡的问题（见 §6 新增坑）；链尾直接起 M3。
  若执行节点已跑过此链，M3 可能已启动/完成，先看 `output/logs/m3_20260911.log` 与 `output/multi_turn_10_agent_rft/`。
- **B 轴（已定稿）**：C1 保持定稿协议；C4@4000 作"省 2.5× token"效率叙事可选评估臂（不进主矩阵）
- **skill 分布**：`output/analysis/skill_usage.json`（base 选率 89.1% / M2 79.8% / M1 84.9%）

### 关键依赖资产（M3 必须有，否则先回到 §0.3/§2 重造）
| 资产 | 路径 | 说明 |
|------|------|------|
| **M2 merged 权重** | `output/rft_sft_conv10turn_e2_merged/` | M3 初始权重（GRPO 的 `--model`） |
| M2 adapter | `output/rft_sft_conv10turn_e2/v*/checkpoint-*` | 重新 merge M2 用（`MERGED_DIR` 覆盖） |
| GRPO 训练数据 | `output/grpo_data.jsonl` | split B，1000 条 |
| RFT SFT 数据 | `output/rft_data_conv_10turn.jsonl` | 重新跑 M2 SFT 用 |
| RFT 采集结果 | `output/rft_collect_conv_10turn/results.jsonl` | 重新跑 M2 SFT 用 |

> **入口 A（推荐，省 20h×3 重跑）**：从 A800 节点取回 `output/` 下上表资产 + §1 评估结果（见 §5 回收），直接进 §2。
> **入口 B（从零）**：重跑 RFT 采集（§2.0）→ M2 SFT（§2.1）→ M3。

---

## 2. A 轴收尾：M3 RFT+GRPO（~20h，接力链停在人工确认闸门处）

### 2.0 （入口 B 才需要）重造 M2：RFT 采集 + SFT + merge
```bash
# 采集（split A 1000 条，base policy，skill_decide@10skills@10轮）
#   前置: guard/target/policy(base) 三服务就绪，policy 必须是 base 模型
COLLECT_WORKERS=12 bash scripts/run_rft_collect_conv.sh
#   产出: output/rft_collect_conv_10turn/results.jsonl + output/rft_data_conv_10turn.jsonl
#   ⚠️ 12路并发下 client 超时已提至 1000s（09-03 丢 332 条事故根治）；逐条落盘，崩溃最多丢一条，可用 --resume 续跑

# SFT（单卡即可，不需要 guard/target；数据 token 长，先量 p50/max 再定 MAX_LENGTH）
bash scripts/rft_sft_conv.sh    # 默认 2 epochs, MAX_LENGTH=6144, GPU3
#   产出: output/rft_sft_conv10turn_e2/

# merge → M2 merged（供 M3 当初始权重 + M2 评估）
MERGED_DIR=output/rft_sft_conv10turn_e2_merged bash scripts/merge_rft_lora.sh
#   ⚠️ merge_rft_lora.sh 的 base 是 BASE_MODEL，本步正确（M2 从 base SFT）
```

### 2.1 M3 训练（GPU2=rollout 8004, GPU3=trainer）

**主路径：直接跑 v2 接力链**（推荐——自带 base@1000/M1@1000 同口径评估、完成即跳过守卫、EngineCore 清理）：
```bash
cd /home/tiger/jailbreak_research/agentic_jailbreak
M2_MERGED=$PWD/output/rft_sft_conv10turn_e2_merged
[ -f "$M2_MERGED/config.json" ] || { echo "缺 M2 merged，先做 §2.0"; exit 1; }

setsid nohup bash scripts/chain_eval1000_then_m3.sh > output/logs/chain_eval_m3.log 2>&1 &
# 流程: base@1000(49.6%) -> M1@1000 -> 起 rollout 8004=M2 merged -> 启动 M3(~20h)
# 日志: output/logs/m3_20260911.log (训练) / chain_eval_m3.log (链)
```

**替代路径：只要 300 口径 / 已评估过 1000（跳过链的前置评估）**：
```bash
cd /home/tiger/jailbreak_research/agentic_jailbreak
V=/home/tiger/jailbreak_research/.venv/bin    # 裸 shell 无 $PROJECT_ROOT，统一用绝对路径
M2_MERGED=$PWD/output/rft_sft_conv10turn_e2_merged
[ -f "$M2_MERGED/config.json" ] || { echo "缺 M2 merged，先做 §2.0"; exit 1; }

# ① 释放 GPU2 的 policy 8003，起 swift rollout 8004 = M2 merged
#    ⚠️ A800 必须 --vllm_max_model_len 32768 --torch_dtype bfloat16
#      （V100 版 m1_launch 的 24576+float16 是历史遗留；10 轮 C1 协议 24576 会越界）
#    ⚠️ 清进程用 v2 链的 kill_port 姿势（§6），别用裸 pgrep -f 自匹配杀自己
kill_port(){ # $1=端口 $2=GPU（v2 链修复版：EngineCore 孤儿是 ps 里可见的）
    fuser -k "$1/tcp" 2>/dev/null || true; sleep 8
    P=$(nvidia-smi -i "$2" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ')
    [ -n "$P" ] && kill $P 2>/dev/null || true
    for ep in $(ps -eo pid,etime,comm | awk '$3=="VLLM::EngineCore" && $2 ~ /^[0-9]+:[0-9]{2}:[0-9]{2}$/ {print $1}'); do
        kill -9 "$ep" 2>/dev/null || true
    done
    sleep 10
}
kill_port 8003 3
CUDA_VISIBLE_DEVICES=2 setsid nohup "$V/swift" rollout \
    --model "$M2_MERGED" \
    --vllm_tensor_parallel_size 1 --port 8004 --vllm_max_model_len 32768 \
    --vllm_gpu_memory_utilization 0.8 --torch_dtype bfloat16 \
    > output/logs/rollout_m3.log 2>&1 &
until [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:8004/health)" = "200" ]; do sleep 15; done

# ② M3 训练（MODEL_TAG=rft → 输出 output/multi_turn_10_agent_rft/）
MODEL="$M2_MERGED" MODEL_TAG=rft nohup bash scripts/exp04_grpo_10turn.sh \
    > output/logs/m3_train.log 2>&1 &
# 300 步 / num_gen16 / lr1e-5 / β0.05 / loss_type=grpo，~20h
# 中途看进度:
tail -f output/logs/m3_train.log | grep -E "Step|reward|grad_norm"
```
> **检查点（可选）**：M1 在 base 初始上 vanilla GRPO 未超越未训练 agent（−3.33pp）。
> M3 的价值在于验证"RFT 初始 + GRPO 叠加"是否还能再推一档。前 10-20 步若
> `frac_reward_zero_std` 持续 ≥80%（组内全零 → 零 advantage → 空转），可考虑停训转 DAPO 臂
> （`DAPO_FLAGS="--dynamic_sample true --max_resample_times 3 --epsilon_high 0.28"`）。

### 2.2 M3 merge（⚠️ base 必须是 M2 merged，不是 BASE_MODEL）
```bash
cd /home/tiger/jailbreak_research/agentic_jailbreak
V=/home/tiger/jailbreak_research/.venv/bin
M2_MERGED=$PWD/output/rft_sft_conv10turn_e2_merged
CKPT=$(ls -dt output/multi_turn_10_agent_rft/*/checkpoint-* | head -1)
[ -n "$CKPT" ] || { echo "找不到 M3 adapter"; exit 1; }
CUDA_VISIBLE_DEVICES=3 "$V/swift" export \
    --model "$M2_MERGED" --adapters "$CKPT" --merge_lora true \
    --output_dir output/m3_10turn_merged --max_length 4096
[ -f output/m3_10turn_merged/config.json ] && echo "M3 merged OK"
```

### 2.3 M3 评估（test C 300，RUN_TAG=m3 防止与 m0/m1/m2 共用目录覆盖）
```bash
cd /home/tiger/jailbreak_research/agentic_jailbreak
V=/home/tiger/jailbreak_research/.venv/bin
# 起 policy = M3 merged @8003（评估要标准 vllm serve，不是 swift rollout）
kill $(lsof -ti:8004 2>/dev/null) 2>/dev/null || true; sleep 10
CUDA_VISIBLE_DEVICES=2 setsid nohup "$V/vllm" serve output/m3_10turn_merged \
    --port 8003 --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --dtype bfloat16 --trust-remote-code > output/logs/policy_m3.log 2>&1 &
until [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:8003/health)" = "200" ]; do sleep 10; done

RUN_TAG=m3 EVAL_WORKERS=12 bash scripts/eval_conv.sh skill_decide 300 10 2 10
# 产出: output/eval_results/conv_skill_decide_top10_10turn_m3/summary.json
```

### 2.4 A 轴四臂汇总（09-14 已回填；M4 DAPO 完成后在此追加一行）
| 臂 | 初始 | 训练 | test C | ASR |
|----|------|------|--------|-----|
| M0 | base | — | 300 | 54.0%（本机复现 51.7%） |
| M1 | base | GRPO | 300 | 50.67% |
| M2 | base→SFT | — | 1000 | 59.5%（@300: 62.3%，本机 61.0%） |
| M3 | M2 merged | GRPO | 300 | **56.67%**（本机 09-14 首发；@1000 待补） |
| M4 | M2 merged | DAPO | 300 | **__?__** |

### 2.5 M4 DAPO 臂（09-14 启动；vanilla GRPO 双负收益后的算法对照臂）

**动机**：M1/M3 的 vanilla GRPO 在 base 与 RFT 初始上均负收益（−3.3pp / −4.3pp），
零组稀疏（frac_reward_zero_std 1/3~2/3）是最大嫌疑 → DAPO 动态采样直接过滤 std=0 组。

**参数链核实（09-14）**：
- `dynamic_sample` / `max_resample_times` 是 **swift 层字段**（TRL GRPOConfig 没有），
  实现于 `swift/rlhf_trainers/grpo_trainer.py::_resample_zero_variance_groups`（std=0 组丢弃，
  从备用 dataloader 换下一组，最多 max_resample_times 轮）
- `epsilon_high` 是 TRL GRPOConfig 字段（`epsilon_high=0.28` 即 DAPO Clip-Higher）
- TRL 0.29 的 `--loss_type` 默认就是 `dapo`（M1/M3 必须显式 `--loss_type grpo` 才是 vanilla）

**命令**（M2 merged 初始；其余超参与 M3 完全一致保证单变量对照）：
```bash
cd /home/tiger/jailbreak_research/agentic_jailbreak
V=/home/tiger/jailbreak_research/.venv/bin
M2_MERGED=$PWD/output/rft_sft_conv10turn_e2_merged
# ① 起 rollout 8004 = M2 merged（GPU2，同 §2.1 姿势: --vllm_max_model_len 32768 --torch_dtype bfloat16）
# ② 训练
export PYTORCH_ALLOC_CONF=expandable_segments:True ROLLOUT_PORT=8004 MASTER_ADDR=127.0.0.1 MASTER_PORT=43210
CUDA_VISIBLE_DEVICES=3 "$V/swift" rlhf \
    --rlhf_type grpo --model "$M2_MERGED" \
    --dataset "$PWD/output/grpo_data.jsonl" \
    --external_plugins "$PWD/src/plugin.py" \
    --multi_turn_scheduler gym_scheduler --gym_env jailbreak_env --use_gym_env true \
    --max_turns 10 --use_vllm true --vllm_mode server \
    --vllm_server_host 127.0.0.1 --vllm_server_port 8004 --vllm_server_timeout 1000 \
    --loss_type dapo --dynamic_sample true --max_resample_times 3 --epsilon_high 0.28 \
    --per_device_train_batch_size 1 --generation_batch_size 16 --gradient_accumulation_steps 8 \
    --max_steps 300 --learning_rate 1e-5 --num_generations 16 --max_completion_length 2048 \
    --fp16 false --bf16 true --gradient_checkpointing true --beta 0.05 \
    --output_dir output/multi_turn_10_agent_rft_dapo --report_to none \
    --run_name multi_turn_10_agent_rft_dapo
# merge → 评估同 §2.2/§2.3（RUN_TAG=m4）
```

**观察点**：① `frac_reward_zero_std` 应显著低于 M1/M3（过滤生效的直接证据）；
② 训练时间因重采样变长（A100 实测 ~15-20h/300 步）；③ clip_high 0.28 放宽上限理论上利于高 advantage 组。**前 20 步若 zero_frac 仍 ≥80% 且重采样告警频繁，需查 env 奖励是否退化（如 target/guard 服务异常）。**

### 2.6 长臂续训实验（09-14 计划，epoch 议题；M5 运行中 09-15）

> **动机**：GRPO 臂 300 步 = 0.3 epoch（B 段 1000 条，1 epoch = 1000 步；对照组 RFT = 2 epochs 全数据）。
> ⚠️ **不用 resume**（本环境 TRL resume + fused adam 报 dtype 错，LOG 09-15）。
> **方案 B**：初始 = 前一臂 merged 权重，全新 run + `--save_steps 300`；**累计步数 = 前臂 + 本 run**。

```bash
cd /home/tiger/jailbreak_research/agentic_jailbreak
V=/home/tiger/jailbreak_research/.venv/bin

# M5: 初始 = m3_10turn_merged（策略权重 = M3 终态），vanilla GRPO 600 步（累计 900 = 0.9 epoch）
CUDA_VISIBLE_DEVICES=3 "$V/swift" rlhf \
    --rlhf_type grpo --model "$PWD/output/m3_10turn_merged" \
    --dataset "$PWD/output/grpo_data.jsonl" --external_plugins "$PWD/src/plugin.py" \
    --multi_turn_scheduler gym_scheduler --gym_env jailbreak_env --use_gym_env true \
    --max_turns 10 --use_vllm true --vllm_mode server --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8004 --vllm_server_timeout 1000 --loss_type grpo \
    --per_device_train_batch_size 1 --generation_batch_size 16 --gradient_accumulation_steps 8 \
    --max_steps 600 --save_steps 300 \
    --learning_rate 1e-5 --num_generations 16 --max_completion_length 2048 \
    --fp16 false --bf16 true --gradient_checkpointing true --beta 0.05 \
    --output_dir output/multi_turn_10_agent_rft_long --run_name multi_turn_10_agent_rft_long
# M6: 同款把 --model 换 output/m4_10turn_merged + --loss_type dapo + DAPO_FLAGS（--dynamic_sample true --max_resample_times 3 --epsilon_high 0.28）
# ⚠️ MASTER_PORT 用 43212（43210 段有残留事故史）；失败重启后须整体干净重启 rollout
# 评估: checkpoint-300（累计 600）→ RUN_TAG=m5_s600；checkpoint-600（累计 900）→ RUN_TAG=m5_s900
```

**判定**：累计 900 步（M5）/600 步（M6）仍无转正迹象 → GRPO 负收益主因 = 算法/稀疏奖励（非训练量），转 C 轴。

---

## 3. C 轴 Reward 组合（A 轴后；⚠️ 需开发机先完成代码改造）

> **代码前置（开发机交付，本节点收到 git 更新后才有对应脚本）**：
> 1. `rewards.py` R2(Process) / R3(Efficiency) 接入 `plugin.py`（现 env.step 只返回 ASR-only 的 0/1）
> 2. R1+λ（AHR 自适应迁移，第一章公式迁移）：env 返回奖励分量 + 训练侧组级自适应混合 λ∈[0.1,0.9]，EMA β
>    （参考 `docs/AHR-GRPO.md` §3.4：α=2.0, δ=-2.0, warmup；代码参考 `RL4jailbreak/experiments/hybrid_reward_exp`）
> 3. 为每臂生成训练脚本（复制 exp04 模板，改 reward 相关 env_config / reward_funcs）

**矩阵**（在 A 轴最优臂的初始权重上训练，各臂 300 步，同 exp04 超参）：
| 臂 | Reward | 代码状态 |
|----|--------|----------|
| R1 ASR-only | 已接入（= M1/M3 现状） | ✅ |
| R1+P 过程奖励 | rewards.py 有类，**plugin 未接** | ⏳ 待开发 |
| R1+E 效率奖励 | rewards.py 有类，**plugin 未接** | ⏳ 待开发 |
| R1+λ AHR 自适应 | 需 env 分量 + 训练侧混合 | ⏳ 待开发（唯一非平凡改造） |

**训练模板**（开发机交付后填实）：
```bash
# 例：R1+E 效率臂（示意，实际以开发机交付的脚本为准）
MODEL=<A轴最优merged> MODEL_TAG=rE bash scripts/exp04_grpo_10turn.sh   # reward 走 env_config
# merge → 评估，同 §2.2/§2.3
```

---

## 4. Phase 4 评估与分析（A 轴定稿后）

### 4.1 多 benchmark 评估（在 M2 / 最优臂上）
```bash
# eval.py 用 obj["prompt"] 解析，benchmark jsonl（{"id","prompt"}）直接兼容
for b in advbench harmbench_standard harmbench_contextual jailbreakBench; do
    DATA_PATH=/home/tiger/jailbreak_research/data/benchmark/$b.jsonl \
    RUN_TAG=m2_$b EVAL_WORKERS=12 \
        bash scripts/eval_conv.sh skill_decide 1000 10 2 10
done
# default 已跑（test.jsonl = §2.3）；五个 benchmark 目录均在 $PROJECT_ROOT/data/benchmark/
# 产出: output/eval_results/conv_skill_decide_top10_10turn_m2_{advbench,harmbench_standard,harmbench_contextual,jailbreakBench}/
```

### 4.2 baseline 重跑（plain-4B target 口径；表 1.4 审计后 Ch1/Ch3 同批产出）
```bash
cd /home/tiger/jailbreak_research
# 前置: guard/target(plain-4B) 就绪（§0.4 的服务即 plain-4B target）
for s in pair autodan crescendo tap genetic deepinception no_rewrite; do
  .venv/bin/python -B -m baselines.cli batch \
      --input self_evolve_skills_jailbreak/data/test_prompts.json \
      --strategy "$s" --output baselines/experiments/${s}_plain4b \
      --limit 1000 --evaluate --max_workers 8 \
      --target-port 8002 --guard-port 8001 \
      $( [ "$s" = crescendo ] && echo --max-turns 8 ) \
      $( [ "$s" = tap ] && echo --max-turns 6 --branching-factor 2 --prune-top-k 1 )
done
# 可用策略: no_rewrite multilingual pair genetic deepinception persona autodan crescendo tap
# 判定口径统一: guard strict Unsafe / 分母 total / 实测调用数（09-09 审计结论）
# ⚠️ 这是旧 PAIR 91.8% 伪值的重测：旧 pair.py 直发原始 prompt 即计成功，新 cli 已修正
```

### 4.3 vs SESS / PAIR / AHR-GRPO 对比
- **PAIR**：见 §4.2（同批产出）
- **SESS（第二章）**：`self_evolve_skills_jailbreak/` 主实验已含；此处产出"第三章 agent vs SESS 规则 skill"对比表
- **AHR-GRPO（第一章）**：表 1.4 全行重测由 `RL4jailbreak/experiments/e2e/e2e_ahr_grpo.py` 承担（Ch1 独占，勿与 Ch3 抢卡；两章并行布局见 LOG 09-09）
- 产出：一张"Ch3 vs 基线 vs 前两章"对比表，进 REPORT 单元 8

### 4.4 行为分析
- 已有 `output/analysis/skill_usage.json`（M0/M1/M2）；M3 完成后补同款
- 分析项：选率随轮次 / 成功 vs 失败轨迹的 skill 依赖（base 成功更依赖 skill，RFT 反转）/ top-10 各 skill 在 agent 语境下的贡献

### 4.5 实验报告
- 补 `exp/README.md` 汇总表 + `docs/REPORT.md` 单元 8（A 轴四臂 + C 轴 + Phase 4）

---

## 5. 结果回收（output/ 不入库，实验完打包带走）
```bash
cd /home/tiger/jailbreak_research
# 模型权重 + 实验数据（评估结果/采集/RFT 数据/skill 分析）
tar -czf /tmp/ch3_output_$(date +%Y%m%d).tar.gz \
    -C agentic_jailbreak/output \
    rft_sft_conv10turn_e2* m3_10turn* multi_turn_10_agent_rft/ \
    rft_collect_conv_10turn/ rft_data_conv_10turn.jsonl \
    eval_results/ baxis_ctx/ analysis/ logs/
# git 代码/文档/日志走 bundle（开发机合并）
git bundle create /tmp/ch3.bundle origin/main..HEAD
# 交接文档：更新 docs/EXECUTION_NODE_*.md + docs/LOG.md 后让开发机取走
```

---

## 6. 已知坑速查

| 坑 | 对策 |
|----|------|
| fastapi≥0.116 → 三服务全 500 | 依赖锁在 `>=0.115,<0.116`，勿升 |
| `wait_for_server` 裸 curl 误判就绪 | 必须判 HTTP 200（common.sh 已修）；swift rollout 常回 307=活着，000=没起 |
| 10 轮 C1 协议 input 可达 14k+ | 服务 max-model-len：policy **32768**（24576 差 1 token 越界，C2 事故）；guard/target 16384 |
| 12 路并发 client 超时 60s 打穿 | 超时已统一 1000s；崩溃用 `--resume` 续跑，逐条落盘 |
| 训练 rollout 用标准 vllm serve | 会 404 —— 训练必须 `swift rollout`（带 communicator 端点），评估才用 vllm serve |
| M3 merge 用 BASE_MODEL 当 base | **错误** —— M3 从 M2 merged 训的 LoRA，base 必须是 `output/rft_sft_conv10turn_e2_merged` |
| 各臂评估共用输出目录覆盖 | 每次评估带 `RUN_TAG=m0/m1/m2/m3/m2_<bench>`；链脚本会先备份 300 子集为 `*_300subset` |
| NUM_GPUS 未前置声明 | 4 卡布局必须在 `source common.sh` 前 `NUM_GPUS=4`（否则走 8 卡布局，NCCL no CUDA-capable device） |
| MASTER_PORT 29500 段预占 | 训练默认 43210（exp04 已写死） |
| 换机器 output/ 丢失 | 必跑 `build_splits.py` 重建 split A/B/C（隔离断言兜底） |
| target 误起 SafeRL 污染口径 | TARGET_MODEL 默认 plain-4B（common.sh 已改） |
| **`pgrep/pkill -f` 匹配到自己 shell** | 命令里出现目标明文就会误杀自己（SIGTERM 143）。安全姿势：模式用字符类打断（`eval[.]py`）、列出与杀死分两条命令、带模式杀进程独占一条命令 |
| **`fuser -k` 后 EngineCore 孤儿占整卡** | vLLM API server 死后 `VLLM::EngineCore` 会 setsid 脱离独占显存。权威判据：`ps -eo pid,ppid,etime,args \| grep VLLM::EngineCore`，杀 <24h 的（常驻引擎均 >1 天，不误伤） |
| **nvidia-smi PID 是残影** | `/proc/<pid>` 不存在且 kill 报 No such process = 已死父进程的 CUDA context 记账。别在假 PID 上白耗，先查 `ps` 找 EngineCore |

---

## 7. 命令速查卡

```bash
# 0. 环境
bash docs/cookbook/bootstrap_env.sh
python agentic_jailbreak/scripts/build_splits.py
NUM_GPUS=4 bash agentic_jailbreak/scripts/start_servers_a800.sh

# 1. 冒烟（M0 重现）
cd agentic_jailbreak && RUN_TAG=m0 EVAL_WORKERS=12 bash scripts/eval_conv.sh skill_decide 300 10 2 10

# 2. M3（入口 B 先 §2.0 重造 M2；cd agentic_jailbreak 后）
#    推荐直接跑 v2 链（含 base@1000/M1@1000 评估 + M3 启动 + EngineCore 清理）
M2_MERGED=$PWD/output/rft_sft_conv10turn_e2_merged
setsid nohup bash scripts/chain_eval1000_then_m3.sh > output/logs/chain_eval_m3.log 2>&1 &
#    （替代：只要 300 口径时手动 rollout+train，见 §2.1 替代路径）
#    merge
CKPT=$(ls -dt output/multi_turn_10_agent_rft/*/checkpoint-* | head -1)
CUDA_VISIBLE_DEVICES=3 "$V/swift" export --model "$M2_MERGED" --adapters "$CKPT" \
    --merge_lora true --output_dir output/m3_10turn_merged --max_length 4096
#    评估
RUN_TAG=m3 EVAL_WORKERS=12 bash scripts/eval_conv.sh skill_decide 300 10 2 10

# 4. 多 benchmark（在 M2/M3 最优臂）
DATA_PATH=/home/tiger/jailbreak_research/data/benchmark/advbench.jsonl RUN_TAG=m2_advbench EVAL_WORKERS=12 \
    bash scripts/eval_conv.sh skill_decide 1000 10 2 10

# 4. baselines 重跑（plain-4B target）
cd /home/tiger/jailbreak_research
.venv/bin/python -B -m baselines.cli batch --input self_evolve_skills_jailbreak/data/test_prompts.json \
    --strategy pair --output baselines/experiments/pair_plain4b --limit 1000 --evaluate --max_workers 8 \
    --target-port 8002 --guard-port 8001
```
