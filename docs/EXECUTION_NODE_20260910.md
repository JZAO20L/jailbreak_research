# A800 执行节点状态交接（2026-09-10 16:20 快照）

> 本节点定位：**只执行实验命令/脚本，不做开发与提交**。所有代码改动已收敛在一个
> 本地提交里，需要由开发机取走（见 §2）。本文档是交接凭证，之后本机只负责跑实验。

---

## 1. 环境与约定

| 项 | 值 |
|---|---|
| 硬件 | 4×A800-SXM4-80GB（SM80，bf16 原生，**不再是 V100**） |
| 仓库 | `/opt/tiger/JudgeForge/jailbreak_research`（= `/home/tiger/jailbreak_research` 符号链接，脚本内硬编码路径依赖它） |
| 依赖 | `requirements.txt` + `requirements.lock.txt`（248 行）+ `docs/cookbook/bootstrap_env.sh`（幂等：venv/符号链接/依赖/冻结） |
| 版本 | vllm 0.18.0 · ms-swift 4.3.2 · trl 0.29.1 · torch 2.10.0 · transformers 4.57.6 · xformers 0.0.35 · liger-kernel 0.8.2 · python 3.12(uv) |
| 模型 | `model/Qwen3-4B`（policy+target）+ `model/Qwen3Guard-Gen-4B`（guard） |
| 端口 | guard 8001 · target 8002 · policy 8003 · swift rollout 8004 · e2e 临时 policy 8005 |
| 布局 | 打包态：GPU0=guard+target(各 0.42)；GPU1 按需（policy/训练）；GPU2=rollout 8004；GPU3=训练 |

### 已封口的坑（勿回退）
- **fastapi 必须 <0.116**：0.116+ 的 `_IncludedRouter` 让 vllm 0.18 中间件在每个请求爆异常 → `/health` 与 `/v1/*` 全 500（pin 在 requirements.txt，注释里有完整故事）
- 依赖安装用 `uv`；模型下载走 hf-mirror 需 `HF_HUB_DISABLE_XET=1`（Xet 通道 401）
- V100 时代的补丁全失活：`v100_attn_patch` 自带 SM<8 守卫；fp16 发散（09-09 GRPO 根因）在 A800 不存在，全程 bf16
- 训练脚本里 `--report_to` 默认已改 none（SwanLab 无 TTY 会直接抛错）
- 本机**无 GitHub 推送凭据**（remote 已切成 `git@github.com:JZAO20L/jailbreak_research.git`，但 `~/.ssh/config` 未配，直接 push 会 publickey 拒绝）——取代码一律用 §2 的 bundle

---

## 2. ⚠️ 未推送的提交（必须带走）

**本地 main 上有 1 个提交未推送：`a13e57f`**（"feat(agentic): A800 环境重建 + C4 压缩式上下文 + Ch1 e2e 单文件脚本"，20 文件 +1658/−68）。

取走方式（在开发机）：
```bash
# 本机生成：
git -C /opt/tiger/JudgeForge/jailbreak_research bundle create /tmp/ch3.bundle origin/main..HEAD
# 开发机接收：
git fetch /tmp/ch3.bundle HEAD:tmp-branch
git merge tmp-branch && git push origin main
```

### `a13e57f` 变更清单
| 文件 | 内容 |
|---|---|
| `requirements.txt` / `requirements.lock.txt` | 依赖唯一来源，严格复刻历史版本；fastapi pin 等坑注释 |
| `docs/cookbook/bootstrap_env.sh` | 幂等环境搭建（uv venv + 依赖 + 冻结 + 符号链接） |
| `agentic_jailbreak/scripts/start_servers_a800.sh` | A800 服务：bf16、16384/32768 ctx、SVC_UTIL/START_POLICY 可打包 |
| `agentic_jailbreak/scripts/infer_replica.sh` | 加 target/guard 副本（吞吐瓶颈在服务不在 trainer） |
| `agentic_jailbreak/scripts/build_splits.py` | 重建 split A/B/C，两两不重叠写成断言（08-26 test 泄漏事故防线） |
| `agentic_jailbreak/scripts/common.sh` | 模型路径自动探测 `model/`；TARGET 默认 plain-4B；SKILLS 默认精选库 |
| `agentic_jailbreak/scripts/eval_conv.sh` | 默认对齐定稿（10 轮/top_k=10）；EVAL_WORKERS；**RUN_TAG**（A 轴各臂不共用目录） |
| `agentic_jailbreak/scripts/run_baxis_ctx_ablation.sh` | 可裁剪臂列表；RESUME 续跑；汇总表加 ctx_max/full_max/folds |
| `agentic_jailbreak/scripts/rft_sft_conv.sh` | bf16 默认（V100 遗留 fp16 清除）；PER_DEVICE/ACCUM 参数化（有效 batch=8 不变） |
| `agentic_jailbreak/scripts/merge_rft_lora.sh` | SFT_DIR/MERGED_DIR/MERGE_GPU 可覆盖 |
| `agentic_jailbreak/scripts/exp04_grpo_10turn.sh` | `--report_to` 可覆盖（默认 none） |
| `agentic_jailbreak/src/conv_eval.py` | **动作解析 bug 修复**（Selection 元信息不再进攻击串）；**C4 压缩式上下文**；**逐条增量落盘 + --resume** |
| `agentic_jailbreak/src/env.py` | 解析同步修复（与 conv_eval 逐字一致） |
| `agentic_jailbreak/src/eval.py` | ctx_compress/ctx_keep/tokenizer_path/resume 参数 |
| `RL4jailbreak/experiments/e2e/e2e_ahr_grpo.py` | Ch1 单文件实验脚本（存档用，见 §4） |
| `RL4jailbreak/TODO.md` / `agentic_jailbreak/docs/LOG.md` | 审计结论与事件记录（含 91.8% 证伪全文） |

---

## 3. 第三章运行状态

### 3.1 已出数字（plain-4B target，official guard 口径，test C 同一前 300）
| 臂 | ASR | avg_turns | 备注 |
|---|---|---|---|
| **M0** base | **54.00%** (162/300) | 5.98 | 09-09 闸门；轨迹存档 `output/eval_results/conv_skill_decide_top10_10turn_m0/` |
| **M2** RFT(516 条,2ep) | **62.33%** (187/300) | 5.43 | 配对 McNemar chi²=6.62, **p≈0.010**, 差值 95%CI **[+2.2,+14.4]pp**（0.05 显著，0.01 不显著，写论文照实） |
| M0 直发对照 | 33.2% | — | 来自 Ch1 评估（不同批，勿混用） |

采集侧：RFT 采集 1000 条 ASR **51.6%**（12 分片，09-09 完成）→ SFT 516 条全轨迹样本（历史 SafeRL 口径仅 82 条，数据量 6.3×）。

### 3.2 正在跑（16:20）
| 作业 | 进度 | ETA | 日志 |
|---|---|---|---|
| **B 轴 4 臂**（C1 / C3win3 / C4@2500 / C4@4000 × 300） | 每臂 97–104/300 | ~19:40 | `output/logs/baxis_20260910.log` + `output/baxis_ctx/<臂>/part*.log` |
| **M1 GRPO@10 轮**（300 步，num_gen 16，bf16） | 122/300（141s/步） | ~23:40 | `output/logs/m1_20260910.log` |

B 轴表会带 5 列：ASR / avg_turns / ctx_max（实际最大输入 token）/ full_max（不压缩会到的）/ folds。
**RESUME=1 + 逐条落盘已生效**：任何分片挂了，重启该片只丢正在跑的一条。

### 3.3 B 轴机制定义（勿混淆）
- **C1** 全量累积（定稿基线）
- **C3** 滑窗 N=3（`_ctx_view`，纯丢弃无摘要）
- **C4** 压缩式记忆（本次新增）：**未越阈值时与 C1 逐字节相同**；越阈才把旧轮折成累计摘要、折无可折再缩窗 → 硬上界。阈值依据实测：C1 十轮输入 token 中位 6.7k / p90 8.7k / max 10.0k（32768 窗口下**不存在溢出问题**，本轴主张 = 同等 ASR 下的上下文成本）
- **C2**（每轮摘要+不删原文）故意不再跑：视图严格大于 C1，当年败是设计必然，写文章时一句话说明即可（需要进表可加回 `ARMS=C2_workmem`）

### 3.4 排队与待办
1. B 轴出表后：**M3**（`MODEL=<m2_merged> MODEL_TAG=rft bash scripts/exp04_grpo_10turn.sh`）；M2 eval 补 1000 条（`RUN_TAG=m2 --eval-samples` 语义，300 条那个已是 `_m2` 标签）
2. **C 轴**：AHR λ 迁进多轮。注意多轮 gym 路径无"组内"统计，落地方式 = plugin 进程内**滑窗方差比**（最近 N 条轨迹的 ASR 分量 vs 过程分量），并在 rewards.py 挂过程奖励分量；判定层需"可信裁判"（Ch1 结论：裁判=target 自身会 reward hacking）
3. **训练侧同步**：C3/C4 目前只接了评估侧（conv_eval），`plugin.py`/`env.py` 只有 C1；若 B 轴结论选非 C1，必须先同步训练侧，否则重演 RFT v2"协议失配"
4. B 轴结论前 **不要动 run_baxis 的阈值**；40 轮上下文里顺带量的 folds 统计是下一步调阈值的依据

---

## 4. 第一章（已定停投，仅留档）

- 结论：ahr 臂 1 epoch（500 步）在 test C 1000 条上 **20.5% < 直发 33.2%**（−12.7pp）；训练奖励 0.05→0.195 上升但 held-out 掉 → **疑"裁判=target 自身"导致 reward hacking**（Ch3 C 轴要可信裁判的动机链）
- **口径税仅 −0.15pp**：PAIR 91.8% 伪值的锅确认是"直发被算成攻击成功 + 泄露思维链当攻击串"，**不是** guard 加不加 system prompt
- `e2e_ahr_grpo.py` 保留：奖励 4 模式、epoch 换算、nproc 自举 torchrun、双口径评估 —— 可作 C 轴移植蓝本
- **欠着的勘误（不占卡但必须做）**：
  - `docs/experiment_results_midterm.md:154,167`（表1.4 与定位结论）
  - `docs/experiment_results_full.md:122-124`、`docs/figures/figure_data_tables.md:99-100`
  - `docs/report/thesis_progress_report.md:36,39`（"成本差两个数量级"）
  - `docs/latex_project/main.tex:330`（"1000+ API 调用/样本"，实测 ~10-30），同文件 285 行写了正确的 ≤20 次
  - `docs/AHR-GRPO.md:25,95`、`docs/cookbook/03_three_chapters.md:25`
  - 修复方案见 `RL4jailbreak/TODO.md` 09-09 节（统一判定层重测基线表是另一件事，用户已决定 baseline 重测不做，**只勘误**）

---

## 5. 常用命令速查（本机执行）

```bash
# 服务（打包态：guard+target 在 GPU0，抠出 GPU1/2/3 给训练）
NUM_GPUS=4 GUARD_GPU=0 TARGET_GPU=0 START_POLICY=0 SVC_UTIL=0.42 bash agentic_jailbreak/scripts/start_servers_a800.sh

# B 轴（断点续跑）
RESUME=1 N_SAMPLES=300 N_PARTS=3 bash agentic_jailbreak/scripts/run_baxis_ctx_ablation.sh

# 单臂评估（A 轴必须带 RUN_TAG，防互相覆盖）
RUN_TAG=m2 EVAL_WORKERS=12 bash agentic_jailbreak/scripts/eval_conv.sh skill_decide 300 10 2 10

# M1/M3 GRPO（M1 前置：GPU2 起 swift rollout 8004）
MAX_TURNS=10 DTYPE=bf16 bash agentic_jailbreak/scripts/exp04_grpo_10turn.sh            # M1
MODEL=agentic_jailbreak/output/rft_sft_conv10turn_e2_merged MODEL_TAG=rft \
  bash agentic_jailbreak/scripts/exp04_grpo_10turn.sh                                   # M3

# Ch1 e2e（仅留档/移植时用）
.venv/bin/python RL4jailbreak/experiments/e2e/e2e_ahr_grpo.py --stage eval \
  --policy <merged> --eval-samples 1000
```

---

## 6. 后续约定

- 本节点**只跑命令**；代码走 §2 的 bundle 带回开发机
- 每次实验产出（`output/` 不入库）如要带走：只带 `*/*/summary.json` 与 `results.jsonl`（`!**/output/**/summary.json` 反而是入库例外的，注意别把 checkpoint 打进 git）
- 关键实验落盘后，往 `agentic_jailbreak/docs/LOG.md` 追加一行事件（只追加不改写）——这是唯一在跑实验时该做的"文档"
---

## 09-11 早间更新（接力链自动执行完毕）

### A 轴数字（plain-4B / skill_decide@top10 / 10 轮 / C1 / official 口径）
| 臂 | 样本 | ASR | avg_turns | 说明 |
|---|---|---|---|---|
| M0 base | 300 | 54.00% | 6.06 | B 轴 C1 复测同数 |
| **M1 GRPO** | 300 | **50.67%** | 6.02 | 低于 base，grad_norm 震荡、未学到选择性 |
| **M2 RFT**(516 条,2ep) | 300 / **1000** | 62.33% / **59.50%**(595/1000) | 5.57 | 全量 CI≈[56.5%,62.5%] |
| **M3** | — | 未跑 | — | 停在人工确认闸门（`chain_after_m1.sh` 设计如此） |

### 状态
- 4 卡当前空闲（服务常驻：GPU0 guard+target / GPU2 rollout 8004 / GPU3 遗留 M2 policy 8003；GPU1 有残留意外的 e2e 引擎 ~70GB，`fuser -k 8003/tcp` 杀 API server 后 EngineCore 成孤儿，与 09-08 同坑）
- M3 启动三步：① 停 8004 rollout → 换 M2 merged 起 rollout；② `MODEL=output/rft_sft_conv10turn_e2_merged MODEL_TAG=rft bash scripts/exp04_grpo_10turn.sh`；③ 观 grad_norm/zero_std，若 M1 的震荡复现应提前停（~30 步内可判）
- **M1 结论待办**：若 M3 也失败 = GRPO 配方问题（lr/β/batch/num_gen 或 ASR-only 奖励），列 C 轴（过程/效率/λ）为修复方向；M3 成功 = "RFT 冷启动是 GRPO 的前提"成为 A 轴主线
