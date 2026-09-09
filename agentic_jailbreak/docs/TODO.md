# Agentic Jailbreak TODO

> 唯一维护的待办清单。事件记录见 [LOG.md](./LOG.md),代码/脚本/成果说明见各层 README。

## 章节核心目标(第三章)

**设计一套好的 jailbreak-agent harness,并在此基础上做奖励建模与 AgenticRL 训练。**

三条主线:
1. **Harness**:observation 标准化(工具返回结构化)、多轮交互、skill 注入、memory —— 让 Agent 的每轮决策有信息可用
2. **奖励建模**:从 ASR-only 起步,逐步引入过程奖励/效率奖励/多样性(对标 AHR 思路),设计 RL 可用的 reward
3. **AgenticRL 训练**:ms-swift GRPO(单轮 → 多轮),RFT 冷启动(必要时),目标是超越 AutoDAN 19% 参照

> 环境:4×V100-SXM2-32GB;服务 guard=GPU0:8001 / target=GPU1:8002 / policy=GPU2:8003 / 训练卡=GPU3(见 `scripts/start_servers_v100.sh`)

## 当前主线:Agent 版本矩阵实验(设计空间消融)

### Agent 版本设计矩阵(2026-08-12)

| 版本 | 设计 | 消融什么 | 状态 |
|------|------|----------|------|
| AV1 | Select+Adapt,top-5 quality 检索,v2 有信息反思 | 基线 | 🔄 v2 3 轮全量对比运行中 |
| AV2 | **No-Skill**:自由生成攻击,无 skill 库 | skill 库价值 | 待实现 |
| AV3 | **Select-Only**:只选 skill,用原始 skill 内容,不适配 | 适配价值 | 待实现 |
| AV4 | **Beam-2**:每轮生成 2 个候选,取更优 | 测试时搜索价值 | 待实现 |
| AV5 | **Adaptive-Retrieval**:按 prompt 相似度检索 top-k | 检索方式(quality vs similarity) | 待实现 |

对照基准:直发 0% / PAIR 0% / AutoDAN 19% / AV1 单轮 4.3% / AV1 3 轮 9.1% / AV1 5 轮 11.1% / AV1 10 轮子集 12.5%

### P0 运行中/待验证

- [x] **v2 3 轮全量(08-12)**:✅ **7.00% < v1 9.10%**——有信息反思对未训练模型有害,REPORT 单元 3 定稿;RL 观察用精简式
- [x] **变体矩阵(无状态 harness,legacy)**:✅ REPORT 单元 4——AV2 18%/23.2%、AV6 35%、AV1 5% 等
- [x] **核心方法形态切换(08-13 决策)**:✅ 对话式前 100——**no_skill 25%(+7pp)、beam-2 35%、select 5%**;主配置定稿(见 REPORT 单元 6)
- [x] **对话式全量验证**:✅ conv_no_skill 3 轮全量(1000 条)= 23.2% —— RL 基线
- [x] **RL 路线选择(定稿)**:✅ action space = 对话式 no_skill(单动作训练);评估 beam-2 放大;**RL 目标 = 超越 25%(单)/35%(beam)**

### Skill 调用时机消融(2026-08-18)

- [x] **三时机实现 + 前 100 对比**:`skill_once`(轨迹只调一次)/ `skill_every_turn`(每轮重选,即旧 select_adapt)/ `skill_decide`(LLM 自决,输出 Selection 标记则用 skill)——前 100:9% / 6% / 28%(no_skill 对照 25%)
- [x] **skill_decide 全量验证**:✅ **19.0% < no_skill 23.2%**——前 100 正收益为子集噪声;全量口径 skill 注入(即使 LLM 自决)仍为负资产,但远好于强制调用(-4.2pp vs -18pp)
- [ ] (可选)similarity 检索 + skill_decide 重测:针对"候选不对齐"根因(54 skills 多 quality=0,top-5 全局最优与 prompt 无关;词袋 Jaccard 检索也粗糙)
- [ ] (可选)动作使用率分析:记录每轮是否输出 Selection,量化 LLM 自决时实际使用 skill 的比例

### Phase 3: 奖励建模 + AgenticRL 训练(ms-swift GRPO)

- [x] **ms-swift 4.3.2 链路修复**:plugin.py 重写为 async Env(已验证注册);exp 脚本 CLI 修正(use_gym_env/gym_env/TRAIN_GPUS/dataset);grpo_data.jsonl 已建(1000 条)
- [x] **对话式 RL 路径对齐(08-13)**:plugin.py 消息协议与 conv_eval 一致(初始消息+最小反馈);eval_conv.sh / chain_test_grpo.sh 已建
- [x] **小型训练链验证**:✅ **08-13 通过**(5/5 步,多轮交互+reward 流入 GRPO;根因: NUM_GPUS 声明顺序,已修复 chain_test+exp01/02/03)
- [x] **正式训练(08-16/17 完成)**:exp01(单轮)= **10.0%** / exp02(3 轮)= **25.8%** / exp03(5 轮)= **33.9%**(beam-2 17.1% / 35.2% / **44.4%**),均为 500 步全量 1000 条,dialogue no_skill;5 轮单次超越目标 23.2%、beam 超越 35%(见 REPORT 草稿单元 8)
- [ ] **未训练 5 轮同口径基线**(可选):conv_no_skill 5 轮未训练对照(现缺,仅有 AV1 无状态 11.1%)
- [ ] 奖励设计:先 ASR-only(与基线可比),再扩展 过程/效率/多样性(harness 已支持 process_reward)
- [ ] (可选)RFT 冷启动:把 AutoDAN 成功样本纳入先验
- [ ] (可选)Agent + Memory(exp04)

### Phase 4: 评估与分析

- [ ] 多 benchmark 评估(default / advbench / harmbench_standard / harmbench_contextual / jailbreakBench)
- [ ] 对比 SESS(第二章)、PAIR、AHR-GRPO(第一章)
- [ ] 分析 Agent 行为(skill 选择分布、失败反馈调整)
- [ ] 撰写实验报告

## 实验矩阵

| 实验 | 配置 | 说明 | 状态 |
|------|------|------|------|
| E1 | 单轮 Agent | baseline | ✅ 完成:10.0%(beam 17.1%) |
| E2 | 多轮 Agent (3 轮) | 验证多轮学习 | ✅ 完成:25.8%(beam 35.2%) |
| E3 | 多轮 Agent (5 轮) | 充分多轮交互 | ✅ 完成:33.9%(beam 44.4%) |
| E4 | Agent + Memory | 验证记忆机制 | 待跑(可选) |
| A1 | 无 SESS skills | 证明 skills 价值 | ✅ 完成:no_skill 23.2% 优于一切 skill 注入;强制调用 6-9%、LLM 自决 19%,全量口径 skill 仍负资产 |
| A2 | 无 Agent 分析 | 证明分析能力价值 | 待跑(反思融入对话,形态上已无独立 analyze) |
| C1 | vs SESS | Agent vs 规则 | 待跑 |
| C2 | vs PAIR | Agent vs 固定 rewrite | 待跑 |
## 2026-09-01 计划：10 轮实验矩阵 M0-M3（DAPO 暂缓）

> **2026-09-02 口径定稿（覆盖此前 no_skill 主配置）**：agent 核心 = **PAIR 骨架 + Skills，LLM 自选**（`skill_decide`），与 SESS 一脉相承。
> Skills 库统一使用 **10-skill 精选库** `exp/skill_asr_sweep/seed_skills_top10.json`（54 库按实测单调用 ASR 筛选）。
> 此前 A1 消融"skill 负资产"的根因是 54 库全局 top-5 候选与 prompt 不对齐；精选 10-skill 后候选全为实测有效技能。
> 训练/评估/RFT 采集统一此协议：plugin.py `DEFAULT_ENV_CONFIG` 已切 `skill_decide@top10@top_k=10@max_turns=10`
> （⚠️ 顺带修复：原 env 默认 max_turns=5，exp04 若不传 env_config 会静默跑成 5 轮）。

### 第三章三轴实验规划（09-02 定稿，漏斗式：每轴只在当前最优配置上做下一轴）

| 轴 | 内容 | 臂 | 排期 |
|----|------|----|------|
| **B 上下文维护**（✅ **09-05 完成,C1 定稿**） | C1 9.33% / C2 5.00% / C3 8.67%（base,test C 前 300,详见表下） | C1 全量累积 / C2 工作记忆 / C3 滑窗 3 轮 | 9/3-9/5 |
| **A 后训练配方**（主轴,**09-05 解除 HOLD**） | RFT/RL 各自贡献与叠加 | M0-M3 主矩阵；M4/M5(DAPO) 视稀疏度 | 9/5 起 |
| **C Reward 组合** | 稀疏缓解 + 第一章方法迁移 | R1 ASR-only / R1+P 过程 / R1+E 效率 / R1+λ **AHR 自适应迁移(章节联动)** | A 轴后，3 新臂全跑 |

**B 轴结果（09-05 定稿,`output/baxis_ctx/summary.json`）**：C1_full **9.33%**(28/300,avg 9.37 轮) / C2_workmem **5.00%**(15/300,9.64 轮) / C3_window3 **8.67%**(26/300,9.51 轮)。判定：C3 与 C1 噪声范围内(无收益),C2 全面落后且工程三重劣势(ASR 低、上下文撑不满、GRPO rollout 无法旁路调 summarizer) → **harness 维持 C1 全量累积协议,RFT 82 条数据无需重采**。
> ⚠️ C2 臂事故：每轮额外插入总结消息使上下文增速超 C1/C3,24576 下 input 22529+2048=24577 越界崩溃(差 1 token),policy 提 **32768** 后补齐 part0/1(`rerun_c2_baxis_20260904.sh`)。

**B 轴前置的理由（09-03）**：M2 的 RFT 数据正按 C1（全量累积）协议落盘；若 B 轴消融选了 C2/C3，这批数据与 M1/M3 训练臂（每条 ~19h）都要在新 harness 下重来。harness 定稿必须先于训练数据定型。附带收益：若 C2/C3 提升基座 ASR，重采 RFT 数据量与 GRPO 奖励稀疏问题同步缓解。

C 轴代码前置：rewards.py R2/R3 接入 plugin（现 ASR-only）；R1+λ 需 env 返回奖励分量 + 训练侧组级自适应混合（唯一非平凡改造）。

**口径变更**：主结果口径 = 单轨迹（beam-2 降级为 test-time scaling 附录分析）；攻击轮数 5 → 10。
**动机**：v10 RFT+GRPO 复训（08-31 完成 200/200 步）暴露 reward 饥饿——66% 步组内奖励全零（ASR-only + num_generations=8 + 冷启动模型 5 轮 ~7% ASR）；10 轮下轨迹成功率上升 + num_generations 提到 16，双重缓解。

| 臂 | 初始权重 | 训练 | 状态 |
|----|----------|------|------|
| M0 直接 agent | base | 无，评估@10 | 部分完成：采集 survivors 61/664≈9.2%（可作 M0 口径参考，正式 300 条 B 轴 C1 臂覆盖） |
| M1 只GRPO | base | vanilla GRPO@10，**显式 `--loss_type grpo`**（TRL 0.29 默认已是 dapo，不设则对照不干净） | 脚本已写 `exp04_grpo_10turn.sh`，**HOLD 至 B 轴出结果** |
| M2 RFT | base→SFT | 无 RL，评估@10 | 采集收尾中（finalize 脚本：补 28 条 + merge assert 1000 + build SFT）；**SFT HOLD 至 B 轴出结果** |
| M3 RFT+GRPO | M2 merged | vanilla GRPO@10（同 M1 超参） | 待 M2 完成 |

统一超参：num_generations=16，其余同 exp03（lr 1e-5 / β 0.05 / temp 0.9 / 300 步）。

**⚠️ RFT 数据量预警（09-03）**：skill_decide@10skills 协议下 split A 1000 条采集 ASR 仅 **~8-9%**（幸存片 61/664=9.2%，补跑片 3.5-11.1%，平均轮数 9.5-9.7）→ M2 SFT 成功轨迹仅 **~85-90 条**，偏薄；M3 GRPO 组内全零梯度风险高。含义：(a) M2 增益可能有限；(b) 若 B 轴 C2/C3 提升基座 ASR，按新 harness 重采可同步缓解数据量与稀疏问题。

**RFT v3 数据链路（08-31/09-01 写代码，09-02 切 skill_decide@10skills 后实跑）**：
- `scripts/run_rft_collect_conv.sh`：split A（train[1000:2000]）→ base + conv_eval **skill_decide@10skill** 10 轮采集 → `output/rft_collect_conv_10turn/`（支持 `VARIANT`/`TOP_K_SKILLS`/`COLLECT_WORKERS` 环境变量；采集并发 12 路）
- `scripts/build_rft_data_conv.py`：conv 轨迹 → **全轨迹样本**（每成功轨迹 1 条）；assistant 轮取 `actions[].raw`（模型原始输出含 Selection 标记，逐字监督防协议漂移）；`load_candidates` 与 env `_load_skills` 排序逻辑逐行对齐（已验证一致）
- `scripts/rft_sft_conv.sh`：SFT 1-2 epochs（默认 2，防坍缩）→ `rft_sft_conv10turn_e2/`
- ⚠️ 关键发现：swift `default` loss_scale 对 messages 中**所有** assistant 轮算 loss（`swift/loss_scale/base.py:28`），故全轨迹样本 = 每个动作恰好监督一次；v2 的"前缀式展开 + default loss"会把早期动作重复监督 T-i+1 次（过加权早期轮），可能是 v2 坍缩贡献因素
- 协议失配（v2 负结果主因）从源头消除：assistant 文本即在其被监督的真实 conv 上下文中生成

**GRPO"9/9 崩溃"误诊修正（09-01 核实）**：早期失败根因 = 端口冲突（EADDRINUSE 43210 / collective_rpc Address already in use）；v10 run 已在 V100 完成 200/200 步；rollout_v100.log 末尾的 EngineCore died = trainer 正常结束后的良性清理。V100 可用，无需算力银行通道。

## 2026-09-05 进展（B 轴完成定稿 C1,A 轴启动）

- [x] **B 轴三臂完成**:C1 9.33% / C2 5.00% / C3 8.67%（base,test C 前 300）→ **C1 定稿**,A 轴解除 HOLD
- [x] **M2 SFT 启动（09-05）**:`rft_sft_conv.sh` LoRA 2 epochs @GPU3;数据 82 条全轨迹样本(token 长度实测 p50=1027/max=5448,`--max_length` 4096→**6144** 防 2 条长轨迹截断)
- [x] **M2 SFT OOM 修复（09-05）**:batch 2 @6144 在 V100-32GB OOM(step 4/22)→ batch 1 + 梯度累积 8 + `--gradient_checkpointing true`,重跑通过
- [x] **M2 评估完成（09-05,`run_m2_eval_20260905.sh`）**:**M2 10.67%**(32/300,avg 9.30 轮) vs base 9.33%(28/300) → **+1.34pp**,82 条薄数据 RFT 有正收益;merged 权重 `output/rft_sft_conv10turn_e2_merged/`(M3 初始权重就绪)
- [ ] **M1 只 GRPO@10**:启动链 `m1_launch_20260905.sh`(停 8003 → swift rollout base @8004 **max_model_len 24576**——8192 会重蹈 10 轮 input 越界 → `exp04_grpo_10turn.sh` 300 步);`--vllm_server_timeout` 600→1000 对齐超时口径
- [ ] M3 RFT+GRPO:`MODEL=<m2_merged> MODEL_TAG=rft bash scripts/exp04_grpo_10turn.sh`
- [ ] M1/M3 完成后:merge → 评估(C1 协议,test C 300)→ A 轴四臂对比表

## 2026-09-03 进展（B 轴前置决策 + 采集收尾 + C3 实现）

- [x] **协议切换全链路执行（09-02）**：`conv_eval.py` / `env.py` / `plugin.py` / `build_rft_data_conv.py` / `run_rft_collect_conv.sh` 统一 `skill_decide@10skills@top_k=10@max_turns=10`；no_skill 旧采集作废重采
- [x] **超时事故与恢复（详见 LOG 09-03）**：12 路并发下 client timeout 60s 被打穿 → 4/12 片崩溃丢 332 条；所有调用点 timeout 提至 **1000s**（用户指定，含 `vllm_client.py` 默认值）；`recover_rft_collect_20260903.sh` 重切重跑，11/12 子任务恢复
- [x] **上下文越界修复**：input_tokens 14337 + max_tokens 2048 = 16385 超 policy `max_model_len 16384`（差 1 个 token 即崩）→ `start_servers_v100.sh` policy 改 **24576**；`finalize_rft_collect_20260903.sh` 自动重启服务并补跑 rerun_0_0
- [x] **RFT 采集收尾（09-03 运行中）**：finalize 流水线 = 补跑 28 条 → merge（`assert total==1000` 防静默丢数据）→ 构建 SFT `rft_data_conv_10turn.jsonl`；采集 ASR ~8-9%（见上"数据量预警"）
- [x] **C3 滑动窗口实现**：`conv_eval.py` `_ctx_view()` + `--ctx_window N`（保留 system+初始 user，尾部只喂最近 N 轮；完整消息仍照常累积用于反馈构建）；summary.json 加 `ctx_window` 字段
- [x] **B 轴 runner 就绪**：`scripts/run_baxis_ctx_ablation.sh` = C1_full / C2_workmem / C3_window3 × test C 前 300（标准 1000 条评估的严格子集）× 3 并行片，自动汇总 ASR + avg_turns 对比表
- [ ] **B 轴三臂运行 + 判定**：finalize 释放 policy 8003 后启动；结果定 harness → 才放行 M1/M2/M3
- [ ] （B 轴若选 C2/C3）M2 RFT 数据按新 harness 重采；C1 数据留作 M2 回退与 C1 臂口径

## 2026-08-25/26 进展（单流重测 + 基线 + RFT 冷启动）

- [x] **单流重测（08-25）**:beam_agent `--beams 1` 全量 1000(seed=42):no_skill 10.7% / skills_step 15.9% / skills_traj 15.4% → **单流下 skill 已是正资产(+4.7~5.2pp)**;beam 放大 skill 增益(+4.0→+8.6pp)。推翻 conv 系"skill 负资产"结论(口径不同:conv 19% vs 23.2%、beam no_skill 10.7%,不可直接比)
- [x] **Crescendo baseline（08-25）**:`baselines/crescendo.py`,全量 1000 = **1.8%**(18/1000,43.9s/样本),脚本化多轮对 SafeRL 几乎无效
- [x] **TAP baseline（08-25）**:`baselines/tap.py`,受限预算(depth6/branch2/prune1,≤12 调用/样本)全量 = **7.2%**(72/1000)
- [x] **thinking 标签根因修复**:Qwen3 vLLM 输出 ` thinking\n\n response\n\n` 标记(5 字母 think),`_strip_thinking` 旧实现只处理 `</t_h>`;已统一修复 `src/vllm_client.py` 与 `RL4jailbreak/src/vllm_client.py`(cli 实际导入后者!)+ `crescendo.py` 本地清洗
- [x] **RFT 数据（08-25）**:从 beam_pilot 成功轨迹提取前缀式 SFT 样本(对话协议,2175 条)
- [x] **RFT SFT + merge（08-26）**:ms-swift sft + LoRA(注意参数是 `--tuner_type lora` 非 `--lora`),816 步完成 → `rft_sft_merged/`
- [x] **数据隔离修正（08-26,关键）**:原 RFT 种子误用 self_evolve test_prompts(=10k test 集)→ **test 泄漏**;改为 RFT 种子=train[1000:2000](`rft_seed_train1000.json`,A),GRPO 训练=train[0:1000](grpo_data.jsonl,B),评估=test(C),A/B/C 两两不重叠
- [ ] **RFT+GRPO 复训（08-26 运行中）**:`rft_grpo.sh` 500 步/5 轮/ASR-only,初始权重=rft_sft_merged;超参与 exp03 完全一致(唯一变量=初始权重)
- [ ] GRPO 完成后:merge checkpoint → 部署 8003 → eval_conv.sh 5 轮单次/beam-2 → **RFT vs 纯 GRPO(33.9%/44.4%) 对比结论**

### 踩坑记录（可复现性）

- swift rlhf `--vllm_mode server` = **连接外部服务**,rollout 必须用 `swift rollout` 启动(带 communicator 端点),裸 `vllm serve` 报 `404 Not Found`
- MASTER_PORT 默认 29500 段被本机系统预占 → 用 `MASTER_PORT=43210`
- 长时间训练的进程用 `setsid` 启动,避免被会话/进程组清理误杀
- `run_beam_pilot.py --outdir` 传相对路径在后台环境会丢结果 → 一律绝对路径;已加 `--data` 参数支持自定义数据源

## Phase 5: RSI 递归自我改进（后续方向，2026-08-27 记录）

- 详见 `docs/RSI_DESIGN.md`（三轴闭环设计、实验矩阵、风险控制）
- 三轴 = 数据重放&合成 / harness 自进化（skill 库）/ agentic 算法&奖励改进（AHR 自适应）
- 当前 RFT+GRPO 完成后作为 E-RSI-3 基线；之后实现循环编排脚本启动 E-RSI-1/2
- 未启动, 等 RFT+GRPO 链路收尾

## 2026-08-29 — RFT 评估（负面结果）+ GRPO 阻塞记录

- RFT(SFT) 5 轮评估: 单次 6.8% / beam-2 5.1%（< 未训练 3 轮 23.2%）—— beam→conv 协议失配导致 RFT 负收益, 详见 exp/README.md §10.1
- 修正方向: conv 协议成功轨迹直接做 RFT / 减 epochs 防坍缩 / 小规模先验
- GRPO 阻塞: swift 4.3.2 GRPO trainer 在本机 V100 首步 rollout 期稳定崩溃(9/9), 判别实验排除环境查杀(满载/空闲均存活); 待算力银行通道或深挖 swift 崩溃根因
## 2026-09-08 进展（M1 启动 + V100 显存三连排障 + 训练稳定）

- [x] **环境恢复**:4×V100 重新拉起,guard/target/policy 三服务就绪
- [x] **M1 启动链修复**:swift rollout 加 `--torch_dtype float16`(V100 SM7.0 不支持 swift 默认 bf16, 09-05 失败根因)
- [x] **OOM 三连排障**(详见 LOG.md 09-08 节): batch×attention → logits 物化(liger) → SDPA math 回退(xformers patch) → bf16 拒绝(patch 内转 fp16)。新增 `src/v100_attn_patch.py`(经 plugin 加载, 注册表+模块双替换, bf16→fp16 透明转换), `--use_liger_kernel true`, xformers 升 0.0.35(--no-deps)。协议零改动
- [ ] **M1 训练中**:第 5 次启动(19:57)稳定,step 6+/300,ETA ~32-48h(09-09 晚出 checkpoint-300);GPU3 18.4GB 稳定
- [ ] **M1 指标检查点(21:47 设)**:zero-std 实测数据出来后决策——≤50% → M3 照跑;≥80% → DAPO 臂 + C 轴提前
- [ ] M1 完成:merge → C1 协议评估(test C 300) → M3 (`MODEL=<m2_merged> MODEL_TAG=rft`)

### 方案讨论记录(09-08,未定案)
- **SESS 被 AAAI 2027 desk reject**(超 9 页, 格式拒, 内容未经评审)→ 第三章可定位"SESS 框架下的 RL 优化"(09-02 口径本就一脉相承, 实验零改动);待定: 重投目标会议 + 论文归属 A(并入 SESS v3)/B(独立成文+毕业论文串联)。方案 B 下 vs SESS 对比实验(C1)优先级提前
- **用户提议降 num_generations 16→8**:已否决——全零组概率 22%→47%(与提升奖励稠密度初衷相反), 轨迹内串行为主, 提速 <20%;v10 的 66% 零组正是 8 条组的数据
- **用户提议换 target 为 plain Qwen3-4B**:暂缓——PAIR@plain4B 已 91.8% 近天花板, 非对齐 target 的 ASR 科学性弱 + 重做成本约一周 4 卡(B 轴/RFT/M0-M3/baselines 全部);替代阶梯: 等 M1 zero-std 数据 → C 轴 reward 提前 / DAPO 臂 / plain4B 只作第二评估 target(transfer 表)
