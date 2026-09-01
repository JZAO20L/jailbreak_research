# Agentic Jailbreak TODO

> 唯一维护的待办清单。事件记录见 [LOG.md](./LOG.md),代码/脚本/成果说明见各层 README。

## 章节核心目标(第三章)

**设计一套好的 jailbreak-agent harness,并在此基础上做奖励建模与 AgenticRL 训练。**

三条主线:
1. **Harness**:observation 标准化(工具返回结构化)、多轮交互、skill 注入、memory —— 让 Agent 的每轮决策有信息可用
2. **奖励建模**:从 ASR-only 起步,逐步引入过程奖励/效率奖励/多样性(对标 AHR 思路),设计 RL 可用的 reward
3. **AgenticRL 训练**:ms-swift GRPO(单轮 → 多轮),RFT 冷启动(必要时),目标是超越 AutoDAN 19% 参照

> 环境:4×A100-SXM4-80GB;服务 guard=8001 / target=8002 / policy=8003(见 LOG 08-10/08-11)

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

**口径变更**：主结果口径 = 单轨迹（beam-2 降级为 test-time scaling 附录分析）；攻击轮数 5 → 10。
**动机**：v10 RFT+GRPO 复训（08-31 完成 200/200 步）暴露 reward 饥饿——66% 步组内奖励全零（ASR-only + num_generations=8 + 冷启动模型 5 轮 ~7% ASR）；10 轮下轨迹成功率上升 + num_generations 提到 16，双重缓解。

| 臂 | 初始权重 | 训练 | 状态 |
|----|----------|------|------|
| M0 直接 agent | base | 无，评估@10 | 待跑（评估脚本现成，eval_conv.sh） |
| M1 只GRPO | base | vanilla GRPO@10，**显式 `--loss_type grpo`**（TRL 0.29 默认已是 dapo，不设则对照不干净） | 待写脚本（参考 exp03 + num_generations=16） |
| M2 RFT | base→SFT | 无 RL，评估@10 | 数据链路已就绪（见下），待 V100 环境 |
| M3 RFT+GRPO | M2 merged | vanilla GRPO@10（同 M1 超参） | 待 M2 完成 |

统一超参：num_generations=16，其余同 exp03（lr 1e-5 / β 0.05 / temp 0.9 / 300 步）。

**RFT v3 数据链路（08-31/09-01 已写代码，待环境跑）**：
- `scripts/run_rft_collect_conv.sh`：split A（train[1000:2000]）→ base + conv_eval no_skill 10 轮采集 → `output/rft_collect_conv_10turn/`
- `scripts/build_rft_data_conv.py`：conv 轨迹 → **全轨迹样本**（每成功轨迹 1 条）
  - ⚠️ 关键发现：swift `default` loss_scale 对 messages 中**所有** assistant 轮算 loss（`swift/loss_scale/base.py:28`），故全轨迹样本 = 每个动作恰好监督一次；v2 的"前缀式展开 + default loss"会把早期动作重复监督 T-i+1 次（过加权早期轮），可能是 v2 坍缩贡献因素
- `scripts/rft_sft_conv.sh`：SFT 1-2 epochs（默认 2，防坍缩）→ `rft_sft_conv10turn_e2/`
- 协议失配（v2 负结果主因）从源头消除：assistant 文本即在其被监督的真实 conv 上下文中生成

**GRPO"9/9 崩溃"误诊修正（09-01 核实）**：早期失败根因 = 端口冲突（EADDRINUSE 43210 / collective_rpc Address already in use）；v10 run 已在 V100 完成 200/200 步；rollout_v100.log 末尾的 EngineCore died = trainer 正常结束后的良性清理。V100 可用，无需算力银行通道。

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
