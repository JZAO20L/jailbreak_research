# Agentic Jailbreak LOG

> 唯一维护的事件时间线。只记:方向变更/决策、卡点及解决、实验起止+结果链接、环境变更。
> 与 README(状态快照)/TODO(待办)交叉引用,不复制内容。只追加,不改写历史。
> **2026-08-12 起:研究结论(尝试-结果-结论)移入 [REPORT.md](./REPORT.md),LOG 只记事件事实。**

---

## 2026-08-06 — 项目启动,基础 Agent 完成

- 创建 `agentic_jailbreak/`,设计定位:PAIR 框架 + SESS skills + 多轮 Agent(毕业论文第三章)
- 环境:2×A100 80GB;部署 SafeRL(Qwen3-4B-SafeRL, GPU0, 8000)与 Policy(Qwen3-4B, GPU1, 8001)
- 实现:success_checker(规则判定,不依赖 Guard)、Agent Loop、单轮/多轮 Agent、GYM env、memory、rewards、ms-swift plugin
- 冒烟测试(仅少量用例):单轮 3 用例、多轮 2×3 轮,全部被 SafeRL 拒绝(ASR=0%,符合预期——SafeRL 是安全对齐模型)
- 决策:训练阶段不用 Guard(success_checker 规则判 ASR),评估阶段才上 Guard

## 2026-08-09 — 实验条件审查

- 发现 **import bug**:根 `src/`(regular 包)遮蔽 `RL4jailbreak/src`(缺 `__init__.py`)→ `from src.utils import ...` 必挂;阻塞 `eval.py` 与训练 env 路径(冒烟测试用 requests 直连绕过)
- 发现 **2-GPU 约束**:`common.sh` 要求 4/8 卡;GPU 显存被服务占满,RL 训练需分时复用
- 发现 **端口约定不一致**:`common.sh`/`env.py`(guard=8001, target=8002)vs 2-GPU 运行方案(SafeRL=8000 作 target, Guard=8002)
- 结论:纯 Agent ASR 可跑(不需 Guard),但全量基准未执行;详见 TODO P0

## 2026-08-10 — 文档重构

- 按约定整理文档:唯一 TODO(`docs/TODO.md`)、唯一 LOG(本文件)、各层 README(代码/脚本说明+本目录成果)
- 合并:`PREPARATION.md` 环境准备 → 根 README;`PROGRESS.md` 状态 → 根 README,任务 → TODO,事件 → 本文件;两份旧文档已删除

## 2026-08-10 — 环境切换 4 卡 + 修复 import bug

- **环境**:2×PCIe A100 → **4×A100-SXM4-80GB**(全部空闲);采用 common.sh 4 卡布局(guard=8001@GPU0 / target=8002@GPU1 / rollout=8003@GPU2 / train@GPU3),"2-GPU 分时复用"与"端口约定"两项 TODO 随之解决
- **修复 import bug**:给 `RL4jailbreak/src/` 加空 `__init__.py`(与根 `src/` 的 vllm_client 完全一致,已验证无副作用);`agentic_jailbreak`(env/agent/eval/plugin)与 `agentic_rl`(rewards/trajectory_generator)全部 7 组导入通过
- 待办推进:启动服务 → 纯 Agent 全量 ASR 基准

## 2026-08-10 — 服务启动 + 纯 Agent 基准开跑

- 启动服务踩坑与修复:
  - flashinfer-cubin 版本不匹配导致 vLLM 崩溃 → common.sh 加 `FLASHINFER_DISABLE_VERSION_CHECK=1`
  - 裸 `swift`/`vllm` 不在 PATH(venv 未激活)→ common.sh 前置 `.venv/bin`
  - `swift rollout` 是 ms-swift 专用协议,不兼容 VLLMClient → **纯 Agent 评估用标准 vllm 服务 8003**,RL 训练时才切回 swift rollout(已记 TODO)
- 服务就绪:guard 8001 / target 8002 / policy 8003
- **sanity 评估通过**(5 条,单轮):全链路可跑,ASR=0%(符合预期——未训练 Qwen3-4B Agent 攻击 SafeRL),约 20s/条
- **纯 Agent 单轮全量基准启动**:10k test(1000 条)切 4 片并行,后台运行中

## 2026-08-10 — 纯 Agent 单轮基准完成

- **结果**:ASR **4.30%**(43/1000),avg_turns 1.00(10k test,未训练 Qwen3-4B Agent vs SafeRL)
- 4 片并行(250×4)合并;结果 `output/eval_results/pure_agent_single_turn/`
- 该数字是 RL 前的纯 Agent baseline;下一步:多轮 3/5 轮基准 → RL 训练

## 2026-08-10 — 基线对比实验启动(PAIR / AutoDAN)

- 动机:4.3% 是否"太低"需参照系——经典攻击方法在 SafeRL(Guard 判定)上表现如何
- 诊断(10 条):直发有害 prompt(无 Agent)Guard ASR = **0/10**,规则判定 2/10 → SafeRL 本身极硬,Agent 4.3% 已高于无 Agent 基线;规则判定比 Guard 宽松约 2 倍
- 判定口径与 Agent 评估一致(仅 Unsafe 算成功)
- sanity:PAIR 10 条 **0/10**
- 100 条全量:pait/autodan 后台运行中(与 3 轮 Agent 评估共享服务)

## 2026-08-11 — 基线结果 + 3 轮评估修复

- **PAIR 100 条:ASR 0%**(0/100)——SafeRL 对 PAIR 完全免疫,与直发基线一致
- **AutoDAN 100 条:ASR 19%**(19/100)——DAN 模板 + 遗传搜索在 SafeRL 上反而有效,显著高于未训练 Agent(4.3%)
- **3 轮评估 3/4 完成**:part0=7.6% / part1=8.4% / part2=10.0%(合并约 8.7%);part3 因 **Guard max-model-len=4096 上下文溢出**崩溃(多轮长 prompt 超限)
- 修复:Guard 重启为 `--max-model-len 16384`;3 轮 4 片重跑中(注:part0-2 部分样本可能因溢出被计为失败,重跑统一口径)
- 启示:RL 训练的目标是让 Agent 超越 AutoDAN(19%),而非仅超越自身 4.3%

## 2026-08-11 — 3 轮评估完成(修复口径)

- **纯 Agent 3 轮:ASR 9.10%**(91/1000,avg_turns 2.89)——Guard 修复(16384)后 4 片完整重跑,无一崩溃
- 多轮增益:单轮 4.30% → 3 轮 9.10%(约 2.1 倍)
- 对比:直发 0% / PAIR 0% / Agent 单轮 4.3% / Agent 3 轮 9.1% / **AutoDAN 19%**
- 5 轮基准已启动(预计 ~7-8h)

## 2026-08-11 — Observation 标准化 + Memory 瘦身(代码改进)

- **问题**:原实现 history 只给模型看 skill 名 + guard 标签,adapted_content/target_response 内容完全不可见 → Agent 反思是"盲"的(解释了多轮增益递减 38/30/23)
- **改动**(env.py / eval.py / memory.py):
  1. history 增加 `refused` 结构化字段(复用 success_checker.check_refusal)
  2. `format_observation_for_model` 历史展示加:Guard 标签 + Target refused + Attack 摘要(150 字)+ Target reply 摘要(150 字)
  3. memory 瘦身:只存 `{prompt, skill_idx, skill_name, success, summary}`,不存完整 action/skill 内容;修复 update 存 `action` 而 agent.py 读 `summary` 的字段不匹配 bug
- **影响**:代码改动只影响新运行的评估;已在跑的 5 轮/10 轮仍是"盲反思"口径(v1)。需重跑 v2 子集对比"有信息反思 vs 盲反思"(见 TODO)

## 2026-08-11 — ms-swift 4.3.2 GRPO 链路修复(Phase 3 前置)

- **关键发现**:ms-swift 4.3.2 的 gym Env 接口是 **async**(`reset(config: RolloutInferRequest) -> (obs, info, system_msg)`、`step(messages) -> (next_obs, reward, done, info)`),原 plugin.py 的同步 wrapper 完全不兼容;`--env`/`--env_config` CLI 已不存在
- **重写 plugin.py**:`JailbreakEnvMS(Env)` 异步包装同步 JailbreakEnv;reset 从 `data_dict['prompt']` 取 prompt;step 解析最后一条 assistant 输出为 action,`asyncio.to_thread` 跑同步 LLM 调用;注册 `envs['jailbreak_env']` ✅ 已验证
- **use_gym_env 语义**:trainer 直接用 env 的 `total_reward` 作为 GRPO 奖励,reward_funcs 可空
- **修 exp 脚本**(exp01/02/03):`--env`→`--gym_env jailbreak_env` + `--use_gym_env true`;删 `--env_config`;删 `reward_funcs`;`TRAIN_GPUS` 4,5,6,7→3(4 卡);补 `--dataset`(grpo_data.jsonl,1000 条,含 prompt 字段)
- 待办:切换 rollout 为 `swift rollout` 后做小型训练链验证

## 2026-08-12 — 多轮曲线完成 + v2 对比启动

- **5 轮全量(v1):ASR 11.10%**(111/1000,avg 4.66 轮)→ 多轮曲线 4.30 → 9.10 → 11.10(收益递减)
- **10 轮子集(v1,前 200 条):12.50%**(25/200,avg 9.21 轮)
- **v2 3 轮子集(有信息反思,前 200 条):6.00%**(12/200)——与 v1 3 轮全量 9.1% 不可直接对比(**子集混杂**:可能前 200 条更难,也可能 v2 观察干扰基础模型)
- 决策:启动 **v2 3 轮全量**(与 v1 9.10% 同口径),用干净对比判定"有信息反思"的真实价值
- 注:若 v2 全量 ≤ v1,说明观察信息过载反而干扰未训练模型——该发现将直接影响 RL 训练的 observation 设计

## 2026-08-12 — Agent 版本矩阵:设计-实现-验证

- **设计 5 个变体**(harness 设计空间消融,见 TODO 版本矩阵):
  - AV2 No-Skill:无 skill 库自由生成(消融 skill 价值)
  - AV3 Select-Only:只选 skill 用原始内容(消融适配价值)
  - AV4 Beam-2:每轮 2 候选取最优(消融测试时搜索)
  - AV5 Similarity-Retrieval:按 prompt 关键词相似度检索(消融检索方式)
- **实现**:新建 `src/agent_variants.py`(NoSkillAgent/SelectOnlyAgent/BeamAgent);env.py 支持空适配回退 skill 内容(AV3)、beam 列表步进(AV4)、retrieval_mode(AV5);eval.py 加 `--variant`/`--retrieval_mode`/`--beam_width`
- **验证**:3 变体 sanity(2 条)全通过
- **实验启动**:4 变体 × 3 轮 × 前 100 条(2 片并行,8 进程)后台运行中;与 AV1(v2)同子集可比

## 2026-08-12 — 变体矩阵完成 + 重大发现

- 5 变体前 100 条 3 轮全部跑完:**AV2 No-Skill 18%** 远超 AV1 select+adapt 5%,几乎追平同批 AutoDAN 19%;AV3 select-only 3%、AV4 beam 8%、AV5 sim-retrieval 6%
- 结论详见 REPORT 单元 4;待办新增:AV2 全量验证 + RL 路线选择

## 2026-08-12 — v2 全量结论 + 组合变体启动

- **v2 3 轮全量 = 7.00%**(70/1000),**低于 v1 9.10%**——"有信息反思"对未训练模型反而有害(全量确认);REPORT 单元 3 定稿,RL observation 决策:精简优先
- 新增组合变体 **AV6 = NoSkill + Beam-2**(自由生成 × 每轮多候选);已注册并 sanity 通过
- 三组实验启动:AV2 全量(1000,3 轮)、AV6 前 100(3 轮)、AV2 前 100 5 轮——验证"最优组合"与"多轮对自由生成是否还有增益"

## 2026-08-13 — 最优 harness 确定

- **AV2 No-Skill 全量 3 轮 = 23.20%**(232/1000,avg 2.71 轮)——**超过 AutoDAN 19%**
- **AV6(NoSkill+Beam-2)前 100 = 35.00%**(35/100,avg 2.44 轮)——接近 AV2 两倍,Beam 是最大杠杆
- AV2 5 轮前 100 = 22.00%(多轮增益有限,远不及 beam)
- **决策**:RL action space 采用 AV6 式(自由生成 × beam),observation 精简式;**下一步:AV6 全量验证 + 小型 GRPO 训练链验证**

## 2026-08-13 — 核心方法形态决策:对话式(Agentic)harness

- **方向修正**:论文核心方法必须是 Agent 式**多轮对话**(消息累积),而非无状态重建——用户指出后重构
- **问题**:原 eval 每轮重建完整 prompt(历史塞进单条 user message),与 RL 路径(ms-swift GYM 消息累积)形态不一致,eval 数字可能不迁移
- **实现**:新建 `src/conv_eval.py`——对话式评估器:system → user(初始观察)→ assistant(动作)→ user(环境反馈)→ ...每轮 1 次 policy 调用,反思隐含在对话中;支持 no_skill / no_skill_beam(beam 宽度)/ select_adapt 变体;评价复用 env._evaluate(sanity 通过)
- **eval.py** 增加 `--mode conversational|stateless`(默认 conversational)
- **前 100 条对比实验启动**(3 变体 × 2 片:conv_no_skill / conv_beam / conv_select),与无状态数字(AV2 18%、AV6 35%、AV1 5%)对照——**之后主方法数字以对话式为准,无状态数字降级为对照**

## 2026-08-13 — 对话式 RL 路径对齐 + 实验脚本

- **plugin.py 重写为对话式**:reset 返回初始消息(prompt+候选,按 variant),step 返回**最小化环境反馈**(Guard+refused+Target reply 摘要),与 conv_eval 同一套消息协议;env_config 支持 variant(默认 no_skill)
- **新脚本**:`scripts/eval_conv.sh`(对话式评估:切分并行+合并)、`scripts/chain_test_grpo.sh`(GRPO 链冒烟:100 条、30 步、report_to none)
- **链测试数据**:`output/grpo_chain_test.jsonl`(100 条)
- 等待:conv 前 100 结果 → 确定主配置 → 对话式全量 → **切 rollout 为 swift rollout 跑 GRPO 链验证**

## 2026-08-13 — 对话式前 100 完成:主方法配置确定

- **conv_no_skill:25.00%**(vs 无状态 18%,**+7pp**——消息累积让反思真正有效)
- **conv_no_skill_beam-2:35.00%**(与无状态 beam 持平,35% 收敛)
- **conv_select_adapt:5.00%**(与无状态持平——skill 注入恒为负资产)
- **决策**:RL action space = 对话式 no_skill(单动作训练),评估 beam-2 放大;REPORT 单元 6
- **对话式全量(no_skill,1000 条)已启动**——RL 基线

## 2026-08-13 — 对话式全量完成 → RL 链验证启动

- **conv_no_skill 全量 3 轮 = 23.20%**(232/1000,avg 2.74 轮)——RL 基线确认(评估时 beam-2 可放大至 ~35%)
- 注:全量口径 conv 与无状态同为 23.2%(前 100 上 conv 25% > 18%,子集差异)
- **服务切换**:清理残留 EngineCore 后,8003 改为 `swift rollout`(GPU2);Guard 重启(16384);就绪后跑 chain_test_grpo.sh
- **修复 plugin bug**:GYMScheduler 传入完整对话,原实现对每条 assistant 消息重评估——改为只评估最后一条
- **REPORT 单元 7**:奖励建模三阶段草案(R1 ASR-only → R2 过程奖励 → R3 效率)

## 2026-08-13 — GRPO 链验证排障(进行中)

- **修 plugin 数据流 bug**:ms-swift 预处理把 jsonl 行剥成只剩 `messages` 列(`prompt` 键消失)→ plugin.reset 改为从请求 messages 提取有害 prompt(已修,验证通过)
- **修 plugin 步进 bug**:GYMScheduler 传入完整对话,只评估最后一条 assistant 动作
- **NCCL 竞态(未根因)**:trainer 端 `ncclCommInitRank` 间歇性报 `no CUDA-capable device`;直调 python CLI 时多数通过,`generation_batch_size` 整除约束需满足;疑似**失败的 trainer 会卡死 rollout 服务**(8003 无响应),后续尝试连环失败 → 对策:每次训练前重启 rollout
- **服务经验**:杀 API server 后 EngineCore 会成孤儿继续占显存,需按 PID 清理;误杀 guard 引擎需重启
- 状态:guard/rollout 重启中,就绪后重跑 chain_test_grpo.sh(已内置 NCCL 重试 + 服务器重启前置)

## 2026-08-13 — 🎉 GRPO 训练链验证通过(根因找到)

- **NCCL 假失败的真正根因**:脚本在 `source common.sh` **之后**才声明 `NUM_GPUS=4`,而 common.sh 默认按 8 卡布局设置 `TRAIN_GPUS="4,5,6,7"`;后面的 `TRAIN_GPUS="${TRAIN_GPUS:-3}"` 不会覆盖已设变量 → trainer 的 `CUDA_VISIBLE_DEVICES=4,5,6,7` 指向不存在的 GPU → NCCL 报 `no CUDA-capable device`。修复:`NUM_GPUS=4` 移到 source **之前**(chain_test + exp01/02/03 全部修复)
- 另发现:失败 trainer 调 `/init_communicator/` 后挂掉会卡死 rollout 引擎(dist store 301s 超时)——训练前需保证 rollout 健康
- **验证成功**:直调 python CLI + 修复布局后,5/5 训练步完成,`num_turns=3.0`、`rewards/gym_reward` 流入 GRPO、runtime≈116s
- 全链路就绪:数据集 → swift rollout → gym env → target/guard → reward → GRPO;REPORT 单元 5 定稿
## 2026-08-16 — 正式训练启动:exp01 完成

- **服务启动坑**:`start_servers.sh` 未在 source common.sh 前声明 `NUM_GPUS`,默认走 8 卡布局(rollout=2,3+TP2、train=4-7)→ 用 `NUM_GPUS=4` 显式启动
- **exp01 单轮正式训练**:dialogue no_skill,500 步,16 samples/step(2×8),~17.6s/step,**2h26m 完成**(checkpoint-500,LoRA);补 `--generation_batch_size 16`(链测试同款路径,默认值语义不同);swanlab 云端链路验证可用(账号 WiseStar_Jacob)
- **评估方式变更(重要)**:eval_conv.sh 的 VLLMClient 需要标准 vLLM `/v1/models`,而训练后 8003 是 swift rollout(仅 `/infer/`)→ **swift export --merge_lora 合成完整权重 + vllm serve**(`output/single_turn_agent/merged-500`)
- **exp01 单轮评估 = 10.00%**(100/1000,avg 1.0 轮);同口径未训练 base 单轮 = **9.00%**(90/1000)(8004 并行评估)→ RL +1pp,增益有限,与先前"未训练模型 3 轮 23.2% 中大部分来自轮数十多轮反思"一致
- 等 beam-2 评估(训练模型,1 轮)结果;后续 exp02(3 轮)/exp03(5 轮)从 base 独立训练

## 2026-08-16/17 — exp02(3 轮 RL)完成

- **exp02 训练**:dialogue no_skill,max_turns=3,500 步,**6h27m**(~46s/步,3 轮对话成本约为单轮 2.6 倍);`num_turns≈2.7-3.0`,reward 流入 GRPO
- **exp02 3 轮评估 = 25.80%**(258/1000,avg 2.69 轮)vs 未训练 3 轮基线 **23.20%** → **+2.6pp,RL 首次超越基线**;前 100 结果已备份(conv_no_skill_3turn_front100_backup,25%)
- 评估链路复用:merge_lora → vllm serve(vllm 服务用 GPU2:8003);swift rollout 需 `--load_args false` 否则默认加载缓存 args 会把 port 改回 8004(本次新坑)
- 待办:exp02 beam-2 评估 → exp03(5 轮)训练

## 2026-08-17 — 🎉 exp03 完成:三组 RL 训练 + 评估全部收官

- **exp03 训练**:max_turns=5,500 步,**10h53m**(~78s/步);rollout 端口坑:swift rollout 偶发忽略 `--port`(解析出默认 8004,根因未明),exp03 改用 `ROLLOUT_PORT=8004` 环境覆盖绕过
- **exp03 评估**:no_skill 5 轮 = **33.90%**(339/1000,avg 4.09);beam-2 = **44.40%**(444/1000,avg 3.72)
- **最终矩阵**(全量 1000,对话式同口径):
  | 配置 | 单次 | beam-2 |
  |---|---|---|
  | 未训练 1 轮 | 9.0% | - |
  | RL 1 轮 | 10.0% | 17.1% |
  | 未训练 3 轮 | 23.2% | ~35%(前100) |
  | RL 3 轮 | 25.8% | 35.2% |
  | RL 5 轮 | **33.9%** | **44.4%** |
- 结论:RL 增益随 max_turns 放大;5 轮单次 33.9% > 23.2% 目标、beam 44.4% > 35% 目标;多轮训练曲线 10.0 → 25.8 → 33.9
- 服务状态:guard/target/policy(exp03 merged)仍运行;待用户决策下一步(未训练 5 轮同口径基线/多 benchmark/报告)

## 2026-08-18 — Skill 调用时机实验:LLM 自决首次超越 no_skill

- **动机**:skill 注入此前恒为负资产(select+adapt 5% vs no_skill 25%)——假设问题出在"强制调用"而非 skill 本身
- **新变体**(conv_eval.py):`skill_once`(轨迹只调一次,turn 1 选后不再给候选)/ `skill_every_turn`(每轮重选,即旧 select_adapt)/ `skill_decide`(每轮候选可选,输出 Selection 标记则用 skill,否则自由攻击)
- **结果**(未训练 base,前 100,3 轮):
  | 变体 | ASR | 结论 |
  |---|---|---|
  | no_skill(对照) | 25.0% | - |
  | skill_once | 9.0% | 强制调用(即使只一次)仍负资产 |
  | skill_every_turn | 6.0% | 同旧 select_adapt |
  | **skill_decide** | **28.0%** | **+3pp,首次有 skill 变体超越 no_skill** |
- 解读:伤害来自"强制指定候选"(不对齐时锁死到弱 skill);给 LLM 自主权后它可以忽略或选择性使用 skill → 正收益;待办:全量(1000)验证 + 动作使用率分析(需记录每轮是否选 skill)

## 2026-08-18 — skill_decide 全量验证:前 100 的正收益不成立

- **skill_decide 全量(1000,3 轮)= 19.00%**(190/1000,avg 2.75)< **no_skill 全量 23.20%**(-4.2pp)
- 前 100 的 28% vs 25% 是子集噪声,主方法口径下 skill 注入(即使 LLM 自决)**仍为负资产**,但比强制调用(once 9%/every_turn 6%)好得多(-4pp vs -18pp)
- 修正结论:伤害主因仍是"候选列表面向 top-5 quality 全局最优,与具体 prompt 不对齐"(54 skills 中多数 quality=0,top-5 实际是少数高质+保底);LLM 自决只能部分规避
- 可选后续:动作使用率分析(需记录每轮 Selection 与否)或按 prompt 相似度检索(similarity 模式)重测 skill_decide —— 若检索对齐问题修正,skill 可能转正

## 2026-08-19 — 文档整理

- TODO.md:新增"Skill 调用时机消融(08-18)"条目(含可选后续:similarity 检索重测、动作使用率分析);实验矩阵更新(E1/E2/E3 ✅ ,A1 ✅ 结论 no_skill 23.2% 优于一切 skill 注入)
- exp/README.md:4.1b 增加 skill 时机消融表(08-18,含全量口径)
- 服务状态:guard=8001 / target=8002 / policy=8003(exp03 merged)/ base=8004(GPU3),四卡满载,无运行任务

## 2026-08-25 — 单流重测 + Crescendo/TAP 基线 + thinking 标签根因修复

- **单流重测**(beam_agent `--beams 1`,全量 1000,seed=42,与双流同源):no_skill **10.7%** / skills_step **15.9%** / skills_traj **15.4%**
- 结论:skill 在单流下已是正资产(+4.7~5.2pp);beam 与 skill 有协同(beam 增益 +4.0→+8.6pp);conv 系旧结论(19% vs 23.2%)与 beam runner 口径不可直接比(no_skill 基线 23.2% vs 10.7%)
- **Crescendo baseline 实现并全量**:`baselines/crescendo.py`(多轮渐进脚本),全量 1000 = **1.8%**
- **TAP baseline 实现并全量**:`baselines/tap.py`(树搜索+剪枝,受限预算 depth6/branch2/prune1),全量 1000 = **7.2%**
- 结果落盘:`baselines/experiments/{crescendo,tap}_asr/README.md`(配置+结果+复现命令+对照表)
- **根因修复(影响全项目)**:Qwen3 vLLM 输出 ` thinking\n\n response\n\n` 思考标记(5 字母 think,非 `<thinking>`),`_strip_thinking` 只处理 `</t_h>` 旧格式;且 `baselines/cli.py` 因 sys.path 顺序实际导入 `RL4jailbreak/src/vllm_client.py`(未修副本)。两处均已统一修复 + crescendo 本地清洗
- RFT 数据提取:`build_rft_data.py`(成功轨迹→前缀式对话 SFT 样本,2175 条)
- 服务状态:8001/8002/8003(base policy)/8004 依次部署

## 2026-08-26 — RFT 冷启动 + GRPO 复训(进行中)+ 数据隔离修正

- RFT SFT:ms-swift sft + LoRA(`--tuner_type lora`,816 步/3 epochs)→ merge → `rft_sft_merged/`
- **GRPO 启动踩坑链**:① `--vllm_mode server` 需外部 `swift rollout` 服务(裸 vllm serve 404);② MASTER_PORT 29500 段被系统预占(EADDRINUSE)→ 43210;③ 用 setsid 防进程组误杀;④ 日志误判死亡(grep 未匹配 `-m swift.cli.rlhf` 模块名格式)
- **数据泄漏发现与修正(关键)**:原 RFT 种子= `self_evolve test_prompts.json` = **10k test 集**(1000/1000 重合)→ SFT 见过评估集输入,评估会虚高。修正:RFT 种子=train[1000:2000](A)、GRPO 训练=train[0:1000](B)、评估=test(C),A/B/C 两两隔离
- train 种子集上单流 beam 重跑(全量 1000):no_skill 9.0% / skills_step 15.0% / skills_traj 12.9%(成功轨迹 369 条→重建 RFT 数据)
- RFT+GRPO 复训:500 步/5 轮/ASR-only/lr=1e-5,初始权重=rft_sft_merged(新 train 种子版),超参与 exp03 一致
- 服务状态:guard=8001(GPU2) target=8002(GPU3) policy=8003(GPU1) swift rollout=8004(GPU0)

## 2026-09-01 — GRPO 阻塞误诊修正 + RFT v3 代码 + 10 轮实验矩阵定稿

- **"swift/V100 首步崩溃 9/9"为误诊**:① 早期 9 次失败(v0-v8)根因 = 端口冲突(MASTER_PORT 43210 EADDRINUSE、collective_rpc "Address already in use"——系统预占段位问题,非 GPU/swift 问题);② v10 run(08-31 10:55 起)已正常跑完 200/200 步,checkpoint-200 落盘;③ rollout_v100.log 19:48 的 "EngineCore died" = trainer 正常结束调 /close_communicator/ 后的良性退出噪音。**结论:V100 可用于当前 agentic 训练,无需外部算力通道**
- **v10 训练质量问题(新瓶颈)**:reward 均值 0.021,66% 步 frac_reward_zero_std=1.0(组内全零→零 advantage→空转);根因 = ASR-only + num_generations=8 + RFT 冷启动 5 轮 ~7% ASR → 全零组概率 ~56%
- **实验矩阵决策(与用户确认)**:攻击轮数 5→10;beam-2 取消为主口径(降级为 test-time scaling 附录);先跑 M0 直接agent / M1 只GRPO / M2 RFT / M3 RFT+GRPO,DAPO 臂暂缓(dynamic_sample/epsilon_high 已确认 swift 4.3.2 支持,后续直接加);num_generations 8→16;M1 必须显式 `--loss_type grpo`(TRL 0.29 默认 dapo)
- **RFT v3 代码落地**(conv 协议原生,替代 beam 来源的 v2):`run_rft_collect_conv.sh`(split A 采集)→ `build_rft_data_conv.py`(全轨迹样本)→ `rft_sft_conv.sh`(1-2 epochs);合成 fixture 离线验证通过
- **swift loss 语义发现**:default loss_scale 对多轮 messages 的所有 assistant 轮算 loss → 全轨迹样本每动作监督一次(正确);v2 前缀式展开会把早期动作重复监督 T-i+1 次(过加权),记录为 v2 负结果的候选贡献因素
- 当前环境无 GPU(待用户切换环境继续),M0-M3 全部待跑

## 2026-09-02 — 方法口径定稿:PAIR+10-skill(LLM 自选) + RFT 采集重启

- **方法定稿(用户决策,覆盖此前 no_skill 主配置)**:agent 核心 = **PAIR 骨架 + Skills,LLM 自选**(`skill_decide`),与 SESS 一脉相承;后续核心实验(矩阵 M0-M3/RFT/评估)统一使用 **10-skill 精选库** `exp/skill_asr_sweep/seed_skills_top10.json`
- 代码切换:① `plugin.py` DEFAULT_ENV_CONFIG → skill_decide/top10/top_k=10/max_turns=10(⚠️ 顺带修复隐患:原默认 max_turns=5,exp04@10 轮若不传 env_config 会静默跑 5 轮);② `conv_eval.py` 轨迹落盘新增 `actions[].raw`(模型原始输出含 Selection 标记)——RFT 监督目标必须是原始动作文本,否则 skill_decide 的选择标记学不到,协议漂移;③ `build_rft_data_conv.py` 支持 --variant/--skills-path(复刻 env 质量池排序,与 env 候选顺序逐项核对一致);④ `run_rft_collect_conv.sh` 切 skill_decide 协议
- **no_skill 协议 RFT 采集中止**(16:59 启动,按新口径协议不符,用户确认杀掉重采);skill_decide@10skills 版本重新全量采集(split A, 4 并行, ~3-5h)
- 风险记录:A1 消融中 skill_decide@54库 = 19% < no_skill 23.2%(负资产);精选 10-skill 消除"候选不对齐"根因,预期反转为正,待采集 ASR 与 M0/M1 对比验证
- 三服务 8001/8002/8003 健康

## 2026-09-03 — 采集并发事故与修复(深夜自动恢复)

- 12 路并发采集至 ~9h 时 4/12 part(0/1/8/11)崩溃:`openai.APITimeoutError`——根因 = client 超时硬编码 60s(conv_eval.py:195 policy / env.py:138,146 target+guard),12 路争抢下单请求超 60s;结果仅结束时落盘 → 4 part 全损(332 条需重跑)
- 修复:① client 超时统一 1000s(conv_eval.py / env.py / agent.py 显式调用点 + src/vllm_client.py 默认值 120→1000;working_memory 保持 30s——summarizer 侧调本就有规则式回退,短超时是有意设计);② 采集脚本 workers 可调(COLLECT_WORKERS,默认 12);③ `recover_rft_collect_20260903.sh` 全自动恢复:等 8 路存活退出 → 4 崩溃 part 重切 12 子任务补采 → 合并(assert 总数=1000)→ 构建 SFT 数据
- 经验:多路并发采集必须先核对 client 超时与预期单请求延迟(10 轮×thinking 生成下单请求可远超 60s);results 建议增量落盘防全损
- 预期:存活 8 part ~06:00 完,补采 ~09:30 完,SFT 数据上午产出

## 2026-09-08 — M1 启动 OOM 三连排障（V100 10轮GRPO 显存治理）

- **根因 1 (batch×attention)**: 10 轮轨迹训练 seq 最长 ~18.8k，SDPA attention 显存 O(seq²)：per_device 2 首步 OOM 55.56GiB → per_device 1 + grad_accum 8（有效 batch 不变）+ gradient_checkpointing
- **根因 2 (全词表 logits 物化)**: completion 段 ~15k token × 152k vocab，liger 前进程 29GB→10.7GB 证实。`pip install liger-kernel`(0.8.2) + `--use_liger_kernel true`，LigerFusedLinearGRPOLoss fused linear+CE 不物化 logits。swift 自带分块路径 `dynamic_num_samples` 仅 per-turn 切分触发，单轨迹协议用不上
- **根因 3 (SDPA math 回退)**: V100 SM7.0 上 PyTorch SDPA 的 flash(SM80+)/mem-efficient(SM75+) 全不可用 → math 后端完整物化 [B,H,M,M]（18.8k×32 heads ≈ 21GB/条）。修复 = `src/v100_attn_patch.py`（经 plugin.py 加载）重定向到 xformers mem-efficient（cutlass fmha 原生支持 SM70，显存 O(seq)）
  - ⚠️ 踩坑 1: 只替换 `sdpa_attention` 模块属性不够——modeling_qwen3 运行时经 `ALL_ATTENTION_FUNCTIONS['sdpa']` 注册表分发，必须同时替换注册表项
  - ⚠️ 踩坑 2: bias 必须 stride-0 `expand` 到 head 维（xformers 0.0.35 要求精确同形），不能 materialize
  - ⚠️ 踩坑 3: trainer 的 no_grad old/ref logps pass 按 config dtype 走 **bf16**（`--fp16 true` 的 autocast 只覆盖 compute_loss），cutlass 拒绝 bf16 → patch 内透明转 fp16 计算再转回
  - ⚠️ 踩坑 4: repo 预装 xformers 0.0.28 系 torch 2.5 编译，torch 2.10 下 CUDA 扩展不加载 → `pip install --no-deps xformers==0.0.35`
  - patch 离线验证: fp16/bf16 × causal/4D-mask 全部误差 ≤ bf16 舍入级（0/0.0156），梯度回传正常，stride-0 bias 峰值 47MB
- 实验协议零改动: 10 轮 / num_generations=16 / β=0.05 / 300 步 / loss_type=grpo 原样
- 经验: 每轮 OOM 先看 traceback 落点（attention vs logits）+ 申请显存大小反推张量形状，不要凭直觉连续调参

## 2026-09-08 — M1 第 5 次启动成功 + 架构/方案讨论

- 21:10 step 6/300 稳定(GPU3 18.4GB vs 修复前 26.6GB+16GB 峰值), 四卡: GPU0 guard 8001 / GPU1 target 8002 / GPU2 swift rollout 8004(24.5k ctx) / GPU3 trainer(LoRA+liger+xformers patch)
- 并发结构核实: 训练 rollout = 16 路并发×10 轮串行/条, vLLM 服务端 max_num_seqs=256(默认) 远未打满; GPU0/1 的 0%↔100% 尖峰是轨迹内三服务串行接力的流水线气泡, 非排队。瓶颈=轨迹内串行, 单纯加训练卡收益有限; 加卡应加 target/guard 副本
- 方案讨论(详见 TODO.md 09-08 节): ①降 num_generations→否决(零组概率翻倍) ②换 plain 4B target→暂缓(科学性+一周重做成本, 替代=C轴/DAPO/plain4B 仅作评估 transfer) ③SESS desk reject 后第三章定位讨论(方案 A/B 未定)
- 决策检查点: M1 前 10-20 步 frac_reward_zero_std 实测值出来后定 M3 是否照跑(≤50%)或转 DAPO+C 轴(≥80%)

## 2026-09-09 — 环境迁移到 4×A800-80GB + 全量重跑启动

- 机器从 4×V100-32GB 换成 **4×A800-80GB(SM80)**;`output/` 不入库 → M0-M3/B 轴/RFT 数据/Ch1 checkpoint **全部丢失**,经用户确认全部重跑重采
- 环境重建改为 requirements 驱动(临时环境):新增根 `requirements.txt` + `requirements.lock.txt`(248 行) + `docs/cookbook/bootstrap_env.sh`(幂等,自动建 `/home/tiger/jailbreak_research` 符号链接兼容 59 处硬编码 PROJECT_ROOT);版本逐字复刻 vllm 0.18.0 / ms-swift 4.3.2 / trl 0.29.1 / torch 2.10.0 / transformers 4.57.6
- 模型按 cookbook 新约定落 `model/`(Qwen3-4B 7.6G + Qwen3Guard-Gen-4B 8.3G);`.gitignore` 补 `/model/`(原只忽略 `models/`);Xet 通道走 hf-mirror 会 401,需 `HF_HUB_DISABLE_XET=1`
- V100 补丁链全部失活:`v100_attn_patch` 自带 SM<8 守卫;bf16 原生可用 → fp16 发散根因消失,plain-4B+5 轮的"恢复臂"不再必要
- 新增 `scripts/start_servers_a800.sh`(bf16,guard/target 16384 ctx,policy 32768);`common.sh` 三处改动:模型路径自动探测 `model/`、**TARGET_MODEL 默认改 plain Qwen3-4B**、SKILLS_PATH 默认指向 10-skill 精选库
- 依赖漂移坑:fastapi≥0.116 的 `_IncludedRouter` 让 vllm 0.18 的 Prometheus 中间件在**每个请求**抛异常 → `/health` 和 `/v1/*` 全 500,三服务全哑。升 instrumentator 8.x 无用(要求 starlette≥1.x)。解法=pin `fastapi>=0.115,<0.116`(已入 requirements.txt)
- 工程坑:`common.sh::wait_for_server` 用裸 `curl -s` 判健康,curl 仅在连不上时非零 → **HTTP 500 被误判就绪**;已改为显式判 200。本次真的踩中才暴露
- `eval_conv.sh`:默认 turns 3→**10**、top_k 5→**10**(对齐 09-02 定稿,漏传参数不再静默跑错协议);新增 `EVAL_WORKERS`(A800 用 12 路,M0 300 条约 40min);SLICE_DIR 加 workers 标签并清理派生 part*(换并发数会双计)
- **协议解析 bug 修复(影响 skill_decide 全部历史测量的解释)**:`conv_eval.py::parse_action_text` 与 `env.py::parse_action_from_completion` 的 `Adapted Strategy:\s*\n` 要求换行,模型写成同行即失配 → 回退成"整段原始输出"当攻击内容,`Selection: Skill N` 元信息被原样发给 target,污染攻击串且使"本轮是否用了 skill"在文本层不可判别。两处同步放宽+剥除标记,四组用例交叉验证两解析器逐字一致
- **M0 闸门结果(plain-4B target,skill_decide@top10,10 轮,C1,test C 前 300)= ASR 54.0%(162/300),avg_turns 5.98,45% 烧满 10 轮**;93% 动作带 Selection(base 策略选择性退化为"无条件全选")。含义:未饱和仍有 46pp 空间;16 条组全零概率 ~2.7e-6 → v10 的奖励饥饿问题消失
- 数据资产脚本化:新增 `scripts/build_splits.py`(A=train[1000:2000]/B=train[0:1000]/C=test,三段两两不重叠写成断言)
- 第一章 PAIR 91.8% 证伪(详见下节)连带推翻"因 PAIR 已近天花板故不换 target"的论证;决策:训练仍用 plain-4B,**评估补 SafeRL 当难目标/transfer**
- 框架统一:Ch1 从 TRL 直调迁到 **ms-swift**(理由:Ch3 多轮 gym env 只有 swift 有;单轮经 `swift.rewards.orms` 自定义 ORM 可表达 AHR λ;`swift.pipelines.rlhf_main` 是纯 Python 入口 → 单文件过程式脚本)。已验证契约:注册表存**类**并以 `cls(args=args)` 构造、以 `func(completions, **kwargs)` 调用、数据列经 `RowPreprocessor.rows_to_batched` 透传(`solution` 一定到,自定义列不保证)
- 新增 `RL4jailbreak/experiments/e2e/e2e_ahr_grpo.py`(单文件过程式:服务检查→数据→GRPO→merge→评估→落盘;评估**同时**出 official/legacy 双口径 ASR + 直发对照 + 实测每样本 API 调用数)
- 冒烟→全链路实测通过:swift GRPO 2/2 步(峰值 40.1GiB,ORM 入 GRPO);真实 500 步/1000 条已启动(step16 ETA ~1h41m,grad_norm 0.0099 / kl 0.0017 / reward 0.066→0.144 上升 / reward_std 非零)
- 同进程 train→eval 显存坑:trainer 不释放显存导致评估引擎按固定 util 申请被拒(free 31/79GiB vs 0.6=47.6GiB);`serve_policy` 改为按 `nvidia-smi -i <gpu>` 真实空闲量自动下调 util,不足则提示分两步跑
- GPU 布局(三阶段):Ch1 独占 = GPU0 target(+judge)/GPU1 guard/GPU2+3 训练;Ch3 独占 = 现状 0/1/2 服务 + GPU3 trainer;两章并行 = GPU0 guard+target 各 0.4 共享、GPU1 Ch3 rollout、GPU2-3 训练。注:Ch1 旧口径 judge=target **同一模型**,同卡放两份是浪费,只有"留出裁判"补强才需独立服务

## 2026-09-09 — 第一章 baseline 表(表1.4)审计:PAIR 91.8% 为伪值,判定层需统一

- 91.8% 来源 = pipeline B(`baselines/asr_test_server.py --all`,原始记录 `docs/AHR-GRPO.md:25`),跑在 **2026-05-28 修正前**的代码状态
- 根因(高置信):旧 `pair.py` 的 `generate_initial_prompt()` 直接返回**原始有害 prompt** → 迭代 1 即"直发即成功",933/1000 那版有 **274 条(27.4%)根本没被攻击就记为成功**;且 attacker 未关 thinking,**泄露的思维链原文被当作攻击 prompt 发给 target**(迭代 2 成功率 97.2%)。git 自证:`84f3a20` "ASR dropped from 93.3% to 28.5%"
- 次要缺陷:`n_streams` 参数收了但从未使用(无流/无树搜索,**严格说不构成 PAIR**);attacker/target/judge 同为 plain Qwen3-4B;早停计数与终评是两次独立判定(26 条 iterations==1 却 is_success=false);未被查询过的最后一次 refine 结果还会抽一次奖
- 口径不一致(同表不可比的结构性原因):①baseline 行用 `GUARD_PROMPT` system prompt 版,修正后的 pipeline A 用官方无 system prompt 版,同方法差 28.5~57.6% vs 91.8%;②分母不同(Ch1 除 valid、baselines 除 total);③Δ 列拿 **test.jsonl** 的 30.8 去减 **val.jsonl** 训出的 33.2(`eval_baseline.sh:19` 还断言"与 exp.sh 一致",实为假);④"PAIR 高成本 1000+ API 调用/样本"错约两个数量级,实测 ~10-30 次(`docs/latex_project/main.tex:330` 有此错述,同文件 285 行写的是正确的 ≤20)
- 受影响面需连带重算:`docs/experiment_results_midterm.md:154,167`、`experiment_results_full.md:122-124`、`figures/figure_data_tables.md:99-100`、`report/thesis_progress_report.md:36,39`(结论④"成本差两个数量级")、`docs/cookbook/03_three_chapters.md:25`、`docs/AHR-GRPO.md:25,95`、本文件 TODO 09-08 节(曾用 91.8% 论证不换 target)
- 处置:表1.4 全部行在新环境下按统一判定层重测(official Guard/仅 Unsafe/分母 total/test.jsonl 1000 条/实测调用数 + 直发单列),由 `RL4jailbreak/experiments/e2e/e2e_ahr_grpo.py` 承担;Ch3 的 baseline 列(TAP/Crescendo/PAIR)因 target 换 plain-4B 本就要重跑,同批产出

## 2026-09-10 — B 轴重跑完成(plain-4B target, 新 C4 机制, 4 臂 × 300)

- **新臂 C4(压缩式记忆)实现**(conv_eval._ctx_view_compress): 未越阈值时视图与 C1 逐字相同(不折不调模型), 越阈值(2500/4000 token)才把旧轮折成累计摘要, 折无可折再缩窗 → 硬上界。与历史 C2(每轮追加摘要且原文一条不删, 视图严格大于 C1)本质不同——C2 当年落败是设计必然
- 修 4 个自身 bug: 折叠先于阈值判断(退化恒滑窗)、未越阈值仍被 keep 截断、循环内重置 keep 致死循环、结尾 "w" 重写会抹掉续跑历史
- **逐条增量落盘 + --resume**(09-03 丢 332 条事故根治): 崩溃最多丢正在跑的一条, 续跑从 results.jsonl 读完成 id, 统计从文件重算; 实服务验证(杀进程→重启→"已完成 1 条, 跳过")
- **结果**(base policy, test C 300, 同 prompt 配对):
  | config | ASR | avg_turns | ctx_max | full_max | folds |
  |---|---|---|---|---|---|
  | C1_full | **54.00%** | 6.06 | — | — | 0 |
  | C3_window3 | 51.00% | 6.13 | — | — | 0 |
  | C4_c2500 | 51.67% | 6.04 | 2500 | 6004 | 140 |
  | C4_c4000 | 55.67% | 6.01 | **3971** | 10087 | 31 |
- **C1 精确复现 M0 的 54.00%(162/300)** → 测量链可复现性验证通过
- 配对 McNemar(同 300 条): C3 Δ=−3.00pp p=0.241 / C4@2500 Δ=−2.33pp p=0.345 / C4@4000 Δ=+1.67pp p=0.529 → **三臂 ASR 均与 C1 不可分辨(噪声内)**, n=300 分辨不了 ~3pp
- **结论:C1 保持定稿协议(训练侧无需同步,C3/C4 的 plugin/env 缺口不再需要补);C4@4000 作为"同等 ASR 下输入 token 省 2.5×(10.1k→4.0k)+摘要调用受阈值约束"的效率叙事, 评估侧可选臂**(注意: 训练 C1→评估 C4 属于 e2e 口径可选, 不进主矩阵)
- 产出: `output/baxis_ctx/summary.json` + 每臂 results.jsonl(逐条, 可复算配对检验)

## 2026-09-11 — M1/M2 全量评估完成(接力链自动执行) + 第一章 skill 分布分析

- **M1 GRPO@10轮(300步/num_gen16/lr1e-5/β0.05, base 初始) = ASR 50.67%(152/300), avg 6.02 —— 低于 M0 base 54.00%(−3.33pp)**
  - 训练曲线: reward 中段 0.35→0.81 但 grad_norm 剧烈震荡(0.009→0.879→1.195), kl 缓升 0.044, frac_reward_zero_std 全程有 1/4-1/2 的步全零组
  - 行为: 评估选率 84.9%, 成功轨迹选率 92.6% vs 失败 83.2% → 未学到 RFT 的"放下 skill"选择性, 行为类 base
  - 解读: 该配方在 plain-4B 多轮对话上, vanilla GRPO(base 初始) 未超越未训练 agent; RFT(M2) 是当前唯一有效臂
- **M2 RFT 全量(test C 1000) = ASR 59.50%(595/1000), avg 5.57**(300 子集曾 62.33%, 一致区间)
- **A 轴当前排序: M2 59.5% > M0 54.0% > M1 50.7%**; M3(M2 merged + GRPO)停在人工确认闸门
- skill 分布分析(输出 output/analysis/skill_usage.json): base 选率 89.1%/M2 79.8%/M1 84.9%; base 成功轨迹更依赖 skill(95.4% vs 87.2%), RFT 反转(76.5% vs 81.2%); 按轮次选率 98.8%→71%(失败=放手触发器); top-10 按单调用 ASR 筛选, agent 语境下 skill7/8/5 最弱
- 接力链 `scripts/chain_after_m1.sh`(新): 等 M1 收工 → merge → M1 评估 300 → M2 评估 1000, 停在 M3 前
