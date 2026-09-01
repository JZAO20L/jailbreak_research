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
