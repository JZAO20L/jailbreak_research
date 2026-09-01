# Agentic-RL（第三章）文献调研

**日期**: 2026-08-03
**方法**: arxiv Atom API（curl），三个候选方向并行调研
**背景**: SESS 系统（skill 库 + 启发式检索/搜索，同族 ASR ~99%，跨架构 gpt-oss 仅 ~30%）基础上做 RL 为主的 agentic 训练研究

---

## 核心发现（TL;DR）

**三个方向的空白收敛到同一个交叉点：skill/strategy library × RL 训练，目前无人占据。**

- 方向 1：RL 训练的攻击模型**都不用** skill 库；用 skill 库的攻击方法**都不做** RL 训练（`skill AND jailbreak AND reinforcement` 检索结果为 0）
- 方向 2：防御训练数据都在 prompt 层面多样化，**无人**用高 ASR 的 skill 积累系统作为定向数据引擎
- 方向 3：攻防对抗训练的 RL 论文中，攻击表示全是自由文本，**无人**用结构化 skill 表示（仅 EvoSafety 沾边，但它不做 GRPO/联合权重训练）

---

## 方向 1：RL 强化 SESS 攻击（把搜索蒸馏进权重）

### 关键论文

| 论文 | arXiv | 日期 | 核心内容 | 与我们的关系 |
|------|-------|------|----------|--------------|
| SEMA | 2602.06854 | 2026-02 | 两阶段：自生成非拒绝 rollout 自调优（冷启动）→ intent-drift 感知 reward 的 RL；平均 ASR@1 80.1%；RL > SFT/DPO | 最接近我们 RFT→RL recipe 的实例，reward 分解（意图对齐+合规风险+细节度）可复用 |
| DC-GRPO (MJ) | 2607.11070 | 2026-07 | turn 级分解 credit assignment 的 GRPO，每 turn 独立 group-relative 信号；ASR5@3 98.26%（当前 SOTA） | 在 SESS 多步搜索轨迹上训练时的 credit assignment 方案 |
| TRACE | 2605.08778 | 2026-05 | leave-one-turn-out 语义掩码估计每 turn 贡献；失败轨迹按有害性/相关性/拒绝感知惩罚 | 诊断了粗粒度 outcome reward 的失败模式，对搜索轨迹（大量死路）直接适用 |
| AdvGRPO | 2606.09701 | 2026-06 | GRPO 攻防联合训练不稳定 → 密集多通道 reward + 解耦优势归一化（GDPO）；单轮→闭环多轮课程 | GRPO 稳定性的权威 fix recipe（Microsoft，含 Russinovich） |
| RL-Hammer | 2510.04885 | 2025-10 | Meta/FAIR，无 SFT 冷启动纯 RL 训练攻击模型，98% ASR vs GPT-4o；记录了 diversity reward 被 hack | recipe 论文；说明冷启动可选（值得 ablation）；多样性 reward 的坑与 SESS skill 库的存在理由直接相关 |
| PISmith | 2603.13026 | 2026-03 | 标准 GRPO 对强防御失败：reward 极度稀疏 → 熵坍缩；fix：自适应熵正则 + 动态优势加权 | 跨架构硬目标（gpt-oss，成功轨迹稀缺）的 reward 稀疏问题，最可迁移的技术内容 |
| Slingshot | 2602.02395 | 2026-02 | 冷启动 RL + 可验证 reward（无 judge）；发现学到的攻击收敛为短指令式模式；跨族迁移存在但衰减 | RL 攻击模型学到什么（紧凑模式而非冗长推理）；跨族迁移衰减印证我们 99%→30% 的问题 |
| TrailBlazer | 2602.06440 | 2026-02 | 注意力机制重加权历史交互的漏洞信号，引导后续动作 | SESS 迭代搜索也积累跨 query 证据，这是"把利用已暴露漏洞的行为内化进权重"的范例 |
| Jailbreak-Zero | 2601.03265 | 2025-12 | 攻击模型大量生成 → 在偏好数据上微调（策略覆盖+多样性+保真度联合优化） | 最接近"蒸馏"思想，但蒸馏的是输出而非搜索过程，无 skill 库；需引用并区分 |
| APD | 2506.17231 | 2025-05 | 大攻击模型 → 小模型：掩码对抗预训练 + 动态温度蒸馏 + RL 模板优化；96.4% ASR_k vs GPT-4 | 唯一明确"把攻击能力蒸馏进小权重"的工作；但教师是单次攻击 LLM，非迭代搜索系统 |
| ACE-Safety | 2511.19218 | 2025-11 | GS-MCTS 在策略空间搜索生成对抗样本 → AC-TGPO 课程 RL 联合训练攻防 | **pipeline 形状最接近**（先策略搜索后 RL），但搜索输出喂给联合进化而非蒸馏检索条件策略；无持久 skill 库；必须显式对比 |
| 系统研究 | 2605.07032 | 2026-05 | 首个 RL-jailbreak 系统消融：密集 reward 和长 episode 是成功主因，算法选择次要 | 支持我们的密集 reward 设计 |

**次要但需引用**：AutoDAN-Reasoning (2510.05379, 策略库 test-time scaling，我们是其参数化互补)、SRTJ (2605.00974, SESS 的 training-free 兄弟)、AutoRISE (2604.22871, 攻击程序进化)、bandit 选择 (2606.26936, 学习选择 jailbreak 库，97% ASR，支持"学习式检索替换启发式")、PASS (2509.23558, GraphRAG 记忆+RL)。

**奠基**：RL-JACK (2406.08725)、RLbreaker (2406.08705)、JailPO (2412.15623, 首个 DPO 攻击模型)。

### Recipe 洞察

1. **冷启动/RFT**：SEMA/AdvGRPO 用自举/课程阶段；RL-Hammer 证明纯 RL 可行。SESS 的成功轨迹是现成 RFT 语料——比已有论文的冷启动都强
2. **GRPO 稳定性**：对抗设定下已知不稳定；fix = 密集多通道 reward + 解耦优势归一化（AdvGRPO）
3. **Reward 稀疏**：熵坍缩是文档化的失败模式；fix = 自适应熵正则 + 动态优势加权（PISmith），或二值→密集（对比打分）
4. **Credit assignment**：2026 前沿（DC-GRPO、TRACE、TROJail）；SESS 搜索轨迹同样稀疏结果结构，预期是我们的核心技术贡献区
5. **多样性**：in-reward 多样性项会被 hack（RL-Hammer）；bandit 式后验选择更干净
6. **格式坍缩**：RL 攻击倾向收敛到短指令式模式（Slingshot）——蒸馏 SESS 轨迹时需显式处理格式多样性

### Gap

- **(a) skill 库 + RL 攻击**：无人做。两条线平行发展从未交汇
- **(b) 迭代搜索 → 单次前向蒸馏**：无人做。APD 蒸馏的是单次攻击 LLM，Jailbreak-Zero 蒸馏的是输出
- **定位**：SESS 用搜索达到 99% ASR；蒸馏摊销计算成本，并假设能突破 30% 的跨架构启发式搜索上限——动机无人占据

---

## 方向 2：SESS → 防御数据合成

### 关键论文

| 论文 | arXiv | 日期 | 核心内容 | 与我们的关系 |
|------|-------|------|----------|--------------|
| ASCoT / Adversarial Déjà Vu | 2510.21910 | 2025-10 | 从 32 篇攻击论文提取对抗 skill 压缩为稀疏字典；防御在 skill 原语的**多样组合**上训练；结论：对抗 skill 覆盖度 > 数据规模 | **概念上最接近 SESS 的防御翻转版**，验证核心假设；但 SESS 是搜索派生的高 ASR 引擎 vs 其事后从论文提取；必须引用并区分 |
| AttackSHAP / A-MESS | 2607.17152 | 2026-07 | 用 Shapley 值评估攻击作为防御数据的下游效用；发现 ASR 排名与防御效用弱相关 | "SESS → 防御"章节现成的评估方法论，可用于 skill 数据选择消融 |
| Rainbow Teaming | 2402.16822 | 2024-02 | MAP-Elites 质量多样性生成多样可迁移对抗 prompt；在其数据上微调提升安全性不损通用性 | "多样攻击数据→微调→更安全模型"的经典结果；SESS skill 库是其 feature-map 的策略级类比 |
| PAD (Purple-teaming) | 2407.01850 | 2024-07 | GAN 式自博弈：attacker 诱导不安全响应，defender 生成安全响应+解释，交替更新 | 攻击生成对→防御训练的早期干净实例 |
| APO | 2311.08045 | 2023-11 | min-max：LLM 与 reward model 交替更新，无需新人工标注 | DPO 防御 pipeline 的基础 recipe |
| Staged-Competence | 2605.26315 | 2026-05 | 课程 DPO：按难度组织偏好数据；OOD 有害响应 -16%，jailbreak ASR -20%，近零 over-refusal | 攻击派生偏好数据的**调度方式**与数据本身同样重要 |
| SGASA | 2511.21214 | 2025-11 | 预合成安全指南+对抗增强 prompt → SFT+DPO 内化；面向推理模型 | 合成数据防御的当前惯例（SFT+DPO 堆叠） |

**次要**：浅层对齐诊断 (2406.05946)、RepBend (2504.01550)、AntiDote (2509.08000)、ADPO (2502.11455)、OPSA (2605.15239)、拒绝触发去活 (2603.11388)、SEAR (2606.31748)、WARDEN (2605.05415)、SAGE-RT (2408.11851)。

### Recipe 谱系（由低到高）

1. SFT on (有害 prompt → 拒绝) 对 — baseline，已知弱点：浅层对齐、拒绝触发过度泛化
2. DPO/偏好优化（safe-unsafe 对）— 静态数据 OOD 脆弱；fix：对抗变体（APO/ADPO）、课程调度、SFT+DPO 堆叠
3. RL 攻防联合训练 — 2025-26 主流（见方向 3），数据在线生成
4. 嵌入/表示空间对抗训练 — CAT/CAPO/WARDEN、RepBend
5. 防篡改训练 — AntiDote 等

### Gap

- **(a) skill/策略多样性的防御数据**：唯一直接命中是 ASCoT；其余都在 prompt 层面多样化。**无人用高 ASR skill 积累系统作为定向数据引擎**——但必须与 ASCoT 区分
- **(b) over-refusal 处理**：现在的惯例是安全指标必须配 XSTest 式 over-refusal + 通用能力数字
- **(c) 跨模型攻击数据（攻击 A → 防御 B）**：都是附带提及，无系统研究，跨族更没有。SESS 的 skill 抽象（策略级而非 token 级扰动）使跨族 skill 迁移做防御数据成为真正开放的贡献点

### 评估惯例

- 安全：HarmBench、AdvBench、JailbreakBench、StrongREJECT；over-refusal：XSTest；通用能力：MMLU/GSM8K/HellaSwag/MT-Bench
- 攻击套件：GCG/PAIR/TAP/AutoDAN/PAP/多轮，越来越强调**自适应攻击**（静态基准会撒谎）
- 新指标：AttackSHAP（攻击子集的下游防御效用）

---

## 方向 3：攻防对抗联合训练

### 关键论文

| 论文 | arXiv | 日期 | 核心内容 | 与我们的关系 |
|------|-------|------|----------|--------------|
| AdvGRPO | 2606.09701 | 2026-06 | 先前工作报告 GRPO 联合训练不稳定 → 密集多通道 reward（攻击/prompt/思考链/有用性，乘法聚合）+ GDPO 解耦优势归一化；课程：单轮→闭环多轮→交替自举 | **我们 GRPO 设计的最重要参考文献**；GAN 式生成器-判别器框架 |
| AdvGame | 2512.20806 | 2025-12 | Meta/FAIR，非零和博弈，attacker+defender **联合/同时**在线 RL（DPO-MD/IPO-MD，Nash 均衡理论）；成对偏好 reward 抗 reward hacking；EMA checkpoint | 论证交替更新低效不稳定；代码开源 |
| Self-RedTeam (SPAG) | 2506.07468 | 2025-06 | 全在线自博弈：**单一共享权重**策略扮演双角色；Re++（无 critic PPO 变体）；零和 min-max + Nash 安全保证；14 个基准最高 95% 安全提升 | 共享权重极端；attacker-only 训练会 mode collapse，协同进化保持多样性（+17.8%） |
| MAGIC | 2602.01539 | 2026-02 | 多轮多智能体非对称对抗博弈；attacker 进化出训练中未见的新组合策略，defender 泛化到这些策略 | 协同进化诱导涌现组合攻击结构——"skill 涌现"的概念桥梁 |
| ARLAS | 2510.05442 | 2025-10 | 零和联合训练（agent 安全）；关键稳定机制：**population-based——defender 对所有历史 attacker checkpoint 训练**防循环学习 | 循环/非平稳动态的最干净已发表解法 |
| ACE-Safety | 2511.19218 | 2025-11 | GS-MCTS 策略空间搜索 + AC-TGPO 课程 RL 联合训练 | 最接近 skill 结构化搜索的类比（但策略是隐式 MCTS 节点，非持久库） |
| TriPlay-RL | 2601.18292 | 2026-01 | 三角色自博弈：attacker + defender + **evaluator** 共同提升；近零人工标注 | evaluator 应入环——GRPO 联合训练需要非平稳下的可信 judge |
| GPT-Red | 2607.26115 | 2026-07 | OpenAI，工业规模自博弈：一个 attacker 对**多样 defender 种群**联合训练，用于 GPT-5.6 对抗训练 | 前沿规模验证 population-based 联合训练 |
| Lifelong Safety Alignment | 2505.20259 | 2025-05 | Meta-Attacker 用论文知识 warm start（粗糙的 skill 先验）；迭代 1 ASR 73% → 最终 defender 压到 7% | 用蒸馏文献知识播种 attacker 的先例；迭代级 ASR 轨迹报告模板 |
| EvoSafety | 2605.13411 | 2026-05 | **最像 SESS**：攻击策略配对抗 skill 库（饱和后扩库）；防御外化为轻量辅助模型+记忆检索（Steer/Guard 模式），37.5% 参数达 99.61% 防御 | 直接证据：(a) 显式 skill 库修复攻击饱和 (b) 防御可外化为记忆/库更新；**最接近的 prior work，必须引用对比** |
| GRTS | 2310.00322 | 2023-09 | 红队博弈理论奠基：多样性度量抗 mode collapse，近似 Nash 收敛保证 | 多样性维护的理论锚点 |

**次要**：EvoDefense (2605.31140, 防御侧经验记忆)、ICAG (2402.13148, in-context 攻防博弈无微调)、RvB (2601.19726)、SPAG 原文 (2401.01335)、好奇心驱动红队 (2402.19464)。
**未验证线索**：DuoGuard（迭代 DPO 攻防联合，arxiv 搜不到，可能在 OpenReview）。

### 机制分类

- **循环结构**：联合更新（AdvGame/MAGIC/GPT-Red）| 交替更新（AdvGRPO/PAD/SecTOW）| 全在线（Self-RedTeam）| 无训练博弈（RvB/ICAG）| 外化记忆（EvoSafety/EvoDefense）
- **权重拓扑**：共享权重角色切换 | 分离模型 | 种群
- **RL 算法**：GRPO 系（AdvGRPO/ACE-Safety）| PPO 系（Self-RedTeam）| DPO/IPO 系（AdvGame）| 进化（GRTS/DARWIN）
- **稳定技巧**：population replay（ARLAS）、EMA（AdvGame）、解耦优势归一化+密集 reward（AdvGRPO）、成对偏好 reward（AdvGame）、多样性/Nash 正则（GRTS）、KL 惩罚、数据门控（Survive or Collapse 2605.22217）、abliterated attacker 初始化（对齐模型会拒绝扮演攻击者！）

### 已知失败模式

1. **GRPO 对抗联合训练不稳定**——已有 fix recipe（AdvGRPO）；AdvGame 直接放弃 GRPO 用 DPO-MD
2. **攻击 mode collapse**——静态 defender 下 attacker 坍缩到少数主导模式；协同进化保持多样性
3. **循环学习/非平稳**——population replay（ARLAS）
4. **over-refusal 与鲁棒性动态耦合**——训练早期 ASR 归零但 over-refusal 最大，后期效用恢复但攻击重新打开（2604.27019）
5. **Reward hacking**——成对偏好 > 单点打分
6. **自博弈坍缩**——数据级门控比 reward 设计更关键（2605.22217）
7. **Attacker 角色拒绝**——对齐模型拒绝扮演攻击者，需 abliterated/unsafe-SFT/文献 warm start 初始化

### Gap

- **(a) skill/策略库作为 RL 环内攻击表示**：仅 EvoSafety 沾边（无 GRPO/联合权重训练）；所有 RL 联合训练论文的攻击都是自由文本——**SESS skill 库 + GRPO 联合训练无人占据**
- **(b) skill 级 credit assignment**：无人追踪哪些 skill 在 defender 适应后存活——skill 存活/淘汰的进化叙事无人做

---

## 三方向综合：定位建议

### 共同空白

skill/strategy library × RL 是三条线的公共空白。SESS 的独特资产（99% ASR 的 skill 积累引擎 + 跨架构 30% 的明确短板）天然适配。

### 最近的竞品（必须引用区分）

| 论文 | 重叠点 | 我们的差异 |
|------|--------|-----------|
| ACE-Safety (2511.19218) | 策略搜索 → RL 训练 | 我们蒸馏检索条件策略进单一模型；有持久 skill 库 |
| EvoSafety (2605.13411) | skill 库 + 攻防外化 | 我们做 GRPO 参数化训练，非外化记忆 |
| ASCoT (2510.21910) | skill 组合 → 防御 | SESS skill 是搜索派生的实测高 ASR，非事后论文提取 |
| APD (2506.17231) / Jailbreak-Zero (2601.03265) | 攻击蒸馏 | 教师是迭代搜索系统，蒸馏搜索过程而非输出 |
| AdvGRPO (2606.09701) | GRPO 攻防联合 | skill 结构化攻击表示 |

### 递进路径（与用户三个想法对应）

1. **方向 1（地基）**：SESS 轨迹 → RFT → GRPO，skill-conditioned 攻击模型。空白最大、工程量最小、对比最清晰（RL-SESS vs SESS vs Jailbreak-R1 类方法）
2. **方向 2（应用延伸）**：SESS/训练后攻击模型作为防御数据引擎。ASCoT 是竞品，跨族攻击→防御数据是空白
3. **方向 3（完全体）**：skill-conditioned attacker + defender 的 GRPO 对抗联合训练。AdvGRPO 提供稳定性 recipe，ARLAS 提供防循环机制

### 优先精读清单

1. AdvGRPO (2606.09701) — GRPO 攻防训练的完整 recipe
2. ACE-Safety (2511.19218) — 最近邻，必须精读以定位
3. SEMA (2602.06854) — RFT→RL 攻击训练范式
4. EvoSafety (2605.13411) — skill 库视角最近邻
5. DC-GRPO (2607.11070) + TRACE (2605.08778) — credit assignment 前沿
6. ASCoT (2510.21910) — 防御侧 skill 视角
7. PISmith (2603.13026) — reward 稀疏 fix（跨架构场景）
8. SecTOW (2507.22037) + DARWIN (2607.19829) — 已读
