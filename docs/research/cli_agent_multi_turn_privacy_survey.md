# 面向 CLI / Coding Agent 的多轮对话隐私窃取攻击：技术调研报告

> 版本：v1.0　日期：2026-08-25
> 文献窗口：2024-01 至 2026-08（少量奠基工作除外）
> 文献量：50 条编号文献（论文 35 + 博客/行业报告 15）+ 若干内联补充
> 可靠性说明：所有 arXiv 条目均通过 arXiv Export API 逐条校验（2026-08-25），所有博客/报告 URL 均通过 HTTP 访问校验（HTTP 200）；无编造条目。

---

## 目录

1. [研究背景与攻击场景定义](#1-研究背景与攻击场景定义)
2. [与本课题已有工作的关系](#2-与本课题已有工作的关系)
3. [调研方法](#3-调研方法)
4. [相关工作分类综述](#4-相关工作分类综述)
   - 4.1 通用 LLM Jailbreak 攻击方法（12 篇）
   - 4.2 Agent 场景下的提示注入与工具攻击（9 篇）
   - 4.3 编码 / Coding Agent 专项安全研究（9 篇）
   - 4.4 隐私窃取、记忆投毒与数据外渗（5 篇）
   - 4.5 CLI Agent 真实攻击案例与安全博客（9 篇）
   - 4.6 防御、评估基准与治理框架（6 篇）
5. [攻击场景威胁建模与文献映射](#5-攻击场景威胁建模与文献映射)
6. [研究空白与潜在创新点](#6-研究空白与潜在创新点)
7. [建议的研究路径（与论文衔接）](#7-建议的研究路径与论文衔接)
8. [参考文献](#8-参考文献)

---

## 1. 研究背景与攻击场景定义

**背景。** CLI/Coding Agent（Claude Code、Cursor、Codex CLI、Gemini CLI、Aider、OpenHands、Copilot 等）正成为主流开发方式。这类 Agent 具备三个特征，使其成为高价值攻击目标：

1. **持有秘密**：会话进程内通常可访问 `~/.env`、`.env`、`~/.ssh/`、`~/.aws/`、`~/.config/`、git credential、npmrc、代理账号 Token、AI 服务商 API Key 等；
2. **高权限执行**：默认可以执行 shell 命令、读写文件、发起网络请求，且部分工具（如 Claude Code `--dangerously-skip-permissions`、Gemini CLI `--yolo`）可跳过人工确认；
3. **多轮会话 + 长上下文**：会话历史、项目文件、工具输出、网页检索内容混合组成上下文，且上下文可能跨会话持久化（记忆、索引、缓存）。

**目标攻击场景（本文调研围绕的核心问题）。** 攻击者与 CLI/Coding Agent 进行**多轮对话**，逐步克服目标模型的对齐/安全审查，**诱导 Agent 输出、读取、或外传用户的隐私信息**（API Key、账号凭据、SSH key、云厂商 Token、会话内出现过的敏感数据等）。攻击者可处于以下角色：

- **直接对话者**：攻击者本人与 Agent 对话（传统 jailbreak 语义的多轮版本）；
- **间接注入者**：攻击者将恶意指令预埋进 Agent 会读取的不可信内容（仓库 README、GitHub issue、网页、工具输出、MCP 工具描述、技能文件），在后续多轮会话中被触发；
- **混合角色**：先通过间接注入劫持会话目标，再用多轮对话逐步诱导外渗。

---

## 2. 与本课题已有工作的关系

本课题（毕业论文）已有章节：
- 第二章 SESS：自进化技能 jailbreak（单轮、面向聊天模型）；
- 第三章 Agentic Jailbreak：基于 PAIR 的多轮 Agent 式越狱攻击（攻击者侧是多轮，攻击对象是普通 chat 模型）。

本调研的新方向把**多轮攻击能力迁移到"带工具/高权限/持有秘密"的 CLI Coding Agent 作为攻击目标**，并将攻击目标从"输出有害内容"扩展为"窃取隐私信息"。第三章的攻击者框架（分析-动作选择-记忆-多轮 RL）可直接复用于新场景，但需要针对工具调用轨迹、上下文污染与秘密外渗重新设计奖励与评测。

---

## 3. 调研方法

- 检索渠道：WebSearch + arXiv Export API 批量校验 + 原文页面抓取；
- 文献窗口：以 2024-01 至 2026-08 为主，仅保留极少数必要奠基工作（如 Greshake 2023 间接注入、Zou 2023 GCG）；
- 分类框架：按"攻击方法 → 攻击对象 → 攻击目标"三个维度组织：通用 jailbreak 方法（方法）、Agent/编码 Agent 场景（对象）、隐私窃取/记忆（目标）、防御与治理（对抗侧）；
- 每条文献标注：标题、作者、年份、venue 或来源、URL、内容摘要、对本课题的意义。

---

## 4. 相关工作分类综述

### 4.1 通用 LLM Jailbreak 攻击方法（12 篇）

该部分是攻击方法学的"弹药库"：多轮对话攻击、自动攻击框架、对抗文本攻击三类，均可迁移到 CLI Agent 场景。

**[1] Universal and Transferable Adversarial Attacks on Aligned Language Models（GCG）**
Zou 等，ICLR 2024。https://arxiv.org/abs/2307.15043
梯度驱动的 token 级对抗后缀优化，附加在任意指令后即可使对齐模型输出有害内容，且可跨模型迁移。GCG 是后续所有自动攻击的基线。
**对本课题的意义**：白盒对抗优化奠基工作；为"在系统提示/工具描述中植入对抗片段"提供理论源头。

**[2] Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack**
Russinovich & Salem（微软），USENIX Security 2025。https://arxiv.org/abs/2404.01833
多轮"对话升级"攻击：每轮在上一轮输出上逐步逼近有害请求，全程保持对话自然；对 GPT-4/Claude/Gemini/Bing Chat 成功率最高 96%。
**对本课题的意义**：与本课题"与 CLI Agent 多轮闲聊逐步套取隐私"的攻击形态直接同构，是方法支柱之一。

**[3] Jailbreaking Black Box Large Language Models in Twenty Queries（PAIR）**
Chao 等，COLM 2024。https://arxiv.org/abs/2310.08419
攻击者 LLM 依据目标模型拒绝反馈迭代改写候选提示，平均约 20 次查询攻破 GPT-3.5/4/Claude 等。本课题第三章即基于该框架。
**对本课题的意义**：是已有工作（Agentic Jailbreak 第三章）的原始出处；其"自适应学习"范式可直接作为 CLI Agent 多轮攻击的复现基座。

**[4] Tree of Attacks: Jailbreaking Black-Box LLMs Automatically（TAP）**
Mehrotra 等，NeurIPS 2024。https://arxiv.org/abs/2312.02119
思维树 + 剪枝：攻击者生成、评估、剪枝候选提示，用评估者 LLM 引导搜索逼近越狱目标，显著降低查询数，对 GPT-4 成功率 80%+。
**对本课题的意义**：树搜索式多轮自动攻击；其"探索-评估-剪枝"循环可迁移到对 Coding Agent 工具调用轨迹的探测。

**[5] AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models**
Liu 等，ICLR 2024。https://arxiv.org/abs/2310.04451
针对 GCG 后缀不自然易被检测的问题，用分层遗传算法进化出语义自然、可读的越狱提示，白盒/黑盒均有效且可转移。
**对本课题的意义**：自然语言层面的进化式攻击；适合生成"看似正常的工作请求"进行隐蔽隐私窃取的攻击语料。

**[6] Many-shot Jailbreaking**
Anil 等（Anthropic），NeurIPS 2024 研究报告。https://www.anthropic.com/research/many-shot-jailbreaking
在单条提示中伪造几十到上百轮"模型已放弃拒绝"的对话示例，诱导目标模型跟随演示模式输出有害内容，可绕过专门的安全训练。
**对本课题的意义**：与"多轮对话隐私窃取"最相关的单发式多轮攻击——Coding Agent 长上下文会读入大量历史/文件，该攻击对其威胁直接。

**[7] GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**
Yu 等，NeurIPS 2023。https://arxiv.org/abs/2309.10253
灰盒模糊测试引入 LLM 红队：从种子模板出发用 LLM 变异生成新模板，按越狱成功率做覆盖引导的语料进化。
**对本课题的意义**：变异 + 覆盖率引导的自动化红队框架，可批量生成针对 Agent 系统提示/工具描述的探测语料。

**[8] ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs**
Jiang 等，ACL 2024。https://arxiv.org/abs/2402.11753
把敏感词编码成 ASCII 艺术，利用模型对非常规输入的不敏感性绕过对齐过滤，黑盒商用模型上高成功率。
**对本课题的意义**：非自然输入编码类攻击典型；可类比为给 CLI Agent 传"变形的命令/文件名/编码内容"绕过安全审查。

**[9] RedAgent: Red Teaming Large Language Models with Context-aware Autonomous Language Agent**
2024（arXiv）。https://arxiv.org/abs/2407.16667
上下文感知的自主红队 Agent：先"挖掘"目标模型系统提示，再基于系统提示生成多轮越狱策略模板，并在真实多轮对话中持续细化攻击。
**对本课题的意义**：把"多轮 + Agent 化攻击者"做成完整框架；是把攻击对象从对话模型换成 Coding Agent 时的直接脚手架。

**[10] DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**
Li 等，2024（arXiv）。https://arxiv.org/abs/2402.16914
将有害提示分解为无害子提示再重建，绕过困惑度过滤类防御。
**对本课题的意义**：分步拆解攻击可借鉴到"把一次外渗任务拆成多轮工具调用"的场景。

**[11] Siege: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search**
2025（arXiv）。https://arxiv.org/abs/2503.10619
树搜索驱动的多轮自主越狱，2025 年多轮攻击前沿代表，与 Crescendo 互补（不依赖预设对话剧本，而是自动探索）。
**对本课题的意义**：可用于设计"自动探索 CLI Agent 工具面"的多轮攻击器。

**[12] 综述两条**
- Jailbreak Attacks and Defenses Against Large Language Models: A Survey（Liu 等，2024，arXiv:2407.04295）：按场景/方法分类攻击（直接提示、对抗优化、多轮、API 保护），按攻击前/中/后分类防御，是领域路线图；
- From LLMs to MLLMs to Agents: A Survey of Emerging Paradigms in Jailbreak Attacks and Defenses within LLM Ecosystem（2025，arXiv:2506.15170）：扩展到多模态与 Agent 生态。
**对本课题的意义**：两篇综述可支撑论文 related work 骨架，尤其第二篇覆盖 Agent 生态。

### 4.2 Agent 场景下的提示注入与工具攻击（9 篇）

该部分定义了"Agent 的上下文被不可信内容污染并导致工具被滥用"的核心威胁模型，是 CLI Agent 攻击面分类学的直接来源。

**[13] Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection**
Greshake 等，CCS 2023（奠基工作）。https://arxiv.org/abs/2302.12173
首次系统定义"间接提示注入"：恶意指令藏入邮件、网页、文档、API 返回等外部数据，LLM 集成应用在检索/调用时被操纵；提出目标劫持（goal hijacking）与提示窃取（prompt stealing）两类攻击。
**对本课题的意义**：CLI Agent 读取的终端输出、日志、仓库文本正是同一条注入通道，是本课题威胁模型的原始定义。

**[14] Prompt Injection Attack against LLM-integrated Applications**
Liu 等，IEEE S&P 2025（扩展版）；arXiv 2306.05499（2023 初版）。
对 LLM 集成应用的（间接）提示注入做系统化调研，给出攻击向量、数据外渗手段与防御分类学并实测大量真实应用。
**对本课题的意义**：提供攻击面分类与术语框架（indirect injection、data exfiltration chain），被引最多综述之一。

**[15] InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents**
Zhan 等，2024（arXiv）。https://arxiv.org/abs/2403.02691
1,554 个测试用例、17 个工具集成场景，区分直接注入（恶意指令出现在工具输出）与间接注入（来自检索内容），按错误信息与隐私窃取两类危害评估 GPT-4 等模型，结果高度脆弱。
**对本课题的意义**：其"工具输出/检索内容"划分直接对应 CLI Agent 每轮 shell 输出与文件读取可被操纵的攻击面。

**[16] AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents**
Debenedetti 等，NeurIPS 2024（Datasets & Benchmarks）。https://arxiv.org/abs/2406.13352
动态基准同时测量"任务效用"与"攻击成功率"双维度；系统评测 spotlighting、过滤、微调等防御，表明多数防御以牺牲效用为代价且可被自适应攻击绕过；官方含 defense track。
**对本课题的意义**：提供了"攻击成功 vs 正常功能保持"双目标评估框架，可移植到 CLI 隐私窃取实验设计。

**[17] ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages**
Ye 等，2024（arXiv）。https://arxiv.org/abs/2402.10753
构造 648 个不安全工具用例，覆盖虚构/金融/法律/医疗等场景与 18 条安全规则，考察恶意用户输入与恶意工具输出两类攻击侧。
**对本课题的意义**："工具响应被恶意构造"即 CLI Agent 的 Bash/文件/网络工具返回被污染的情形。

**[18] Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents**
Zhang 等，IEEE S&P 2025。https://arxiv.org/abs/2410.02644
形式化定义 Agent 四类攻击面（提示注入、恶意工具使用、记忆投毒、多 Agent 攻击），构建 10 个现实场景、312 个工具、1,323 个测试用例。
**对本课题的意义**：覆盖最全面的 Agent 攻击基准之一；"隐私数据被攻击者获取"的威胁建模可直接对应 CLI Agent 会话中代码/密钥/环境变量的窃取目标。

**[19] Identifying the Risks of LM Agents with an LM-Emulated Sandbox（ToolEmu）**
Ruan 等，ICLR 2024。https://arxiv.org/abs/2309.15817
用 LM 仿真工具沙箱规模化评估 Agent 风险：36 个仿真工具、144 个失败案例，人工验证约半数失败可造成不可逆危害（欺诈、数据丢失、隐私泄露）。
**对本课题的意义**：低成本、可复现地测试 CLI Agent 多轮攻击与后果评估的现成方法论模板。

**[20] Malla: Demystifying Real-world Large Language Model Integrated Malicious Services**
Lin 等，USENIX Security 2024。https://arxiv.org/abs/2401.03315
首次系统化收集现实世界集成 LLM 的恶意服务（网络犯罪、欺诈、虚假信息、隐私侵犯），刻画其能力、定价与运作方式。
**对本课题的意义**：展示"恶意工具/服务"作为供应链上游；CLI Agent 自动执行中误调用此类 API/服务即构成隐私外泄通道。

**[21] AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases**
Chen 等，NeurIPS 2024。https://arxiv.org/abs/2407.12784
首个针对 Agent 记忆/RAG 知识库的投毒攻击：将含触发器的恶意指令注入少量记忆条目或知识库片段，用户正常提问命中触发器即劫持 Agent，同时保持良性任务性能。
**对本课题的意义**：对应 CLI Agent 的持久化攻击形态——在工作区文件、配置、索引或历史记忆预埋触发条件，后续会话触发隐私数据窃取。

**[22] Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems**
Lee 等，2024（arXiv）。https://arxiv.org/abs/2410.07283
恶意 Agent 通过 Agent 间消息向其他 Agent 传播注入指令形成级联感染；多 Agent 拓扑可放大攻击。
**对本课题的意义**：CLI Agent 编排子 Agent / MCP 工具时，"一条恶意输出扩散成整轮会话窃取"的攻击模型。

### 4.3 编码 / Coding Agent 专项安全研究（9 篇）

该部分是距离本课题最近的学术工作：从"代码模型生成漏洞"演进到"编码 Agent 被操纵、被投毒、被用于窃取凭据"。

**[23] CodeLMSec Benchmark: Systematically Evaluating and Finding Security Vulnerabilities in Black-Box Code Language Models**
Hajipour 等，IEEE SaTML 2024。https://arxiv.org/abs/2302.04012
150 条 C/C++ 生成提示与三类测试场景，评测 GPT-3.5/4、Copilot、StarCoder 等 9 个代码模型的漏洞生成率。
**对本课题的意义**：代码模型安全基准源头（锚点工作，唯一 2024 年前工作）。

**[24] Poisoning Programs by Un-Repairing Code: Security Concerns of AI-generated Code**
Improta 等，2024（arXiv）。https://arxiv.org/abs/2403.06675
"un-repairing"数据投毒：对真实漏洞修复补丁逆向引入漏洞、还原成看似合法的漏洞代码注入训练数据，使模型倾向生成易受攻击代码。
**对本课题的意义**：代码生成投毒方向代表工作，为训练层面的污染威胁建模提供基础。

**[25] XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants**
Štorek 等，2025（arXiv）。https://arxiv.org/abs/2503.14281
针对编码助手"自动汇聚多来源上下文"（跨文件、跨项目、跨贡献者）的新攻击面提出跨源上下文投毒：无需与受害者修改意图相关即可隐蔽注入，可持续跨会话生效。
**对本课题的意义**：直接研究"恶意仓库/文档经上下文注入编码助手"链路，2025 仓库投毒方向代表作。

**[26] Red-Teaming Coding Agents from a Tool-Invocation Perspective: An Empirical Security Assessment**
Xie 等，2025（arXiv，v6 2026-06）。https://arxiv.org/abs/2509.05755
首个从工具调用视角对 6 个主流编码 Agent（Cursor、Claude Code、Copilot、Windsurf、Cline、Trae）的系统性红队评测：提出 ToolLeak（借工具参数模式缺口外泄系统提示），并用"工具描述+工具返回"双通道提示注入实现 RCE。
**对本课题的意义**：与本课题最贴切的实证研究——直接覆盖多款 CLI/IDE 编码 Agent 的多轮工具调用攻击面，是本课题方法学基座。

**[27] SecureVibeBench: Benchmarking Secure Vibe Coding of AI Agents via Reconstructing Vulnerability-Introducing Scenarios**
Chen 等，2025（arXiv）。https://arxiv.org/abs/2509.22097
从 OSS-Fuzz 41 个 C/C++ 项目还原真实漏洞引入场景，构建 105 个安全编码任务基准，评估编码 Agent 在 vibe coding 流程下生成代码的安全性。
**对本课题的意义**：2025 年编码 Agent 安全基准代表，可作攻击危害度度量工具。

**[28] SeCodePLT: A Unified Platform for Evaluating the Security of Code GenAI**
Nie 等，2024（arXiv）。https://arxiv.org/abs/2410.11096
统一评测平台，同时覆盖代码生成模型的安全风险（生成漏洞）与安全能力（漏洞检测修正），以动态分析替代纯静态判定。
**对本课题的意义**：提供评测范式参考（内联补充，不占编号）。

**[29] Prompt Injection Attacks on Agentic Coding Assistants: A Systematic Analysis of Vulnerabilities in Skills, Tools, and Protocol Ecosystems**
Maloyan & Namiot，2026（arXiv）。https://arxiv.org/abs/2601.17548
面向 Claude Code、Copilot、Cursor 及 MCP 的 skill 架构，系统化梳理提示注入攻击面（Skills、Tools、MCP 协议生态）；归类攻击向量、影响（含凭据/密钥窃取）与缓解措施。
**对本课题的意义**：SoK 式综述直接覆盖本课题（API Key 窃取、skills/MCP 攻击面），是最佳入门文献。

**[30] SkillJect: Effectively Automating Skill-Based Prompt Injection for Skill-Enabled Agents**
Jia 等，2026（arXiv）。https://arxiv.org/abs/2602.14211
指出 Agent skill（含可执行脚本与指令）会被反复当作可信指导加载，形成供应链攻击面；提出自动生成"中毒 skill"的框架，绕过显式恶意指令被拒绝的问题。
**对本课题的意义**：自动化 skill 级提示注入，与 Claude Code/Cursor 的 skills 体系高度契合，是隐私窃取的具体实现技术。

**[31] Supply-Chain Poisoning Attacks Against LLM Coding Agent Skill Ecosystems**
Qu 等，2026（arXiv）。https://arxiv.org/abs/2604.03081
针对三方 coding-agent skills 的供应链投毒：提出 Document-Driven Implicit Payload Execution（DDIPE），在 skill 文档隐藏恶意负载，借 Agent 自身动作（文件写入、shell、网络请求）执行，实现动作空间劫持。
**对本课题的意义**：供应链视角的 skill 投毒，与 Snyk Nx 事件形成学术-现实对照。

**[32] IssueTrojanBench: Benchmarking AI Coding Agents Against Malicious Issue Requests**
Singh 等，2026（arXiv）。https://arxiv.org/abs/2607.20759
针对"编码 Agent 读取并执行 GitHub issue"的真实入口构建恶意 issue 基准：覆盖 prompt 对抗/投毒/后门触发词与 agentic 架构下工具误用、数据外泄（含凭据）与持久化控制。
**对本课题的意义**：与"issue/README 中毒 + 凭据窃取"选题几乎一一对应，提供现成评测集与攻击分类法。

### 4.4 隐私窃取、记忆投毒与数据外渗（5 篇）

该部分直接支撑"多轮会话中从 Agent 处窃取秘密"的目标建模，2024→2026 记忆投毒是最活跃子线。

**[33] Hidden in Memory: Sleeper Memory Poisoning in LLM Agents**
Pulipaka 等（CISPA），2026（arXiv）。https://arxiv.org/abs/2605.15338
"潜伏记忆投毒"：攻击者操纵文档、网页或代码仓库等外部上下文，诱导 Agent 存储关于用户的伪造记忆，攻击可休眠并在之后多次会话中重新浮现触发恶意行为，比常规注入更具持久性。
**对本课题的意义**：与 CLI Agent 场景几乎一一对应——编码 Agent 读取仓库/文档即被植入记忆，后续多轮会话提取秘密。

**[34] Poison Once, Exploit Forever: Environment-Injected Memory Poisoning Attacks on Web Agents（eTAMP）**
Zou 等，2026（arXiv）。https://arxiv.org/abs/2604.02623
首个无需直接访问记忆存储、仅通过一次被污染的"环境观察"（如浏览恶意网页）即可实现跨会话、跨站点记忆投毒的攻击。
**对本课题的意义**："观察即接触面"——对应编码 Agent 浏览网页/读取日志后记忆被污染再到窃取秘密的攻击链。

**[35] MemLeak: Diagnosing Information Leaks in Multimodal Agent Memory**
Wang & Zhang，2026（arXiv）。https://arxiv.org/abs/2606.29788
提出信息溯源图（IPG）分类学，诊断多模态 Agent 记忆中隐私信息的持续可恢复性：即使文本条目被删除，相关信息仍可从保留图片等关联表征恢复（18.3%–70%+）。
**对本课题的意义**：覆盖"记忆读取/泄露"侧——用户删除 secret 后记忆系统仍留可被窃取的残余信息。

**[36] Are AI-assisted Development Tools Immune to Prompt Injection?**
Huang 等，2026（arXiv）。https://arxiv.org/abs/2603.21642
首个对七个 MCP 客户端（含 Claude Code 等 CLI 编码 Agent）的实证提示注入研究，聚焦 tool-poisoning 向量，检验开发工具能否被诱导泄露敏感数据或触发未授权工具调用。
**对本课题的意义**：目前最直接针对"CLI/Coding Agent 被诱导泄露数据"的学术实证，是本课题定位参照。

**[37] Scalable Extraction of Training Data from (Production) Language Models**
Nasr & Carlini 等，ICLR 2025。https://arxiv.org/abs/2311.17035
把训练数据提取扩大到数十亿参数、经对齐/安全训练的生产级模型，可恢复数百 MB 训练数据，包括姓名、邮箱等个人信息。
**对本课题的意义**：理论支撑——编码 Agent 上下文/记忆里的 API Key 本质上是模型可内化的内容，提取与诱导输出是同一风险谱系。

**（内联补充）** EchoLeak: The First Real-World Zero-Click Prompt Injection Exploit in a Production LLM System（2025，arXiv:2509.10540）——首次在真实产品（浏览器 Agent 的 MCP）中复现零点击注入，说明 MCP 通道的现实危险性；ChatGPT 长期记忆被注入后形成持久外渗信道（见 4.5 [45]）为该子线最著名公开案例；JetBrains 恶意插件窃取 AI API key 事件见 4.5 内联补充。

### 4.5 CLI Agent 真实攻击案例与安全博客（9 篇）

该部分是本课题"现实可行性"的证据链：2025-2026 年已有多起真实 PoC/CVE 与供应链事件表明上述攻击不是理论推演。

**[38] Caught in the Hook: RCE and API Token Exfiltration Through Claude Code Project Files**
Check Point Research，2026-02。https://research.checkpoint.com/2026/rce-and-api-token-exfiltration-through-claude-code-project-files-cve-2025-59536/
披露 Claude Code 多个高危漏洞（CVE-2025-59536、CVE-2026-21852）：仓库内投毒的 `.claude/settings.json` 可定义 Hooks、恶意 MCP server 与环境变量，clone 并打开不可信仓库即触发任意 shell 命令执行并窃取 Anthropic API key。
**对本课题的意义**：为"CLI Agent 多轮会话中经项目文件渗透并窃取凭证"提供真实 CVE 级实证。

**[39] MCP Security Notification: Tool Poisoning Attacks**
Invariant Labs，2025-04。https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks
提出"Tool Poisoning Attack"：恶意指令隐藏在 MCP 工具描述中（对用户不可见、对模型可见），可指挥模型读取 `~/.cursor/mcp.json`、`~/.ssh/id_rsa` 等敏感文件并外传；已在 Cursor 上演示成功窃取 SSH 私钥与凭据。
**对本课题的意义**："经工具描述在对话中被诱导交出密钥"的教科书级演示，与本课题核心机制几乎一一对应。

**[40] Code Execution Through Deception: Gemini AI CLI Hijack**
Tracebit Research，2025-07。https://tracebit.com/blog/code-exec-deception-gemini-ai-cli-hijack
在 README.md/GEMINI.md 中藏匿提示注入（用 GPL 全文作掩护），结合权限白名单前缀匹配缺陷实现两阶段攻击：先诱导用户放行无害命令，再静默执行恶意 shell 窃取环境变量与凭据；Google 评定 P1/S1。
**对本课题的意义**：演示了"审查不可信代码时默认会话中静默窃取凭据"的完整链路。

**[41] Weaponizing AI Coding Agents for Malware in the Nx Malicious Package**
Liran Tal（Snyk），2025-08。https://snyk.io/blog/weaponizing-ai-coding-agents-for-malware-in-the-nx-malicious-package/
复盘 2025-08 npm Nx 投毒事件：恶意 postinstall 代码在本机调用 Claude Code、Gemini CLI、Amazon q（跳过权限确认），驱使 Agent 全盘搜索 SSH key、.env、钱包文件并上传到攻击者仓库；被称为"首个公开记录的使用 AI CLI Agent 做侦察与数据外泄的恶意软件"。
**对本课题的意义**：真实供应链事件证明"攻击者通过 prompt 控制本机 coding agent 窃取密钥"已是现实威胁，是论文引言的最佳案例。

**[42] New Prompt Injection Attack Vectors Through MCP Sampling**
Unit 42（Palo Alto Networks），2025-12。https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/
研究 MCP sampling 原语（server 反向请求模型补全）缺陷，在某主流编码 copilot 上演示三类攻击：资源盗用、对话劫持（注入持久指令、操纵回复并窃取敏感数据）、隐蔽工具调用。
**对本课题的意义**："conversation hijacking"向量与多轮对话中持续控制/窃取目标直接吻合。

**[43] MCP Horror Stories: The GitHub Prompt Injection Data Heist**
Docker Blog，2025-08。https://www.docker.com/blog/mcp-horror-stories-github-prompt-injection/
复盘 Invariant Labs 2025-05 发现的官方 GitHub MCP 集成漏洞：攻击者在公开 repo issue 植入注入文本，开发者让 Agent"查一下 open issues"即被劫持，借助宽权限 PAT 跨仓库读取私有数据（API key 等）并外泄。
**对本课题的意义**：真实发生的"外部不可信内容在会话中被读取后触发隐私外泄"事件，印证间接注入对凭据的现实杀伤力。

**[44] Protecting against indirect prompt injection attacks in MCP**
Microsoft Developer Blog，2025-04。https://developer.microsoft.com/blog/protecting-against-indirect-injection-attacks-mcp/
微软官方梳理 MCP 场景下间接注入与 Tool Poisoning 风险模型，指出托管式 MCP server 可被动态篡改工具定义形成"rug pull"致数据外泄，给出 Azure AI Prompt Shields、spotlighting、供应链管控对策。
**对本课题的意义**：头部厂商的权威风险框架，论证 MCP 工具元数据是隐私窃取攻击面的行业共识（内联补充，不占编号）。

**[45] Hacker plants false memories in ChatGPT to steal user data**
Ars Technica（原研究：Embrace The Red），2024-09。https://arstechnica.com/security/2024/09/false-memories-planted-in-chatgpt-give-hacker-persistent-exfiltration-channel/
公开演示通过 prompt injection 向 ChatGPT 长期记忆写入虚假记忆，形成跨会话持久注入信道，攻击者可事后远程操纵模型外泄用户数据。
**对本课题的意义**："对话记忆被投毒进而窃取数据"的最直接公开案例，适合作为论文动机性案例。

**[46] New prompt injection papers: Agents Rule of Two and The Attacker Moves Second**
Simon Willison，2025-11。https://simonwillison.net/2025/Nov/2/new-prompt-injection-papers/
解读 Meta 团队"Agents Rule of Two"（Agent 单会话内至多满足：可处理不可信输入/可访问敏感数据/可改变状态或对外通信 三条中的两条）与 OpenAI/Anthropic/Google 合作论文"Attacker Moves Second"（自适应攻击以 >90% 成功率击穿 12 种防护）：提示注入尚未解决，只能靠架构隔离。
**对本课题的意义**：权威安全社区视角的理论依据——CLI Agent 恰好同时满足三条性质，是隐私窃取高风险对象。

**[47] Prompt Injection and AI Agent Security Risks: A Claude Code Guide for Enterprise Teams**
TrueFoundry，2026-06。https://www.truefoundry.com/blog/claude-code-prompt-injection
面向企业的 Claude Code 提示注入风险综述：引用 OWASP Top 10 for Agentic Applications 2026（Agent Goal Hijacking 居首）与 2026-03 Oasis Security 的"Claudy Day"攻击（chain 式注入+数据外泄，默认配置下盗走 claude.ai 会话历史），给出网关/权限/白名单/审计等缓解层。
**对本课题的意义**：最新行业综述 + 2026 新案例，可作为研究动机与相关工作收尾引用。

**（内联补充）**
- JetBrains Marketplace 15 个恶意插件窃取 AI API key（Aikido Security/Infosecurity Magazine，2026-06）：https://www.aikido.dev/blog/multiple-jetbrains-ide-plugins-caught-stealing-ai-keys ——工具/插件供应链窃取编码者密钥的最新实证；
- HiddenLayer《The Next AI Supply Chain Risk: Malicious Skills in Agentic AI》（2025）：https://www.hiddenlayer.com/research/the-next-ai-supply-chain-risk-malicious-skills-in-agentic-ai ——恶意 skill 供应链分析；
- Trend Micro《Slopsquatting: When AI Agents Hallucinate Malicious Packages》（2025）：https://www.trendaisecurity.com/pl/resources-insights/deep-research/slopsquatting-when-ai-agents-hallucinate-malicious-packages ——Agent 幻觉出恶意 npm 包；
- Endor Labs《AI Code Security Benchmark》：https://www.endorlabs.com/research/ai-code-security-benchmark ——编码 Agent 安全基准。

### 4.6 防御、评估基准与治理框架（6 篇）

该部分用于：a) 设计实验时作为被评估的防御基线；b) 论文中威胁建模与缓解建议的行业对标。

**[48] CYBERSECEVAL 3: Advancing the Evaluation of Cybersecurity Risks and Capabilities in Large Language Models**
Meta PurrlLlama 团队，2024（arXiv 工业报告）。https://arxiv.org/abs/2408.01605
Meta 第三代网络安全评估套件，新增 Agent 场景风险：提示注入鲁棒性、代码解释器沙箱逃逸、终端/系统提示词注入、越狱等。
**对本课题的意义**：最贴近 CLI/代码 Agent 的工业级评估集，可作基准指标与场景设计参照。

**[49] Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations**
Meta（Llama Guard 3 发布于 2024）。https://arxiv.org/abs/2312.06674
基于 Llama 的判别式内容安全分类器，将输入/输出转为安全标签分类任务作为生产过滤器；Llama Guard 3 扩展代码解释器滥用等 Agent 相关违规类别（同类工业检测器：微软 Prompt Shields）。
**对本课题的意义**："检测类防御"代表基线——可在攻击实验中作为被评估对象，测其多轮漏检率。

**[50] LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet**
Zeng 等，2024（arXiv）。https://arxiv.org/abs/2408.15221
用人类规划者构造多轮对话越狱（MHJ），评估经数据训练、RLHF、LLM 检测器等防御后的模型，防御在一至三轮内显著退化，并发布 MHJ 基准。
**对本课题的意义**：多轮越狱防御评估的代表性研究，为"多轮对话安全"主题提供动机与基线对照。

**（内联补充）**
- **OWASP LLM Top 10（2025 版）**（https://owasp.org/www-project-top-10-for-large-language-model-applications/）与 **OWASP Agentic AI – Threats and Mitigations / Top 10 for Agentic Applications（2026）**（https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/）：提示注入居首（LLM01），配套最小权限、tool 白名单、上下文隔离、人工确认等缓解控制；
- **MITRE ATLAS**（https://atlas.mitre.org/）：基于真实事件构建的对抗战术矩阵，含提示注入、工具操纵、元提示劫持等 Agent 手法；
- **NIST AI 100-2e2025: Adversarial Machine Learning**（https://csrc.nist.gov/pubs/ai/100/2/e2025/final）：官方攻击/缓解分类学，含 LLM 提示注入、越狱、数据投毒缓解清单；
- **Security of LLM-based Agents regarding Attacks, Defenses, and Applications: A Comprehensive Survey**（Information Fusion 2025，https://www.sciencedirect.com/science/article/abs/pii/S1566253525010036）：2025 年 Agent 安全综述，梳理威胁面与防御手段；
- **Microsoft MSRC: How Microsoft Defends Against Indirect Prompt Injection Attacks**（2025-07，https://www.microsoft.com/en-us/msrc/blog/2025/07/how-microsoft-defends-against-indirect-prompt-injection-attacks）：生产级分层防御（隔离、过滤、权限分离）工程实证。

---

## 5. 攻击场景威胁建模与文献映射

将本课题的攻击场景分解为攻击链，并映射到调研文献：

| 阶段 | 攻击行为（CLI Agent 场景） | 支撑文献 |
|---|---|---|
| S1 侦察 | 探测 Agent 类型、可用工具、上下文来源、权限模式（是否 yolo/skip-permissions）；通过 ToolLeak 式手段外泄系统提示 | [9][26][42] |
| S2 初始接触 | 将恶意指令送入 Agent 上下文：对话诱导（直接）、恶意仓库/README/issue（间接）、MCP 工具描述、skill 文件、网页/日志 | [2][3][6][13][25][29][30][31][32][38][40][43] |
| S3 上下文操纵 | 多轮逐步改写会话目标：目标劫持、提示窃取、伪多轮历史、潜伏记忆植入 | [2][6][13][33][34][45] |
| S4 多轮诱导 | 对话升级逼近"读取/输出/外传秘密"；分解拆解任务绕过检测；树搜索/Agent 化自动探索 | [2][3][4][10][11][9][47] |
| S5 秘密定位 | 定位 API Key/.env/SSH key/云凭据：工具调用、文件读取、环境变量枚举；记忆残留恢复 | [37][35][39] |
| S6 秘密外渗 | 经工具调用外传：网络请求、写文件到可控位置、上传到攻击者仓库、经回复内容带出 | [14][41][38][39][43] |
| S7 持久化 | 写入记忆/配置/skill 实现跨会话复发：记忆投毒、环境注入、sleeper 攻击 | [21][31][33][34][45] |

关键观察：**单发注入 → 多轮持久化**是当前研究从"已做"到"缺失"的过渡带；真实攻击（[38]-[43]）多为单轮触发或单 PoC，而**系统化的"多轮对话 + 跨会话 + 隐私定向"攻击研究在 CLI/Coding Agent 场景中尚属空白**。

---

## 6. 研究空白与潜在创新点

基于上述调研，归纳以下空白（即本课题潜在创新点）：

1. **对象空白**：Crescendo/PAIR/TAP 等多轮攻击研究以聊天模型为对象；针对 CLI Agent 的实证研究（[26][36]）聚焦注入触发与 RCE，未系统研究"多轮对话逐步诱导隐私窃取"。**→ 可提出：面向 CLI Coding Agent 的多轮隐私窃取攻击框架（MTPE：Multi-Turn Privacy Exfiltration）。**
2. **目标空白**：现有编码 Agent 攻击目标多为"执行恶意代码/RCE/生成漏洞代码"；以"会话内秘密（API Key、凭据）定向窃取"为目标的系统研究少，且无统一基准。**→ 可构建 CLI 场景数据外渗基准（含多轮、多注入通道、多目标秘密类型）。**
3. **持续性空白**：记忆投毒研究（[33][34][35]）对象是 web agent/通用记忆，未针对编码 Agent 的 skills、`.claude/` 配置、索引与 shell 历史等特有持久化载体；**→ 可研究"跨会话潜伏窃取"变体。**
4. **防御空白**：CLI Agent 默认权限过大的矛盾（[46] Rule of Two）无解；检测类防御（[49]）对多轮退化（[50]）；**→ 可提出/评估面向 CLI 场景的分层防御（输出过滤 + 秘密标记 + 隔离 + 最小权限）并做攻防对照实验。**
5. **自动化空白**：可将第三章已有的 PAIR/RF Agent 攻击者迁移到工具调用轨迹上，学习"哪类工具调用序列最易解锁秘密"（与 [26] ToolLeak、[11] Siege 结合）；**→ 与论文前两章形成方法继承，是毕业论文"连续性叙事"的自然延伸。**

---

## 7. 建议的研究路径（与论文衔接）

1. **威胁模型与场景设计**：以 Claude Code / Codex CLI / Cursor CLI 为靶（或仿真沙箱，参照 [19] ToolEmu 思路降低风险），定义秘密集合（`OPENAI_API_KEY`、`.env`、`~/.ssh/id_rsa`、git credential、云 CLI 凭据）；
2. **攻击框架**：复用第三章 Agent 攻击器（分析-动作选择-记忆-多轮 RL），新增"工具轨迹感知"模块与"秘密外渗检测"奖励（替代原有 guard label）；
3. **注入通道矩阵**：覆盖 4 通道——直接对话、恶意仓库/README/issue、MCP 工具描述、skill 文件——并组合多轮策略（Crescendo 式升级 + Many-shot 伪历史 + 分解 + 潜伏记忆）；
4. **评估协议**：双指标（攻击成功率、任务效用保持率，参照 [16] AgentDojo）；防御基线为 Llama Guard/Prompt Shields 类检测器 + 输出过滤 + 权限白名单（[48][49][50]）；
5. **现实对标**：以 [41] Nx 事件为现实动机案例，[38]-[43] 作为攻击面证据。

---

## 8. 参考文献

**分类 1：通用 LLM Jailbreak 攻击方法**
1. Zou et al. Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG). ICLR 2024. https://arxiv.org/abs/2307.15043
2. Russinovich & Salem. Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack. USENIX Security 2025. https://arxiv.org/abs/2404.01833
3. Chao et al. Jailbreaking Black Box Large Language Models in Twenty Queries (PAIR). COLM 2024. https://arxiv.org/abs/2310.08419
4. Mehrotra et al. Tree of Attacks: Jailbreaking Black-Box LLMs Automatically (TAP). NeurIPS 2024. https://arxiv.org/abs/2312.02119
5. Liu et al. AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models. ICLR 2024. https://arxiv.org/abs/2310.04451
6. Anil et al. Many-shot Jailbreaking. Anthropic, NeurIPS 2024. https://www.anthropic.com/research/many-shot-jailbreaking
7. Yu et al. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts. NeurIPS 2023. https://arxiv.org/abs/2309.10253
8. Jiang et al. ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs. ACL 2024. https://arxiv.org/abs/2402.11753
9. RedAgent: Red Teaming Large Language Models with Context-aware Autonomous Language Agent. arXiv 2024. https://arxiv.org/abs/2407.16667
10. Li et al. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers. arXiv 2024. https://arxiv.org/abs/2402.16914
11. Siege: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search. arXiv 2025. https://arxiv.org/abs/2503.10619
12. Liu et al. Jailbreak Attacks and Defenses Against Large Language Models: A Survey. arXiv 2024. https://arxiv.org/abs/2407.04295 ；From LLMs to MLLMs to Agents: A Survey of … Jailbreak Attacks and Defenses within LLM Ecosystem. arXiv 2025. https://arxiv.org/abs/2506.15170

**分类 2：Agent 场景提示注入与工具攻击**
13. Greshake et al. Not what you've signed up for: … Indirect Prompt Injection. CCS 2023. https://arxiv.org/abs/2302.12173
14. Liu et al. Prompt Injection Attack against LLM-integrated Applications. IEEE S&P 2025 扩展版. https://arxiv.org/abs/2306.05499
15. Zhan et al. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents. arXiv 2024. https://arxiv.org/abs/2403.02691
16. Debenedetti et al. AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. NeurIPS 2024 D&B. https://arxiv.org/abs/2406.13352
17. Ye et al. ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages. arXiv 2024. https://arxiv.org/abs/2402.10753
18. Zhang et al. Agent Security Bench (ASB). IEEE S&P 2025. https://arxiv.org/abs/2410.02644
19. Ruan et al. Identifying the Risks of LM Agents with an LM-Emulated Sandbox (ToolEmu). ICLR 2024. https://arxiv.org/abs/2309.15817
20. Lin et al. Malla: Demystifying Real-world LLM Integrated Malicious Services. USENIX Security 2024. https://arxiv.org/abs/2401.03315
21. Chen et al. AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases. NeurIPS 2024. https://arxiv.org/abs/2407.12784
22. Lee et al. Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems. arXiv 2024. https://arxiv.org/abs/2410.07283

**分类 3：编码 / Coding Agent 专项安全研究**
23. Hajipour et al. CodeLMSec Benchmark. IEEE SaTML 2024. https://arxiv.org/abs/2302.04012
24. Improta et al. Poisoning Programs by Un-Repairing Code. arXiv 2024. https://arxiv.org/abs/2403.06675
25. Štorek et al. XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants. arXiv 2025. https://arxiv.org/abs/2503.14281
26. Xie et al. Red-Teaming Coding Agents from a Tool-Invocation Perspective. arXiv 2025. https://arxiv.org/abs/2509.05755
27. Chen et al. SecureVibeBench: Benchmarking Secure Vibe Coding of AI Agents. arXiv 2025. https://arxiv.org/abs/2509.22097
28. Nie et al. SeCodePLT: A Unified Platform for Evaluating the Security of Code GenAI. arXiv 2024. https://arxiv.org/abs/2410.11096
29. Maloyan & Namiot. Prompt Injection Attacks on Agentic Coding Assistants: A Systematic Analysis of Vulnerabilities in Skills, Tools, and Protocol Ecosystems. arXiv 2026. https://arxiv.org/abs/2601.17548
30. Jia et al. SkillJect: Effectively Automating Skill-Based Prompt Injection for Skill-Enabled Agents. arXiv 2026. https://arxiv.org/abs/2602.14211
31. Qu et al. Supply-Chain Poisoning Attacks Against LLM Coding Agent Skill Ecosystems. arXiv 2026. https://arxiv.org/abs/2604.03081
32. Singh et al. IssueTrojanBench: Benchmarking AI Coding Agents Against Malicious Issue Requests. arXiv 2026. https://arxiv.org/abs/2607.20759

**分类 4：隐私窃取、记忆投毒与数据外渗**
33. Pulipaka et al. Hidden in Memory: Sleeper Memory Poisoning in LLM Agents. arXiv 2026. https://arxiv.org/abs/2605.15338
34. Zou et al. Poison Once, Exploit Forever: Environment-Injected Memory Poisoning Attacks on Web Agents (eTAMP). arXiv 2026. https://arxiv.org/abs/2604.02623
35. Wang & Zhang. MemLeak: Diagnosing Information Leaks in Multimodal Agent Memory. arXiv 2026. https://arxiv.org/abs/2606.29788
36. Huang et al. Are AI-assisted Development Tools Immune to Prompt Injection? arXiv 2026. https://arxiv.org/abs/2603.21642
37. Nasr & Carlini et al. Scalable Extraction of Training Data from (Production) Language Models. ICLR 2025. https://arxiv.org/abs/2311.17035

**分类 5：CLI Agent 真实攻击案例与安全博客**
38. Donenfeld & Vanunu. Caught in the Hook: RCE and API Token Exfiltration Through Claude Code Project Files. Check Point Research, 2026. https://research.checkpoint.com/2026/rce-and-api-token-exfiltration-through-claude-code-project-files-cve-2025-59536/
39. Invariant Labs. MCP Security Notification: Tool Poisoning Attacks. 2025. https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks
40. Cox. Code Execution Through Deception: Gemini AI CLI Hijack. Tracebit, 2025. https://tracebit.com/blog/code-exec-deception-gemini-ai-cli-hijack
41. Tal. Weaponizing AI Coding Agents for Malware in the Nx Malicious Package. Snyk, 2025. https://snyk.io/blog/weaponizing-ai-coding-agents-for-malware-in-the-nx-malicious-package/
42. Huang et al. New Prompt Injection Attack Vectors Through MCP Sampling. Unit 42, 2025. https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/
43. Raina. MCP Horror Stories: The GitHub Prompt Injection Data Heist. Docker Blog, 2025. https://www.docker.com/blog/mcp-horror-stories-github-prompt-injection/
44. Microsoft Developer Blog. Protecting against indirect prompt injection attacks in MCP. 2025. https://developer.microsoft.com/blog/protecting-against-indirect-injection-attacks-mcp/
45. Edwards. Hacker plants false memories in ChatGPT to steal user data. Ars Technica, 2024（原始研究：embracethered.com/blog/posts/2024/chatgpt-hacking-memories/）. https://arstechnica.com/security/2024/09/false-memories-planted-in-chatgpt-give-hacker-persistent-exfiltration-channel/
46. Willison. New prompt injection papers: Agents Rule of Two and The Attacker Moves Second. 2025. https://simonwillison.net/2025/Nov/2/new-prompt-injection-papers/
47. Dubey. Prompt Injection and AI Agent Security Risks: A Claude Code Guide for Enterprise Teams. TrueFoundry, 2026. https://www.truefoundry.com/blog/claude-code-prompt-injection

**分类 6：防御、评估基准与治理框架**
48. Meta PurrlLlama. CYBERSECEVAL 3: Advancing the Evaluation of Cybersecurity Risks and Capabilities in LLMs. arXiv 2024. https://arxiv.org/abs/2408.01605
49. Inan et al. Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. Meta, 2023/2024. https://arxiv.org/abs/2312.06674
50. Zeng et al. LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet. arXiv 2024. https://arxiv.org/abs/2408.15221

**内联补充（未编号）**
- Simulation/Oasis: EchoLeak (arXiv:2509.10540)；Aikido/JetBrains（https://www.aikido.dev/blog/multiple-jetbrains-ide-plugins-caught-stealing-ai-keys）；HiddenLayer 恶意 Skills（https://www.hiddenlayer.com/research/the-next-ai-supply-chain-risk-malicious-skills-in-agentic-ai）；Trend Micro Slopsquatting（https://www.trendaisecurity.com/pl/resources-insights/deep-research/slopsquatting-when-ai-agents-hallucinate-malicious-packages）；Endor Labs AI Code Security Benchmark（https://www.endorlabs.com/research/ai-code-security-benchmark）
- OWASP LLM Top 10 2025 / Agentic AI Threats（https://owasp.org/www-project-top-10-for-large-language-model-applications/，https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/）；MITRE ATLAS（https://atlas.mitre.org/）；NIST AI 100-2e2025（https://csrc.nist.gov/pubs/ai/100/2/e2025/final）；Agent 安全综述（Information Fusion 2025，https://www.sciencedirect.com/science/article/abs/pii/S1566253525010036）；MSRC 间接注入防御（https://www.microsoft.com/en-us/msrc/blog/2025/07/how-microsoft-defends-against-indirect-prompt-injection-attacks）
- 其他可补充：DeepInception（arXiv:2311.03191）、Agent Smith（arXiv:2402.08567）、AgentPoint 类后门工作（检索未获权威出处，引用需谨慎）

---

*报告完*