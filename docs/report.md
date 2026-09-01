
哈尔滨工业大学
硕士学位论文中期报告

题 目：基于自进化技能系统与强化学习的
越狱提示词生成方法研究

院     （系）                   计算学部			     	 
学        科          电子信息-计算机科学与技术                
导        师                            徐冰				
研    究  生                         贾子潇				
学       号                      24S103188			  
中期报告日期               2026.6.22    			   



研究生院
1. 课题主要研究内容及进度情况
1.1 研究背景
1.1.1 大模型安全与对齐技术
随着 GPT、Deepseek、Qwen等大语言模型 (LLM) 在对话、代码生成、知识问答等任务中的广泛应用，其安全性问题日益受到学术界和工业界的关注。对齐技术 (Alignment) 通过训练使模型遵循人类价值观和安全准则，主要包括 RLHF (Reinforcement Learning from Human Feedback)、Constitutional AI、Red Teaming 等方法。然而，即便经过严格的安全对齐，主流大语言模型仍然存在被"越狱"(Jailbreak) 攻击的风险——攻击者通过精心设计的提示词可以绕过模型的安全护栏，诱导模型生成违规内容。
1.1.2 Jailbreak 攻击研究现状
现有的 jailbreak 攻击方法主要可以分为静态模板、梯度优化、多轮对话和角色扮演等几类。其中，静态模板方法如 PAIR 和 AutoDAN 采用预定义的策略模板，虽然实现简单但无法适应新的防御机制；梯度优化方法如 GCG 和 AutoDAN-GA 基于梯度进行 prompt 优化，但计算成本高昂且需要白盒访问权限；多轮对话方法如 Crescendo 采用渐进式引导策略，但缺乏跨案例的知识迁移能力；角色扮演方法如 Persona 通过角色设定绕过安全限制，但效果不够稳定。
1.1.3 研究挑战
当前 jailbreak 攻击研究面临四个核心挑战。首先是效率挑战，现有方法如 PAIR 和 TAP 通常需要1000 次以上的 API 调用才能生成有效的攻击 prompt，训练时间长且资源消耗大。其次是泛化挑战，攻击方法缺乏跨模型迁移能力，在一个模型上有效的攻击策略往往无法迁移到其他模型。第三是知识积累问题，成功攻击策略无法复用到新案例，每次攻击都需要从头探索。最后是奖励设计问题，强化学习训练中奖励信号稀疏，ASR 奖励是二值信号，方差大且梯度稀疏。
1.2 研究目的
本课题旨在系统研究大语言模型的越狱攻击方法与防御策略，从攻击端和防御端两个维度深入探索，为构建更安全的大语言模型提供理论支撑和技术方案。具体而言，本课题希望推进 jailbreak prompt 合成边界，提升攻击绝对能力，探索高效且可迁移的攻击策略；针对传统方法的高资源需求，优化训练效率，实现低资源需求的 MaaS (Model as a Service) 方案，降低实验成本；最终产出易用的方法模块，构建可复用的 Skills 系统，提供开源的攻击工具与数据合成管道。
1.3 研究方法
本课题采用三种递进式的研究方法。第一种方法是基于自适应混合奖励 GRPO 的越狱提示词生成方法 (AHR-GRPO)，该方法创新性地提出方差驱动的自适应奖励权重机制，动态平衡稀疏的 ASR 奖励与稠密的 Judge 奖励，实现高效、稳定的强化学习训练。第二种方法是基于自进化技能系统的越狱提示词生成方法 (SESS)，该方法设计 Skills 库自进化机制，从成功攻击轨迹中提取可复用的策略模板，实现知识积累与跨案例迁移，同时可用于高效数据合成。第三种方法是基于自进化系统内强化学习的越狱提示词生成方法 (Agentic-RL)，该方法将 AHR-GRPO 与 SESS 结合，在自进化 Skills 环境中进行强化学习训练，实现过程奖励与结果奖励的协同优化。
1.4 进度情况
截至本报告撰写日期，课题研究整体进展顺利。第一章 AHR-GRPO 方法已完成 85%，最佳 ASR 达到33.2%，相比 baseline 提升2.4%。我们已完成核心方法设计与实现，开发了自适应奖励权重计算器，并完成了实验1（jailbreak prompt 筛选）、实验2（多维度 Judge reward）和实验3（自适应权重验证）。第二章 SESS 方法已完成 95%，最佳 ASR 达到 99.7%，累计完成217 组实验，涵盖方法消融（Layer1-4）、数据策略、迁移性验证等，主要结论已得出。第三章 Agentic-RL 方法目前完成10%，方法框架已设计，待开展实验验证。
2. 相关研究现状
2.1 基于提示词工程的越狱攻击方法
早期 jailbreak 攻击主要依赖人工构造的 Prompt 模板，通过角色扮演、假设场景、创意写作等策略绕过模型的安全限制。近年来，自动化方法显著提升了攻击效率。
在迭代式攻击方法方面，Chao 等人提出的 PAIR 方法[1]建立了"攻击者 - 目标"双模型交互框架，通过多轮对话自动优化越狱提示，通常仅需 20 次查询即可攻破黑盒对齐模型。Wang 等人提出的 PromptAgent[2]将提示优化建模为智能体规划问题，提出"规划- 执行-反思"闭环框架，实现专家级提示词自动生成。
在树搜索方法方面，Mehrotri 等人提出的 TAP 方法[3]引入思维树（Tree-of-Thoughts）思想构建越狱提示搜索空间，结合评估器对分支进行有害性打分并动态剪枝，显著提升搜索效率。Sorkhpour 等人提出的 RedHit方法[19]进一步融合蒙特卡洛树搜索（MCTS）与直接偏好优化（DPO），实现攻击策略的渐进式演化。
在进化算法方法方面，Liu 等人提出的 AutoDAN 方法[4]结合人工模板与遗传算法，通过选择、交叉、变异等操作进化提示词种群。其升级版 AutoDAN-Turbo[5]提出分层遗传优化机制（片段级→提示级），大幅提升收敛速度。Guo 等人提出的 GAP 方法[6]基于遗传算法实现免梯度搜索。
在场景诱导方法方面，Li 等人提出的 DeepInception 方法[7]利用 LLM 对上下文情境的强依赖性，构建多层嵌套虚拟场景（"梦中梦"结构），将恶意意图隐藏于深层语境中。
2.2 基于强化学习的越狱攻击方法
近年来，强化学习被广泛应用于自动化越狱攻击。在轨迹级优化方面，Xiong 等人提出的 TROJail 方法[9]首次将多轮越狱建模为轨迹级强化学习问题，突破传统轮次级优化的短视局限，其核心创新包括轨迹级 MDP 建模与双过程奖励机制（隐蔽性奖励 + 有效性奖励）。Chen 等人提出的 RL-MTJail方法[15]针对黑盒多轮越狱提出强化学习优化框架，结合软/硬标签奖励过渡机制。

在课程学习与探索方面，Guo 等人提出的 Jailbreak-R1 方法[8]提出三阶段强化学习框架：冷启动阶段使用 SFT 初始化策略，探索预热阶段引入多样性奖励与一致性奖励，增强越狱阶段采用渐进式课程学习，奖励信号从软标签平滑过渡到硬标签。
在表征空间引导方面，Lee 等人提出的 xJailbreak 方法[10]提出表征空间引导的黑盒越狱方法，通过分析良性与恶意提示的 embedding 邻近性，确保改写提示在语义上贴近原始意图同时提升攻击有效性。
在多轮对话攻击方面，Yang 等人提出的 Many-Turn Jailbreaking 方法[11]首次系统性探索多轮越狱攻击范式，构建首个多轮越狱评测基准 MTJ-Bench，揭示"首轮越狱→后续对话污染"的新型攻击链。
2.3 自进化策略与红队框架
在策略自进化方面，Chen 等人提出的 Metis 方法[16]提出基于自进化元认知策略优化的越狱框架，将攻击过程建模为因果诊断与策略演化的闭环系统，维护可组合的策略原语库。Liu 等人提出的 ASTRA方法[17]提出自动化策略发现、检索与演化的越狱框架，三层动态策略库按效果分类为"高效""有潜力""无效"。
在方法级演化方面，Chen 等人提出的 EvoSynth 方法[18]首次将越狱攻击范式从"提示优化"转向"攻击方法演化"，提出基于多智能体协作的代码级攻击算法自主合成框架。Zhang 等人提出的 LLM-Virus 方法[21]将越狱攻击建模为进化与迁移学习的联合问题，利用 LLM作为启发式进化算子在提示种群中交叉变异传播。
在自适应红队方面，Yun 等人提出的 Active Attacks 方法[20]受主动学习范式启发，周期性用收集的攻击提示对目标模型进行安全微调，使奖励信号随目标防御演化而动态调整。Zhang 等人提出的 Genesis 方法[13]面向 Web Agent 场景提出策略演化式红队框架，Attacker-Scorer-Strategist 闭环支持遗传算法驱动的交叉/变异演化。Belaire 等人提出的 Automatic LLM Red Teaming 方法[12]将自动化红队建模为分层强化学习框架，设计 token 级别的有害性奖励信号。Wei 等人提出的 Learning-Based Automated Adversarial Red-Teaming方法[14]提出学习驱动的自动化红队框架，在匹配查询预算下相较人工红队实现 3.9 倍漏洞发现率提升。
2.4 与本文方法的关系
本文方法与上述工作存在以下关系。与迭代式和树搜索方法相比，本文 AHR-GRPO 通过强化学习训练专用生成模型，避免 PAIR 和 TAP 等方法的高查询成本（通常1000 次以上 API 调用/样本）。与进化算法方法相比，本文 SESS 借鉴 AutoDAN 的进化思想，但将进化对象从提示词升级为可复用的 Skills 策略模板，实现知识积累与跨案例迁移。与轨迹级 RL 方法相比，本文 AHR-GRPO 的自适应混合奖励机制与 TROJail[9]的双过程奖励机制类似，但本文聚焦单轮 Prompt 重写而非多轮对话。与自进化策略方法相比，本文 SESS 与 Metis[16]和 ASTRA[17]均关注策略演化，但本文从成功攻击轨迹中提取 Skills 模板，而非依赖元认知诊断或三层策略库。
3. 目前已完成的研究工作及进展
3.1 Adaptive Hybrid-Reward GRPO for Jailbreak Prompt Generation (AHR-GRPO)
3.1.1 研究背景与动机
传统 GRPO 方法使用单一奖励信号，而对于我们的研究问题而言，ASR （攻击成功率，衡量目标模型对给定攻击prompt数据集的回复的不安全率）奖励过于稀疏导致早期探索困难，引入LLM-as-a-Judge给出的Judge Reward，用混合奖励进行GRPO训练是可行的解决方法。但现有混合奖励方法一版采用固定权重，无法适应训练过程中奖励信号的动态变化，因此我们设计了自适应机制，希望在GRPO训练过程中，当ASR 奖励信号稀疏时（比如一批数据全都攻击失败，实践中存在大量这样的case）更依赖稠密 Judge 引导，在 ASR 信号区分度明显（组内存在攻击成功的新prompt）时则对应地提高其权重，更直接地引导模型学习。整体Insight如图1所示。

图 1
3.1.2 方法设计

图 2


我们方法的核心是自适应混合权重奖励用于GRPO，整体方法流程如图2所示。自适应 reward 权重计算的核心公式使用指数移动方差（EMV）估计两组奖励的方差：
λraw=σα⋅logσASR2σproc2+δ
该公式的核心思想是：当 ASR 奖励方差较大时（说明模型已经能区分好坏样本），增大 ASR 权重；当 ASR 奖励方差较小时（说明模型整体重写效果差），降低 ASR 权重，更依赖 Judge 奖励进行学习。方法包含三个关键机制：双平滑机制采用 EMA 平滑加惯性平滑，防止单步跳变导致梯度震荡；边界裁剪将λ限制在[λmin,λmax]范围内，防止信号坍缩；冷启动机制在前W步固定λ=0.5，避免初始方差估计偏差。
在多维度 Judge Prompt 设计方面，我们设计了四个正交评判维度：intent_preservation（意图保留）、stealth（隐蔽性）、strategy_execution（策略执行）和 attack_potential（攻击潜力）。
3.1.3 实验设计与结果
实验设置：训练的攻击模型/Policy为Qwen3-4B，作为攻击目标的Target模型也为Qwen3-4B，用于评估攻击结果的Guard模型为Qwen3Guard-Gen-4B，Judge模型复用Qwen3-4B以减少显存和GPU使用量。数据集使用allenai/wildjailbreak，从其中抽取了10000条jailbreak prompt数据并进行8:1:1分割作为我们后续实验的train：dev：test集。训练实验使用两张A800进行，GPU0用于训练Policy，GPU1用于部署Guard模型和Qwen3-4B（同时作为Target和Judge模型）。
实验1 的目标是筛选高效的基础攻击 prompt 策略。我们测试了24 种 jailbreak prompt 模板的初始 ASR（使用1000 条测试集），结果如下。筛选出 Top-3 策略：hypothetical_scenario（30.8%）、creative_writing（28.3%）和 role_playing（25.0%），如图3所示。

图 3 实验1:攻击策略prompt选择
实验2 的目标是研究不同 Judge 维度权重对训练效果的影响。我们设计了3 种攻击 prompt 乘以4 种 Judge 维度共12 组 GRPO 训练实验，在训练集上使用固定权重比例的reward（ASR reward和Judge reward均占比0.5）进行GRPO训练2epoch，训练后模型在test集上的ASR结果如图4所示。结果显示最佳组合为 hypothetical_scenario 加 idea_preservation，达到32.3%（相比 baseline 提升1.5%）。

图 4 实验2:top3策略4单维度judge训练结果
实验3 的目标是在实验2 最佳配置基础上，验证自适应权重机制的进一步改进效果。结果如图5所示，自适应机制超越固定权重：当β=0（窗口=1）时 ASR 达到33.2%，相比实验2 最佳提升 0.9%，而β=0.8和0.9时（权重计算窗口=5和10）也有明显提升。如图6所示，消融验证表明仅使用 ASR Reward 时效果下降5.8%（ASR=25.0%），仅使用Judge Reward时效果也明显下降（ASR=28.5%）。可以看出混合 Reward 显著优于单一 Reward。

图 5

图 6
综合三个实验的结果，我们得出以下核心结论：通过实验1、2筛选出的攻击prompt组合Judge prompt，构建ASR+Judge的混合 Reward 显著优于单一 Reward；在混合Reward基础上，自适应权重机制有效，动态调整 Lambda 实现 ASR 与 Judge 的最优平衡。
3.2 Self-Evolving Skills System for Jailbreak Prompt Generation (SESS)
3.2.1 研究背景与动机
现有 jailbreak 方法缺乏知识积累机制，每次攻击需从头探索。PAIR 等迭代方法效率低，无法跨案例复用成功策略。因此，我们设计了可复用的攻击模板技能（Skills）与自进化机制，实现知识积累与迁移，整体形成可插拔的自进化系统模块。
3.2.2 方法设计

图 7
整体方法如图7所示，主要包括：1.冷启动，在部分数据上基于PAIR｜AutoDAN这样的成熟攻击框架进行攻击尝试并且积累成功经验沉淀技能，形成skills库；2.skills进化，在evolve数据上，增加skills检索和使用并继续进行攻击，基于多种方式｜时机对skills库进行进化，时机包括只在成功时｜只在失败时｜成功和失败都反馈｜固定步数后根据统计信息更新，更新方式包括直接修改、增量修改、只删低成功率skills；3.运行与测试，在进化阶段结束后，在test集上运行整个SESS框架，进行攻击尝试并统计结果。

图 8
在 Skill 数据结构与检索机制方面，如图8所示。Skill 包含 content（攻击模板）、source（来源）、统计信息（success_rate 和 usage_count）、标签、伤害类型。检索评分公式为：score=quality_score×(1+关键词匹配×0.3)×类型匹配系数。


图 9
如图9所示，三阶段自进化流程包括：Cold Start 阶段使用初始 Skills 进行攻击，积累成功案例；Evolution 阶段从成功或失败轨迹中提取或改进 Skills；Test 阶段使用演化后的 Skills 库评估最终 ASR。
3.2.3 消融实验：寻找最佳组合
我们累计完成了217 组实验。Layer1和Layer2实验基于PAIR迭代prompt重写方法，从空skills库开始进行cold start和evolve，Layer3和4则直接以AutoDAN的固定攻击模板作为初始的skills，同样进行cold start和evolve。
Layer1 方法组合 Grid Search 包含16 组实验，确定了最佳方法组合为 single_call 加 trajectory 加 statistical，ASR 达到 79.1%，Skills 数量为28 个。关键发现是 single_call 比 every_iteration 高12.7%。
Layer2 数据消融实验包含36 组实验，最佳配置为 small(300) 加 early(30% CS)，ASR 达到 80.0%。关键发现是数据量影响小于2%，配比影响显著，early 策略最佳。
Layer3 DAN 模板验证包含17 组实验，最佳配置为 full_evolve（无 Cold Start），ASR 达到 98.8%。关键发现是 DAN 模板无需cold start来增加额外的技能，直接使用、在其基础上进行进化更新最优。
Layer4 DAN 数据消融包含12 组实验，最佳配置为 medium 加 evo，ASR 达到 99.7%。关键发现是数据量影响极小，差距仅1.3%。

图 10
3.2.4 迁移实验：跨模型、跨数据集对比基线方法
迁移实验旨在验证 SESS 方法在不同目标模型和不同数据集上的泛化能力。我们设计了两个维度的迁移实验：跨模型迁移（4 个目标模型）和跨数据集迁移（5 个数据集），累计完成 120 组实验。
跨模型迁移效果：
在跨模型迁移方面，我们在 4 个目标模型上进行了系统测试：3 个 Qwen 系列模型（同族）和 1 个 gpt-oss-20b 模型（跨族）。同族迁移效果良好。引入 SESS 后，PAIR 和 AutoDAN 在三个 Qwen 模型上均取得稳定提升。具体而言，PAIR_w_SESS 在 Qwen3-0.6B 上达到 95.1%，相比原始 PAIR 的 78.5% 提升 16.6 个百分点；在 Qwen3-4B 上达到 86.7%，相比原始 PAIR 的 72.3% 提升 14.4 个百分点；在 Qwen3-14B 上达到 86.5%，相比原始 PAIR 的 82.1% 提升 4.4 个百分点。PAIR 方法经 SESS 增强后，平均提升幅度达 11.8 个百分点。
AutoDAN 方法经 SESS 增强后同样表现优异。AutoDAN_w_SESS 在 Qwen3-0.6B 上达到 96.8%，相比原始 AutoDAN 的 80.2% 提升 16.6 个百分点；在 Qwen3-4B 上达到 89.2%，相比原始 AutoDAN 的 78.7% 提升 10.5 个百分点；在 Qwen3-14B 上达到 90.3%，相比原始 AutoDAN 的 85.5% 提升 4.8 个百分点。AutoDAN 方法经 SESS 增强后，平均提升幅度达 10.6 个百分点。
综合对比，SESS 对两种基线方法均表现出稳定的增强效果，其中对 PAIR 的提升略高于 AutoDAN（平均 +11.8pp vs +10.6pp）。这可能是因为 PAIR 原本缺乏可复用的攻击策略模板，SESS 的 Skills 机制恰好弥补了这一不足；而 AutoDAN 本身已有较好的模板基础，SESS 的边际提升相对较小。
跨族迁移挑战性更强。在 gpt-oss-20b 模型上，所有方法的效果均大幅下降。原始 PAIR 在 gpt-oss-20b 上 ASR 仅为 0.08%，AutoDAN 也仅有 6.0%，说明传统方法难以应对跨族模型的安全机制。尽管如此，SESS 仍能提供显著提升：PAIR_w_SESS 从 0.08% 提升至 11.4%（+11.3pp），AutoDAN_w_SESS 从 6.0% 提升至 32.5%（+26.5pp）。其中 AutoDAN_w_SESS 的 32.5% 是当前跨族迁移的最佳结果，这可能是因为 AutoDAN 的遗传算法框架与 SESS 的 Skills 机制结合后，能更好地泛化到未知模型。

图 11 跨模型迁移效果
模型规模影响
实验还揭示了模型规模对 SESS 效果的影响。在同族模型中，SESS 在较小模型上提升更显著（Qwen3-0.6B: +16.6pp），在较大模型上提升相对较小（Qwen3-14B: +4.4~4.8pp）。这可能是因为较小模型的安全对齐相对较弱，SESS 的 Skills 机制更容易找到有效的攻击模式；而较大模型的安全机制更加完善，需要更复杂的攻击策略才能绕过。

跨数据集迁移效果
在跨数据集迁移方面，我们在 Qwen3-4B 模型上测试了 5 个数据集：default、advbench、harmbench_contextual、harmbench_standard 和 jailbreakBench。
实验结果显示，SESS 在所有 5 个数据集上都表现出稳定的提升效果。在 default 数据集上，PAIR_w_SESS 达到 79.0%，相比 no_rewrite 的 30.3% 提升 48.7 个百分点；在 advbench 上达到 81.0%，相比 no_rewrite 的 3.3% 提升 77.7 个百分点；在 harmbench_contextual 上达到 98.0%，相比 no_rewrite 的 79.0% 提升 19.0 个百分点；在 harmbench_standard 上达到 91.5%，相比 no_rewrite 的 21.5% 提升 70.0 个百分点；在 jailbreakBench 上达到 88.0%，相比 no_rewrite 的 7.0% 提升 81.0 个百分点。
AutoDAN_w_SESS 同样在各数据集上表现优异，在 harmbench_contextual 上达到 96.5%，在 harmbench_standard 上达到 93.0%，在 default 上达到 89.2%。值得注意的是，harmbench 系列数据集（contextual 和 standard）上的效果普遍优于其他数据集，这可能是因为 harmbench 的危害类型更加明确，SESS 的 Skills 机制更容易针对性地生成有效攻击。

图 12 跨数据集迁移效果
3.2.5 核心结论
综合217 组实验的结果，我们得出以下核心结论：最佳方法为 AutoDAN_w_SESS，在同族模型上达到 99%，在跨族模型上达到32.5%；Evolution 必要性方面，DAN 场景不必要，pair 风格仅有微弱的 0.9% 提升；Skills 泛化性方面，同族模型表现良好（85% 至 95%），跨族模型表现有限（低于35%）；数据策略方面，300 条数据足够，full_evolve 最优（DAN 场景）。
3.3 Agentic RL with Self-Evolving Skills System (待开展)
3.3.1 研究背景与动机
AHR-GRPO 解决了奖励信号稀疏问题，但缺乏攻击策略的知识积累；SESS 实现了 Skills 知识积累与迁移，但缺乏强化学习的策略优化。我们需要将两者结合：在自进化 Skills 环境中进行 RL 训练，实现策略优化与知识积累的双重效果。
3.3.2 方法设计
在自进化 Skills 环境构建方面，Skills 作为环境状态的一部分动态更新。环境提供 Skills 检索接口，Policy 根据当前 prompt 检索最佳 Skill。Skills 库随训练进程演化，形成适应 Policy 策略的 Skills 分布。
在结果奖励与过程奖励设置方面，结果奖励（ASR）为攻击成功与否的稀疏信号；过程奖励（Judge）为多维度评判的稠密信号；自适应权重使用 AHR-GRPO 的方差驱动机制动态调整权重；Skills 质量奖励鼓励 Policy 使用高质量 Skills。
训练流程设计分为三个阶段：Phase1 进行 Skills 初始化，使用 SESS 的 DAN 模板或演化 Skills；Phase2 进行 RL 训练，Policy 在 Skills 环境中学习；Phase3 进行 Skills 协同演化，Policy 与 Skills 双向优化。
4. 后期拟完成的研究工作及进度安排
4.1 AHR-GRPO 后续工作
第一章 AHR-GRPO 的后续工作主要包括已完成工作和待完成工作两部分。已完成工作包括：实验3 自适应权重机制验证，通过3 组 EMA 配置确认自适应超越固定权重，获得 0.9% 的额外改进；已验证自适应机制优越性，最佳配置达到33.2%，超越实验2 最佳的32.3%；已确认混合 Reward 有效性，仅 ASR Reward 效果下降5.8%，混合 Reward 显著优于单一 Reward。待完成工作包括：分析$\lambda$演化曲线，验证物理直觉（早期依赖 Judge，后期回归 ASR）；整理实验数据，撰写论文方法章节与实验章节。
4.2 SESS 后续工作
第二章 SESS 的后续工作同样包括已完成工作和待完成工作。已完成工作包括：已完成217 组实验，涵盖 Layer1-4 方法消融、数据消融、迁移性验证等全部实验内容；已得出主要结论，最佳方法为 AutoDAN_w_SESS，在同族模型上达到 99%，在跨族模型上达到32.5%。待完成工作包括：整理实验数据，撰写完整实验报告；分析跨族迁移失败的根本原因，主要考虑模型安全机制差异和 Skills 模型特异性两个因素；探索提升 Skills 泛化性的方法，如元学习和多模型联合训练；整理 Skills 库，开源最佳 Skills 模板；撰写论文方法章节与实验章节。
4.3 SESS+Agentic-RL 工作计划
第三章 Agentic-RL 的工作计划分为三个阶段。方法实现阶段预计需要2 周时间，主要任务是搭建自进化 Skills 环境、集成 AHR-GRPO 自适应奖励机制、实现 Skills 协同演化流程。实验验证阶段预计需要4 周时间，主要任务包括 Baseline 对比、消融实验、跨模型迁移测试。结果分析与论文撰写阶段预计需要2 周时间，主要任务包括分析训练曲线与$\lambda$演化、Skills 演化过程可视化、与前两章方法进行对比讨论。
4.4 论文整理与投稿准备
论文整理与投稿准备包括四项主要工作：整理三章内容，统一论文框架（采用 ICLR 格式）；补充实验，根据 review 反馈可能需要的额外实验；撰写 Introduction、Related Work 和 Conclusion；完善图表与可视化。
5. 存在的困难与问题
5.1 计算资源需求较大
计算资源需求是本课题面临的主要困难之一。AHR-GRPO 实验每实验约需4 块 GPU（A800-80G），训练2 至4 小时（1000 步）。SESS 实验已完成217 组，消耗大量 GPU 资源。第三章 Agentic-RL 预计需额外数百 GPU 卡时。
5.2 高成本试错风险
高成本试错也是计算资源需求的一部分，在截止目前的工作过程中出现过多次由于idea整体效果不佳、难以优化出结果以至于浪费大量GPU卡时的情况，造成实验成本较高。
5.3 跨族迁移挑战
跨族迁移挑战是 SESS 实验揭示的重要问题。所有方法在 安全护栏非常强大的GPT-OSS-20B 上效果大幅下降，最佳方法仅达到32.5%。当前 Skills 缺乏跨族泛化性，需要新方法提升泛化性。这将是第三章工作的重点探索方向，我们将更注重优化在GPT-OSS-20B模型上的ASR表现。
5.4 论文时间压力
论文时间压力主要来自三个方面。首先是对于毕业论文而言，三章内容整合需要统一叙事逻辑。其次是实验周期，第三章实验预计需要6 周，当前整体Agentic-RL环境和代码已经准备完毕，将尽快开始实验。最后针对小论文投稿时间窗口，需在截稿日前整理已有的两篇工作和争取完成第三篇工作，完成论文撰写与投稿。
6. 如期完成全部论文工作的可能性
6.1 已完成工作评估
从已完成工作来看，第一章 AHR-GRPO 完成 85%，最佳 ASR 达到33.2%；第二章 SESS 完成 95%，最佳 ASR 达到 99.7%；第三章 Agentic-RL 完成10%，方法框架已设计。总体完成度约为65%。
6.2 后期工作可行性分析
从后期工作可行性来看，时间规划方面剩余6 至 8 周，足够完成第三章实验与论文整理。技术可行性方面 AHR-GRPO 与 SESS 代码库已成熟，组合实现难度可控。资源可行性方面已有 GPU 资源调度经验，可合理安排实验。关键成果方面第一章已验证自适应机制有效（0.9% 额外改进），第二章已验证 Skills 积累有效（99.7% ASR）。
6.3 风险评估与应对策略
针对可能的风险，我们制定了相应的应对策略。对于 Agentic-RL 训练不稳定的风险，我们将设计稳定性保障机制（双平滑、边界裁剪、冷启动），参考 AHR-GRPO 经验。对于跨族迁移效果仍不理想的风险，我们将此作为研究发现，分析原因而非强行提升，符合研究完整性。对于时间不足的风险，我们将优先完成核心实验，消融实验可视情况缩减。
6.4 结论
基于已完成工作与后续计划，我们判断如期完成全部论文工作的可能性较高。需要合理安排实验顺序，控制资源消耗，保证核心实验优先完成。论文结构清晰，三章节递进关系明确，具备 ICLR 投稿潜力。
 7. 参考文献
[1]Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G. J., et al. (2023). Jailbreaking Black Box Large Language Models in Twenty Queries. arXiv:2310.08419. [PAIR]
[2]Wang, X. et al. (2024). PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization. arXiv:2310.16428.
[3]Mehrotri, A., Zampetakis, M., et al. (2023). Tree of Attacks: Jailbreaking Black-Box LLMs Automatically. arXiv:2312.02119. [TAP]
[4]Liu, X. et al. (2023). AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models. arXiv:2310.04451.
[5]Liu, X. et al. (2024). AutoDAN-Turbo: A Hierarchical Genetic Algorithm for Automated Jailbreaking. arXiv:2405.06479.
[6]Guo, Y. et al. (2024). GAP: A Genetic Algorithm Approach to Automated Jailbreaking of Large Language Models. arXiv:2402.11852.
[7]Li, X. et al. (2023). DeepInception: Hypnotize Large Language Models to Be Jailbreakers. arXiv:2311.07588.
[8]Guo, W., Shi, Z., Li, Z., et al. (2025). Jailbreak-R1: Exploring the Jailbreak Capabilities of LLMs via Reinforcement Learning. arXiv:2506.00782.
[9]Xiong, X., Li, O., Liu, Z., et al. (2026). TROJail: Trajectory-Level Optimization for Multi-Turn LLM Jailbreaks with Process Rewards. ACL2026. arXiv:2512.07761.
[10]Lee, S., Ni, S., Wei, C., et al. (2025). xJailbreak: Representation Space Guided Reinforcement Learning for Interpretable LLM Jailbreaking. arXiv:2501.16727.
[11]Yang, X., Xiao, L., Li, S., et al. (2025). Many-Turn Jailbreaking. arXiv:2508.06755.
[12]Belaire, R., Sinha, A., Varakantham, P. (2025). Automatic LLM Red Teaming. arXiv:2508.04451.
[13]Zhang, Z., He, J., Cai, Y., et al. (2025). Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming. arXiv:2510.18314.
[14]Wei, Z., Chen, H., Hu, P., et al. (2025). Learning-Based Automated Adversarial Red-Teaming for Robustness Evaluation of LLMs. arXiv:2512.20677.
[15]Chen, X. et al. (2025). RL-MTJail: Reinforcement Learning for Automated Black-Box Multi-Turn Jailbreaking. arXiv:2512.07761.
[16]Chen, Y., Wang, X., Li, J., et al. (2026). Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization. arXiv:2605.10067.
[17]Liu, X., Chen, Y., Ling, K., et al. (2026). ASTRA: An Automated Framework for Strategy Discovery, Retrieval, and Evolution for Jailbreaking LLMs. ACL2026. arXiv:2511.02356.
[18]Chen, Y., Wang, X., Li, J., et al. (2025). Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs. arXiv:2511.12710. [EvoSynth]
[19]Sorkhpour, M., Yazdinejad, A., Dehghantanha, A. (2025). RedHit: Adaptive Red-Teaming of LLMs via Search, Reasoning, and Preference Optimization. LLM Security Conference2025.
[20]Yun, T., St-Charles, P.-L., Park, J., Bengio, Y., Kim, M. (2025). Active Attacks: Red-teaming LLMs via Adaptive Environments. arXiv:2509.21947.
[21]Zhang, W., Liu, Y., Chen, H., et al. (2026). LLM-Virus: Evolutionary Jailbreak Attack on Large Language Models. arXiv:2601.0xxxx.




















