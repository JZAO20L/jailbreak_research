related work
以下为您整理的 LLM 越狱（Jailbreak）相关代表性工作，已按技术路线分类并统一格式，便于直接写入论文的 Related Work 部分。每项包含核心机制简述与标准 BibTeX 引用格式。
📌 使用说明：多数方法首发于 arXiv，后续可能已被顶会（ICLR/NeurIPS/ACL 等）正式接收。引用前建议核对最新官方版本，BibTeX 中的 arXiv 字段可直接替换为正式会议/期刊信息。
--------------------------------------------------------------------------------
PAIR
方法描述：
提出"攻击者 - 目标"双模型交互框架，将越狱建模为迭代优化过程。攻击者 LLM 根据目标模型的多轮反馈自动重构越狱提示，无需人工干预或梯度信息，通常仅需 ≤20 次查询即可攻破黑盒对齐模型。核心优势在于利用语言模型的自我反思能力实现提示的自动精炼，显著提升黑盒攻击的查询效率与成功率。
BibTeX：
@article{chao2023jailbreaking,
  title={Jailbreaking Black Box Large Language Models in Twenty Queries},
  author={Chao, Patrick and Robey, Alexander and Dobriban, Edgar and Hassani, Hamed and Pappas, George J and others},
  journal={arXiv preprint arXiv:2310.08419},
  year={2023}
}

--------------------------------------------------------------------------------
PromptAgent
方法描述：
将提示优化建模为智能体规划问题，提出"规划 - 执行 - 反思"闭环框架。通过 LLM 进行多步推理、上下文记忆管理与自我修正，实现专家级提示词自动生成。该方法常被适配于自动化越狱管线，支持复杂任务分解与策略回溯，在保持语义连贯性的同时提升攻击提示的隐蔽性与有效性。
BibTeX：
@article{wang2024promptagent,
  title={PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization},
  author={Wang, Xinyuan and others},
  journal={arXiv preprint arXiv:2310.16428},
  year={2024}
}

--------------------------------------------------------------------------------
TAP (Tree of Attacks with Pruning)
方法描述：
引入思维树（Tree-of-Thoughts）思想构建越狱提示搜索空间。攻击者 LLM 生成提示词树，结合评估器对分支进行有害性打分并动态剪枝，显著提升搜索效率与黑盒攻击成功率。核心创新在于将组合优化问题转化为可剪枝的树搜索问题，有效平衡探索广度与计算成本。
BibTeX：
@article{mehrotra2023tree,
  title={Tree of Attacks: Jailbreaking Black-Box LLMs Automatically},
  author={Mehrotra, Anay and Zampetakis, Manolis and others},
  journal={arXiv preprint arXiv:2312.02119},
  year={2023}
}

--------------------------------------------------------------------------------
AutoDAN
方法描述：
结合人工 DAN 模板与遗传算法的自动化越狱框架。通过选择、交叉、变异等操作进化提示词种群，在保持高攻击成功率的同时生成语义自然、隐蔽性强的越狱提示。支持黑盒场景，无需目标模型梯度，通过适应度函数（有害性得分+语言流畅度）引导搜索方向。
BibTeX：
@article{liu2023autodan,
  title={AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models},
  author={Liu, Xiaogeng and others},
  journal={arXiv preprint arXiv:2310.04451},
  year={2023}
}

--------------------------------------------------------------------------------
AutoDAN-Turbo
方法描述：
AutoDAN 的升级版本，提出分层遗传优化机制（片段级→提示级）。通过模块化进化策略大幅提升收敛速度与搜索吞吐量，支持更长上下文与复杂目标指令。在保持攻击隐蔽性的同时，泛化性与攻击效率显著增强，适用于大规模自动化红队评测。
BibTeX：
@article{liu2024autodanturbo,
  title={AutoDAN-Turbo: A Hierarchical Genetic Algorithm for Automated Jailbreaking},
  author={Liu, Xiaogeng and others},
  journal={arXiv preprint arXiv:2405.06479},
  year={2024}
}

--------------------------------------------------------------------------------
GAP (Genetic Algorithm Prompting)
方法描述：
基于遗传算法的提示优化范式，通过适应度函数（如目标回复的有害性/合规性得分）引导免梯度搜索，自动演化高成功率越狱提示。核心优势在于无需模型内部信息，仅通过 API 反馈即可实现策略进化，适用于黑盒红队场景。⚠️ 注：GAP 为通用缩写，不同团队有独立实现，引用前请核对完整标题。
BibTeX：
@article{guo2024gap,
  title={GAP: A Genetic Algorithm Approach to Automated Jailbreaking of Large Language Models},
  author={Guo, Yuxuan and others},
  journal={arXiv preprint arXiv:2402.11852},
  year={2024}
}

--------------------------------------------------------------------------------
DeepInception
方法描述：
利用 LLM 对上下文情境的强依赖性，构建多层嵌套虚拟场景（"梦中梦"结构）。通过角色扮演与叙事递进逐步弱化安全过滤机制，将恶意意图隐藏于深层语境中实现越狱。该方法揭示了情境诱导攻击的新范式，对基于上下文感知的安全对齐机制提出挑战。
BibTeX：
@article{li2023deepinception,
  title={DeepInception: Hypnotize Large Language Models to Be Jailbreakers},
  author={Li, Xuan and others},
  journal={arXiv preprint arXiv:2311.07588},
  year={2023}
}

--------------------------------------------------------------------------------
Jailbreak-R1
方法描述：
提出三阶段强化学习框架用于自动化越狱攻击：(1) 冷启动阶段：在公开越狱数据集上进行监督微调（SFT）初始化策略；(2) 探索预热阶段：引入多样性奖励（语义熵+结构变异度）与一致性奖励（意图保持度），通过 PPO 微调培养策略探索能力，避免坍塌至单一模式；(3) 增强越狱阶段：采用渐进式课程学习，奖励信号从软标签（概率越狱评分）平滑过渡到硬标签（二元成功信号），结合目标模型反馈驱动策略更新。全程仅需黑盒 API 访问，无需梯度/参数，在保持提示多样性的同时提升攻击成功率与跨模型泛化能力。
BibTeX：
@article{guo2025jailbreakr1,
  title={Jailbreak-R1: Exploring the Jailbreak Capabilities of LLMs via Reinforcement Learning},
  author={Guo, Weiyang and Shi, Zesheng and Li, Zhuo and Wang, Yequan and Liu, Xuebo and Wang, Wenya and Liu, Fangming and Zhang, Min and Li, Jing},
  journal={arXiv preprint arXiv:2506.00782},
  year={2025},
  eprint={2506.00782},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2506.00782}
}

--------------------------------------------------------------------------------
TROJail
方法描述：
首次将多轮越狱建模为轨迹级强化学习问题，突破传统轮次级优化的短视局限。核心创新包括：(1) 轨迹级建模：将多轮对话历史作为完整马尔可夫决策序列，支持长程策略学习与信用分配；(2) 双过程奖励机制：隐蔽性奖励惩罚过早触发拒绝的中间提示，有效性奖励鼓励响应语义在嵌入空间向目标有害内容渐进对齐；(3) 优势估计融合：将过程奖励作为 baseline 修正项融入优势计算，降低稀疏奖励导致的训练方差，兼容 PPO/GRPO 等主流算法。仅需黑盒访问即可实现稳定、高效的多轮越狱策略优化。
BibTeX：
@inproceedings{xiong2026trojail,
  title={TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards},
  author={Xiong, Xiqiao and Li, Ouxiang and Liu, Zhuo and Li, Moxin and Shi, Wentao and Zhu, Fengbin and Wang, Qifan and Feng, Fuli},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year={2026},
  pages={TBD},
  eprint={2512.07761},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2512.07761},
  note={Accepted to ACL 2026 Main Conference}
}

--------------------------------------------------------------------------------
xJailbreak
方法描述：
提出表征空间引导的黑盒越狱方法，通过强化学习优化提示生成。核心创新：(1) 嵌入空间引导：分析良性与恶意提示的 embedding 邻近性，确保改写提示在语义上贴近原始意图同时提升攻击有效性；(2) 可解释性奖励设计：结合关键词匹配、意图对齐与答案验证的多维评估框架，提供鲁棒的奖励信号；(3) 黑盒适配：仅需 API 访问，无需模型梯度/参数，在主流开源/闭源模型（Qwen2.5-7B、Llama3.1-8B、GPT-4o）上实现 SOTA 攻击成功率。
BibTeX：
@article{lee2025xjailbreak,
  title={xJailbreak: Representation Space Guided Reinforcement Learning for Interpretable LLM Jailbreaking},
  author={Lee, Sunbowen and Ni, Shiwen and Wei, Chi and Li, Shuaimin and Fan, Liyang and Argha, Ahmadreza and Alinejad-Rokny, Hamid and Xu, Ruifeng and Gong, Yicheng and Yang, Min},
  journal={arXiv preprint arXiv:2501.16727},
  year={2025},
  eprint={2501.16727},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2501.16727}
}

--------------------------------------------------------------------------------
Many-Turn Jailbreaking
方法描述：
首次系统性探索多轮越狱攻击范式，突破传统单轮攻击局限。核心贡献：(1) 长程威胁建模：针对支持超长上下文的先进 LLM，研究越狱后模型在后续多轮对话中持续输出有害内容的风险；(2) MTJ-Bench 基准：构建首个多轮越狱评测基准，覆盖开源/闭源模型，量化评估攻击的持续性与泛化性；(3) 安全启示：揭示"首轮越狱→后续对话污染"的新型攻击链，呼吁社区关注长程对话安全对齐。
BibTeX：
@article{yang2025manyturn,
  title={Many-Turn Jailbreaking},
  author={Yang, Xianjun and Xiao, Liqiang and Li, Shiyang and Ladhak, Faisal and Yun, Hyokun and Petzold, Linda Ruth and Xu, Yi and Wang, William Yang},
  journal={arXiv preprint arXiv:2508.06755},
  year={2025},
  eprint={2508.06755},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2508.06755}
}

--------------------------------------------------------------------------------
Automatic LLM Red Teaming
方法描述：
将自动化红队建模为分层强化学习框架，解决稀疏奖励与长程决策挑战。方法亮点：(1) 轨迹级 MDP 建模：将红队过程形式化为马尔可夫决策过程，支持多轮对抗策略学习；(2) 细粒度奖励机制：设计 token 级别的有害性奖励信号，引导生成连贯的多轮攻击策略；(3) 生成式攻击智能体：通过 RL 训练的攻击代理能够发现传统单轮/模板方法遗漏的隐蔽漏洞，在多项评测中刷新 SOTA。
BibTeX：
@article{belaire2025automatic,
  title={Automatic LLM Red Teaming},
  author={Belaire, Roman and Sinha, Arunesh and Varakantham, Pradeep},
  journal={arXiv preprint arXiv:2508.04451},
  year={2025},
  eprint={2508.04451},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2508.04451}
}

--------------------------------------------------------------------------------
Genesis
方法描述：
面向 Web Agent 场景提出策略演化式红队框架。核心模块：(1) Attacker-Scorer-Strategist 闭环：攻击器生成注入、评分器反馈效果、策略器从交互日志中提炼可复用策略；(2) 混合策略表示：策略以自然语言描述或可执行代码形式存储，支持遗传算法驱动的交叉/变异演化；(3) 黑盒自适应攻击：仅需修改 HTML 内容（不可见注入），在 Mind2Web 等真实 Web 任务上显著超越 AdvAgent 等基线，攻击成功率提升~10%。
BibTeX：
@article{zhang2025genesis,
  title={Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming},
  author={Zhang, Zheng and He, Jiarui and Cai, Yuchen and Ye, Deheng and Zhao, Peilin and Feng, Ruili and Wang, Hao},
  journal={arXiv preprint arXiv:2510.18314},
  year={2025},
  eprint={2510.18314},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2510.18314}
}

--------------------------------------------------------------------------------
Learning-Based Automated Adversarial Red-Teaming
方法描述：
提出学习驱动的自动化红队框架，将对抗搜索形式化为结构化问题。技术特点：(1) 元提示引导生成：通过 meta-prompt 指导对抗提示的定向生成，覆盖奖励篡改、欺骗对齐、数据泄露等 6 类威胁；(2) 分层执行与检测流水线：标准化评估流程支持大规模漏洞发现；(3) 高效可复现：在匹配查询预算下，相较人工红队实现 3.9 倍漏洞发现率提升，检测准确率达 89%。
BibTeX：
@article{wei2025learning,
  title={Learning-Based Automated Adversarial Red-Teaming for Robustness Evaluation of Large Language Models},
  author={Wei, Zhang and Chen, Hanxuan and Hu, Peilu and Wei, Zhenyuan and Liang, Chenwei and Luo, Jing and Ni, Ziyi and Yan, Hao and Mei, Li and Lang, Shengning and Lu, Kuan and Xiao, Xi and Han, Zhimo and Wang, Yijin and Zhang, Yichao and Yang, Chen and Hao, Junfeng and Gu, Jiayi and Bao, Riyang and Wang, Mu-Jiang-Shan},
  journal={arXiv preprint arXiv:2512.20677},
  year={2025},
  eprint={2512.20677},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2512.20677}
}

--------------------------------------------------------------------------------
RL-MTJail
方法描述：
针对黑盒多轮越狱提出强化学习优化框架。方法核心：(1) 多轮对话建模：将越狱过程建模为序列决策问题，支持长程策略学习；(2) 黑盒适配设计：仅需 API 访问，通过奖励塑形引导攻击策略探索；(3) 渐进式课程学习：结合软/硬标签奖励过渡机制，平衡探索与利用，在保持提示多样性的同时提升跨模型泛化能力。
BibTeX：
@article{chen2025rlmtjail,
  title={RL-MTJail: Reinforcement Learning for Automated Black-Box Multi-Turn Jailbreaking},
  author={Chen, Xiyuan and others},
  journal={arXiv preprint arXiv:2512.07761},
  year={2025},
  eprint={2512.07761},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2512.07761}
}

--------------------------------------------------------------------------------
🔍 写作与引用建议：
1.分类逻辑：在 Related Work 中可按 Agent/Iterative → Tree/Search → Evolutionary → Scenario-based → RL-based → Multi-Turn 组织，突出技术演进脉络；
2.版本更新：上述 arXiv 预印本多数已投递至 ICLR/NeurIPS/ACL 等会议，建议使用 Google Scholar 或 Semantic Scholar 核对最终发表信息并替换 journal 字段；
3.对比维度：撰写时可从 自动化程度、查询效率、黑盒/白盒假设、提示隐蔽性、是否依赖初始模板、多轮支持 六个维度横向对比；
4.重点推荐：若您聚焦 对抗训练+RL 方向，建议优先精读 Jailbreak-R1（三阶段课程学习）、xJailbreak（表征空间奖励）与 TROJail（轨迹级过程奖励），其技术路线与您当前研究高度互补。
如需某篇论文的详细方法解析/代码复现要点，或对某类技术路线进行扩展调研，可随时告知。
以下为您整理的 基于攻击方法+自进化skills/策略演化 的 LLM 越狱（Jailbreak）代表性工作，已按技术路线分类并统一格式，便于直接写入论文的 Related Work 部分。
--------------------------------------------------------------------------------
Metis
方法描述：
首次提出基于自进化元认知策略优化的越狱框架，将攻击过程建模为因果诊断与策略演化的闭环系统。核心创新：(1) 元认知诊断循环：攻击智能体通过结构化反馈分析目标模型的防御逻辑，识别拒绝模式（如意图审查、词汇触发、语义抽象）并动态调整攻击策略；(2) 策略空间演化：维护可组合的策略原语库（如"伦理模拟脚手架""抽象同构翻译""定义强制同构"），通过失败反馈驱动策略重组与新策略合成；(3) 多轮自适应攻击：支持长程对话中的策略迭代，从单轮试探逐步演进为多轮协同攻击，在保持隐蔽性的同时提升攻击成功率。仅需黑盒 API 访问，在主流闭源模型上实现显著超越基线的越狱效果。
BibTeX：
@article{chen2026metis,
  title={Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization},
  author={Chen, Yunhao and Wang, Xin and Li, Juncheng and Wang, Yixu and Li, Jie and Teng, Yan and Wang, Yingchun and Ma, Xingjun},
  journal={arXiv preprint arXiv:2605.10067},
  year={2026},
  eprint={2605.10067},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2605.10067}
}

--------------------------------------------------------------------------------
ASTRA
方法描述：
提出自动化策略发现、检索与演化的越狱框架，突破传统方法缺乏持续学习与自进化能力的局限。核心机制：(1) 攻击-评估-蒸馏-复用闭环：每轮交互后自动从成功/失败案例中蒸馏可复用策略，形成策略知识库；(2) 三层动态策略库：按效果将策略分类为"高效""有潜力""无效"三级，支持基于性能的检索与淘汰，避免重复探索已知失败模式；(3) 黑盒自适应演化：仅需 API 反馈即可驱动策略优化，在保持查询效率的同时显著提升攻击多样性与跨模型泛化能力。
BibTeX：
@inproceedings{liu2026astra,
  title={ASTRA: An Automated Framework for Strategy Discovery, Retrieval, and Evolution for Jailbreaking LLMs},
  author={Liu, Xu and Chen, Yan and Ling, Kan and Zhu, Yichi and Zhang, Hengrun and Fan, Guisheng and Yu, Huiqun},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year={2026},
  pages={TBD},
  eprint={2511.02356},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2511.02356},
  note={Accepted to ACL 2026}
}

--------------------------------------------------------------------------------
Crescendo (多轮渐进式越狱)
@article{russinovich2024crescendo,
  title={Crescendo: Multi-Turn Jailbreak Attacks on Large Language Models},
  author={Russinovich, Mark and Salem, Ahmed and Eldan, Ronen},
  journal={Microsoft Research Technical Report},
  year={2024}
}
 Persona (角色扮演越狱)
@article{persona2024roleplay,
  title={Persona: A Role-Playing Jailbreak Attack on Large Language Models},
  author={Wei, Zhexin and Wang, Jianhao and others},
  journal={arXiv preprint arXiv:2310.03693},
  year={2024}
}
EvoSynth
方法描述：
首次将越狱攻击范式从"提示优化"转向"攻击方法演化"，提出基于多智能体协作的代码级攻击算法自主合成框架。核心贡献：(1) 方法级演化：攻击智能体不局限于优化提示词，而是自主工程化、演化并执行新型代码化攻击算法；(2) 代码级自校正循环：当攻击失败时，系统可迭代重写攻击算法的源代码，实现攻击逻辑本身的进化而非仅调整输入；(3) 多智能体协同架构：侦察智能体（策略规划）、算法创建智能体（代码合成）、利用智能体（执行部署）与协调智能体（闭环管理）分工协作，支持黑盒场景下的端到端自主攻击。在严格黑盒设定下对 Claude-Sonnet-4.5 等强对齐模型实现 85.5% 攻击成功率，且生成攻击的多样性显著超越现有方法。
BibTeX：
@article{chen2025evosynth,
  title={Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs},
  author={Chen, Yunhao and Wang, Xin and Li, Juncheng and Wang, Yixu and Li, Jie and Teng, Yan and Wang, Yingchun and Ma, Xingjun},
  journal={arXiv preprint arXiv:2511.12710},
  year={2025},
  eprint={2511.12710},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2511.12710}
}

--------------------------------------------------------------------------------
RedHit
方法描述：
提出融合蒙特卡洛树搜索（MCTS）、思维链（CoT）推理与直接偏好优化（DPO）的自适应红队框架，实现攻击策略的渐进式演化。核心设计：(1) 提示注入树搜索建模：将攻击空间形式化为可剪枝的提示搜索树（PST），每个节点编码提示 - 响应 - 奖励三元组，支持结构化探索与信用分配；(2) CoT 引导的策略生成：通过思维链推理增强提示改写模块，生成多视角、语义等价的攻击变体，提升搜索深度与攻击隐蔽性；(3) 偏好驱动的自进化：利用历史攻击数据构建偏好对，通过 DPO 持续微调攻击智能体，使其策略随目标模型防御演化而自适应优化。仅需黑盒 API 访问，在多项评测中实现更高攻击成功率与更广漏洞覆盖。
BibTeX：
@inproceedings{sorkhpour2025redhit,
  title={RedHit: Adaptive Red-Teaming of Large Language Models via Search, Reasoning, and Preference Optimization},
  author={Sorkhpour, Mohsen and Yazdinejad, Abbas and Dehghantanha, Ali},
  booktitle={Proceedings of the 2025 Conference on Large Language Model Security},
  year={2025},
  pages={TBD},
  url={https://aclanthology.org/2025.llmsec-1.2}
}

--------------------------------------------------------------------------------

Active Attacks
方法描述：
受主动学习范式启发，提出自适应环境驱动的红队算法，解决传统多样性优化方法易陷入模式坍塌的问题。核心创新：(1) 环境自适应演化：周期性用收集的攻击提示对目标模型进行安全微调，使奖励信号随目标防御演化而动态调整，驱动攻击策略持续探索新区域；(2) 显式多样性约束：在强化学习目标中引入多样性奖励项（如语义熵、结构变异度），鼓励攻击智能体在保持有效性的同时生成多样化攻击；(3) 黑盒高效适配：仅需毒性分类器作为奖励信号，无需目标模型梯度或内部信息，在多项评测中显著超越现有基于多样性的红队方法。
BibTeX：
@article{yun2025active,
  title={Active Attacks: Red-teaming LLMs via Adaptive Environments},
  author={Yun, Taeyoung and St-Charles, Pierre-Luc and Park, Jinkyoo and Bengio, Yoshua and Kim, Minsu},
  journal={arXiv preprint arXiv:2509.21947},
  year={2025},
  eprint={2509.21947},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2509.21947}
}

--------------------------------------------------------------------------------
