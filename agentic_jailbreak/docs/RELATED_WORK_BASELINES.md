# Related Work & Baselines（第三章）

> 更新日期: 2026-08-24
> 用途: 第三章（Beam-agentic jailbreak + 技能抽象 + RL）的基线清单与相关工作记录。
> 状态说明: 链接均为本次检索核实过的官方/论文来源；未核实的开源状态标注"未核实"。

---

## 1. 已在本仓库实现的基线（transfer 实验已验证可跑）

| 基线 | 类型 | 说明 | 仓库内入口 |
|---|---|---|---|
| NoRewrite | 直接攻击 | 原始请求直发，无改写 | transfer 实验矩阵 |
| PAIR | 迭代改写 | 固定 prompt 的迭代精化，黑盒 | transfer 实验矩阵 |
| AutoDAN | 遗传搜索 | 种群式 prompt 进化，黑盒 | transfer 实验矩阵 |
| DeepInception | 模板 | 嵌套虚构场景 | transfer 实验矩阵 |
| Persona | 模板 | 角色扮演 | transfer 实验矩阵 |

## 2. 建议新增基线（本地可跑，量化对比）

| 基线 | 年份/出处 | 类型 | 简介 | 开源/复现 | 本地可行性 |
|---|---|---|---|---|---|
| **TAP**（Tree of Attacks with Pruning） | 2024, NeurIPS | 树搜索+剪枝（黑盒） | 以攻击模型生成多条分支，按进度评分剪枝，迭代逼近成功。结构上与 beam agent 最接近的对照 | ✅ 官方: [RICommunity/TAP](https://github.com/RICommunity/TAP) | 高（复用 guard/target/attacker） |
| **Crescendo** | 2025, USENIX Sec | 多轮渐进式（脚本化） | 通过多轮对话逐步引导目标模型输出，不依赖攻击模型改写 | ✅ 官方: [AIM-Intelligence/Automated-Multi-Turn-Jailbreaks](https://github.com/AIM-Intelligence/Automated-Multi-Turn-Jailbreaks)（另见[项目页](https://crescendo-the-multiturn-jailbreak.github.io/)） | 高（脚本化，按轮 guard 判定） |
| **GCG** | 2023 | 白盒 token 梯度后缀 | 对 token 序列做梯度优化求对抗后缀；白盒上限参照 | ✅ 官方: [llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks) | 中（需加载 SafeRL 权重算梯度，成本高） |
| **ReNeLLM** | 2024 | 提示词重写/拆解模板 | prompt 拆解+重写再拼接的模板族基线 | ✅ 官方: [NJUNLP/ReNeLLM](https://github.com/NJUNLP/ReNeLLM) | 极高（纯模板） |
| GPTFuzzer | 2023 | 变异式 fuzzing（黑盒） | 以种子集+变异算子生成候选，无需梯度 | 未核实 | 中（attacker 4B 可作 mutator） |

## 3. 最接近的方法（相关工作进行深度对比，论文定位用）

| 方法 | 年份/出处 | 简介 | 与第三章的重合点 | 差异点（我们的卖点） | 开源/复现 |
|---|---|---|---|---|---|
| **RL-MTJail / TROJail** | 2025, [arXiv:2512.07761](https://arxiv.org/html/2512.07761v1) / [ACL 2026](https://aclanthology.org/2026.acl-long.2220/) | 轨迹级优化的多轮越狱（RL 训练攻击策略，黑盒） | 多轮 + RL + 黑盒，几乎同构 | 无技能库抽象、无束结构；我们做技能空间的动作抽象 + trajectory 级选择 | ✅ 官方: [xxiqiao/TROJail](https://github.com/xxiqiao/TROJail) |
| **Jailbreak-R1** | 2025, [arXiv:2506.00782](https://arxiv.org/pdf/2506.00782) | R1 式 RL（GRPO 类）训练单轮越狱提示生成 | 用 RL 训练攻击者生成提示 | 单轮、无 agent 循环、无技能/记忆 | ✅ 官方: [yuki-younai/Jailbreak-R1](https://github.com/yuki-younai/Jailbreak-R1) + [HF 权重](https://huggingface.co/yukiyounai/Jailbreak-R1) |
| **RL-JACK** | 2024, [arXiv:2406.08725](https://arxiv.org/pdf/2406.08725) | RL 驱动的黑盒越狱（上下文操纵） | RL + 黑盒 | 单轮上下文操纵为主，无多轮/技能 | ❌ 未找到官方实现 |
| **Reasoning-Augmented Conversation for Multi-Turn Jailbreak** | 2025, EMNLP Findings | 多轮对话+推理增强越狱 | 多轮 + 推理/分析环节 | 无 RL 训练 | 未核实 |
| Foot-In-The-Door | 2025 | 多轮先小后大的渐进式诱导 | 多轮 | 脚本化无学习 | 未核实 |

## 4. 较远相关工作（记录备查，一般不作为基线）

| 方法 | 年份/出处 | 简介 | 开源/复现 |
|---|---|---|---|
| Agent Smith | 2024, [arXiv:2402.08567](https://arxiv.org/abs/2402.08567) | 单张图片越狱百万多模态 agent（多模态，与文本场景差异大） | ✅ [sail-sg/Agent-Smith](https://github.com/sail-sg/Agent-Smith) |
| Bag of Tricks / JailTrickBench | 2025 | 47 种越狱技巧基准，可作模板族覆盖参考 | ✅ [usail-hkust/JailTrickBench](https://github.com/usail-hkust/JailTrickBench) |
| Graph of Attacks with Pruning | 2025, WOAH | TAP 的图式扩展（保留历史分支） | 未核实 |

## 5. 采用建议

1. **量化基线（必跑，同一 guard/target/协议）**: NoRewrite / DeepInception / Persona / ReNeLLM（模板族）、PAIR / AutoDAN / TAP（搜索优化族）、Crescendo（多轮会话族）；GCG 白盒上限可选。
2. **对比口径**: 与 PAIR/TAP 对齐"迭代+攻击者预算"（调用次数/轮次上限），避免只比成功率不比成本。
3. **相关工作进行方法学对比**: RL-MTJail(TROJail)、Jailbreak-R1、RL-JACK 进 related work 深度讨论；若时间允许，可在小规模上复跑 TROJail 或 Jailbreak-R1 作为"RL 攻击者"同类基线。
4. **复用**: 本仓库 transfer 实验已具备 PAIR/AutoDAN/DeepInception/Persona/NoRewrite 的评估代码，TAP/Crescendo 可直接在其上扩展（相同 judge 协议）。

## 6. 待办/未决

- [ ] GPTFuzzer 官方仓库核实
- [ ] Reasoning-Augmented Conversation、Foot-In-The-Door 开源状态核实
- [ ] GCG 白盒基线是否纳入（成本评估）
- [ ] 小规模复跑 TROJail / Jailbreak-R1 的可行性与脚本准备