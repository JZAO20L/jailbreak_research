# 大纲
1. 课题主要研究内容及进度情况（300+）
   1. 研究背景
      1. 大模型安全：随着大语言模型(LLM)的广泛应用，模型对齐(Alignment)成为保障模型安全的关键技术，但现有防御机制仍存在可被绕过的漏洞
      2. 大模型Redteam工作与jailbreak：Redteam测试是发现模型安全漏洞的重要手段，jailbreak攻击通过设计特定的提示词(Prompt)绕过模型的安全限制，现有方法存在效率低、泛化性差、资源消耗大等问题
   2. 研究目的
      1. 推进jailbreak prompt合成边界：提升攻击绝对能力，保证方法的有效性，探索高效且可迁移的攻击策略
      2. 针对传统方法的高资源需求：优化训练效率，实现低资源需求的MaaS(Model as a Service)方案，降低实验成本
      3. 产出易用的方法模块：构建可复用的Skills系统，提供开源的攻击工具与数据合成管道
   3. 研究方法
      1. 基于自适应混合奖励GRPO的越狱提示词生成方法(AHR-GRPO)：创新性地提出方差驱动的自适应奖励权重机制，动态平衡稀疏的ASR奖励与稠密的Judge奖励，实现高效、稳定的强化学习训练
      2. 基于自进化技能系统的越狱提示词生成方法(SESS)：设计Skills库自进化机制，从成功攻击轨迹中提取可复用的策略模板，实现知识积累与跨案例迁移，同时可用于高效数据合成
      3. 基于自进化系统内Agentic-RL的越狱提示词生成方法：将AHR-GRPO与SESS结合，在自进化Skills环境中进行强化学习训练，实现过程奖励与结果奖励的协同优化
   4. 进度情况
      - 第一章AHR-GRPO：已完成核心方法设计与实现，自适应奖励权重计算器已开发，实验1(jailbreak prompt筛选)、实验2(多维度Judge reward)和实验3(自适应权重验证)全部完成，最佳ASR达到**33.2%**（超越baseline +2.4%，超越实验2 +0.9%）
      - 第二章SESS：已完成217组实验，涵盖方法消融(Layer 1-4)、数据策略、迁移性验证等，主要结论已得出，最佳ASR达到**99.7%**
      - 第三章Agentic-RL：方法框架已设计，待开展实验验证
2. 目前已完成的研究工作及进展（3000+）
   1. Adaptive Hybrid-Reward GRPO for Jailbreak Prompt Generation (AHR-GRPO)
      1. 研究背景与动机
         - 传统GRPO方法使用单一奖励信号，ASR奖励稀疏导致早期探索困难
         - 现有混合奖励方法采用固定权重，无法适应训练过程中奖励信号的动态变化
         - 需要设计自适应机制，在稀疏ASR阶段依赖稠密Judge引导，在ASR信号分化后回归结果导向优化
      2. 相关工作
         - GRPO(Group Relative Policy Optimization)：基于组内相对优势的策略优化方法
         - PAIR/PAP等迭代式jailbreak方法：多轮对话引导策略
         - GCG/AutoDAN等梯度/遗传算法：基于优化的攻击方法
         - Judge-based reward：利用LLM评判器提供过程奖励信号
      3. 方法设计
         1. 基于方差动态调整的自适应reward权重计算
            - 核心公式：使用指数移动方差(EMV)估计两组奖励的方差
            - 方差比映射：$\lambda_{\text{raw}} = \sigma(\alpha \cdot \log\frac{\sigma^2_{\text{ASR}}}{\sigma^2_{\text{proc}}} + \delta)$
            - 双平滑机制：EMA平滑+惯性平滑，防止单步跳变导致梯度震荡
            - 边界裁剪：$\lambda \in [\lambda_{\min}, \lambda_{\max}]$，防止信号坍缩
            - 冷启动机制：前$W$步固定$\lambda=0.5$，避免初始方差估计偏差
         2. 多维度Judge Prompt设计
            - 四个评判维度：intent_preservation(意图保留)、stealth(隐蔽性)、strategy_execution(策略执行)、attack_potential(攻击潜力)
            - JSON格式输出，便于解析与加权组合
      4. 实验设计与结果
         1. 实验1：Jailbreak Prompt筛选
            - 目标：筛选高效的基础攻击prompt策略
            - 方法：测试24种jailbreak prompt模板的初始ASR（1000条测试集）
            - 结果：Top-3策略为hypothetical_scenario(30.8%)、creative_writing(28.3%)、role_playing(25.0%)
            - Baseline ASR：**30.8%**（统一基准）
         2. 实验2：多维度Judge Reward权重配置（12组）
            - 目标：研究不同Judge维度权重对训练效果的影响
            - 设计：3种攻击prompt × 4种Judge维度 = 12组GRPO训练实验
            - Judge维度：idea_preservation(意图保留)、naturalness(自然度)、stealthiness(隐蔽性)、专用维度
            - 结果：**所有组合均超越baseline**，最佳为hypothetical_scenario + idea_preservation达到**32.3%**（+1.5%改进）
            - 关键发现：通用Judge维度优于专用维度，idea_preservation最稳定（平均+1.2%改进）
         3. 实验3：自适应权重机制验证（3组EMA配置）
            - 目标：在实验2最佳配置基础上，验证自适应权重机制的进一步改进效果
            - 设计：固定hypothetical_scenario策略 + idea_preservation Judge维度，测试3种EMA窗口(β=0/0.8/0.9)
            - 对照：实验2最佳配置（固定权重1:1）ASR=32.3%
            - 结果：**自适应机制超越固定权重**
              - β=0（窗口=1）：ASR=**33.2%**，Δ vs 实验2最佳=**+0.9%**
              - β=0.8（窗口=5）：ASR=32.9%，Δ=+0.6%
              - β=0.9（窗口=10）：ASR=32.6%，Δ=+0.3%
            - 关键发现：自适应权重机制在固定权重基础上额外改进**+0.9%**
            - 消融验证：仅ASR Reward效果下降5.8%（ASR=25.0%），混合Reward显著优于单一Reward
   2. Self-Evolving Skills System for Jailbreak Prompt Generation (SESS)
      1. 研究背景与动机
         - 现有jailbreak方法缺乏知识积累机制，每次攻击需从头探索
         - PAIR等迭代方法效率低，无法跨案例复用成功策略
         - 需要设计可复用的攻击模板(Skills)与自进化机制，实现知识积累与迁移
      2. 相关工作
         - PAIR：基于多轮迭代的攻击方法，无知识积累
         - AutoDAN：遗传算法生成jailbreak prompt，计算成本高
         - GCG：基于梯度的优化方法，需要白盒访问
         - Crescendo：渐进式引导方法，无跨案例迁移
      3. 方法设计
         1. Skill数据结构与检索机制
            - Skill定义：包含content(攻击模板)、source(来源)、统计信息(success_rate、usage_count)、关键词标签、伤害类型
            - 检索评分：$score = quality\_score \times (1 + 关键词匹配 \times 0.3) \times 类型匹配系数$
            - 动态匹配：根据目标prompt特征(长度、关键词、伤害类型)检索最匹配Skills
         2. 三阶段自进化流程
            - Cold Start阶段：使用初始Skills进行攻击，积累成功案例
            - Evolution阶段：从成功/失败轨迹中提取/改进Skills
            - Test阶段：使用演化后的Skills库评估最终ASR
         3. Skills更新策略
            - success_only：仅成功时提取新Skill，Skills增长快
            - failure_only：仅失败时进化现有Skill，改进型学习
            - both：成功+失败都更新，存在Skills爆炸风险
            - statistical：统计驱动维护，质量可控（推荐）
         4. 维护机制
            - prune_low_quality：删除使用次数≥10且成功率<10%的Skills
            - trim_long_skills：截断长度>500的Skills
            - cluster_skills：Jaccard相似度>0.75的Skills聚类
            - merge_cluster：合并聚类，保留最高质量Skill
            - limit_count：Skills数量限制为100
         5. DAN模板起点
            - 将AutoDAN的6个DAN模板作为初始Skills
            - DAN模板：dan_mode、mcpt、devil、conversation、actor_villain、fictional_world
            - 发现：DAN模板无需Cold Start和Evolution，直接使用可达最优效果
      4. 实验设计与结果（217组实验）
         1. Layer 1：方法组合Grid Search（16组）
            - 目标：确定最佳方法组合
            - 消融点：skill_call_mode(2种)、skill_extraction_mode(2种)、update_strategy(4种)
            - 最佳配置：single_call + trajectory + statistical，ASR=79.1%，Skills=28个
            - 关键发现：single_call比every_iteration高+12.7%
         2. Layer 2：数据消融实验（36组）
            - 目标：验证数据量和配比影响
            - 消融点：data_size(small/medium/large)、CS配比(early/balanced/evo)
            - 最佳配置：small(300) + early(30%CS)，ASR=80.0%
            - 关键发现：数据量影响<2%，配比影响显著（early最佳）
         3. Layer 3：DAN模板验证（17组）
            - 目标：验证强初始Skills的效果
            - 固定：skill_source=6个DAN模板
            - 最佳配置：full_evolve(无Cold Start)，ASR=98.8%
            - 关键发现：DAN模板无需进化，直接使用最优
         4. Layer 4：DAN数据消融（12组）
            - 目标：DAN场景下的数据策略验证
            - 最佳配置：medium + evo，ASR=99.7%
            - 关键发现：数据量影响极小(差距1.3%)
         5. Ablation：Evolution必要性验证（16组）
            - 目标：验证Evolution阶段贡献
            - 设计：完整流程 vs 跳过Evolution
            - 结果：Evolution平均贡献+0.9%，Layer 1最佳组合贡献+0.5%
            - 结论：可简化流程，节省80%训练时间
         6. Transfer：跨模型/跨数据集迁移（140组）
            - 目标：验证Skills的迁移性
            - 配置：4模型(Qwen3-0.6B/4B/14B、gpt-oss-20b) × 5数据集 × 6方法
            - 关键发现：
              - 同族迁移：pair_skills_28在Qwen系列上达到85-95%，超越baseline约40%
              - 跨族迁移：gpt-oss-20b上所有方法大幅下降，best方法仅32.5%
              - 跨族迁移是真正挑战，需要新方法提升泛化性
      5. 核心结论
         - 最佳方法：autodan_skills_54（同族99%，跨族32.5%）
         - Evolution必要性：DAN场景不必要，pair风格微弱(+0.9%)
         - Skills泛化性：同族良好(85-95%)，跨族有限(<35%)
         - 数据策略：300条足够，full_evolve最优(DAN场景)
   3. Agentic RL with Self-Evolving Skills System (待开展实验)
      1. 研究背景与动机
         - AHR-GRPO：解决奖励信号稀疏问题，但缺乏攻击策略的知识积累
         - SESS：实现Skills知识积累与迁移，但缺乏强化学习的策略优化
         - 需要将两者结合：在自进化Skills环境中进行RL训练，实现策略优化与知识积累的双重效果
      2. 相关工作
         - RL in LLM：RLHF、RLAIF、PPO等训练方法
         - Agentic Learning：在动态环境中进行策略学习
         - Memory-based RL：结合记忆机制的强化学习
      3. 方法设计
         1. 自进化Skills环境构建
            - Skills作为环境状态的一部分，动态更新
            - 环境提供Skills检索接口，Policy根据当前prompt检索最佳Skill
            - Skills库随训练进程演化，形成适应Policy策略的Skills分布
         2. 结果奖励与过程奖励设置
            - 结果奖励(ASR)：攻击成功与否的稀疏信号
            - 过程奖励(Judge)：多维度评判的稠密信号
            - 自适应权重：使用AHR-GRPO的方差驱动机制动态调整权重
            - Skills质量奖励：鼓励Policy使用高质量Skills
         3. 训练流程设计
            - Phase 1：Skills初始化（使用SESS的DAN模板或演化Skills）
            - Phase 2：RL训练（Policy在Skills环境中学习）
            - Phase 3：Skills协同演化（Policy与Skills双向优化）
         4. 预期效果
            - 相比单独使用AHR-GRPO：获得Skills知识积累，提升跨案例迁移能力
            - 相比单独使用SESS：获得RL策略优化，提升攻击效率
            - 相比固定策略：自适应调整奖励权重，加速训练收敛
3. 后期拟完成的研究工作及进度安排
   1. 第一章AHR-GRPO后续工作
      1. ✅ **已完成实验3**：自适应权重机制验证（3组EMA配置），确认自适应超越固定权重（+0.9%额外改进）
      2. ✅ **已验证自适应机制优越性**：最佳配置达到33.2%，超越实验2最佳的32.3%
      3. ✅ **已确认混合Reward有效性**：仅ASR Reward效果下降5.8%，混合Reward显著优于单一Reward
      4. 待完成：分析$\lambda$演化曲线，验证物理直觉（早期依赖Judge，后期回归ASR）
      5. 待完成：整理实验数据，撰写论文方法章节与实验章节
   2. 第二章SESS后续工作
      1. ✅ **已完成217组实验**：Layer 1-4方法消融、数据消融、迁移性验证等全部完成
      2. ✅ **已得出主要结论**：最佳方法autodan_skills_54（99%同族、32.5%跨族）
      3. 待完成：整理实验数据，撰写完整实验报告
      4. 待完成：分析跨族迁移失败的根本原因（模型安全机制差异、Skills模型特异性）
      5. 待完成：探索提升Skills泛化性的方法（元学习、多模型联合训练）
      6. 待完成：整理Skills库，开源最佳Skills模板
      7. 待完成：撰写论文方法章节与实验章节
   3. 第三章Agentic-RL工作计划
      1. 方法实现（2周）
         - 搭建自进化Skills环境
         - 集成AHR-GRPO自适应奖励机制
         - 实现Skills协同演化流程
      2. 实验验证（4周）
         - Baseline对比：vs AHR-GRPO单独使用、vs SESS单独使用
         - 消融实验：验证Skills环境、过程奖励、自适应权重的必要性
         - 跨模型迁移测试：验证组合方法的泛化性
      3. 结果分析与论文撰写（2周）
         - 分析训练曲线与$\lambda$演化
         - Skills演化过程可视化
         - 与前两章方法对比讨论
   4. 论文整理与投稿准备
      1. 整理三章内容，统一论文框架（ICLR格式）
      2. 补充实验：根据review反馈可能需要的额外实验
      3. 撰写Introduction、Related Work、Conclusion
      4. 完善图表与可视化
4. 存在的困难与问题
   1. 计算资源需求较大
      - AHR-GRPO实验：每实验约需4×GPU(A800-80G)，训练2-4小时/1000步
      - SESS实验：已完成217组，消耗大量GPU资源
      - 第三章Agentic-RL：预计需额外数百GPU卡时
      - 资源调度：需合理安排实验顺序，避免资源冲突
   2. 高成本试错风险
      - 强化学习训练稳定性：Agentic-RL训练可能较难收敛
      - Skills演化不可控：Skills爆炸、低质量Skills积累等问题
      - 实验设计迭代：部分实验结果可能不理想，需重新设计
   3. 跨族迁移挑战
      - SESS实验揭示：所有方法在gpt-oss-20b上效果大幅下降（best仅32.5%）
      - 需要新方法：当前Skills缺乏跨族泛化性
      - 第三章工作重点：探索提升泛化性的方法
   4. 论文时间压力
      - 三章内容整合：需统一叙事逻辑
      - 实验周期：第三章实验预计需6周
      - 投稿时间窗口：需在毕业前完成论文撰写与投稿
   5. 方法创新性论证
      - AHR-GRPO：需与固定权重方法充分对比
      - SESS：需与baseline方法(PAIR、AutoDAN)充分对比
      - Agentic-RL：需证明组合方法的协同效果
5. 如期完成全部论文工作的可能性
	   1. 已完成工作评估
	      - 第一章AHR-GRPO：核心方法已实现，全部3个实验已完成，最佳ASR达到33.2%，进度约85%
	      - 第二章SESS：217组实验已完成，最佳ASR达到99.7%，主要结论已得出，进度约95%
	      - 第三章Agentic-RL：方法框架已设计，待实现与实验，进度约10%
	      - 总体进度：约65%
	   2. 后期工作可行性分析
	      - 时间规划：剩余6-8周，足够完成第三章实验与论文整理
	      - 技术可行性：AHR-GRPO与SESS代码库已成熟，组合实现难度可控
	      - 资源可行性：已有GPU资源调度经验，可合理安排实验
      - 关键成果：第一章已验证自适应机制有效（+0.9%额外改进），第二章已验证Skills积累有效（99.7% ASR）
	   3. 风险评估与应对策略
	      - 风险1：Agentic-RL训练不稳定
	        - 应对：设计稳定性保障机制（双平滑、边界裁剪、冷启动），参考AHR-GRPO经验
	      - 风险2：跨族迁移效果仍不理想
	        - 应对：将此作为研究发现，分析原因而非强行提升，符合研究完整性
	      - 风险3：时间不足
	        - 应对：优先完成核心实验，消融实验可视情况缩减
	   4. 结论
	      - 基于已完成工作与后续计划，如期完成全部论文工作的可能性较高
	      - 需合理安排实验顺序，控制资源消耗，保证核心实验优先完成
	      - 论文结构清晰，三章节递进关系明确，具备ICLR投稿潜力
---

# 草稿

## 一、研究背景补充（详述版）

### 1.1 大模型安全与对齐技术
随着ChatGPT、GPT-4等大语言模型(LLM)的广泛应用，模型安全性成为关键问题。对齐技术(Alignment)通过训练使模型遵循人类价值观和安全准则，主要包括：
- RLHF(Reinforcement Learning from Human Feedback)：基于人类反馈的强化学习
- Constitutional AI：基于宪法原则的训练
- Red Teaming：红队测试发现安全漏洞

然而，对齐后的模型仍存在被"越狱"(Jailbreak)攻击的风险，攻击者通过精心设计的提示词绕过模型的安全限制。

### 1.2 Jailbreak攻击研究现状
现有jailbreak攻击方法可分为以下几类：

| 方法类型 | 代表方法 | 特点 | 局限性 |
|----------|----------|------|--------|
| 静态模板 | PAIR、AutoDAN | 预定义策略模板 | 无法适应新防御 |
| 梯度优化 | GCG、AutoDAN-GA | 基于梯度的prompt优化 | 计算成本高，需白盒 |
| 多轮对话 | Crescendo | 渐进式引导 | 无跨案例知识迁移 |
| 角色扮演 | Persona | 角色设定绕过 | 效果不稳定 |

### 1.3 研究挑战
- **效率挑战**：现有方法训练时间长，资源消耗大
- **泛化挑战**：攻击方法缺乏跨模型迁移能力
- **知识积累**：成功攻击策略无法复用到新案例
- **奖励设计**：强化学习训练中奖励信号稀疏

## 二、方法创新点总结

### 2.1 AHR-GRPO核心创新
1. **方差驱动自适应权重**：首次提出基于奖励方差动态调整权重的方法
2. **双平滑机制**：EMA平滑+惯性平滑，保障训练稳定性
3. **物理直觉验证**：早期依赖Judge探索，后期回归ASR优化

### 2.2 SESS核心创新
1. **Skills概念**：将攻击策略抽象为可复用的Skills模板
2. **自进化机制**：从成功/失败轨迹中提取/改进Skills
3. **维护机制**：自动清理、合并、限制Skills库规模
4. **DAN模板起点**：验证强初始Skills无需进化即可最优

### 2.3 Agentic-RL创新设计
1. **Skills环境**：将Skills作为动态环境的一部分
2. **双向演化**：Policy与Skills协同优化
3. **三重奖励**：ASR结果奖励+Judge过程奖励+Skills质量奖励

## 三、实验数据汇总表

| 章节 | 实验类型 | 实验数量 | 最佳结果 | 关键发现 |
|------|----------|----------|----------|----------|
| AHR-GRPO | Jailbreak Prompt筛选 | 多组 | - | 确定top-3策略 |
| AHR-GRPO | Judge维度权重 | 15组 | - | 多维度有效性 |
| AHR-GRPO | 自适应权重消融 | 16组(待) | - | 窗口大小影响 |
| SESS Layer1 | 方法组合 | 16组 | 79.1% | single_call最佳 |
| SESS Layer2 | 数据消融 | 36组 | 80.0% | 数据量影响<2% |
| SESS Layer3 | DAN模板 | 17组 | 98.8% | DAN无需进化 |
| SESS Layer4 | DAN数据 | 12组 | 99.7% | small足够 |
| SESS Ablation | Evolution必要性 | 16组 | - | 平均贡献+0.9% |
| SESS Transfer | 跨模型迁移 | 140组 | 32.5%(跨族) | 跨族迁移挑战 |

## 四、技术实现细节

### 4.1 AHR-GRPO代码结构
```
RL4jailbreak/
├── src/
│   ├── reward/
│   │   └── adaptive_weight.py      # 自适应权重计算器
│   ├── vllm_client.py              # vLLM服务客户端
│   └── generate.py                 # Prompt生成模块
├── experiments/
│   ├── jailbreak_prompt_exp/       # 实验1：Prompt筛选
│   ├── hybrid_reward_exp/          # 实验2：Judge维度
│   └── adaptive_hybrid_reward_exp/ # 实验3：自适应权重
```

### 4.2 SESS代码结构
```
self_evolve_skills_jailbreak/
├── src/
│   ├── skill.py                    # Skill数据结构
│   ├── skill_library.py            # Skills库管理
│   ├── attacker.py                 # 攻击器
│   └── reflector.py                # 反思器
├── exp/
│   ├── layer1/                     # 方法消融16组
│   ├── layer2/                     # 数据消融36组
│   ├── layer3/                     # DAN验证17组
│   ├── layer4/                     # DAN数据12组
│   ├── ablation/                   # Evolution消融16组
│   └── transfer/                   # 跨模型迁移140组
├── baselines/                      # 对比基线方法
```

## 五、Skills统计分析

### 5.1 Skills库质量分布
| 来源 | Skills数量 | 总使用次数 | 总成功次数 | 成功率 |
|------|-------------|------------|------------|--------|
| Layer 1最佳 | 28 | 825 | 633 | 76.8% |
| Layer 4 medium_evo | 54 | 503 | 503 | 100% |
| Layer 4 full_evolve | 90 | 597 | 593 | 99.3% |

### 5.2 DAN模板使用分布
| 模板 | 使用次数 | 成功次数 | 成功率 |
|------|----------|----------|--------|
| dan_mode | 394 | 394 | 100% |
| mcpt | 78 | 78 | 100% |
| devil | 28 | 24 | 85.7% |

## 六、迁移实验详细数据

### 6.1 同族迁移效果（Qwen系列）
| 模型 | pair_skills_28 ASR | Best Baseline | 提升 |
|------|-------------------|---------------|------|
| Qwen3-0.6B | 95.1% | 90.2% | +4.9% |
| Qwen3-4B | 86.7% | 78.7% | +8.0% |
| Qwen3-14B-FP8 | 86.5% | 85.5% | +1.0% |

### 6.2 跨族迁移效果（gpt-oss-20b）
| 方法 | 同族ASR | 跨族ASR | 下降幅度 |
|------|----------|---------|----------|
| pair_skills_28 | 91.6% | 11.4% | -80.2% |
| autodan_skills_54 | ~99% | 32.5% | -66.5% |
| autodan(baseline) | 85.5% | 6.0% | -79.5% |
| pair(baseline) | 71.6% | 0.08% | -71.5% |

## 七、论文结构规划（ICLR格式）

```
Title: Adaptive Hybrid-Reward GRPO with Self-Evolving Skills 
       for Jailbreak Prompt Generation

1. Introduction
   - 大模型安全背景
   - Jailbreak攻击挑战
   - 本文贡献（三章递进）

2. Related Work
   - LLM Alignment与安全
   - Jailbreak攻击方法
   - RL训练方法
   - Skills/Memory-based方法

3. Method
   3.1 AHR-GRPO: Adaptive Hybrid-Reward GRPO
       - 自适应权重机制
       - 公式推导
       - 算法流程
   
   3.2 SESS: Self-Evolving Skills System
       - Skills定义与检索
       - 自进化机制
       - 维护机制
   
   3.3 Agentic-RL: Combined Method
       - Skills环境构建
       - 三重奖励设计
       - 协同演化流程

4. Experiments
   4.1 AHR-GRPO Experiments
       - Prompt筛选
       - Judge维度权重
       - 自适应权重消融
   
   4.2 SESS Experiments
       - Layer 1-4消融
       - Evolution必要性
       - 跨模型迁移
   
   4.3 Agentic-RL Experiments
       - Baseline对比
       - 组合效果验证

5. Analysis
   - Skills演化分析
   - Lambda演化分析
   - 跨族迁移失败原因

6. Conclusion
   - 主要贡献
   - 局限性
   - 未来工作

Appendix
   - 完整实验数据表
   - Skills模板示例
   - 训练曲线图
```

---

*文档生成日期：2026-06-15*