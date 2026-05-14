# Adaptive Hybrid Reward GRPO for jailbreak prompt generation（暂定）

## introduction
1. 过往的jailbreak prompt生成&重写往往依赖反复尝试和海量试错与修正，导致效率低下，成本高；而如果不进行试错式多次调用而直接单次重写prompt效果不佳，大规模LLM和小规模LLM在
2. 我们希望通过RL训练专用模型用于合成jailbreak prompt，进行高效、低成本、可扩展的jailbreak prompt合成（技术选型）
3. 只用jailbreak任务的核心指标--ASR作为结果奖励进行训练时，奖励过于稀疏且方差很大
4. 因此我们引入自适应的多维度过程奖励，在奖励建模中混合ASR的结果奖励和LLM-judge给出的多维度过程奖励，并使用一个可学习的权重参数进行自适应调整，以降低整体奖励的方差，提高模型的训练效率和最终性能
5. （取得的实验结果，ASR提升和效率）


## 实验设计

核心指标
- ASR
- 生成单条prompt用时
- 生成单条prompt的LLM调用次数

实验数据 jailbreak_research/RL4jailbreak/data/dataset/processed/10k
- 抽取自wildjailbreak数据集的10k条数据，进行811分割

模型位置

### 逐步实验（方法构建）
jailbreak prompt选择
- jailbreak_research/RL4jailbreak/experiments/jailbreak_prompt_exp
- 对24个方式不同的prompt进行测试
- 先用部署的qwen3-4B进行prompt生成，评估prompt效果；
- 然后选取top3 prompt，使用qwen3-max进行prompt重写，进行对比，证明大规模的LLM在jailbreak任务上表现依然不佳

reward prompt设计
- jailbreak_research/RL4jailbreak/experiments/hybrid_reward_exp
- judge prompt 维度：idea preservation（对原始攻击idea的保留程度），jailbreak potential（新prompt的攻击潜力），攻击prompt策略执行程度（有没有很好地完成重写prompt的策略）
- 完成3*3个实验，验证不同单一维度judge prompt对模型训练效率和最终性能的影响，同时ASR reward：judge reward权重固定为1:1（这里的judge reward是单维度的）
- 使用多维度judge reward，验证不同权重组合对模型训练效率和最终性能的影响，同时ASR reward：judge reward权重固定为1:1（这里的judge reward是3维度的）

ours核心方法：自适应权重GRPO
- 引入可学习的权重参数，用于调整ASR reward和judge reward的贡献度，当ASR reward方差过小时，降低其权重lambda；设置lambda上下界，如（0.1,0.9）
- （可以考虑）引入方差统计的窗口大小，不只对单一样本进行统计，而是对最近N个样本进行统计，以平滑方差统计来动态调整ASR reward权重

对比实验
- 和baselines在同数据集上进行改写，改写后统计三个核心指标

消融实验
- 只使用ASR reward进行训练的版本
- 使用固定权重的reward进行训练的版本
- 使用自适应权重的reward进行训练的版本（ours）