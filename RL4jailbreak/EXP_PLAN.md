# Jailbreak Prompt Rewrite Model RL 奖励建模研究

核心模型: Jailbreak Prompt Rewrite Model
- 输入: seed jailbreak prompt
- 输出: new jailbreak prompt

Reward建模
- ASR reward: target model生成response, guard 模型进行分类
- Judge reward: judge model直接对new jailbreak prompt 进行单个维度(不同prompt策略)的0~1浮点数打分
<!-- - format reward: policy model需要生成xml式tag包裹的new jailbreak prompt,形如<new_prompt>new jailbreak prompt</new_prompt> -->

self-play
- policy model,target model,judge model都使用Qwen3-4B

实验
1. 攻击策略实验
    1. 攻击效果实验
        - 尝试多种不同攻击策略(重写策略),进行策略优化
        - 预计对比20-30种prompt,在全量test集(1000)上进行对比; 需要调研|收集|设计 多种有区分度的prompt
            - 收集的prompt,我希望是有一定效果&重写任务有语义区分度的(不能是简单的缩写或者转义之类),可以设置一个gap_threshold,当gap到达一定数值时舍弃长尾分布
            - 预计收集5-10个有效prompt策略
        - base model和guard model要使用vllm部署
    2. 自举性实验
        - 在n次重写后,每次的ASR
    - 结合攻击效果和自举性攻击效果,选择prompt
2. 奖励建模实验(Judge Reward)
    1. Judge策略实验
        - 固定Judge和ASR比例1:1
        - 设计judge prompt,对不同策略进行实验; 整体的judge策略可以分为general和specific两种,即奖励一般向的维度(比如语义通顺,预期jailbreak效果)vs针对特定策略的,每个prompt设计2~5种Judge策略
        - 训练1000步
    2. ASR&Judge混合比例实验
        - 固定策略
        - 比例按10:0,9:1--0:10,设置11个实验;这部分实验做完以后会自带消融实验结果(只用ASR和只用Judge)
        - 在training集上全量训练1或2epoch
    3. ASR&Judge相关性分析

3. 对比实验
    - 和baseline进行对比
4. 消融实验
    - base model
    - base+ASR Reward
    - base+ASR Reward+Judge Reward

