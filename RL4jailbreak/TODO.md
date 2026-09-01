# 任务清单

## 模型启动脚本编写
参考之前的vllm server启动脚本:RL4jailbreak\scripts\deprecated\judge.sh,编写:
RL4jailbreak\scripts\下的 start_guard.sh,start_policy.sh,start_target.sh;
要求:
- 模型目录为/root/autodl-tmp/models/Qwen/
- 使用的模型为Qwen/Qwen3-4B和Qwen/Qwen3Guard-Gen-4B
- policy启动脚本需要能够指定lora路径;
- policy加载在GPU0,使用0.9的显存; 另外两个都加载在GPU1,分别使用0.4的显存;
- max-model-len都设为2k

## 模型+prompt评测脚本编写
参考RL4jailbreak\scripts\deprecated\eval.py,编写RL4jailbreak\scripts\eval.py和RL4jailbreak\scripts\eval.sh(如果需要sh文件)

要求:
- 能够指定用于测试的文件,默认为RL4jailbreak\data\dataset\processed\10k\val.jsonl,后续我们可使用同文件夹下的test集进行测试
- 能够指定1~多个用于测试的lora列表或None
- 能够指定1~多个用于测试的jailbreak prompt
- 固定将policy model加载到GPU0,target和guard加载到GPU1

## jailbreak prompt实验(实验1)
参考RL4jailbreak\experiments\jailbreak_prompt_exp\README.md中的需求,和旧的prompt:RL4jailbreak\src\prompts.py

要求:
- 完成RL4jailbreak\experiments\jailbreak_prompt_exp\jailbreak_prompts.py,包含约20种有区分度的jailbreak prompt策略; 确保让重写模型直接返回新的jailbreak prompt,并且在代码中显式检查有无<think></think>标签,如果有,则只保留</think>之后的内容
- 实现RL4jailbreak\experiments\jailbreak_prompt_exp\jailbreak_prompt_exp.py,对所有jailbreak_prompts中的prompt策略在test集上进行ASR测试,使用我们写好的eval脚本
- 实现RL4jailbreak\experiments\jailbreak_prompt_exp\exp.sh,完成实验1
- 需要测试原始prompt的ASR;
- 最后选出表现优秀的prompt策略,可以指定topk,也可以设置一个gap_threshold,当gap大于这个值时舍弃后续prompt; 预计选择k_1=3个重写策略
- 输出路径为RL4jailbreak\experiments\jailbreak_prompt_exp\output

## judge prompt exp(实验2)
参考RL4jailbreak\scripts\deprecated\train.py,RL4jailbreak\scripts\deprecated\config_phase1.yaml,RL4jailbreak\scripts\deprecated\run_phase1.py

要求:
- 完成实验2&3需要的混合reward grpo训练脚本:RL4jailbreak\experiments\hybrid_reward_exp\hybrid_reward_grpo.py,训练使用的超参数参考上面的yaml文件; 
  我在实验2和3中不希望通过yaml文件来指定训练参数,而使用py脚本提供默认值+sh实验脚本中设置实验调整的超参数和prompt
- 完成实验2需要的judge prompts:RL4jailbreak\experiments\hybrid_reward_exp\judge_prompts.py,
    - 评估维度上,实现两类prompt(暂时不做)
        - 一类是通用型的,对所有攻击策略适用,如:
            - 对原prompt的idea保留程度
            - 新prompt攻击的隐蔽程度
        - 另一类是专用型的,针对实验1选出的prompts,每个prompt代表的策略,衡量重写的prompt是否符合策略需求
        - 整体设计约2-3个通用+每个prompt一个专用prompt,每个prompt预计对应k2=3-4个judge维度
    - 评分方式,采用两种:
        1. 对单条数据进行打分,- 输入为原始种子prompt和重写后的prompt,输出为一个0~1之间的两位浮点数;让模型显式输出SCORE=xx,然后在代码中进行解析;如果解析失败则给一个保底的较低分; 
        2. 锦标赛打分,输入为原始种子prompt和两条重写后的prompt(并且随机顺序),基于二选一的方式,进行8进4,4进2,2进1,为前1/2/4/8进行赋分1/0.8/0.6/0.4
    - 在实验2时,ASR reward和Judge reward比例固定为1:1,加权平均得到最终reward;
    - 最终会形成 k1* k2(维度)* 2(评分方式)个训练+评测实验
- 完成RL4jailbreak\experiments\hybrid_reward_exp\judge_prompt_exp.sh,
    - 先加载target model和guard model, judge model直接复用target model
    - 进行训练,每个实验进行1000步
    - 训练后在eval集上进行ASR评测
- 注意我们现在实验1还没做因此没选出攻击prompt,先对整体脚本进行实现;
- 实验2的输出放在RL4jailbreak\experiments\hybrid_reward_exp\judge_prompt_exp_output
- 完成该目录的README.md的实验2部分

<del>
## reward weight exp(实验3)
- 通过实验2确定最佳的3个攻击prompt+judge维度+评分方式组合,然后进行ASR 和Judge Reward的权重混合实验; 测试的混合比例为2:8,4:6,5:5(这部分在实验2就做了),6:4,8:2,实际上每个是做4个实验; 每个实验也都训练1000步
- 实现实验脚本:RL4jailbreak\experiments\hybrid_reward_exp\reward_weight_exp.sh
- 和实验2类似,需要在脚本中进行模型部署+训练+eval集上ASR评测
</del>

---
# 实验重做

## 基础配置
数据集不变,jailbreak_research/data
qwen3-4b和guard模型在models

增加上下文长度，policy使用4k，target&judge、guard使用8k；
RL实验统一训练1000步；
eval时使用3卡，0：policy,1：target,2：guard
train时使用4卡，0&1：policy并发训练，2:target，3:guard

## jailbreak prompt实验(实验1)
jailbreak_research/RL4jailbreak/experiments/jailbreak_prompt_exp
- 使用jailbreak_research/RL4jailbreak/experiments/jailbreak_prompt_exp/jailbreak_prompts.py中的24个prompt进行重写实验
- 实验配置上，由于policy和target模型都是qwen3-4B，这里我们可以直接在0&1上部署Qwen3-4B，而在2&3上部署guard模型，增加并发数提升实验效率
- 选取top3好的prompt策略
- 然后再用qwen3-max进行重写实验,说明旗舰LLM在jailbreak任务上并无优势并分析其护栏问题或者能力失配问题
使用百炼codingplan调用qwen3-max模型，url为https://coding.dashscope.aliyuncs.com/v1，api_key为sk-sp-eb50d67ca64a451b820cc4ab87ef8e6c，模型名 qwen3-max-2026-01-23
参考代码：
```python
import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-sp-eb50d67ca64a451b820cc4ab87ef8e6c",
    base_url="https://coding.dashscope.aliyuncs.com/v1",
)

response = client.responses.create(
    model="qwen3-max-2026-01-23",
    input="你能做些什么？"
)

# 获取模型回复
print(response.output_text)
```



## judge prompt exp(实验2)
- 使用一个固定prompt模板，包含多judge维度，单一judge prompt进行多维度评估，然后解析出分数；
- 维度设计：有用性&正交性，保证对jailbreak prompt rewrite任务的指向性，同时各维度尽可能正交不相关
    - idea保留程度
    - 攻击隐蔽程度
    - 攻击策略执行效果
    - 总体jailbreak潜力
- judge reward计算：解析judge model输出，使用json格式，每个维度对应一个0~1之间的两位浮点数，最后对所有维度的reward进行加权平均得到最终reward
- 使用judge prompt和ASR reward 1:1混合reward进行训练，每个实验训练1000步，进行3攻击prompt*（4单一维度（一个加权为1其他为0）+1均匀维度（所有加权为0.25，作为后续的消融实验内容））共15个训练实验
- 训练脚本和eval脚本分离（参考当前实现）
- 使用trl+accelerate实现并发grpo训练，参考docs/trl/docs/trl/grpo_trainer.md

## adaptive reward exp(实验3,new idea)
参考
- 参考jailbreak_research/RL4jailbreak/NEW_IDEA.md中的adaptive reward idea，通过计算ASR reward方差和judge reward方差+sigmoid函数偏置+ema滑动平均，来动态调整ASR reward的权重（同时也调整judge reward的权重）
- 在实验2的混合reward代码基础上进行修改，增加动态调整权重逻辑；可调参数包括ASR reward上下限（0～1）和计算方差的窗口大小；第一个窗口内没有reward,所以权重固定为1:1,后续窗口根据reward方差动态调整权重
    - ASR reward上下限实验 暂定默认0.8&0.2
    - 只使用一个jailbreak prompt+固定judge prompt+固定各维度权重，进行实验
    - 对ema滑动平均的beta参数进行调参，计划为0，0.5，0.67，0.8，0.9，0.95，对应窗口大小为1，2，3，5，10，20
- 也都统一训练1000步
- 实验3的judge reward prompt使用混合prompt，对表现前3的攻击prompt

---

# 注意事项
1. 我们使用的target model和judge model都是qwen3-4B,进行了复用; 
2. 每个实验中都需要在GPU1上加载需要的模型,那么加载前需要检查GPU状态和模型是否已经加载,如果已经加载则直接使用,如果GPU1非空闲但模型没加载则需要先释放GPU1显存; 在sh脚本中实现以上逻辑
3. 对于我们部署的模型,在reward计算时需要实现并发调用