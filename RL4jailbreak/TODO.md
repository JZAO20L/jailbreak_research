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