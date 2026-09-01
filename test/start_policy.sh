#!/bin/bash
# Start Policy Model Server (GPU 1)
# 用于 Agent 推理和训练

# 激活虚拟环境
source /home/tiger/jailbreak_research/.venv/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export VLLM_USE_MODELSCOPE=true
export FLASHINFER_DISABLE_VERSION_CHECK=1

MODEL_PATH="/home/tiger/models/Qwen/Qwen3-4B"
PORT=8001

echo "=========================================="
echo "启动 Policy 模型服务"
echo "模型路径: $MODEL_PATH"
echo "端口: $PORT"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "虚拟环境: $VIRTUAL_ENV"
echo "=========================================="

# 使用虚拟环境的 vllm 命令
vllm serve ${MODEL_PATH} \
  --max-model-len 8192 \
  --port ${PORT} \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --trust-remote-code
