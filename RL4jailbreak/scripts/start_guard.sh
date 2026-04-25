#!/bin/bash
# Start Guard Model Server (GPU1)
export CUDA_VISIBLE_DEVICES=1
export VLLM_USE_MODELSCOPE=true

MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B"

vllm serve ${MODEL_PATH} \
  --served-model-name guard \
  --max-model-len 2048 \
  --port 8002 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.4 \
  --host 0.0.0.0
