#!/bin/bash
# Start Guard Model Server (GPU1)
export CUDA_VISIBLE_DEVICES=2
export VLLM_USE_MODELSCOPE=true

MODEL_PATH="/home/tiger/models/Qwen3Guard-Gen-4B"

vllm serve ${MODEL_PATH} \
  --served-model-name guard \
  --max-model-len 8192 \
  --port 8002 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0
