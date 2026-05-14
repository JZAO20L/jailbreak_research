#!/bin/bash
# Start Target Model Server (GPU1)
export CUDA_VISIBLE_DEVICES=1
export VLLM_USE_MODELSCOPE=true

MODEL_PATH="/home/tiger/models/Qwen3-4B"

vllm serve ${MODEL_PATH} \
  --served-model-name target \
  --max-model-len 8192 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0
