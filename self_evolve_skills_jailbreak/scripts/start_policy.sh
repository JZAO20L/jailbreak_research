#!/bin/bash
# Start Target Model Server (GPU1)
export CUDA_VISIBLE_DEVICES=1,2
export VLLM_USE_MODELSCOPE=true
export FLASHINFER_DISABLE_VERSION_CHECK=1

MODEL_PATH="/home/tiger/models/Qwen/Qwen3-4B"

vllm serve ${MODEL_PATH} \
  --max-model-len 16384 \
  --port 8001 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0