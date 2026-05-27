#!/bin/bash
# Start Guard Model Server (GPU0)
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_MODELSCOPE=true
export FLASHINFER_DISABLE_VERSION_CHECK=1


MODEL_PATH="/home/tiger/models/Qwen/Qwen3Guard-Gen-4B"

vllm serve ${MODEL_PATH} \
  --max-model-len 8192 \
  --port 8002 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0