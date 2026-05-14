#!/bin/bash
# Start Policy Model Server (GPU0)
# Usage: bash start_policy.sh [--lora-path /path/to/lora]
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_MODELSCOPE=true

MODEL_PATH="/home/tiger/models/Qwen3-4B"
LORA_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --lora-path)
      LORA_PATH="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

VLLM_ARGS="--served-model-name policy \
  --max-model-len 4096 \
  --port 8003 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0"

if [ -n "${LORA_PATH}" ]; then
  echo "Loading LoRA from: ${LORA_PATH}"
  vllm serve ${MODEL_PATH} \
    --enable-lora \
    --lora-path ${LORA_PATH} \
    ${VLLM_ARGS}
else
  echo "Starting policy model without LoRA"
  vllm serve ${MODEL_PATH} \
    ${VLLM_ARGS}
fi
