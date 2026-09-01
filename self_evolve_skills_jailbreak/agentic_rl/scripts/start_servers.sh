#!/bin/bash
# Start vLLM servers for Target and Guard models
#
# Usage:
#   bash scripts/start_servers.sh  # Start both
#   bash scripts/start_servers.sh guard  # Start guard only
#   bash scripts/start_servers.sh target  # Start target only

set -e

GUARD_MODEL="/home/tiger/models/Qwen/Qwen3Guard-Gen-4B"
TARGET_MODEL="/home/tiger/models/Qwen/Qwen3-4B"

GUARD_PORT=8001
TARGET_PORT=8002

start_guard() {
    echo "Starting Guard server on port $GUARD_PORT (GPU 0)..."
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model "$GUARD_MODEL" \
        --port $GUARD_PORT \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --dtype auto &
    echo "Guard PID: $!"
}

start_target() {
    echo "Starting Target/Judge server on port $TARGET_PORT (GPU 0)..."
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model "$TARGET_MODEL" \
        --port $TARGET_PORT \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --dtype auto &
    echo "Target PID: $!"
}

case "${1:-all}" in
    guard)
        start_guard
        ;;
    target)
        start_target
        ;;
    all)
        start_guard
        sleep 5
        start_target
        ;;
    *)
        echo "Usage: $0 [all|guard|target]"
        exit 1
        ;;
esac

echo "Servers started. Use 'kill %1 %2' to stop."
