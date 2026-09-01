#!/bin/bash
# =============================================================================
# 2-GPU 服务布局(已验证,2026-08-20)
#   GPU0: Policy (Qwen3-4B, port 8003, 16k, util 0.9)
#   GPU1: Guard  (Qwen3Guard-Gen-4B, port 8001, 8k, util 0.42)
#   GPU1: Target (Qwen3-4B-SafeRL, port 8002, 8k, util 0.55)
# 注意:
#   - 必须用 venv 的 vllm(PATH 中的 vllm 是 python3.11 旧版本,EngineCore 会挂)
#   - 必须顺序启动:guard 完全就绪后再起 target(vLLM profiling 与并发启动会瞬态
#     显存竞争导致 "Available KV cache memory" 为负;target util 0.38 也不够,
#     其 profiling non-KV 占用约 32.5GB,需 0.55)
# Usage:
#   bash scripts/start_servers_2gpu.sh
# =============================================================================

set -e

source "$(dirname "$0")/common.sh"

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
export FLASHINFER_DISABLE_VERSION_CHECK=1
VLLM_BIN="$PROJECT_ROOT/.venv/bin/vllm"

log_section "启动 2-GPU 服务 (GPU0: policy, GPU1: guard+target)"

# 1. Policy (GPU0)
CUDA_VISIBLE_DEVICES=0 "$VLLM_BIN" serve "$BASE_MODEL" \
    --port 8003 --max-model-len 16384 --gpu-memory-utilization 0.9 \
    --trust-remote-code --dtype auto \
    > "$LOG_DIR/policy_2gpu.log" 2>&1 &
log_info "policy: GPU0 port 8003"
wait_for_server 8003 "Policy"

# 2. Guard (GPU1) —— 先启动并等待就绪
CUDA_VISIBLE_DEVICES=1 "$VLLM_BIN" serve "$GUARD_MODEL" \
    --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.42 \
    --trust-remote-code --dtype auto \
    > "$LOG_DIR/guard_2gpu.log" 2>&1 &
log_info "guard: GPU1 port 8001 (waiting ready before target)"
wait_for_server 8001 "Guard"

# 3. Target (GPU1) —— guard 完全就绪后再启动
CUDA_VISIBLE_DEVICES=1 "$VLLM_BIN" serve "$TARGET_MODEL" \
    --port 8002 --max-model-len 8192 --gpu-memory-utilization 0.55 \
    --trust-remote-code --dtype auto \
    > "$LOG_DIR/target_2gpu.log" 2>&1 &
log_info "target: GPU1 port 8002"
wait_for_server 8002 "Target"

log_section "全部服务已就绪"
log_info "policy=8003(GPU0) guard=8001(GPU1) target=8002(GPU1)"