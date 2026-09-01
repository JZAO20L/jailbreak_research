#!/bin/bash
# =============================================================================
# 4-GPU 服务布局(单流重测用,2026-08-25)
#   GPU0: Policy #1  (Qwen3-4B, port 8003, 16k, util 0.9)
#   GPU1: Policy #2  (Qwen3-4B, port 8004, 16k, util 0.9)  <- 双实例负载均衡
#   GPU2: Guard      (Qwen3Guard-Gen-4B, port 8001, 8k, util 0.9)
#   GPU3: Target     (Qwen3-4B-SafeRL, port 8002, 8k, util 0.9)
# 注意:
#   - 必须用 venv 的 vllm(PATH 中的 vllm 是 python3.11 旧版本,EngineCore 会挂)
#   - guard 先启动并完全就绪后再起 target(vLLM profiling 与并发启动会瞬态
#     显存竞争)
# Usage:
#   bash scripts/start_servers_4gpu.sh
# =============================================================================

set -e

source "$(dirname "$0")/common.sh"

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
export FLASHINFER_DISABLE_VERSION_CHECK=1
VLLM_BIN="$PROJECT_ROOT/.venv/bin/vllm"

log_section "启动 4-GPU 服务 (GPU0/1: policy x2, GPU2: guard, GPU3: target)"

# 1. Guard (GPU2) —— 先启动并等待就绪
CUDA_VISIBLE_DEVICES=2 "$VLLM_BIN" serve "$GUARD_MODEL" \
    --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --trust-remote-code --dtype auto \
    > "$LOG_DIR/guard_4gpu.log" 2>&1 &
log_info "guard: GPU2 port 8001 (waiting ready before target)"
wait_for_server 8001 "Guard"

# 2. Target (GPU3) —— guard 完全就绪后再启动
CUDA_VISIBLE_DEVICES=3 "$VLLM_BIN" serve "$TARGET_MODEL" \
    --port 8002 --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --trust-remote-code --dtype auto \
    > "$LOG_DIR/target_4gpu.log" 2>&1 &
log_info "target: GPU3 port 8002"
wait_for_server 8002 "Target"

# 3. Policy #1 (GPU0)
CUDA_VISIBLE_DEVICES=0 "$VLLM_BIN" serve "$BASE_MODEL" \
    --port 8003 --max-model-len 16384 --gpu-memory-utilization 0.9 \
    --trust-remote-code --dtype auto \
    > "$LOG_DIR/policy1_4gpu.log" 2>&1 &
log_info "policy#1: GPU0 port 8003"
wait_for_server 8003 "Policy#1"

# 4. Policy #2 (GPU1)
CUDA_VISIBLE_DEVICES=1 "$VLLM_BIN" serve "$BASE_MODEL" \
    --port 8004 --max-model-len 16384 --gpu-memory-utilization 0.9 \
    --trust-remote-code --dtype auto \
    > "$LOG_DIR/policy2_4gpu.log" 2>&1 &
log_info "policy#2: GPU1 port 8004"
wait_for_server 8004 "Policy#2"

log_section "全部服务已就绪"
log_info "guard=8001(GPU2) target=8002(GPU3) policy#1=8003(GPU0) policy#2=8004(GPU1)"