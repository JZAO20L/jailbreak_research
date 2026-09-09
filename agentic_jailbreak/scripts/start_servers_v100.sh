#!/bin/bash
# =============================================================================
# V100 服务布局 v2 (2026-08-27, 4×V100-SXM2-32GB, 每卡一模型)
#   GPU0: Guard (Qwen3Guard-Gen-4B, port 8001, fp16, util 0.9)
#   GPU1: Target (Qwen3-4B-SafeRL, port 8002, fp16, util 0.9)
#   GPU2: Policy (Qwen3-4B, port 8003, fp16, util 0.9)
#   GPU3: 预留训练卡 (SFT/GRPO train; GRPO 时 GPU2 换 swift rollout 8004)
# 注意:
#   - V100 不支持 bf16, 显式 --dtype float16
#   - V100 无 FA2, vLLM 首次启动 triton kernel 编译慢, 超时设 1500s
# Usage: bash scripts/start_servers_v100.sh
# =============================================================================
set -e
source "$(dirname "$0")/common.sh"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
export FLASHINFER_DISABLE_VERSION_CHECK=1
VLLM_BIN="$PROJECT_ROOT/.venv/bin/vllm"

log_section "启动 V100 服务 (guard=GPU0, target=GPU1, policy=GPU2)"

CUDA_VISIBLE_DEVICES=0 "$VLLM_BIN" serve "$GUARD_MODEL" \
    --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --dtype float16 --trust-remote-code \
    > "$LOG_DIR/guard_v100.log" 2>&1 &
log_info "guard: GPU0 port 8001"
wait_for_server 8001 "Guard" 1500

CUDA_VISIBLE_DEVICES=1 "$VLLM_BIN" serve "$TARGET_MODEL" \
    --port 8002 --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --dtype float16 --trust-remote-code \
    > "$LOG_DIR/target_v100.log" 2>&1 &
log_info "target: GPU1 port 8002"
wait_for_server 8002 "Target" 1500

CUDA_VISIBLE_DEVICES=2 "$VLLM_BIN" serve "$BASE_MODEL" \
    --port 8003 --max-model-len 24576 --gpu-memory-utilization 0.9 \
    --dtype float16 --trust-remote-code \
    > "$LOG_DIR/policy_v100.log" 2>&1 &
log_info "policy: GPU2 port 8003"
wait_for_server 8003 "Policy" 1500

log_section "V100 服务就绪"
log_info "guard=8001(GPU0) target=8002(GPU1) policy=8003(GPU2) 训练卡=GPU3"