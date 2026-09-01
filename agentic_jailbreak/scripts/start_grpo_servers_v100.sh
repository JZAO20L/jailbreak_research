#!/bin/bash
# GRPO 阶段服务: guard(GPU0 8001) + target(GPU1 8002) + swift rollout(GPU2 8004)
set -e
source "$(dirname "$0")/common.sh"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
export FLASHINFER_DISABLE_VERSION_CHECK=1
VLLM_BIN="$PROJECT_ROOT/.venv/bin/vllm"
MERGED="$AGENTIC_DIR/output/rft_sft_train1000_v2_merged"

log_section "GRPO 阶段服务 (guard=GPU0, target=GPU1, rollout=GPU2)"

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

CUDA_VISIBLE_DEVICES=2 "$PROJECT_ROOT/.venv/bin/swift" rollout \
    --model "$MERGED" --vllm_tensor_parallel_size 1 --port 8004 \
    --vllm_max_model_len 8192 --vllm_gpu_memory_utilization 0.8 \
    > "$LOG_DIR/rollout_v100.log" 2>&1 &
log_info "rollout: GPU2 port 8004 (v2 merged)"
wait_for_server 8004 "Rollout" 1500

log_section "GRPO 阶段服务就绪"
log_info "guard=8001(GPU0) target=8002(GPU1) rollout=8004(GPU2) train=GPU3"