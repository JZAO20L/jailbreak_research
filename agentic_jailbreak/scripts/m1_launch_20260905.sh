#!/bin/bash
# =============================================================================
# A 轴 M1 (只 GRPO@10) 启动链 2026-09-05:
#   1) 停 GPU2:8003 的 M2 policy (B 轴/M2 评估已结束, GPU2 让给 swift rollout)
#   2) swift rollout base @8004 (GPU2) — max_model_len 必须 24576:
#      10 轮 C1 协议实测 input 可达 14337+, 8192 会 BadRequest 崩溃 (09-03 教训)
#      ⚠️ 必须 --torch_dtype float16: V100 (SM7.0) 不支持 bf16, swift 默认 bf16
#      会在 EngineCore 初始化时 ValueError 崩溃 (09-05 首次启动失败根因)
#   3) exp04_grpo_10turn.sh (M1, base, GPU3 训练, 300 步)
# Usage: bash scripts/m1_launch_20260905.sh
# =============================================================================
set -e
cd /home/tiger/jailbreak_research/agentic_jailbreak
source "$(dirname "$0")/common.sh"
LOG_DIR="$AGENTIC_DIR/output/logs"
BASE_MODEL=/home/tiger/models/Qwen/Qwen3-4B

# --- 1. 释放 GPU2 ---
for pid in $(pgrep -f "port 8003"); do kill "$pid" 2>/dev/null || true; done
sleep 10
for pid in $(pgrep -f "port 8003"); do kill -9 "$pid" 2>/dev/null || true; done
sleep 5

# --- 2. swift rollout (GPU2:8004, setsid 防误杀, 长跑) ---
CUDA_VISIBLE_DEVICES=2 setsid /home/tiger/jailbreak_research/.venv/bin/swift rollout \
    --model "$BASE_MODEL" \
    --vllm_tensor_parallel_size 1 --port 8004 --vllm_max_model_len 24576 \
    --vllm_gpu_memory_utilization 0.8 \
    --torch_dtype float16 \
    > "$LOG_DIR/rollout_exp04.log" 2>&1 &
echo "[m1-launch] $(date) swift rollout 启动中 (8004, max_model_len 24576)"
until curl -s http://127.0.0.1:8004/health > /dev/null 2>&1; do sleep 15; done
echo "[m1-launch] $(date) rollout 就绪"

# --- 3. M1 训练 (前台, 由 nohup 托管; 训练用 timeout 1000 由 exp04 内部传递) ---
bash scripts/exp04_grpo_10turn.sh
