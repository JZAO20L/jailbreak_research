#!/bin/bash
# =============================================================================
# Start the three vLLM services required by SESS.
#   guard    : Qwen3Guard-Gen-4B  (safety judge)   -> port 8002 (GPU 0)
#   target   : Qwen3-4B           (victim model)   -> port 8001 (GPU 1,2)
#   attacker : Qwen3-4B           (policy/refiner) -> port 8003 (GPU 3)
#
# Override model paths / ports / dtype via env, e.g.:
#   GUARD_MODEL=/path/Qwen3Guard-Gen-4B TARGET_MODEL=/path/Qwen3-4B \
#   ATK_MODEL=/path/Qwen3-4B bash start_services.sh
#
# Note for V100 (SM 7.0): add `--dtype float16` (see DTYPE below).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

GUARD_MODEL="${GUARD_MODEL:-Qwen/Qwen3Guard-Gen-4B}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
ATK_MODEL="${ATK_MODEL:-Qwen/Qwen3-4B}"
DTYPE="${DTYPE:-auto}"          # use float16 on V100 / bf16 on A100
GUARD_PORT="${GUARD_PORT:-8002}"
TARGET_PORT="${TARGET_PORT:-8001}"
ATK_PORT="${ATK_PORT:-8003}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

echo "== starting guard ($GUARD_MODEL) on :$GUARD_PORT (GPU0) =="
CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$GUARD_MODEL" --port "$GUARD_PORT" \
    --max-model-len 8192 --dtype "$DTYPE" \
    > "$LOG_DIR/guard.log" 2>&1 &

echo "== starting target ($TARGET_MODEL) on :$TARGET_PORT (GPUs 1,2) =="
CUDA_VISIBLE_DEVICES=1,2 nohup vllm serve "$TARGET_MODEL" --port "$TARGET_PORT" \
    --max-model-len 16384 --tensor-parallel-size 2 --dtype "$DTYPE" \
    > "$LOG_DIR/target.log" 2>&1 &

echo "== starting attacker ($ATK_MODEL) on :$ATK_PORT (GPU3) =="
CUDA_VISIBLE_DEVICES=3 nohup vllm serve "$ATK_MODEL" --port "$ATK_PORT" \
    --max-model-len 8192 --dtype "$DTYPE" \
    > "$LOG_DIR/attacker.log" 2>&1 &

echo "waiting for services to become ready..."
for port in "$GUARD_PORT" "$TARGET_PORT" "$ATK_PORT"; do
    until curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
        sleep 2
    done
    echo "  ok 127.0.0.1:${port}"
done
echo "== all services ready =="
