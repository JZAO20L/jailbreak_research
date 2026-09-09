#!/bin/bash
# =============================================================================
# 09-09: target (GPU1, 8002) 从 Qwen3-4B-SafeRL 换成 plain Qwen3-4B
# 动机: SafeRL target 太难破解 -> ASR 奖励稀疏; plain 4B 提升奖励稠密度
#       (用户决策, 见 docs/TODO.md 09-09 节)
# =============================================================================
cd /home/tiger/jailbreak_research/agentic_jailbreak
source scripts/common.sh

VLLM_BIN="$PROJECT_ROOT/.venv/bin/vllm"
TARGET_MODEL="${TARGET_MODEL:-/home/tiger/models/Qwen/Qwen3-4B}"
LOG_DIR="$OUTPUT_DIR/logs"

echo "== 停旧 target (SafeRL) =="
pkill -f 'Qwen3-4B-SafeRL' 2>/dev/null
sleep 5
pkill -9 -f 'Qwen3-4B-SafeRL' 2>/dev/null
sleep 2
if pgrep -f 'Qwen3-4B-SafeRL' >/dev/null; then
    echo "STILL-ALIVE"
else
    echo "old target stopped"
fi

echo "== 起新 target (plain Qwen3-4B, port $TARGET_PORT) =="
CUDA_VISIBLE_DEVICES=1 setsid nohup "$VLLM_BIN" serve "$TARGET_MODEL" \
    --port $TARGET_PORT --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --dtype float16 --trust-remote-code \
    > "$LOG_DIR/target_plain4b.log" 2>&1 < /dev/null &
echo "new target pid $! (log: $LOG_DIR/target_plain4b.log)"

echo "== 等健康 =="
for i in $(seq 1 600); do
    if curl -s "http://127.0.0.1:$TARGET_PORT/health" >/dev/null 2>&1; then
        echo "target ready after ${i}s"; break
    fi
    sleep 1
done
curl -s "http://127.0.0.1:$TARGET_PORT/v1/models" | head -c 400; echo
