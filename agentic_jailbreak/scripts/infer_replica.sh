#!/bin/bash
# =============================================================================
# 给 target / guard 加一个推理副本(提奖励侧吞吐用的那个开关)
# =============================================================================
# 为什么需要: 两章 GRPO 的每步都是同步打 target/guard 的 HTTP 往返, 瓶颈在服务容量
#   而不在 trainer 卡(实测 Ch1 15.2s/步里 GPU 常在等回包)。再加训练卡几乎无收益,
#   加一份 target/guard 副本才是真提速。
#
# Usage:
#   bash scripts/infer_replica.sh target <gpu> <port> [util]   # util 默认 0.45
#   bash scripts/infer_replica.sh guard  <gpu> <port> [util]
#   例: 采集结束后把 GPU2 变成两个副本
#     bash scripts/infer_replica.sh target 2 8012 0.45
#     bash scripts/infer_replica.sh guard  2 8013 0.45
#   然后训练侧把端口写成列表:  --target-port 8002,8012 --guard-port 8001,8013
#
# 停止: 脚本会打印 PID, kill 该 PID 即可(不要 pkill -f vllm, 会误伤主服务)
# =============================================================================
set -e
NUM_GPUS=4
source "$(dirname "$0")/common.sh"

ROLE="${1:?用法: infer_replica.sh <target|guard> <gpu> <port> [util]}"
GPU="${2:?需要 GPU 号}"
PORT="${3:?需要端口}"
UTIL="${4:-0.45}"

case "$ROLE" in
    target) MODEL="$TARGET_MODEL" ;;
    guard)  MODEL="$GUARD_MODEL" ;;
    *) echo "ROLE 只能是 target 或 guard, 收到 $ROLE"; exit 1 ;;
esac

if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "端口 $PORT 已有服务在跑, 放弃启动副本"; exit 1
fi

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
echo "== 启动 $ROLE 副本: GPU$GPU port $PORT util=$UTIL model=$MODEL =="
CUDA_VISIBLE_DEVICES=$GPU setsid nohup "$PROJECT_ROOT/.venv/bin/vllm" serve "$MODEL" \
    --port $PORT --max-model-len 16384 --gpu-memory-utilization $UTIL \
    --dtype bfloat16 --trust-remote-code \
    > "$LOG_DIR/${ROLE}_replica_${PORT}.log" 2>&1 < /dev/null &
PID=$!
echo "pid=$PID  日志=$LOG_DIR/${ROLE}_replica_${PORT}.log  (停止: kill $PID)"

for i in $(seq 1 900); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
        "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
    [ "$code" = "200" ] && { echo "$ROLE 副本就绪 (port $PORT, ${i}s)"; exit 0; }
    if ! kill -0 $PID 2>/dev/null; then
        echo "副本进程已退出, 见日志尾部:"; tail -5 "$LOG_DIR/${ROLE}_replica_${PORT}.log"; exit 1
    fi
    sleep 1
done
echo "启动超时 900s"; exit 1
