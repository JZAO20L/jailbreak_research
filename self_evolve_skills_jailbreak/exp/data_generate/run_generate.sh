#!/bin/bash
# =============================================================================
# Data Generation Script
# =============================================================================
#
# 使用 Skills 系统生成合成数据
# - 输入：训练集的一半数据（500条）
# - 输出：种子 prompt -> 最终攻击成功 prompt 数据集
#
# Usage:
#   bash run_generate.sh --method pair_skills_28
#   bash run_generate.sh --method autodan_skills_54
#   bash run_generate.sh --method both
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/tiger/jailbreak_research"

# =============================================================================
# 清理 GPU 进程
# =============================================================================

echo "清理 GPU 上的 vLLM 进程..."

pkill -f "vllm serve" || true
pkill -f "vllm.api_server" || true
sleep 2

for port in 8001 8002 8003; do
    pid=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        kill -9 $pid 2>/dev/null || true
        echo "  清理端口 $port: PID $pid"
    fi
done

sleep 5

echo "✓ GPU 清理完成"

# =============================================================================
# 启动 vLLM 服务
# =============================================================================

echo ""
echo "启动 vLLM 服务..."

export VLLM_USE_MODELSCOPE=true
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Guard (GPU 0)
CUDA_VISIBLE_DEVICES=0 vllm serve "/home/tiger/models/Qwen/Qwen3Guard-Gen-4B" \
    --served-model-name "Qwen3Guard-Gen-4B" \
    --port 8002 \
    --host 0.0.0.0 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    > "$SCRIPT_DIR/logs/guard.log" 2>&1 &
GUARD_PID=$!

# Attacker (GPU 1)
CUDA_VISIBLE_DEVICES=1 vllm serve "/home/tiger/models/Qwen/Qwen3-4B" \
    --served-model-name "Qwen3-4B" \
    --port 8003 \
    --host 0.0.0.0 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    > "$SCRIPT_DIR/logs/attacker.log" 2>&1 &
ATTACKER_PID=$!

# Target (GPU 2,3, TP=2)
CUDA_VISIBLE_DEVICES=2,3 vllm serve "/home/tiger/models/Qwen/Qwen3-4B" \
    --served-model-name "Qwen3-4B" \
    --port 8001 \
    --host 0.0.0.0 \
    --max-model-len 4096 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    > "$SCRIPT_DIR/logs/target.log" 2>&1 &
TARGET_PID=$!

# 等待服务启动
wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=600
    local wait_time=0

    echo "Waiting for $name (port $port)..."

    while true; do
        if curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo "✓ $name ready (port $port)!"
            return 0
        fi
        sleep 5
        wait_time=$((wait_time + 5))
        if [ $wait_time -ge $max_wait ]; then
            echo "Error: $name not ready after ${max_wait}s"
            return 1
        fi
        echo "  waited ${wait_time}s..."
    done
}

mkdir -p "$SCRIPT_DIR/logs"

wait_for_server 8002 "Guard"
wait_for_server 8003 "Attacker"
wait_for_server 8001 "Target"

echo "All servers ready!"

# =============================================================================
# 运行数据生成
# =============================================================================

echo ""
echo "============================================================================"
echo "开始生成数据"
echo "============================================================================"

cd "$PROJECT_ROOT"

# 解析参数
METHOD="both"
MAX_WORKERS=32

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --max_workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Method: $METHOD"
echo "Max workers: $MAX_WORKERS"

# 运行生成脚本（使用全量训练数据）
python self_evolve_skills_jailbreak/exp/data_generate/generate_data.py \
    --method "$METHOD" \
    --max_workers "$MAX_WORKERS"

# =============================================================================
# 停止服务
# =============================================================================

echo ""
echo "============================================================================"
echo "停止 vLLM 服务"
echo "============================================================================"

for pid in $GUARD_PID $ATTACKER_PID $TARGET_PID; do
    if [ -n "$pid" ]; then
        kill $pid 2>/dev/null || true
    fi
done

sleep 3

for port in 8001 8002 8003; do
    pid=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        kill -9 $pid 2>/dev/null || true
    fi
done

echo ""
echo "============================================================================"
echo "数据生成完成"
echo "============================================================================"
echo ""
echo "结果保存在: $SCRIPT_DIR/results/"
echo ""
echo "查看结果:"
echo "  ls $SCRIPT_DIR/results/"
echo "  cat $SCRIPT_DIR/results/*/summary.json"