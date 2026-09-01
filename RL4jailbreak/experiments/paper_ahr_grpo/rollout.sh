#!/bin/bash
# =============================================================================
# Rollout Service Startup Script for AHR-GRPO Paper Experiments
#
# GPU Allocation (3 cards):
# - GPU 0: Guard (Qwen3Guard-Gen-4B) + Target&Judge (Qwen3-4B) - vLLM servers
# - GPU 1-2: Policy (Qwen3-4B) training with TP=2 (colocate mode)
#
# This script starts vLLM inference services on GPU 0:
# 1. Guard model on port 8001
# 2. Target&Judge model on port 8002
#
# Usage:
#   bash rollout.sh start    # Start all services
#   bash rollout.sh stop     # Stop all services
#   bash rollout.sh status   # Check service status
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
GUARD_MODEL="/home/tiger/models/Qwen/Qwen3Guard-Gen-4B"
TARGET_MODEL="/home/tiger/models/Qwen/Qwen3-4B"

GUARD_PORT=8001
TARGET_PORT=8002

GUARD_GPU=0
TARGET_GPU=0  # Same GPU as guard, serial inference

# Model context lengths
GUARD_MAX_MODEL_LEN=8192
TARGET_MAX_MODEL_LEN=8192

LOG_DIR="output/logs"
mkdir -p $LOG_DIR

# =============================================================================
# Functions
# =============================================================================
start_guard() {
    echo "Starting Guard model on GPU $GUARD_GPU, port $GUARD_PORT..."
    echo "  Model: $GUARD_MODEL"
    echo "  Max model len: $GUARD_MAX_MODEL_LEN"
    
    CUDA_VISIBLE_DEVICES=$GUARD_GPU \
    swift rollout \
        --model $GUARD_MODEL \
        --port $GUARD_PORT \
        --vllm_tensor_parallel_size 1 \
        --vllm_gpu_memory_utilization 0.4 \
        --vllm_max_model_len $GUARD_MAX_MODEL_LEN \
        > $LOG_DIR/guard_rollout.log 2>&1 &

    echo $! > $LOG_DIR/guard.pid
    echo "Guard started with PID $(cat $LOG_DIR/guard.pid)"
}

start_target_judge() {
    echo "Starting Target&Judge model on GPU $TARGET_GPU, port $TARGET_PORT..."
    echo "  Model: $TARGET_MODEL"
    echo "  Max model len: $TARGET_MAX_MODEL_LEN"
    
    CUDA_VISIBLE_DEVICES=$TARGET_GPU \
    swift rollout \
        --model $TARGET_MODEL \
        --port $TARGET_PORT \
        --vllm_tensor_parallel_size 1 \
        --vllm_gpu_memory_utilization 0.4 \
        --vllm_max_model_len $TARGET_MAX_MODEL_LEN \
        > $LOG_DIR/target_judge_rollout.log 2>&1 &

    echo $! > $LOG_DIR/target_judge.pid
    echo "Target&Judge started with PID $(cat $LOG_DIR/target_judge.pid)"
}

stop_guard() {
    if [ -f $LOG_DIR/guard.pid ]; then
        PID=$(cat $LOG_DIR/guard.pid)
        echo "Stopping Guard (PID: $PID)..."
        kill $PID 2>/dev/null || true
        rm $LOG_DIR/guard.pid
        echo "Guard stopped."
    else
        echo "Guard PID file not found."
    fi
}

stop_target_judge() {
    if [ -f $LOG_DIR/target_judge.pid ]; then
        PID=$(cat $LOG_DIR/target_judge.pid)
        echo "Stopping Target&Judge (PID: $PID)..."
        kill $PID 2>/dev/null || true
        rm $LOG_DIR/target_judge.pid
        echo "Target&Judge stopped."
    else
        echo "Target&Judge PID file not found."
    fi
}

check_status() {
    echo "=== Service Status ==="

    if [ -f $LOG_DIR/guard.pid ]; then
        PID=$(cat $LOG_DIR/guard.pid)
        if kill -0 $PID 2>/dev/null; then
            echo "Guard: RUNNING (PID: $PID, Port: $GUARD_PORT)"
        else
            echo "Guard: STOPPED (stale PID: $PID)"
        fi
    else
        echo "Guard: NOT STARTED"
    fi

    if [ -f $LOG_DIR/target_judge.pid ]; then
        PID=$(cat $LOG_DIR/target_judge.pid)
        if kill -0 $PID 2>/dev/null; then
            echo "Target&Judge: RUNNING (PID: $PID, Port: $TARGET_PORT)"
        else
            echo "Target&Judge: STOPPED (stale PID: $PID)"
        fi
    else
        echo "Target&Judge: NOT STARTED"
    fi

    echo "======================"
}

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=60
    local waited=0
    
    echo "Waiting for $name server on port $port..."
    while ! curl -s "http://localhost:$port/health" > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            echo "ERROR: $name server failed to start within ${max_wait}s"
            echo "Check logs: $LOG_DIR/${name}_rollout.log"
            return 1
        fi
    done
    echo "$name server is ready!"
    return 0
}

# =============================================================================
# Main
# =============================================================================
case "$1" in
    start)
        echo "================================================================"
        echo "Starting rollout services on GPU 0..."
        echo "================================================================"
        echo "Guard:    $GUARD_MODEL -> port $GUARD_PORT"
        echo "Target:   $TARGET_MODEL -> port $TARGET_PORT"
        echo "================================================================"
        
        start_guard
        wait_for_server $GUARD_PORT "guard"
        
        start_target_judge
        wait_for_server $TARGET_PORT "target_judge"
        
        echo ""
        echo "================================================================"
        echo "All services started successfully!"
        echo "  Guard:        http://localhost:$GUARD_PORT"
        echo "  Target&Judge: http://localhost:$TARGET_PORT"
        echo "Logs: $LOG_DIR/"
        echo "================================================================"
        ;;
    stop)
        echo "Stopping rollout services..."
        stop_guard
        stop_target_judge
        echo "All services stopped."
        ;;
    status)
        check_status
        ;;
    restart)
        echo "Restarting rollout services..."
        stop_guard
        stop_target_judge
        sleep 2
        start_guard
        wait_for_server $GUARD_PORT "guard"
        start_target_judge
        wait_for_server $TARGET_PORT "target_judge"
        echo "All services restarted."
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        echo ""
        echo "Commands:"
        echo "  start    - Start Guard and Target&Judge services on GPU 0"
        echo "  stop     - Stop all services"
        echo "  status   - Check service status"
        echo "  restart  - Restart all services"
        exit 1
        ;;
esac
