#!/bin/bash
# =============================================================================
# Agentic Jailbreak 共享函数
# =============================================================================
#
# 所有实验脚本通过 source common.sh 使用这些函数
#
# =============================================================================

set -e

# 本机 flashinfer-cubin 与 flashinfer 版本不匹配，跳过版本检查（否则 vLLM 启动即崩）
export FLASHINFER_DISABLE_VERSION_CHECK=1

# 显式使用项目 venv 的 vllm/swift（当前 shell 未激活 venv，裸 swift 会 command not found）
export PATH="/home/tiger/jailbreak_research/.venv/bin:$PATH"

# =============================================================================
# 路径配置
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="/home/tiger/jailbreak_research"
AGENTIC_DIR="$PROJECT_ROOT/agentic_jailbreak"

# =============================================================================
# 默认配置（可被实验脚本覆盖）
# =============================================================================

# GPU 配置
NUM_GPUS="${NUM_GPUS:-8}"

# 模型路径
BASE_MODEL="${BASE_MODEL:-/home/tiger/models/Qwen/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/home/tiger/models/Qwen/Qwen3Guard-Gen-4B}"
TARGET_MODEL="${TARGET_MODEL:-/home/tiger/models/Qwen/Qwen3-4B-SafeRL}"

# 端口
GUARD_PORT="${GUARD_PORT:-8001}"
TARGET_PORT="${TARGET_PORT:-8002}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8003}"

# 根据 GPU 数量自动分配
if [ "$NUM_GPUS" -eq 8 ]; then
    GUARD_GPU="${GUARD_GPU:-0}"
    TARGET_GPU="${TARGET_GPU:-1}"
    ROLLOUT_GPUS="${ROLLOUT_GPUS:-2,3}"
    TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
    TRAIN_TP=4
elif [ "$NUM_GPUS" -eq 4 ]; then
    GUARD_GPU="${GUARD_GPU:-0}"
    TARGET_GPU="${TARGET_GPU:-1}"
    ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
    TRAIN_GPUS="${TRAIN_GPUS:-3}"
    TRAIN_TP=1
else
    echo "ERROR: NUM_GPUS must be 4 or 8, got $NUM_GPUS"
    exit 1
fi

# 数据路径
SKILLS_PATH="${SKILLS_PATH:-$AGENTIC_DIR/data/skills.json}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-$AGENTIC_DIR/output}"

# =============================================================================
# 日志函数
# =============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_section() {
    echo ""
    echo "============================================================================="
    echo "$*"
    echo "============================================================================="
    echo ""
}

# =============================================================================
# GPU 清理
# =============================================================================

cleanup_gpus() {
    log_section "清理 GPU 进程"
    
    # 杀掉 vLLM 进程
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "vllm.api_server" 2>/dev/null || true
    sleep 2
    
    # 清理特定端口
    for port in 8001 8002 8003; do
        pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill -9 $pid 2>/dev/null || true
            log_info "清理端口 $port: PID $pid"
        fi
    done
    
    sleep 5
    log_info "GPU 状态:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "  (nvidia-smi 不可用)"
}

# =============================================================================
# vLLM Server 管理
# =============================================================================

start_guard_server() {
    local gpu="${1:-$GUARD_GPU}"
    local port="${2:-$GUARD_PORT}"
    local log_dir="${3:-$OUTPUT_DIR/logs}"
    
    mkdir -p "$log_dir"
    
    log_info "启动 Guard server: GPU $gpu, port $port"
    
    CUDA_VISIBLE_DEVICES=$gpu vllm serve "$GUARD_MODEL" \
        --port $port \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --dtype auto \
        > "$log_dir/guard.log" 2>&1 &
    
    GUARD_PID=$!
    wait_for_server $port "Guard"
}

start_target_server() {
    local gpu="${1:-$TARGET_GPU}"
    local port="${2:-$TARGET_PORT}"
    local model="${3:-$TARGET_MODEL}"
    local log_dir="${4:-$OUTPUT_DIR/logs}"
    
    mkdir -p "$log_dir"
    
    log_info "启动 Target server: GPU $gpu, port $port, model=$model"
    
    CUDA_VISIBLE_DEVICES=$gpu vllm serve "$model" \
        --port $port \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --dtype auto \
        > "$log_dir/target.log" 2>&1 &
    
    TARGET_PID=$!
    wait_for_server $port "Target"
}

start_rollout_server() {
    local gpus="${1:-$ROLLOUT_GPUS}"
    local port="${2:-$ROLLOUT_PORT}"
    local model="${3:-$BASE_MODEL}"
    local log_dir="${4:-$OUTPUT_DIR/logs}"
    
    mkdir -p "$log_dir"
    
    local num_gpus=$(echo "$gpus" | tr ',' '\n' | wc -l)
    
    log_info "启动 Rollout server: GPU $gpus, port $port, TP=$num_gpus"
    
    CUDA_VISIBLE_DEVICES=$gpus swift rollout \
        --model "$model" \
        --vllm_tensor_parallel_size $num_gpus \
        --port $port \
        --vllm_max_model_len 8192 \
        --vllm_gpu_memory_utilization 0.8 \
        > "$log_dir/rollout.log" 2>&1 &
    
    ROLLOUT_PID=$!
    wait_for_server $port "Rollout"
}

wait_for_server() {
    local port=$1
    local name=$2
    local timeout=${3:-600}
    
    log_info "等待 $name server (port $port) 启动..."
    
    local start_time=$(date +%s)
    while true; do
        if curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            log_info "$name server 已就绪 (port $port)"
            return 0
        fi
        
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $timeout ]; then
            log_error "$name server 启动超时 (${timeout}s)"
            return 1
        fi
        
        sleep 5
    done
}

stop_all_servers() {
    log_section "停止所有 servers"
    
    [ -n "$GUARD_PID" ] && kill $GUARD_PID 2>/dev/null || true
    [ -n "$TARGET_PID" ] && kill $TARGET_PID 2>/dev/null || true
    [ -n "$ROLLOUT_PID" ] && kill $ROLLOUT_PID 2>/dev/null || true
    
    sleep 3
    
    # 强制清理
    for port in 8001 8002 8003; do
        pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill -9 $pid 2>/dev/null || true
        fi
    done
    
    log_info "所有 servers 已停止"
}

# =============================================================================
# 实验完成
# =============================================================================

experiment_complete() {
    local exp_name=$1
    local output_dir=$2
    
    log_section "实验完成: $exp_name"
    log_info "输出目录: $output_dir"
    
    if [ -f "$output_dir/summary.json" ]; then
        log_info "结果摘要:"
        cat "$output_dir/summary.json" | python -m json.tool 2>/dev/null || cat "$output_dir/summary.json"
    fi
}
