#!/bin/bash
# =============================================================================
# Layer 3.1: AutoDAN Skills 核心机制 Grid 实验
# =============================================================================
#
# 实验设计：Grid 2 × 4 × 2 = 16 组
# - skill_call_mode: single_call / every_iteration (2)
# - update_strategy: success_only / failure_only / both / statistical (4)
# - cs_ratio: full_evolve(0%) / early(30%) (2)
#
# 固定配置：
# - data_size: large (1000)
# - skill_source: DAN模板 (6个)
# - skill_extraction_mode: trajectory
#
# Usage:
#   bash run_layer3.sh                              # 启动服务 + 运行全部 16 组
#   bash run_layer3.sh --skip_launch --max_workers 64  # 跳过启动，使用已有服务
#   bash run_layer3.sh --skip_launch --single single_call statistical full_evolve  # 单实验
#   bash run_layer3.sh --skip_launch --resume_from 9  # 断点续跑
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$EXP_DIR")")"

# 服务配置
GUARD_GPU="0"
TARGET_GPU="1,2"
GUARD_PORT=8002
TARGET_PORT=8001

# 模型路径
GUARD_MODEL_PATH="/home/tiger/models/Qwen/Qwen3Guard-Gen-4B"
TARGET_MODEL_PATH="/home/tiger/models/Qwen/Qwen3-4B"

# vLLM 参数
MAX_MODEL_LEN=8192
GUARD_TP=1
TARGET_TP=2
GPU_MEMORY_UTIL=0.9

# 实验参数
NUM_EPOCHS=1
MAX_ITERATIONS=10
MAX_WORKERS=64
MIN_SUCCESS_RATE=0.7
MAINTENANCE_INTERVAL=100

# 日志目录
LOG_DIR="$SCRIPT_DIR/logs"
RESULT_DIR="$SCRIPT_DIR/results_core"
PROCESS_LOG="$SCRIPT_DIR/process_core.log"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$RESULT_DIR/skills"

# PID 记录
GUARD_PID=""
TARGET_PID=""
SKIP_LAUNCH=false
RESUME_FROM=0
SINGLE_MODE=""

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --max_workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --skip_launch)
            SKIP_LAUNCH=true
            shift
            ;;
        --resume_from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --single)
            SINGLE_MODE="$2 $3 $4"
            shift 4
            ;;
        --guard_gpu)
            GUARD_GPU="$2"
            shift 2
            ;;
        --target_gpu)
            TARGET_GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_layer3.sh [--skip_launch] [--max_workers N] [--single call_mode strategy ratio] [--resume_from N]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=300
    local wait_time=0

    echo "等待 $name vLLM server 就绪 (port $port)..."

    while ! curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
        sleep 3
        wait_time=$((wait_time + 3))

        if [ $wait_time -ge $max_wait ]; then
            echo "错误: Server 在 ${max_wait} 秒内未就绪 (port $port)"
            return 1
        fi

        echo "  已等待 ${wait_time}s..."
    done

    echo "✓ $name Server 已就绪 (port $port)!"
    return 0
}

kill_server() {
    local port=$1
    echo "关闭 vLLM server (port $port)..."

    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        kill -9 $pid 2>/dev/null
        echo "已关闭进程 $pid (port $port)"
    else
        echo "未找到占用 port $port 的进程"
    fi
}

check_server_running() {
    local port=$1
    curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1
}

# =============================================================================
# Step 1: Start vLLM Servers
# =============================================================================

echo ""
echo "============================================================================"
echo "Layer 3.1: AutoDAN Skills 核心机制 Grid 实验"
echo "============================================================================"
echo ""
echo "Step 1: 启动 vLLM Servers"
echo "============================================================================"
echo "Guard:  GPU $GUARD_GPU, Port $GUARD_PORT, $GUARD_MODEL_PATH (TP=$GUARD_TP)"
echo "Target: GPU $TARGET_GPU, Port $TARGET_PORT, $TARGET_MODEL_PATH (TP=$TARGET_TP)"
echo "============================================================================"
echo ""

if [ "$SKIP_LAUNCH" = true ]; then
    echo "跳过服务启动 (--skip_launch)"

    if check_server_running $GUARD_PORT; then
        echo "✓ Guard 服务已在端口 $GUARD_PORT 运行"
    else
        echo "✗ Guard 服务未运行，请先启动"
        exit 1
    fi

    if check_server_running $TARGET_PORT; then
        echo "✓ Target 服务已在端口 $TARGET_PORT 运行"
    else
        echo "✗ Target 服务未运行，请先启动"
        exit 1
    fi
else
    export VLLM_USE_MODELSCOPE=true
    export FLASHINFER_DISABLE_VERSION_CHECK=1

    # 启动 Guard
    echo "启动 Qwen3Guard (GPU $GUARD_GPU, port $GUARD_PORT)..."
    CUDA_VISIBLE_DEVICES=$GUARD_GPU vllm serve "$GUARD_MODEL_PATH" \
        --served-model-name "Qwen3Guard-Gen-4B" \
        --port $GUARD_PORT \
        --host 0.0.0.0 \
        --max-model-len $MAX_MODEL_LEN \
        --tensor-parallel-size $GUARD_TP \
        --gpu-memory-utilization $GPU_MEMORY_UTIL \
        --dtype auto \
        > "$LOG_DIR/guard.log" 2>&1 &

    GUARD_PID=$!
    echo "Guard PID: $GUARD_PID"

    # 启动 Target
    echo "启动 Qwen3-4B (GPU $TARGET_GPU, port $TARGET_PORT)..."
    CUDA_VISIBLE_DEVICES=$TARGET_GPU vllm serve "$TARGET_MODEL_PATH" \
        --served-model-name "Qwen3-4B" \
        --port $TARGET_PORT \
        --host 0.0.0.0 \
        --max-model-len $MAX_MODEL_LEN \
        --tensor-parallel-size $TARGET_TP \
        --gpu-memory-utilization $GPU_MEMORY_UTIL \
        --dtype auto \
        > "$LOG_DIR/target.log" 2>&1 &

    TARGET_PID=$!
    echo "Target PID: $TARGET_PID"

    # 等待服务就绪
    echo ""
    wait_for_server $GUARD_PORT "Guard"
    wait_for_server $TARGET_PORT "Target"
fi

echo ""
echo "所有服务已就绪!"
echo ""

# =============================================================================
# Step 2: Run Layer 3.1 Grid Search
# =============================================================================

echo ""
echo "============================================================================"
echo "Step 2: 运行 Layer 3.1 Grid Search (16 组)"
echo "============================================================================"
echo ""
echo "Grid 变量:"
echo "  - skill_call_mode: single_call / every_iteration (2)"
echo "  - update_strategy: success_only / failure_only / both / statistical (4)"
echo "  - cs_ratio: full_evolve(0%) / early(30%) (2)"
echo ""
echo "固定配置:"
echo "  - data_size: large (1000)"
echo "  - skill_source: DAN模板 (6个)"
echo "  - skill_extraction_mode: trajectory"
echo ""
echo "并发数: $MAX_WORKERS"
echo "轮数: $NUM_EPOCHS"
echo "============================================================================"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_ROOT"

GRID_SEARCH_CMD="python self_evolve_skills_jailbreak/exp/layer3/scripts/grid_search_layer3.py \
    --num_epochs $NUM_EPOCHS \
    --max_iterations $MAX_ITERATIONS \
    --max_workers $MAX_WORKERS \
    --min_success_rate $MIN_SUCCESS_RATE \
    --maintenance_interval $MAINTENANCE_INTERVAL \
    --output_dir $RESULT_DIR \
    --guard_port $GUARD_PORT \
    --target_port $TARGET_PORT"

if [ "$RESUME_FROM" -gt 0 ]; then
    GRID_SEARCH_CMD="$GRID_SEARCH_CMD --resume_from $RESUME_FROM"
fi

if [ -n "$SINGLE_MODE" ]; then
    GRID_SEARCH_CMD="$GRID_SEARCH_CMD --single $SINGLE_MODE"
fi

echo "执行: $GRID_SEARCH_CMD"
echo "过程日志: $PROCESS_LOG"
eval $GRID_SEARCH_CMD 2>&1 | tee "$PROCESS_LOG"

# =============================================================================
# Step 3: Summary (内置在 Python 脚本中)
# =============================================================================

echo ""
echo "============================================================================"
echo "Step 3: 结果汇总"
echo "============================================================================"
echo ""
echo "结果已自动汇总到: $RESULT_DIR"
echo "查看汇总文件: ls $RESULT_DIR/layer3_core_summary_*.json"
echo ""
echo "============================================================================"

# =============================================================================
# Step 4: Stop Servers
# =============================================================================

if [ "$SKIP_LAUNCH" = false ]; then
    echo ""
    echo "============================================================================"
    echo "Step 4: 关闭 vLLM Servers"
    echo "============================================================================"
    echo ""

    if [ -n "$GUARD_PID" ]; then
        echo "关闭 Guard (PID: $GUARD_PID)..."
        kill $GUARD_PID 2>/dev/null || true
        wait $GUARD_PID 2>/dev/null || true
        echo "Guard 已关闭"
    fi

    if [ -n "$TARGET_PID" ]; then
        echo "关闭 Target (PID: $TARGET_PID)..."
        kill $TARGET_PID 2>/dev/null || true
        wait $TARGET_PID 2>/dev/null || true
        echo "Target 已关闭"
    fi

    sleep 3
    kill_server $GUARD_PORT
    kill_server $TARGET_PORT
fi

echo ""
echo "============================================================================"
echo "实验完成!"
echo "============================================================================"
echo ""
echo "结果目录: $RESULT_DIR"
echo "Skills目录: $RESULT_DIR/skills"
echo "过程日志: $PROCESS_LOG"
echo ""
echo "查看结果汇总:"
echo "  cat $RESULT_DIR/RESULTS_SUMMARY.md"
echo "  或打开 $RESULT_DIR/RESULTS_SUMMARY.html"
echo ""
echo "============================================================================"