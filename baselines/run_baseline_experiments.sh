#!/bin/bash
# =============================================================================
# Baseline ASR Experiments Launcher Script
# =============================================================================
#
# GPU配置：
# - Guard: GPU 0, 端口 8002, Qwen3Guard-Gen-4B
# - Target: GPU 1-2, 端口 8001, Qwen3-4B (TP=2)
#
# 测试策略：pair, autodan, genetic, deepinception
# 轮次上限：10
#
# Usage:
#   # 快速测试（50条数据）
#   bash baselines/run_baseline_experiments.sh --limit 50
#
#   # 完整测试（200条数据）
#   bash baselines/run_baseline_experiments.sh --limit 200
#
#   # 测试特定策略
#   bash baselines/run_baseline_experiments.sh --strategies pair autodan --limit 100
#
#   # 设置并发数（默认8）
#   bash baselines/run_baseline_experiments.sh --workers 4 --limit 100
#
#   # 手动启动服务后测试（跳过服务启动）
#   bash baselines/run_baseline_experiments.sh --skip_launch --limit 100
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# GPU配置
GUARD_GPU="0"
TARGET_GPU="1,2"

# 模型路径
QWEN3_4B_PATH="/home/tiger/models/Qwen/Qwen3-4B"
QWEN3GUARD_PATH="/home/tiger/models/Qwen/Qwen3Guard-Gen-4B"

# 端口配置
TARGET_PORT=8001
GUARD_PORT=8002

# vLLM参数
GPU_MEMORY_UTIL=0.9
MAX_MODEL_LEN=8192
GUARD_TP=1
TARGET_TP=2

# 测试配置
TEST_FILE="data/dataset/processed/10k/test.jsonl"
OUTPUT_DIR="experiments/baseline_asr"
STRATEGIES="pair autodan genetic deepinception deepinception_multilayer persona"  # 默认测试所有非多语种方法
LIMIT=""
MAX_ITERATIONS=10  # 轮次上限
MAX_WORKERS=8      # 并发数（同时执行的攻击流程数）

# 日志目录
LOG_DIR="baselines/logs"
mkdir -p "$LOG_DIR"

# PID记录
GUARD_PID=""
TARGET_PID=""
SKIP_LAUNCH=false

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --strategies)
            STRATEGIES="$2"
            shift 2
            ;;
        --max_iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip_launch)
            SKIP_LAUNCH=true
            shift
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
            echo "Usage: bash baselines/run_baseline_experiments.sh [--limit N] [--strategies ...] [--max_iterations N] [--workers N] [--skip_launch]"
            exit 1
            ;;
    esac
done

# 创建输出目录（在参数解析后）
mkdir -p "$OUTPUT_DIR"

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
echo "Step 1: 启动 vLLM Servers"
echo "============================================================================"
echo "Guard:  GPU $GUARD_GPU, Port $GUARD_PORT, $QWEN3GUARD_PATH (TP=$GUARD_TP)"
echo "Target: GPU $TARGET_GPU, Port $TARGET_PORT, $QWEN3_4B_PATH (TP=$TARGET_TP)"
echo "============================================================================"
echo ""

if [ "$SKIP_LAUNCH" = true ]; then
    echo "跳过服务启动 (--skip_launch)"

    # 检查服务是否已运行
    if check_server_running $GUARD_PORT; then
        echo "✓ Guard 服务已在端口 $GUARD_PORT 运行"
    else
        echo "✗ Guard 服务未运行，请先启动: bash RL4jailbreak/scripts/start_guard.sh"
        exit 1
    fi

    if check_server_running $TARGET_PORT; then
        echo "✓ Target 服务已在端口 $TARGET_PORT 运行"
    else
        echo "✗ Target 服务未运行，请先启动: bash RL4jailbreak/scripts/start_target.sh"
        exit 1
    fi
else
    # 设置环境变量
    export VLLM_USE_MODELSCOPE=true
    export FLASHINFER_DISABLE_VERSION_CHECK=1

    # 启动 Guard
    echo "启动 Qwen3Guard (GPU $GUARD_GPU, port $GUARD_PORT)..."
    CUDA_VISIBLE_DEVICES=$GUARD_GPU vllm serve "$QWEN3GUARD_PATH" \
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
    CUDA_VISIBLE_DEVICES=$TARGET_GPU vllm serve "$QWEN3_4B_PATH" \
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
# Step 2: Run ASR Tests
# =============================================================================

echo ""
echo "============================================================================"
echo "Step 2: 执行 Baseline ASR 测试"
echo "============================================================================"
echo "测试数据: $TEST_FILE"
echo "策略: $STRATEGIES"
echo "最大迭代次数: $MAX_ITERATIONS"
echo "并发数: $MAX_WORKERS"
if [ -n "$LIMIT" ]; then
    echo "数据量限制: $LIMIT"
fi
echo "输出目录: $OUTPUT_DIR"
echo "============================================================================"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$OUTPUT_DIR/baseline_asr_summary_${TIMESTAMP}.json"

# 创建汇总文件初始结构
echo '{"timestamp": "'$TIMESTAMP'", "config": {"max_iterations": '$MAX_ITERATIONS', "test_file": "'$TEST_FILE'"}, "results": {}}' > "$SUMMARY_FILE"

for strategy in $STRATEGIES; do
    echo ""
    echo "----------------------------------------"
    echo "测试策略: $strategy"
    echo "----------------------------------------"

    # 构建 Python 测试命令
    TEST_CMD="python baselines/test_single_strategy.py \
        --strategy $strategy \
        --test_path $TEST_FILE \
        --target_port $TARGET_PORT \
        --guard_port $GUARD_PORT \
        --max_iterations $MAX_ITERATIONS \
        --workers $MAX_WORKERS \
        --output_dir $OUTPUT_DIR"

    if [ -n "$LIMIT" ]; then
        TEST_CMD="$TEST_CMD --test_limit $LIMIT"
    fi

    echo "执行: $TEST_CMD"
    eval $TEST_CMD

    echo ""
done

# =============================================================================
# Step 3: Generate Summary
# =============================================================================

echo ""
echo "============================================================================"
echo "Step 3: 生成结果汇总"
echo "============================================================================"
echo ""

python baselines/summarize_asr_results.py --input "$OUTPUT_DIR" --output "$SUMMARY_FILE"

echo ""
echo "结果汇总:"
cat "$SUMMARY_FILE" | python -c "import sys,json; d=json.load(sys.stdin); print('Method       ASR      Avg Iter    Avg Time'); print('-'*42); [print(f'{k:<12} {v[\"asr\"]*100:>9.1f}% {v[\"avg_iterations\"]:>10.1f} {v[\"avg_time\"]:>10.1f}s') for k,v in d['results'].items()]"

# =============================================================================
# Step 4: Stop Servers (if not skip_launch)
# =============================================================================

if [ "$SKIP_LAUNCH" = false ]; then
    echo ""
    echo "============================================================================"
    echo "Step 4: 关闭 vLLM Servers"
    echo "============================================================================"
    echo ""

    # 关闭 Guard
    if [ -n "$GUARD_PID" ]; then
        echo "关闭 Guard (PID: $GUARD_PID)..."
        kill $GUARD_PID 2>/dev/null || true
        wait $GUARD_PID 2>/dev/null || true
        echo "Guard 已关闭"
    fi

    # 关闭 Target
    if [ -n "$TARGET_PID" ]; then
        echo "关闭 Target (PID: $TARGET_PID)..."
        kill $TARGET_PID 2>/dev/null || true
        wait $TARGET_PID 2>/dev/null || true
        echo "Target 已关闭"
    fi

    # 确保端口释放
    sleep 3
    kill_server $GUARD_PORT
    kill_server $TARGET_PORT
fi

echo ""
echo "============================================================================"
echo "实验完成!"
echo "============================================================================"
echo ""
echo "输出目录: $OUTPUT_DIR"
echo "汇总文件: $SUMMARY_FILE"
echo "日志目录: $LOG_DIR"
echo ""
echo "============================================================================"