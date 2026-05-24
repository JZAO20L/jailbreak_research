#!/bin/bash
# =============================================================================
# Baseline Experiments Launcher Script
# =============================================================================
# 
# 在单卡GPU0上同时加载Qwen3-4B（port 8001）和Qwen3Guard（port 8002）
# 各模型占用0.45显存，使用vLLM server
#
# 流程：
# 1. 启动两个vLLM servers
# 2. 等待servers就绪
# 3. 调用rewrite_prompts_server_concurrent.py进行prompt重写（并发版本）
# 4. 调用asr_test_server_concurrent.py进行ASR测试（并发版本）
# 5. 关闭servers
#
# Usage:
#   # 快速测试（100条数据）
#   bash baselines/run_baseline_experiments.sh --limit 100
#
#   # 完整测试（1000条数据）
#   bash baselines/run_baseline_experiments.sh --limit 1000
#
#   # 使用特定策略
#   bash baselines/run_baseline_experiments.sh --strategies deepinception multilingual --limit 100
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# GPU配置
GPU_ID="0"

# 模型路径（根据服务器环境调整）
QWEN3_4B_PATH="/root/autodl-tmp/models/Qwen/Qwen3-4B"
QWEN3GUARD_PATH="/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B"

# 端口配置
TARGET_PORT=8001
GUARD_PORT=8002

# vLLM参数
GPU_MEMORY_UTIL=0.45  # 各模型占用45%显存
MAX_MODEL_LEN=4096
TENSOR_PARALLEL_SIZE=1

# 数据配置
INPUT_FILE="data/dataset/processed/10k/test.jsonl"
OUTPUT_DIR="baselines/output"
STRATEGIES="deepinception multilingual pair genetic"
LIMIT=""  # 默认不限制，可通过参数设置
MAX_WORKERS=8  # 默认并发数，可通过参数设置
BATCH_SIZE=10  # ASR测试批大小

# 日志目录
LOG_DIR="baselines/logs"
mkdir -p "$LOG_DIR"

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
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash baselines/run_baseline_experiments.sh [--limit N] [--max-workers N] [--batch-size N]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================

wait_for_server() {
    local port=$1
    local max_wait=240
    local wait_time=0
    
    echo "等待vLLM server就绪 (port $port)..."
    
    while ! curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
        sleep 2
        wait_time=$((wait_time + 2))
        
        if [ $wait_time -ge $max_wait ]; then
            echo "错误: Server在${max_wait}秒内未就绪 (port $port)"
            return 1
        fi
        
        echo "  已等待 ${wait_time}s..."
    done
    
    echo "Server已就绪 (port $port)!"
    return 0
}

kill_server() {
    local port=$1
    echo "关闭vLLM server (port $port)..."
    
    # 查找占用该端口的进程
    local pid=$(lsof -ti:$port)
    if [ -n "$pid" ]; then
        kill -9 $pid
        echo "已关闭进程 $pid (port $port)"
    else
        echo "未找到占用port $port的进程"
    fi
}

# =============================================================================
# Step 1: Start vLLM Servers
# =============================================================================

echo ""
echo "============================================================================"
echo "Step 1: 启动vLLM Servers"
echo "============================================================================"
echo "GPU: $GPU_ID"
echo "Qwen3-4B: port $TARGET_PORT ($QWEN3_4B_PATH)"
echo "Qwen3Guard: port $GUARD_PORT ($QWEN3GUARD_PATH)"
echo "GPU显存占用: $GPU_MEMORY_UTIL"
echo "============================================================================"
echo ""

# 设置CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GPU_ID
export VLLM_USE_MODELSCOPE=true

# 启动Qwen3-4B（用于rewrite和target）
echo "启动Qwen3-4B (port $TARGET_PORT)..."
vllm serve "$QWEN3_4B_PATH" \
    --served-model-name "Qwen3-4B" \
    --port $TARGET_PORT \
    --host 127.0.0.1 \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --dtype auto \
    > "$LOG_DIR/qwen3_4b.log" 2>&1 &

QWEN3_PID=$!
echo "Qwen3-4B PID: $QWEN3_PID"

# 启动Qwen3Guard（用于guard）
echo "启动Qwen3Guard (port $GUARD_PORT)..."
vllm serve "$QWEN3GUARD_PATH" \
    --served-model-name "Qwen3Guard-Gen-4B" \
    --port $GUARD_PORT \
    --host 127.0.0.1 \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --dtype auto \
    > "$LOG_DIR/qwen3guard.log" 2>&1 &

GUARD_PID=$!
echo "Qwen3Guard PID: $GUARD_PID"

# =============================================================================
# Step 2: Wait for Servers Ready
# =============================================================================

echo ""
echo "============================================================================"
echo "Step 2: 等待Servers就绪"
echo "============================================================================"
echo ""

wait_for_server $TARGET_PORT
wait_for_server $GUARD_PORT

echo ""
echo "所有Servers已就绪!"
echo ""

# =============================================================================
# Step 3: Rewrite Prompts
# =============================================================================

echo ""
echo "============================================================================"
echo "Step 3: 执行Prompt重写"
echo "============================================================================"
echo "输入文件: $INPUT_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "策略: $STRATEGIES"
if [ -n "$LIMIT" ]; then
    echo "数据量限制: $LIMIT"
fi
echo "============================================================================"
echo ""

# 构建rewrite命令（使用并发版本）
REWRITE_CMD="python baselines/rewrite_prompts_server_concurrent.py \
    --input $INPUT_FILE \
    --output $OUTPUT_DIR \
    --strategies $STRATEGIES \
    --rewrite-port $TARGET_PORT \
    --guard-port $GUARD_PORT \
    --max-iterations 5 \
    --max-workers $MAX_WORKERS"

if [ -n "$LIMIT" ]; then
    REWRITE_CMD="$REWRITE_CMD --limit $LIMIT"
fi

echo "执行命令: $REWRITE_CMD"
echo ""

eval $REWRITE_CMD

echo ""
echo "Prompt重写完成!"
echo ""

# =============================================================================
# Step 4: ASR Testing
# =============================================================================

echo ""
echo "============================================================================"
echo "Step 4: 执行ASR测试"
echo "============================================================================"
echo "输出目录: $OUTPUT_DIR"
echo "策略: $STRATEGIES"
echo "============================================================================"
echo ""

# 构建ASR测试命令
# 执行ASR测试（并发版本，逐个策略）
echo "开始ASR测试..."
echo ""

for strategy in $STRATEGIES; do
    echo "测试策略: $strategy"
    
    STRATEGY_INPUT="$OUTPUT_DIR/${strategy}.jsonl"
    STRATEGY_OUTPUT="$OUTPUT_DIR/${strategy}_asr.jsonl"
    
    if [ ! -f "$STRATEGY_INPUT" ]; then
        echo "输入文件不存在: $STRATEGY_INPUT"
        continue
    fi
    
    ASR_CMD="python baselines/asr_test_server_concurrent.py \
        --input $STRATEGY_INPUT \
        --output $STRATEGY_OUTPUT \
        --target-port $TARGET_PORT \
        --guard-port $GUARD_PORT \
        --max-workers $MAX_WORKERS \
        --batch-size $BATCH_SIZE"
    
    if [ -n "$LIMIT" ]; then
        echo "(并发版本不支持limit参数，处理全部数据)"
    fi
    
    echo "执行: $ASR_CMD"
    eval $ASR_CMD
    echo ""
done

# 生成对比结果（使用原版本的--all功能汇总）
python baselines/asr_test_server.py --input $OUTPUT_DIR --output $OUTPUT_DIR --strategies $STRATEGIES --all --target-port $TARGET_PORT --guard-port $GUARD_PORT

echo ""
echo "ASR测试完成!"
echo ""

# =============================================================================
# Step 5: Stop Servers
# =============================================================================

echo ""
echo "============================================================================"
echo "Step 5: 关闭vLLM Servers"
echo "============================================================================"
echo ""

# 关闭Qwen3-4B
if [ -n "$QWEN3_PID" ]; then
    echo "关闭Qwen3-4B (PID: $QWEN3_PID)..."
    kill $QWEN3_PID
    wait $QWEN3_PID 2>/dev/null || true
    echo "Qwen3-4B已关闭"
fi

# 关闭Qwen3Guard
if [ -n "$GUARD_PID" ]; then
    echo "关闭Qwen3Guard (PID: $GUARD_PID)..."
    kill $GUARD_PID
    wait $GUARD_PID 2>/dev/null || true
    echo "Qwen3Guard已关闭"
fi

# 确保端口释放
sleep 5
kill_server $TARGET_PORT
kill_server $GUARD_PORT

echo ""
echo "============================================================================"
echo "实验完成!"
echo "============================================================================"
echo ""
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "各策略rewrite结果:"
for strategy in $STRATEGIES; do
    echo "  - $strategy: $OUTPUT_DIR/$strategy.jsonl"
done
echo ""
echo "各策略ASR测试结果:"
for strategy in $STRATEGIES; do
    echo "  - $strategy: $OUTPUT_DIR/$strategy_asr.jsonl"
done
echo ""
echo "ASR对比结果: $OUTPUT_DIR/asr_comparison.json"
echo ""
echo "日志目录: $LOG_DIR"
echo "============================================================================"
echo ""