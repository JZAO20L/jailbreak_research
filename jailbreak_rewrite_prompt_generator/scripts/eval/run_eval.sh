#!/bin/bash
# =============================================================================
# ASR 测试 - 完整流程
# =============================================================================
#
# 评估 Rewrite Prompt Generator 的性能
#
# 流程：
#   1. 清理 GPU 进程
#   2. 启动服务（Generator, Executor, Target, Guard）
#   3. 运行 ASR 测试
#   4. 关闭服务
#
# GPU 配置（4 GPU）：
#   GPU 0: Generator (port 8000) - 待评测模型
#   GPU 1: Executor (port 8001) - Qwen3-4B
#   GPU 2: Target (port 8002) - 目标模型
#   GPU 3: Guard (port 8003) - Qwen3Guard
#
# Usage:
#   bash run_eval.sh --generator_model base --max_samples 100
#   bash run_eval.sh --generator_model sft --generator_path /path/to/sft/model
#   bash run_eval.sh --generator_model rl --generator_path /path/to/rl/model
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# =============================================================================
# 默认配置
# =============================================================================

# GPU 分配
GENERATOR_GPU=0
EXECUTOR_GPU=1
TARGET_GPU=2
GUARD_GPU=3

# 服务端口
GENERATOR_PORT=8000
EXECUTOR_PORT=8001
TARGET_PORT=8002
GUARD_PORT=8003

# 默认模型路径
GENERATOR_MODEL_PATH="/home/tiger/models/Qwen/Qwen3-4B"
EXECUTOR_MODEL_PATH="/home/tiger/models/Qwen/Qwen3-4B"
TARGET_MODEL_PATH="/home/tiger/models/Qwen/Qwen3-4B"
GUARD_MODEL_PATH="/home/tiger/models/Qwen/Qwen3Guard-Gen-4B"

# vLLM 参数
MAX_MODEL_LEN=4096
GPU_MEMORY_UTIL=0.85

# 测试参数
GENERATOR_MODEL="base"
LORA_PATH=""  # LoRA adapter 路径（可选）
SEED_DATASET="$PROJECT_DIR/../data/dataset/processed/10k/test.jsonl"
OUTPUT_DIR="$PROJECT_DIR/exp/results/eval"
MAX_SAMPLES=0  # 0 表示使用全量数据
BATCH_SIZE=16
SAVE_INTERMEDIATE=false

# 日志目录
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# =============================================================================
# 解析参数
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --generator_model)
            GENERATOR_MODEL="$2"
            shift 2
            ;;
        --generator_path)
            GENERATOR_MODEL_PATH="$2"
            shift 2
            ;;
        --lora_path)
            LORA_PATH="$2"
            shift 2
            ;;
        --target_model)
            TARGET_MODEL_PATH="$2"
            shift 2
            ;;
        --seed_dataset)
            SEED_DATASET="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --save_intermediate)
            SAVE_INTERMEDIATE=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# 根据模型类型调整输出目录
OUTPUT_DIR="$OUTPUT_DIR/$GENERATOR_MODEL"

# LoRA 名称（用于 API 请求）
LORA_NAME=""
if [ -n "$LORA_PATH" ] && [ -d "$LORA_PATH" ]; then
    LORA_NAME="sft-lora"
fi

# =============================================================================
# Helper Functions
# =============================================================================

# GPU 内存检查所需的最小空闲内存（MiB）
MIN_FREE_MEMORY=60000  # 约 60GB，足够加载 Qwen3-4B
MAX_GPU_WAIT=60  # 最大等待时间（秒）

check_gpu_free_memory() {
    echo "检查 GPU 空闲内存..."

    all_ready=true
    for gpu in $GENERATOR_GPU $EXECUTOR_GPU $TARGET_GPU $GUARD_GPU; do
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu 2>/dev/null | tr -d ' ')
        if [ -z "$free_mem" ]; then
            free_mem=0
        fi

        if [ "$free_mem" -lt "$MIN_FREE_MEMORY" ]; then
            echo "  GPU $gpu: 空闲 ${free_mem} MiB < 需要 ${MIN_FREE_MEMORY} MiB ⚠️"
            all_ready=false
        else
            echo "  GPU $gpu: 空闲 ${free_mem} MiB ✓"
        fi
    done

    if [ "$all_ready" = false ]; then
        echo ""
        echo "警告: GPU 内存不足，可能无法启动 vLLM 服务"
        echo "建议: 等待其他进程释放 GPU 内存，或手动清理"
        return 1
    fi

    echo "✓ GPU 内存充足"
    return 0
}

wait_for_gpu_free() {
    local max_wait=$MAX_GPU_WAIT
    local wait_time=0

    echo "等待 GPU 空闲（最多 ${max_wait} 秒）..."

    while [ $wait_time -lt $max_wait ]; do
        if check_gpu_free_memory > /dev/null 2>&1; then
            echo "✓ GPU 已空闲，可以启动服务"
            return 0
        fi

        sleep 5
        wait_time=$((wait_time + 5))
        echo "  已等待 ${wait_time}s..."
    done

    echo "错误: GPU 在 ${max_wait} 秒内未空闲"
    return 1
}

kill_gpu_processes() {
    echo "清理 GPU 进程..."

    # 方式1: 通过端口清理（更可靠）
    for port in $GENERATOR_PORT $EXECUTOR_PORT $TARGET_PORT $GUARD_PORT; do
        pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill -9 $pid 2>/dev/null || true
            echo "  Port $port: 已杀死进程 $pid"
        fi
    done

    # 方式2: 通过 GPU ID 清理（兜底）
    for gpu in $GENERATOR_GPU $EXECUTOR_GPU $TARGET_GPU $GUARD_GPU; do
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $gpu 2>/dev/null || true)
        for pid in $pids; do
            kill -9 $pid 2>/dev/null || true
            echo "  GPU $gpu: 已杀死 $pid"
        done
    done

    # 等待进程完全退出
    sleep 5
    echo "✓ GPU 进程清理完成"
}

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=300
    local wait_time=0

    echo "等待 $name vLLM server 就绪 (port $port)..."

    # 尝试多个端点检查服务状态
    while true; do
        # 尝试 /health 端点
        if curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            break
        fi

        # 尝试 /v1/models 端点
        if curl -s "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
            break
        fi

        # 尝试 /ping 端点
        if curl -s "http://127.0.0.1:$port/ping" > /dev/null 2>&1; then
            break
        fi

        sleep 3
        wait_time=$((wait_time + 3))

        if [ $wait_time -ge $max_wait ]; then
            echo "错误: $name Server 在 ${max_wait} 秒内未就绪 (port $port)"
            return 1
        fi

        echo "  已等待 ${wait_time}s..."
    done

    echo "✓ $name Server 已就绪 (port $port)!"

    # 额外等待，确保服务完全稳定
    sleep 10
}

# =============================================================================
# 主流程
# =============================================================================

echo ""
echo "============================================================================"
echo "ASR 测试 - $GENERATOR_MODEL Generator"
echo "============================================================================"
echo ""
echo "GPU 配置:"
echo "  GPU $GENERATOR_GPU: Generator (port=$GENERATOR_PORT)"
echo "  GPU $EXECUTOR_GPU: Executor (port=$EXECUTOR_PORT)"
echo "  GPU $TARGET_GPU: Target (port=$TARGET_PORT)"
echo "  GPU $GUARD_GPU: Guard (port=$GUARD_PORT)"
echo ""
echo "模型:"
echo "  Generator: $GENERATOR_MODEL_PATH"
echo "  Executor: $EXECUTOR_MODEL_PATH"
echo "  Target: $TARGET_MODEL_PATH"
echo "  Guard: $GUARD_MODEL_PATH"
echo ""
echo "测试:"
echo "  数据: $SEED_DATASET"
echo "  样本数: $MAX_SAMPLES"
echo "  并发: $BATCH_SIZE"
echo "  输出: $OUTPUT_DIR"
echo "============================================================================"
echo ""

# Step 1: 清理 GPU
echo ""
echo "Step 1: 清理 GPU"
echo "============================================================================"
echo ""
kill_gpu_processes

# Step 1.5: 检查 GPU 空闲内存
echo ""
echo "Step 1.5: 检查 GPU 空闲内存"
echo "============================================================================"
echo ""
if ! check_gpu_free_memory; then
    echo ""
    echo "GPU 内存不足，尝试等待其他进程释放..."
    if ! wait_for_gpu_free; then
        echo ""
        echo "无法启动评测：GPU 内存不足"
        echo "请手动清理占用 GPU 的进程后重试"
        exit 1
    fi
fi

# Step 2: 启动服务
echo ""
echo "Step 2: 启动服务"
echo "============================================================================"
echo ""

export VLLM_USE_MODELSCOPE=true
export FLASHINFER_DISABLE_VERSION_CHECK=1

echo "启动 Generator (GPU $GENERATOR_GPU)..."

# 构建 LoRA 参数
LORA_ARGS=""
if [ -n "$LORA_PATH" ] && [ -d "$LORA_PATH" ]; then
    echo "  启用 LoRA: $LORA_PATH"
    LORA_ARGS="--enable-lora --lora-modules sft-lora=$LORA_PATH"
fi

CUDA_VISIBLE_DEVICES=$GENERATOR_GPU vllm serve "$GENERATOR_MODEL_PATH" \
    --served-model-name "generator" \
    --port $GENERATOR_PORT \
    --host 0.0.0.0 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --dtype auto \
    $LORA_ARGS \
    > "$LOG_DIR/generator.log" 2>&1 &
GENERATOR_PID=$!

echo "启动 Executor (GPU $EXECUTOR_GPU)..."
CUDA_VISIBLE_DEVICES=$EXECUTOR_GPU vllm serve "$EXECUTOR_MODEL_PATH" \
    --served-model-name "executor" \
    --port $EXECUTOR_PORT \
    --host 0.0.0.0 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --dtype auto \
    > "$LOG_DIR/executor.log" 2>&1 &
EXECUTOR_PID=$!

echo "启动 Target (GPU $TARGET_GPU)..."
CUDA_VISIBLE_DEVICES=$TARGET_GPU vllm serve "$TARGET_MODEL_PATH" \
    --served-model-name "target" \
    --port $TARGET_PORT \
    --host 0.0.0.0 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --dtype auto \
    > "$LOG_DIR/target.log" 2>&1 &
TARGET_PID=$!

echo "启动 Guard (GPU $GUARD_GPU)..."
CUDA_VISIBLE_DEVICES=$GUARD_GPU vllm serve "$GUARD_MODEL_PATH" \
    --served-model-name "guard" \
    --port $GUARD_PORT \
    --host 0.0.0.0 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --dtype auto \
    > "$LOG_DIR/guard.log" 2>&1 &
GUARD_PID=$!

echo ""
wait_for_server $GENERATOR_PORT "Generator"
wait_for_server $EXECUTOR_PORT "Executor"
wait_for_server $TARGET_PORT "Target"
wait_for_server $GUARD_PORT "Guard"

echo "所有服务就绪!"

# Step 3: 运行测试
echo ""
echo "Step 3: 运行 ASR 测试"
echo "============================================================================"
echo ""

SAVE_FLAG=""
if [ "$SAVE_INTERMEDIATE" = true ]; then
    SAVE_FLAG="--save_intermediate"
fi

LORA_FLAG=""
if [ -n "$LORA_NAME" ]; then
    LORA_FLAG="--generator_lora_name $LORA_NAME"
fi

python "$SCRIPT_DIR/evaluate.py" \
    --seed_dataset "$SEED_DATASET" \
    --generator_port $GENERATOR_PORT \
    --executor_port $EXECUTOR_PORT \
    --target_port $TARGET_PORT \
    --guard_port $GUARD_PORT \
    --max_samples $MAX_SAMPLES \
    --batch_size $BATCH_SIZE \
    --output_dir "$OUTPUT_DIR" \
    $SAVE_FLAG \
    $LORA_FLAG

# Step 4: 关闭服务
echo ""
echo "Step 4: 关闭服务"
echo "============================================================================"
echo ""

for pid in $GENERATOR_PID $EXECUTOR_PID $TARGET_PID $GUARD_PID; do
    kill $pid 2>/dev/null || true
done

sleep 3

echo "服务已关闭"

# =============================================================================
# 完成
# =============================================================================

echo ""
echo "============================================================================"
echo "ASR 测试完成"
echo "============================================================================"
echo ""
echo "结果目录: $OUTPUT_DIR"
echo ""
echo "查看统计:"
echo "  cat $OUTPUT_DIR/test_stats.json"
echo ""
echo "查看报告:"
echo "  cat $OUTPUT_DIR/asr_report.md"
echo ""
echo "============================================================================"