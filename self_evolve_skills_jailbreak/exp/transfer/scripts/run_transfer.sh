#!/bin/bash
# =============================================================================
# Transfer Experiment: 单模型全矩阵测试
# =============================================================================
#
# 对单个 target 模型，测试所有方法 × 所有数据集
#
# 方法：8 种
#   - 5 baselines: no_rewrite, pair, autodan, deepinception, persona
#   - 3 our methods:
#     - pair_skills_28: Layer1演化28个Skills（有CS+Evo）
#     - autodan_skills_1: DAN单个模板（无自进化）
#     - autodan_skills_54: Layer4演化54个Skills（有CS+Evo）
#
# 数据集：5 个
#   - default (原有 test 集)
#   - advbench
#   - harmbench_contextual
#   - harmbench_standard
#   - jailbreakBench
#
# GPU 分配 (固定):
#   - GPU 0: Guard (Qwen3Guard-Gen-4B, Port 8002)
#   - GPU 1: Attacker (Qwen3-4B, Port 8003, 用于 rewrite)
#   - GPU 2: Target (可变, Port 8001)
#
# 结果目录:
#   - results/<模型名>/<数据集>/<方法>/
#
# Usage:
#   bash run_transfer.sh --target_model Qwen3-0.6B
#   bash run_transfer.sh --target_model Qwen3-14B-FP8 --skip_launch
#   bash run_transfer.sh --target_model gemma-4-12B-it --mode baselines
#   bash run_transfer.sh --target_model gpt-oss-20b --method pair --dataset advbench
#
# =============================================================================

set -e

# =============================================================================
# 清理 GPU 上的 vLLM 进程
# =============================================================================

echo "清理 GPU 上的 vLLM 进程..."

# 1. 杀掉所有 vllm 相关进程
pkill -f "vllm serve" || true
pkill -f "vllm.api_server" || true
sleep 2

# 2. 清理特定端口上的进程（确保端口释放）
for port in 8001 8002 8003; do
    pid=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        kill -9 $pid 2>/dev/null || true
        echo "  清理端口 $port: PID $pid"
    fi
done

# 3. 等待 GPU 内存释放
sleep 5

# 4. 检查 GPU 状态
echo ""
echo "GPU 状态检查:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "  (nvidia-smi 不可用)"
echo ""

echo "✓ GPU 清理完成"

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="/home/tiger/jailbreak_research"

# 固定配置：Guard 和 Attacker
GUARD_GPU="0"
GUARD_PORT=8002
GUARD_MODEL_PATH="/home/tiger/models/Qwen/Qwen3Guard-Gen-4B"
GUARD_MODEL_NAME="Qwen3Guard-Gen-4B"

ATTACKER_GPU="1"
ATTACKER_PORT=8003
ATTACKER_MODEL_PATH="/home/tiger/models/Qwen/Qwen3-4B"
ATTACKER_MODEL_NAME="Qwen3-4B"

# Target 默认配置
TARGET_GPU="2,3"  # 使用两张 GPU 加速
TARGET_TP=2       # Target tensor_parallel_size
TARGET_PORT=8001

# 模型家族映射（模型名 -> 路径）
declare -A MODEL_PATHS
MODEL_PATHS["Qwen3-4B"]="/home/tiger/models/Qwen/Qwen3-4B"
MODEL_PATHS["Qwen3-0.6B"]="/home/tiger/models/Qwen/Qwen3-0.6B"
MODEL_PATHS["Qwen3-14B-FP8"]="/home/tiger/models/Qwen/Qwen3-14B-FP8"
MODEL_PATHS["gemma-4-12B-it"]="/home/tiger/models/google/gemma-4-12B-it"
MODEL_PATHS["gpt-oss-20b"]="/home/tiger/models/openai-mirror/gpt-oss-20b"

# 数据集映射（数据集名 -> 路径）
declare -A DATASET_PATHS
DATASET_PATHS["default"]="$PROJECT_ROOT/self_evolve_skills_jailbreak/data/test_prompts.json"
DATASET_PATHS["advbench"]="$PROJECT_ROOT/data/benchmark/advbench.jsonl"
DATASET_PATHS["harmbench_contextual"]="$PROJECT_ROOT/data/benchmark/harmbench_contextual.jsonl"
DATASET_PATHS["harmbench_standard"]="$PROJECT_ROOT/data/benchmark/harmbench_standard.jsonl"
DATASET_PATHS["jailbreakBench"]="$PROJECT_ROOT/data/benchmark/jailbreakBench.jsonl"

# 数据集列表（固定顺序）
DATASETS=("default" "advbench" "harmbench_contextual" "harmbench_standard" "jailbreakBench")

# 方法列表
BASELINE_METHODS=("no_rewrite" "pair" "autodan" "deepinception" "persona")  # 添加 no_rewrite 作为基准
OUR_METHODS=("pair_skills_28" "autodan_skills_1" "autodan_skills_54")  # pair用28个, autodan用单个/54个演化
ALL_METHODS=("no_rewrite" "pair" "autodan" "deepinception" "persona" "pair_skills_28" "autodan_skills_1" "autodan_skills_54")

# 实验参数
MAX_ITERATIONS=10
MAX_WORKERS=64
NUM_EPOCHS=1

# vLLM 参数
MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.9

# 结果和日志目录
RESULT_DIR="$SCRIPT_DIR/../results"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

# 解析参数
TARGET_MODEL=""
RUN_MODE="all"         # all, baselines, ours
RUN_METHOD=""           # 指定单个方法
RUN_DATASET=""          # 指定单个数据集
SKIP_LAUNCH=false
SKIP_EXISTING=false     # 跳过已存在的结果

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --target_model)
            TARGET_MODEL="$2"
            shift 2
            ;;
        --mode)
            RUN_MODE="$2"
            shift 2
            ;;
        --method)
            RUN_METHOD="$2"
            shift 2
            ;;
        --dataset)
            RUN_DATASET="$2"
            shift 2
            ;;
        --skip_launch)
            SKIP_LAUNCH=true
            shift
            ;;
        --skip_existing)
            SKIP_EXISTING=true
            shift
            ;;
        --target_gpu)
            TARGET_GPU="$2"
            shift 2
            ;;
        --target_port)
            TARGET_PORT="$2"
            shift 2
            ;;
        --max_workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash run_transfer.sh [options]"
            echo ""
            echo "Options:"
            echo "  --target_model <name>     Target model name"
            echo "                            Available: ${!MODEL_PATHS[@]}"
            echo "  --mode <mode>             Experiment mode: all, baselines, ours"
            echo "  --method <method>         Run single method only"
            echo "                            Available: ${ALL_METHODS[@]}"
            echo "  --dataset <dataset>       Run single dataset only"
            echo "                            Available: ${DATASETS[@]}"
            echo "  --skip_launch             Use existing servers"
            echo "  --skip_existing           Skip if results already exist"
            echo "  --max_workers <n>         Concurrent workers (default: 64)"
            echo ""
            echo "Example:"
            echo "  # Full test for a model"
            echo "  bash run_transfer.sh --target_model Qwen3-0.6B"
            echo ""
            echo "  # Test only baselines"
            echo "  bash run_transfer.sh --target_model Qwen3-14B-FP8 --mode baselines"
            echo ""
            echo "  # Test single method on single dataset"
            echo "  bash run_transfer.sh --target_model gemma-4-12B-it --method pair --dataset advbench"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate Target Model
# =============================================================================

if [ -z "$TARGET_MODEL" ]; then
    echo "Error: --target_model is required"
    echo "Available models: ${!MODEL_PATHS[@]}"
    exit 1
fi

if [[ -v MODEL_PATHS[$TARGET_MODEL] ]]; then
    TARGET_MODEL_PATH="${MODEL_PATHS[$TARGET_MODEL]}"
else
    echo "Error: Unknown model '$TARGET_MODEL'"
    echo "Available models: ${!MODEL_PATHS[@]}"
    exit 1
fi

# 检查模型路径是否存在
if [ ! -d "$TARGET_MODEL_PATH" ]; then
    echo "Warning: Model path not found: $TARGET_MODEL_PATH"
    echo "vLLM will attempt to download from ModelScope/HuggingFace"
fi

# 模型结果目录（每个模型独立）
MODEL_RESULT_DIR="$RESULT_DIR/$TARGET_MODEL"
mkdir -p "$MODEL_RESULT_DIR"

# =============================================================================
# Print Experiment Configuration
# =============================================================================

echo ""
echo "============================================================================"
echo "Transfer Experiment: $TARGET_MODEL"
echo "============================================================================"
echo ""
echo "GPU Configuration:"
echo "  GPU 0 (Guard):    $GUARD_MODEL_NAME (Port $GUARD_PORT)"
echo "  GPU 1 (Attacker): $ATTACKER_MODEL_NAME (Port $ATTACKER_PORT)"
echo "  GPU 2,3 (Target):  $TARGET_MODEL (Port $TARGET_PORT, TP=$TARGET_TP)"
echo ""
echo "Model Path: $TARGET_MODEL_PATH"
echo "Result Dir: $MODEL_RESULT_DIR"
echo ""
echo "Datasets: ${DATASETS[@]}"
echo "Methods (baseline): ${BASELINE_METHODS[@]}"
echo "Methods (ours): ${OUR_METHODS[@]}"
echo ""
echo "Run Mode: $RUN_MODE"
echo "Skip Launch: $SKIP_LAUNCH"
echo "Skip Existing: $SKIP_EXISTING"
echo "============================================================================"

# =============================================================================
# Helper Functions
# =============================================================================

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=600  # 增加到 10 分钟，模型加载需要时间
    local wait_time=0

    echo "Waiting for $name (port $port)..."

    while true; do
        # 尝试多种健康检查 URL
        if curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo "✓ $name ready (port $port)!"
            return 0
        fi
        if curl -s "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
            echo "✓ $name ready (port $port)!"
            return 0
        fi
        if curl -s "http://127.0.0.1:$port/ping" > /dev/null 2>&1; then
            echo "✓ $name ready (port $port)!"
            return 0
        fi

        sleep 5
        wait_time=$((wait_time + 5))

        if [ $wait_time -ge $max_wait ]; then
            echo "Error: $name not ready after ${max_wait}s"
            # 打印调试信息
            echo "Debug: checking port $port..."
            lsof -i :$port 2>/dev/null || echo "  No process on port $port"
            curl -v "http://127.0.0.1:$port/health" 2>&1 | tail -5
            return 1
        fi

        echo "  waited ${wait_time}s..."
    done
}

kill_server() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        kill -9 $pid 2>/dev/null
        echo "Killed server on port $port (PID: $pid)"
    fi
}

check_server_running() {
    curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1
}

check_result_exists() {
    local method=$1
    local dataset=$2
    local method_dir="$MODEL_RESULT_DIR/$dataset/$method"

    # 检查目录是否存在且有结果文件
    if [ ! -d "$method_dir" ]; then
        return 1  # 目录不存在，需要运行
    fi

    # 检查是否有结果文件
    # baselines 输出 summary.json，pipeline 输出 result_*.json
    if [ -f "$method_dir/summary.json" ]; then
        return 0  # baselines 结果存在
    fi

    # 检查 result_*.json 文件（pipeline 输出）
    local result_files=$(find "$method_dir" -maxdepth 1 -name "result_*.json" -type f 2>/dev/null)
    if [ -n "$result_files" ]; then
        return 0  # pipeline 结果存在
    fi

    return 1  # 没有结果文件，需要运行
}

# =============================================================================
# Step 1: Start vLLM Servers
# =============================================================================

GUARD_PID=""
ATTACKER_PID=""
TARGET_PID=""

if [ "$SKIP_LAUNCH" = false ]; then
    echo ""
    echo "Step 1: Starting vLLM Servers"
    echo "============================================================================"

    export VLLM_USE_MODELSCOPE=true
    export FLASHINFER_DISABLE_VERSION_CHECK=1

    # Guard (GPU 0)
    echo "Starting Guard..."
    CUDA_VISIBLE_DEVICES=$GUARD_GPU vllm serve "$GUARD_MODEL_PATH" \
        --served-model-name "$GUARD_MODEL_NAME" \
        --port $GUARD_PORT \
        --host 0.0.0.0 \
        --max-model-len 2048 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization $GPU_MEMORY_UTIL \
        --dtype auto \
        > "$LOG_DIR/${TARGET_MODEL}_guard.log" 2>&1 &
    GUARD_PID=$!

    # Attacker (GPU 1)
    echo "Starting Attacker..."
    CUDA_VISIBLE_DEVICES=$ATTACKER_GPU vllm serve "$ATTACKER_MODEL_PATH" \
        --served-model-name "$ATTACKER_MODEL_NAME" \
        --port $ATTACKER_PORT \
        --host 0.0.0.0 \
        --max-model-len $MAX_MODEL_LEN \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization $GPU_MEMORY_UTIL \
        --dtype auto \
        > "$LOG_DIR/${TARGET_MODEL}_attacker.log" 2>&1 &
    ATTACKER_PID=$!

    # Target (GPU 2,3, TP=2)
    echo "Starting Target: $TARGET_MODEL (TP=$TARGET_TP)..."
    CUDA_VISIBLE_DEVICES=$TARGET_GPU vllm serve "$TARGET_MODEL_PATH" \
        --served-model-name "$TARGET_MODEL" \
        --port $TARGET_PORT \
        --host 0.0.0.0 \
        --max-model-len $MAX_MODEL_LEN \
        --tensor-parallel-size $TARGET_TP \
        --gpu-memory-utilization $GPU_MEMORY_UTIL \
        --dtype auto \
        > "$LOG_DIR/${TARGET_MODEL}_target.log" 2>&1 &
    TARGET_PID=$!

    # Wait for servers
    wait_for_server $GUARD_PORT "Guard"
    wait_for_server $ATTACKER_PORT "Attacker"
    wait_for_server $TARGET_PORT "Target"
    echo "All servers ready!"
else
    echo ""
    echo "Skipping server launch (--skip_launch)"
    for port in $GUARD_PORT $ATTACKER_PORT $TARGET_PORT; do
        if curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo "✓ Server running on port $port"
        else
            echo "✗ No server on port $port"
            exit 1
        fi
    done
fi

# =============================================================================
# Step 2: Run Experiments
# =============================================================================

cd "$PROJECT_ROOT"

run_baseline() {
    local method=$1
    local dataset=$2
    local output_dir="$MODEL_RESULT_DIR/$dataset/$method"
    local log_file="$LOG_DIR/${TARGET_MODEL}_${dataset}_${method}.log"
    local test_data="${DATASET_PATHS[$dataset]}"

    # Check skip
    if [ "$SKIP_EXISTING" = true ] && check_result_exists "$method" "$dataset"; then
        echo "  [SKIP] $method/$dataset (results exist)"
        return 0
    fi

    echo ""
    echo "Running: $method on $dataset"
    echo "  Test data: $test_data"
    echo "  Output: $output_dir"
    mkdir -p "$output_dir"

    python -m baselines.cli batch \
        --input "$test_data" \
        --strategy "$method" \
        --output "$output_dir" \
        --target-model "$TARGET_MODEL_PATH" \
        --target-port "$TARGET_PORT" \
        --guard-model "$GUARD_MODEL_PATH" \
        --guard-port "$GUARD_PORT" \
        --max_workers "$MAX_WORKERS" \
        --evaluate \
        2>&1 | tee "$log_file"
}

run_our_method() {
    local method=$1
    local dataset=$2
    local output_dir="$MODEL_RESULT_DIR/$dataset/$method"
    local log_file="$LOG_DIR/${TARGET_MODEL}_${dataset}_${method}.log"
    local test_data="${DATASET_PATHS[$dataset]}"

    # Check skip
    if [ "$SKIP_EXISTING" = true ] && check_result_exists "$method" "$dataset"; then
        echo "  [SKIP] $method/$dataset (results exist)"
        return 0
    fi

    # Parse method config
    # 注意：使用精选的单个最佳 skill，不使用整个 skills 库
    local skill_call_mode=""
    local skill_source=""
    local skill_library_path=""

    case $method in
        "pair_skills_28")
            skill_call_mode="single_call"
            skill_source="default"
            # 使用 Layer 1 演化后的 28 个 skills
            skill_library_path="$PROJECT_ROOT/self_evolve_skills_jailbreak/exp/layer1/results/skills/skills_single_call_trajectory_statistical.json"
            ;;
        "autodan_skills_1")
            skill_call_mode="every_iteration"
            skill_source="dan_templates"
            # 使用单个最佳 DAN skill（无自进化）
            skill_library_path="$EXP_DIR/skills/transfer_autodan_skill.json"
            ;;
        "autodan_skills_54")
            skill_call_mode="every_iteration"
            skill_source="default"  # 关键：使用 default 而不是 dan_templates，避免清空 skills
            # 使用 Layer 4 演化后的 54 个 skills（有 Cold Start + Evolution）
            skill_library_path="$PROJECT_ROOT/self_evolve_skills_jailbreak/exp/layer4/results/skills/skills_dan_data_medium_evo.json"
            ;;
    esac

    echo ""
    echo "Running: $method on $dataset"
    echo "  Test data: $test_data"
    echo "  skill_call_mode: $skill_call_mode"
    echo "  skill_source: $skill_source"
    echo "  Mode: test only (using existing skills, no cold start/evolution)"
    echo "  Output: $output_dir"
    mkdir -p "$output_dir"

    # Build command - 跳过 cold start 和 evolution，直接使用已有 skills 进行测试
    local cmd="python self_evolve_skills_jailbreak/scripts/pipeline.py \
        --target_model_path $TARGET_MODEL_PATH \
        --target_model_name $TARGET_MODEL \
        --target_port $TARGET_PORT \
        --attacker_port $ATTACKER_PORT \
        --guard_port $GUARD_PORT \
        --skill_call_mode $skill_call_mode \
        --skill_source $skill_source \
        --skill_extraction_mode trajectory \
        --max_iterations $MAX_ITERATIONS \
        --max_workers $MAX_WORKERS \
        --output_dir $output_dir \
        --exp_name ${TARGET_MODEL}_${dataset}_${method} \
        --test_data_path $test_data \
        --skip_cold_start \
        --skip_evolution \
        --skip_launch"

    # 对于 pair_skills_28，使用 Layer 1 演化后的 skills 文件
    if [ -n "$skill_library_path" ] && [ -f "$skill_library_path" ]; then
        cmd="$cmd --skill_library_path $skill_library_path"
    fi

    eval $cmd 2>&1 | tee "$log_file"
}

# =============================================================================
# Execute Experiments
# =============================================================================

echo ""
echo "Step 2: Running Experiments"
echo "============================================================================"
echo ""

# Determine which datasets to run
DATASETS_TO_RUN=()
if [ -n "$RUN_DATASET" ]; then
    DATASETS_TO_RUN=("$RUN_DATASET")
else
    DATASETS_TO_RUN=("${DATASETS[@]}")
fi

# Determine which methods to run
METHODS_TO_RUN=()
if [ -n "$RUN_METHOD" ]; then
    METHODS_TO_RUN=("$RUN_METHOD")
elif [ "$RUN_MODE" = "baselines" ]; then
    METHODS_TO_RUN=("${BASELINE_METHODS[@]}")
elif [ "$RUN_MODE" = "ours" ]; then
    METHODS_TO_RUN=("${OUR_METHODS[@]}")
else
    METHODS_TO_RUN=("${ALL_METHODS[@]}")
fi

TOTAL_EXPS=$((${#DATASETS_TO_RUN[@]} * ${#METHODS_TO_RUN[@]}))
CURRENT_EXP=0

echo "Plan: ${#METHODS_TO_RUN[@]} methods × ${#DATASETS_TO_RUN[@]} datasets = $TOTAL_EXPS experiments"
echo ""

for dataset in "${DATASETS_TO_RUN[@]}"; do
    # Check dataset exists
    if [[ ! -v DATASET_PATHS[$dataset] ]]; then
        echo "Warning: Unknown dataset '$dataset', skipping"
        continue
    fi

    dataset_path="${DATASET_PATHS[$dataset]}"
    if [ ! -f "$dataset_path" ]; then
        echo "Warning: Dataset file not found: $dataset_path, skipping"
        continue
    fi

    echo "=== Dataset: $dataset ==="
    mkdir -p "$MODEL_RESULT_DIR/$dataset"

    for method in "${METHODS_TO_RUN[@]}"; do
        CURRENT_EXP=$((CURRENT_EXP + 1))
        echo "[$CURRENT_EXP/$TOTAL_EXPS]"

        if [[ " ${BASELINE_METHODS[@]} " =~ " $method " ]]; then
            run_baseline "$method" "$dataset"
        elif [[ " ${OUR_METHODS[@]} " =~ " $method " ]]; then
            run_our_method "$method" "$dataset"
        else
            echo "Warning: Unknown method '$method', skipping"
        fi
    done
done

# =============================================================================
# Step 3: Summary
# =============================================================================

echo ""
echo "============================================================================"
echo "Step 3: Generating Summary"
echo "============================================================================"
echo ""

python "$SCRIPT_DIR/summarize.py" \
    --model "$TARGET_MODEL" \
    --all_datasets \
    --result_dir "$RESULT_DIR"

echo ""
echo "Summary files:"
echo "  $MODEL_RESULT_DIR/ALL_DATASETS_SUMMARY.json"

# =============================================================================
# Step 4: Stop Servers
# =============================================================================

if [ "$SKIP_LAUNCH" = false ]; then
    echo ""
    echo "============================================================================"
    echo "Step 4: Stopping Servers"
    echo "============================================================================"

    for pid in $GUARD_PID $ATTACKER_PID $TARGET_PID; do
        if [ -n "$pid" ]; then
            kill $pid 2>/dev/null || true
        fi
    done
    sleep 3
    kill_server $GUARD_PORT
    kill_server $ATTACKER_PORT
    kill_server $TARGET_PORT
fi

echo ""
echo "============================================================================"
echo "Experiment Complete: $TARGET_MODEL"
echo "============================================================================"
echo ""
echo "Results saved in: $MODEL_RESULT_DIR"
echo ""
echo "Directory structure:"
find "$MODEL_RESULT_DIR" -type d | head -20
echo ""
echo "View summary:"
echo "  cat $MODEL_RESULT_DIR/ALL_DATASETS_SUMMARY.json"
echo ""
echo "============================================================================"