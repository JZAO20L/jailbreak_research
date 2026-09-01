#!/bin/bash
# =============================================================================
# Layer Ablation Experiment Launcher
# =============================================================================
#
# 组件消融实验：验证 Evolution 阶段的必要性
# - 跳过 Evolution 阶段，只做 Cold Start
# - Cold Start 数据量限制为 300 条
# - 使用 Layer 1 的所有实验组合（16组）
#
# 消融点 A: skill_call_mode (single_call | every_iteration)
# 消融点 B: skill_extraction_mode (final_prompt | trajectory)
# 消融点 C: update_strategy (success_only | failure_only | both | statistical)
#
# 与 Layer 1 对比：
# - Layer 1: 200 条 cold start + 800 条 evolution
# - 组件消融: 300 条 cold start + 0 条 evolution（跳过）
#
# Usage:
#   # 完整实验（16组）
#   bash run_ablation.sh --skip_launch
#
#   # 快速验证
#   bash run_ablation.sh --skip_launch --test_limit 100 --eval_limit 20
#
#   # 断点续跑
#   bash run_ablation.sh --skip_launch --resume_from 8
#
# =============================================================================

set -e

# =============================================================================
# 配置
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

# 组件消融数据配置
COLD_START_LIMIT=200  # 与 Layer 1 保持一致（200条）
TEST_LIMIT=           # Test 数据量限制（默认使用全部）
EVAL_LIMIT=100        # 中间评估数据量
NUM_EPOCHS=1          # 进化轮数（跳过，不生效）
MAX_ITERATIONS=10     # 最大攻击迭代次数
MAX_WORKERS=32         # 轨迹级并发数
MIN_SUCCESS_RATE=0.7  # Skills 清理阈值
MAINTENANCE_INTERVAL=100  # 维护间隔步数

# 日志目录（组件消融专属）
LOG_DIR="$SCRIPT_DIR/logs"
RESULT_DIR="$SCRIPT_DIR/results"
PROCESS_LOG="$SCRIPT_DIR/process.log"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$RESULT_DIR/skills"

# PID 记录
GUARD_PID=""
TARGET_PID=""
SKIP_LAUNCH=false
RESUME_FROM=0
SINGLE_MODE=""

# =============================================================================
# 解析参数
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --test_limit)
            TEST_LIMIT="$2"
            shift 2
            ;;
        --eval_limit)
            EVAL_LIMIT="$2"
            shift 2
            ;;
        --max_workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --min_success_rate)
            MIN_SUCCESS_RATE="$2"
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
            echo "Usage: bash run_ablation.sh [--test_limit N] [--eval_limit N] [--max_workers N] [--skip_launch] [--resume_from N] [--single CALL_MODE EXTRACTION_MODE UPDATE_STRATEGY]"
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
# Step 1: 启动 vLLM Servers
# =============================================================================

echo ""
echo "============================================================================"
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
# Step 2: 运行组件消融 Grid Search
# =============================================================================

echo ""
echo "============================================================================"
echo "Step 2: 运行组件消融 Grid Search (16 组)"
echo "============================================================================"
echo "组件消融配置:"
echo "  Cold Start: ${COLD_START_LIMIT} 条（限制）"
echo "  Evolution:  跳过 (--skip_evolution)"
echo "  Test:       test_prompts.json (1000 条，可限制)"
echo ""
echo "实验参数:"
echo "  Eval Limit: $EVAL_LIMIT"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Max Workers: $MAX_WORKERS"
echo "  Min Success Rate: $MIN_SUCCESS_RATE"
echo "  Maintenance Interval: $MAINTENANCE_INTERVAL"
if [ -n "$TEST_LIMIT" ]; then
    echo "  Test Limit: $TEST_LIMIT"
fi
echo ""
echo "消融点:"
echo "  A: skill_call_mode (single_call | every_iteration)"
echo "  B: skill_extraction_mode (final_prompt | trajectory)"
echo "  C: update_strategy (success_only | failure_only | both | statistical)"
echo ""
echo "结果目录: $RESULT_DIR"
echo "============================================================================"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 构建 Python 命令
cd "$PROJECT_ROOT"

GRID_SEARCH_CMD="python self_evolve_skills_jailbreak/scripts/grid_search.py \
    --eval_limit $EVAL_LIMIT \
    --num_epochs $NUM_EPOCHS \
    --max_iterations $MAX_ITERATIONS \
    --max_workers $MAX_WORKERS \
    --min_success_rate $MIN_SUCCESS_RATE \
    --maintenance_interval $MAINTENANCE_INTERVAL \
    --output_dir $RESULT_DIR \
    --guard_port $GUARD_PORT \
    --target_port $TARGET_PORT \
    --skip_evolution \
    --cold_start_limit $COLD_START_LIMIT"

if [ -n "$TEST_LIMIT" ]; then
    GRID_SEARCH_CMD="$GRID_SEARCH_CMD --test_limit $TEST_LIMIT"
fi

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
# Step 3: 统计结果
# =============================================================================

echo ""
echo "============================================================================"
echo "Step 3: 统计结果"
echo "============================================================================"
echo ""

# 使用 layer1 的汇总脚本（结构相同）
python "$SCRIPT_DIR/../layer1/scripts/summarize_layer1.py" \
    --input "$RESULT_DIR" \
    --output "$RESULT_DIR/ablation_summary_${TIMESTAMP}.json"

# 生成 Markdown 报告
python "$SCRIPT_DIR/../layer1/scripts/generate_report.py" \
    --input "$RESULT_DIR/ablation_summary_${TIMESTAMP}.json" \
    --output "$RESULT_DIR/ablation_report_${TIMESTAMP}.md"

echo ""
echo "结果汇总:"
cat "$RESULT_DIR/ablation_summary_${TIMESTAMP}.json" | python -c "
import sys, json
d = json.load(sys.stdin)
print('Method                          ASR      Avg Iter    Skill Count')
print('-' * 60)
for k, v in d.get('results', {}).items():
    asr = v.get('asr', 0) * 100
    avg_iter = v.get('avg_iterations', 0)
    skill_count = v.get('final_skill_count', 0)
    print(f'{k:<30} {asr:>9.1f}% {avg_iter:>10.2f} {skill_count:>12}')
"

# =============================================================================
# Step 4: 关闭服务
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
echo "汇总文件: $RESULT_DIR/ablation_summary_${TIMESTAMP}.json"
echo "报告文件: $RESULT_DIR/ablation_report_${TIMESTAMP}.md"
echo "过程日志: $PROCESS_LOG"
echo "vLLM日志: $LOG_DIR"
echo ""
echo "对比说明："
echo "  - Layer 1: 200条 Cold Start + 800条 Evolution (完整流程)"
echo "  - 组件消融: 200条 Cold Start + 0条 Evolution (跳过进化)"
echo "  - 通过对比可验证 Evolution 阶段的效果贡献（控制 Cold Start 数据量相同）"
echo ""
echo "============================================================================"