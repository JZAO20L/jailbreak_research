#!/bin/bash
# Jailbreak Prompt 实验脚本 - 实验1
# 启动模型服务 → 调用Python实验脚本 → 关闭模型服务
#
# 用法:
#   bash experiments/jailbreak_prompt_exp/exp.sh              # 运行所有策略
#   bash experiments/jailbreak_prompt_exp/exp.sh --topk 5     # 只输出top5
#   bash experiments/jailbreak_prompt_exp/exp.sh s1 s2        # 运行指定策略

set -e

# =========================
# 配置区
# =========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

TEST_SET="${TEST_SET:-$BASE_DIR/../data/dataset/processed/10k/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"

POLICY_MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
TARGET_MODEL="${TARGET_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B}"

POLICY_PORT=8003
TARGET_PORT=8001
GUARD_PORT=8002

SLEEP_BETWEEN_EVALS=5

# =========================
# 策略列表
# =========================
SELECTED_STRATEGIES=()
TOPK=""
GAP_THRESHOLD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --topk)
            TOPK="$2"
            shift 2
            ;;
        --gap_threshold)
            GAP_THRESHOLD="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项] [策略1 策略2 ...]"
            echo ""
            echo "选项:"
            echo "  --topk N                只输出表现最好的N个策略"
            echo "  --gap_threshold FLOAT   差距阈值 (如 0.05 表示5%)"
            echo "  --help                  显示帮助"
            exit 0
            ;;
        *)
            SELECTED_STRATEGIES+=("$1")
            shift
            ;;
    esac
done

if [ ${#SELECTED_STRATEGIES[@]} -gt 0 ]; then
    STRATEGIES=("${SELECTED_STRATEGIES[@]}")
else
    STRATEGIES=(
        "urgent_situation" "academic_research" "creative_writing"
        "cybersecurity_defense" "historical_analysis" "legal_compliance"
        "psychological_counseling" "medical_research" "journalistic_investigation"
        "policy_analysis" "role_playing" "hypothetical_scenario"
        "technical_documentation" "reverse_engineering" "debate_preparation"
        "ethics_discussion" "risk_assessment" "training_materials"
        "data_protection" "social_engineering_defense" "comparative_analysis"
        "case_study" "system_testing" "red_teaming"
    )
fi

# =========================
# 辅助函数
# =========================
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

check_port_active() {
    local port=$1
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q "200"
}

# 等待端口就绪 (最多等待240秒)
wait_for_port() {
    local port=$1
    local name=$2
    for i in $(seq 1 120); do
        if check_port_active $port; then
            log "$name 服务就绪 (端口:$port)"
            return 0
        fi
        sleep 2
    done
    log "$name 服务启动超时!"
    return 1
}

# 关闭指定端口的服务
stop_service() {
    local port=$1
    local name=$2
    if check_port_active $port; then
        log "关闭 $name 服务 (端口:$port)..."
        curl -s -X POST "http://127.0.0.1:$port/v1/shutdown" 2>/dev/null || true
        # 等待端口释放
        for i in $(seq 1 30); do
            if ! check_port_active $port; then
                log "$name 服务已关闭"
                return 0
            fi
            sleep 1
        done
        log "警告: $name 服务可能未完全关闭"
    else
        log "$name 服务未在运行 (端口:$port)"
    fi
}

# =========================
# 实验主流程
# =========================
log "============================================================"
log "Jailbreak Prompt 实验 - 实验1"
log "============================================================"
log "测试集: $TEST_SET"
log "策略数量: ${#STRATEGIES[@]}"
log "总评估次数: $(( ${#STRATEGIES[@]} + 1 )) (含原始基线)"
log "输出目录: $OUTPUT_DIR"
if [ -n "$TOPK" ]; then log "Top-K筛选: 只保留前 $TOPK 个策略"; fi
if [ -n "$GAP_THRESHOLD" ]; then log "Gap筛选: 差距阈值 = $GAP_THRESHOLD"; fi
log "============================================================"

mkdir -p "$OUTPUT_DIR"
START_TIME=$(date +%s)

# ----------------------------------------------------------------
# Step 1: 启动三个模型服务
# ----------------------------------------------------------------
log "Step 1/3: 启动模型服务..."

log "启动Policy (GPU0:$POLICY_PORT)..."
unset OMP_NUM_THREADS
CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$POLICY_MODEL" \
    --host 127.0.0.1 --port $POLICY_PORT \
    --max-model-len 4096 --gpu-memory-utilization 0.9 \
    > "$OUTPUT_DIR/policy_vllm.log" 2>&1 &
wait_for_port $POLICY_PORT "Policy"

log "启动Target (GPU1:$TARGET_PORT)..."
unset OMP_NUM_THREADS
CUDA_VISIBLE_DEVICES=1 nohup vllm serve "$TARGET_MODEL" \
    --host 127.0.0.1 --port $TARGET_PORT \
    --max-model-len 4096 --gpu-memory-utilization 0.4 \
    > "$OUTPUT_DIR/target_vllm.log" 2>&1 &
wait_for_port $TARGET_PORT "Target"

log "启动Guard (GPU1:$GUARD_PORT)..."
unset OMP_NUM_THREADS
CUDA_VISIBLE_DEVICES=1 nohup vllm serve "$GUARD_MODEL" \
    --host 127.0.0.1 --port $GUARD_PORT \
    --max-model-len 4096 --gpu-memory-utilization 0.4 \
    > "$OUTPUT_DIR/guard_vllm.log" 2>&1 &
wait_for_port $GUARD_PORT "Guard"

log "所有模型服务已就绪!"

# ----------------------------------------------------------------
# Step 2: 运行Python实验脚本 (只连接，不启动/关闭服务)
# ----------------------------------------------------------------
log "Step 2/3: 运行实验..."

python "$SCRIPT_DIR/jailbreak_prompt_exp.py" \
    --test_set "$TEST_SET" \
    --output_dir "$OUTPUT_DIR" \
    --policy_port $POLICY_PORT \
    --target_port $TARGET_PORT \
    --guard_port $GUARD_PORT \
    --sleep_between_evals $SLEEP_BETWEEN_EVALS \
    ${TOPK:+--topk $TOPK} \
    ${GAP_THRESHOLD:+--gap_threshold $GAP_THRESHOLD} \
    ${SELECTED_STRATEGIES:+--strategies ${SELECTED_STRATEGIES[@]}}

PYTHON_EXIT=$?

# ----------------------------------------------------------------
# Step 3: 关闭所有模型服务
# ----------------------------------------------------------------
log "Step 3/3: 关闭模型服务..."

stop_service $POLICY_PORT "Policy"
stop_service $TARGET_PORT "Target"
stop_service $GUARD_PORT "Guard"

# 清理显存
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
log "显存已清理"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

log ""
log "============================================================"
log "实验完成!"
log "============================================================"
log "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
log "结果保存在: $OUTPUT_DIR"
log "Python脚本退出码: $PYTHON_EXIT"
log "============================================================"

log "\n查看结果:"
log "  cat $OUTPUT_DIR/experiment_summary.json | python -m json.tool"
log "  ls -la $OUTPUT_DIR/*/"

exit $PYTHON_EXIT
