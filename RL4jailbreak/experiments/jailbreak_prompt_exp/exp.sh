# =============================================================================
# Jailbreak Prompt 实验脚本 - 实验1 (重做版本)
#
# 根据 TODO.md "实验重做" 部分：
# GPU配置: policy&target共用GPU0&1(Qwen3-4B tensor parallel), guard用GPU2&3(tensor parallel)
# - 模型路径: models
# - 上下文长度: policy 4k, target&guard 8k
# - 新增: qwen3-max对比实验
#
# 用法:
#   bash experiments/jailbreak_prompt_exp/exp.sh              # 运行所有策略
#   bash experiments/jailbreak_prompt_exp/exp.sh --topk 3     # 只输出top3
#   bash experiments/jailbreak_prompt_exp/exp.sh --qwen3_max  # 包含qwen3-max对比
#   bash experiments/jailbreak_prompt_exp/exp.sh s1 s2        # 运行指定策略
# =============================================================================

set -e

# =========================
# 配置区
# =========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 测试集
TEST_SET="${TEST_SET:-$BASE_DIR/../data/dataset/processed/10k/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"

# 模型路径 (根据TODO.md "实验重做") - policy和target共用Qwen3-4B
POLICY_MODEL="${POLICY_MODEL:-/home/tiger/models/Qwen/Qwen3-4B}"
TARGET_MODEL="${TARGET_MODEL:-/home/tiger/models/Qwen/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/home/tiger/models/Qwen/Qwen3Guard-Gen-4B}"

# 端口配置 - policy和target共用GPU0&1上的同一个Qwen3-4B服务
POLICY_PORT=8003
TARGET_PORT=8003  # 与policy共用同一个服务
GUARD_PORT=8002   # GPU2&3上的guard

# 上下文长度 (根据TODO.md - policy和target共用，统一使用8k)
POLICY_MAX_MODEL_LEN=8192
TARGET_MAX_MODEL_LEN=8192
GUARD_MAX_MODEL_LEN=8192

# GPU显存利用率
POLICY_GPU_UTIL=0.9
GUARD_GPU_UTIL=0.9

# qwen3-max API配置 (根据TODO.md)
QWEN3_MAX_API_KEY="sk-sp-eb50d67ca64a451b820cc4ab87ef8e6c"
QWEN3_MAX_BASE_URL="https://coding.dashscope.aliyuncs.com/v1"
QWEN3_MAX_MODEL="qwen3-max-2026-01-23"

# 其他配置
SLEEP_BETWEEN_EVALS=5

# =========================
# 参数解析
# =========================
SELECTED_STRATEGIES=()
TOPK=""
GAP_THRESHOLD=""
INCLUDE_QWEN3_MAX=false
RESET_CKPT=false

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
        --qwen3_max)
            INCLUDE_QWEN3_MAX=true
            shift
            ;;
        --reset)
            RESET_CKPT=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项] [策略1 策略2 ...]"
            echo ""
            echo "选项:"
            echo "  --topk N                只输出表现最好的N个策略"
            echo "  --gap_threshold FLOAT   差距阈值 (如 0.05 表示5%)"
            echo "  --qwen3_max             包含qwen3-max对比实验"
            echo "  --reset                 清空checkpoint重头开始"
            echo "  --help                  显示帮助"
            exit 0
            ;;
        *)
            SELECTED_STRATEGIES+=("$1")
            shift
            ;;
    esac
done

# =========================
# 策略列表
# =========================
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

wait_for_port() {
    local port=$1
    local name=$2
    local timeout=120
    
    log "等待 $name 服务启动 (端口:$port)..."
    for i in $(seq 1 $timeout); do
        if check_port_active $port; then
            log "$name 服务就绪!"
            return 0
        fi
        sleep 2
    done
    log "ERROR: $name 服务启动超时!"
    return 1
}

stop_service() {
    local port=$1
    local name=$2
    
    if check_port_active $port; then
        log "关闭 $name 服务 (端口:$port)..."
        # 尝试优雅关闭
        curl -s -X POST "http://127.0.0.1:$port/v1/shutdown" 2>/dev/null || true
        # 强制终止进程
        pkill -f "vllm.*$port" 2>/dev/null || true
        sleep 3
        
        if check_port_active $port; then
            log "警告: $name 可能未完全关闭"
        else
            log "$name 已关闭"
        fi
    else
        log "$name 未运行"
    fi
}

# =========================
# Checkpoint管理
# =========================
CKPT_FILE="$OUTPUT_DIR/exp_checkpoint.json"

load_checkpoint() {
    if [ -f "$CKPT_FILE" ] && [ "$RESET_CKPT" = false ]; then
        python3 -c "
import json
with open('$CKPT_FILE') as f:
    ckpt = json.load(f)
print(' '.join(ckpt.get('completed', [])))
" 2>/dev/null || echo ""
    else
        echo ""
    fi
}

save_checkpoint() {
    local completed_list="$1"
    mkdir -p "$OUTPUT_DIR"
    python3 -c "
import json
completed = '$completed_list'.split() if '$completed_list' else []
ckpt = {'completed': completed, 'timestamp': '$(date -Iseconds)'}
with open('$CKPT_FILE', 'w') as f:
    json.dump(ckpt, f, indent=2)
"
}

# =========================
# 模型服务管理
# =========================
start_all_services() {
    log "============================================================"
    log "启动模型服务"
    log "============================================================"
    log "GPU0&1: Qwen3-4B (tensor parallel, policy&target共用, $POLICY_MAX_MODEL_LEN/$TARGET_MAX_MODEL_LEN context)"
    log "GPU2&3: Guard (tensor parallel, $GUARD_MAX_MODEL_LEN context)"
    log "============================================================"

    # 设置环境变量
    export VLLM_USE_MODELSCOPE=true
    export FLASHINFER_DISABLE_VERSION_CHECK=1

    # Qwen3-4B (GPU0&1, tensor parallel) - policy和target共用
    log "启动 Qwen3-4B (GPU0&1, tensor parallel, 端口 $POLICY_PORT)..."
    if ! check_port_active $POLICY_PORT; then
        CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve "$POLICY_MODEL" \
            --host 127.0.0.1 --port $POLICY_PORT \
            --tensor-parallel-size 2 \
            --max-model-len $TARGET_MAX_MODEL_LEN \
            --gpu-memory-utilization $POLICY_GPU_UTIL \
            --served-model-name policy \
            > "$OUTPUT_DIR/qwen3_vllm.log" 2>&1 &
        log "  Qwen3-4B 进程已启动 (等待就绪)"
    else
        log "  Qwen3-4B 已在运行"
    fi

    # 等待 Qwen3-4B 就绪
    wait_for_port $POLICY_PORT "Qwen3-4B"

    # Guard (GPU2&3, tensor parallel)
    log "启动 Guard (GPU2&3, tensor parallel, 端口 $GUARD_PORT)..."
    if ! check_port_active $GUARD_PORT; then
        CUDA_VISIBLE_DEVICES=2,3 nohup vllm serve "$GUARD_MODEL" \
            --host 127.0.0.1 --port $GUARD_PORT \
            --tensor-parallel-size 2 \
            --max-model-len $GUARD_MAX_MODEL_LEN \
            --gpu-memory-utilization $GUARD_GPU_UTIL \
            --served-model-name guard \
            > "$OUTPUT_DIR/guard_vllm.log" 2>&1 &
        wait_for_port $GUARD_PORT "Guard"
    else
        log "Guard 已在运行"
    fi

    log "所有模型服务已就绪!"
}

stop_all_services() {
    log "============================================================"
    log "关闭所有模型服务"
    log "============================================================"

    stop_service $POLICY_PORT "Qwen3-4B"
    stop_service $GUARD_PORT "Guard"

    log "所有服务已关闭"
}

# =========================
# 主流程
# =========================
mkdir -p "$OUTPUT_DIR"

# 加载checkpoint
COMPLETED_STR=$(load_checkpoint)
COMPLETED_COUNT=0
if [ -n "$COMPLETED_STR" ]; then
    COMPLETED_COUNT=$(echo "$COMPLETED_STR" | wc -w)
    log "Checkpoint: 已完成 $COMPLETED_COUNT 个策略"
fi

if [ "$RESET_CKPT" = true ]; then
    log "重置checkpoint"
    rm -f "$CKPT_FILE"
    COMPLETED_STR=""
    COMPLETED_COUNT=0
fi

TOTAL_EXPS=$(( ${#STRATEGIES[@]} + 1 ))  # +1 for baseline
REMAINING=$((TOTAL_EXPS - COMPLETED_COUNT))

log "============================================================"
log "Jailbreak Prompt 实验 - 实验1 (重做版本)"
log "============================================================"
log "测试集: $TEST_SET"
log "策略数量: ${#STRATEGIES[@]}"
log "总评估次数: $TOTAL_EXPS (含基线)"
log "已完成: $COMPLETED_COUNT"
log "剩余: $REMAINING"
log "输出目录: $OUTPUT_DIR"
if [ -n "$TOPK" ]; then log "Top-K筛选: $TOPK"; fi
if [ -n "$GAP_THRESHOLD" ]; then log "Gap阈值: $GAP_THRESHOLD"; fi
if [ "$INCLUDE_QWEN3_MAX" = true ]; then log "包含qwen3-max对比"; fi
log "============================================================"

START_TIME=$(date +%s)

# Step 1: 启动服务
start_all_services

# Step 2: 运行Python实验脚本 (连接已有服务)
log ""
log "============================================================"
log "运行实验..."
log "============================================================"

python "$SCRIPT_DIR/jailbreak_prompt_exp.py" \
    --test_set "$TEST_SET" \
    --output_dir "$OUTPUT_DIR" \
    --policy_port $POLICY_PORT \
    --target_port $TARGET_PORT \
    --guard_port $GUARD_PORT \
    --policy_max_model_len $POLICY_MAX_MODEL_LEN \
    --target_max_model_len $TARGET_MAX_MODEL_LEN \
    --guard_max_model_len $GUARD_MAX_MODEL_LEN \
    --sleep_between_evals $SLEEP_BETWEEN_EVALS \
    ${TOPK:+--topk $TOPK} \
    ${GAP_THRESHOLD:+--gap_threshold $GAP_THRESHOLD} \
    ${SELECTED_STRATEGIES:+--strategies ${SELECTED_STRATEGIES[@]}} \
    ${RESET_CKPT:+--reset}

PYTHON_EXIT=$?

# Step 3: qwen3-max对比实验 (可选)
if [ "$INCLUDE_QWEN3_MAX" = true ]; then
    log ""
    log "============================================================"
    log "qwen3-max 对比实验"
    log "============================================================"
    
    # 获取top3策略
    if [ -f "$OUTPUT_DIR/experiment_summary.json" ]; then
        TOP3_STRATEGIES=$(python3 -c "
import json
with open('$OUTPUT_DIR/experiment_summary.json') as f:
    summary = json.load(f)
results = summary.get('filtered_results', summary.get('all_results', {}))
top3 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
print(' '.join([s[0] for s in top3]))
" 2>/dev/null || echo "hypothetical_scenario creative_writing role_playing")
        
        log "Top3策略: $TOP3_STRATEGIES"
        
        python "$SCRIPT_DIR/qwen3_max_comparison.py" \
            --test_set "$TEST_SET" \
            --output_dir "$OUTPUT_DIR/qwen3_max_comparison" \
            --strategies $TOP3_STRATEGIES \
            --api_key "$QWEN3_MAX_API_KEY" \
            --base_url "$QWEN3_MAX_BASE_URL" \
            --model "$QWEN3_MAX_MODEL" \
            --target_port $TARGET_PORT \
            --guard_port $GUARD_PORT
    else
        log "警告: 未找到实验结果文件，跳过qwen3-max对比"
    fi
fi

# Step 4: 关闭服务
stop_all_services

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
log "============================================================"

log "\n查看结果:"
log "  cat $OUTPUT_DIR/experiment_summary.json | python -m json.tool"

exit $PYTHON_EXIT