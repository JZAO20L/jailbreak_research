#!/bin/bash
# Judge Prompt 实验脚本 - 实验2
# 对5个Top策略 × 2种评分方式 × 4个维度(3通用+1专用) 进行混合奖励GRPO训练
#
# 用法:
#   bash experiments/hybrid_reward_exp/judge_prompt_exp.sh              # 运行全部40个实验
#   bash experiments/hybrid_reward_exp/judge_prompt_exp.sh --strategy hypothetical_scenario  # 只运行指定策略
#   bash experiments/hybrid_reward_exp/judge_prompt_exp.sh --max_steps 1000  # 调整步数

set -e

# =========================
# 配置区
# =========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 训练数据
TRAIN_DATA="${TRAIN_DATA:-$BASE_DIR/../data/dataset/processed/10k/train.jsonl}"

# 评估数据
EVAL_DATA="${EVAL_DATA:-$BASE_DIR/../data/dataset/processed/10k/eval.jsonl}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/judge_prompt_exp_output}"

# 模型路径
POLICY_MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
TARGET_MODEL="${TARGET_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B}"

# 端口
TARGET_JUDGE_PORT=8001
GUARD_PORT=8002

# 训练超参数
MAX_STEPS="${MAX_STEPS:-500}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
BETA="${BETA:-0.05}"

# 奖励权重 (实验2固定为1:1)
ASR_WEIGHT=0.5
JUDGE_WEIGHT=0.5
FORMAT_WEIGHT=0.1

# =========================
# 策略 × 维度定义
# =========================
# 基于实验1 Top 5策略
# 每个策略对应: 3个通用维度 + 1个专用维度
STRATEGIES=(
    "hypothetical_scenario"
    "creative_writing"
    "role_playing"
    "red_teaming"
    "urgent_situation"
)

# 通用维度 (对所有策略适用)
GENERAL_DIMENSIONS=(
    "idea_preservation"
    "stealthiness"
    "naturalness"
)

# 评分方式
SCORING_METHODS=(
    "single"      # 单条打分
    "tournament"  # 锦标赛打分
)

# 解析命令行参数
SELECTED_STRATEGY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            SELECTED_STRATEGY="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --strategy NAME      只运行指定策略"
            echo "  --max_steps N        训练步数 (默认500)"
            echo "  --help               显示帮助"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# 如果指定了策略，只运行该策略
if [ -n "$SELECTED_STRATEGY" ]; then
    STRATEGIES=("$SELECTED_STRATEGY")
fi

# =========================
# 辅助函数
# =========================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

print_progress() {
    local current=$1
    local total=$2
    local pct=$((current * 100 / total))
    local filled=$((pct / 2))
    local empty=$((50 - filled))

    printf -v bar '%*s' "$filled" ''
    bar=${bar// /#}
    printf -v spaces '%*s' "$empty" ''

    printf "\r  [%s%s] %d%% (%d/%d)" "$bar" "$spaces" "$pct" "$current" "$total"
}

# =============================================================================
# GPU状态检查和模型加载函数
# =============================================================================
check_gpu1_status() {
    local gpu1_memory
    gpu1_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d '[:space:]')
    if [ -z "$gpu1_memory" ]; then echo "error"; return 1; fi
    if [ "$gpu1_memory" -lt 2000 ]; then echo "idle"
    elif [ "$gpu1_memory" -gt 15000 ]; then echo "loaded"
    else echo "busy"; fi
}

check_port_active() {
    local port=$1
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q "200"
}

release_gpu1_memory() {
    log "检查GPU1状态..."
    local status
    status=$(check_gpu1_status)
    log "GPU1当前状态: $status"
    case $status in
        "idle") log "GPU1空闲,无需释放" ;;
        "loaded")
            log "GPU1已加载模型,检查服务是否活跃..."
            if check_port_active "$TARGET_JUDGE_PORT" && check_port_active "$GUARD_PORT"; then
                log "所有服务已就绪,无需重新启动"; return 0
            else
                log "部分服务未就绪,需要重新启动"; return 1
            fi
            ;;
        *) return 1 ;;
    esac
    return 0
}

start_target_service() {
    log "启动Target模型服务 (端口: $TARGET_JUDGE_PORT)..."
    if check_port_active "$TARGET_JUDGE_PORT"; then log "Target服务已在运行,跳过启动"; return 0; fi
    if [ -f "$BASE_DIR/scripts/start_target.sh" ]; then
        bash "$BASE_DIR/scripts/start_target.sh" --port "$TARGET_JUDGE_PORT" --gpu 1 &
        for i in $(seq 1 120); do
            if check_port_active "$TARGET_JUDGE_PORT"; then log "Target服务启动成功!"; return 0; fi
            sleep 2
        done
        log "Target服务启动超时!"; return 1
    else
        log "错误: 找不到 start_target.sh"; return 1
    fi
}

start_guard_service() {
    log "启动Guard模型服务 (端口: $GUARD_PORT)..."
    if check_port_active "$GUARD_PORT"; then log "Guard服务已在运行,跳过启动"; return 0; fi
    if [ -f "$BASE_DIR/scripts/start_guard.sh" ]; then
        bash "$BASE_DIR/scripts/start_guard.sh" --port "$GUARD_PORT" --gpu 1 &
        for i in $(seq 1 120); do
            if check_port_active "$GUARD_PORT"; then log "Guard服务启动成功!"; return 0; fi
            sleep 2
        done
        log "Guard服务启动超时!"; return 1
    else
        log "错误: 找不到 start_guard.sh"; return 1
    fi
}

ensure_services_running() {
    log "============================================================"
    log "确保GPU1上的服务就绪..."
    log "============================================================"
    release_gpu1_memory
    start_target_service
    start_guard_service
    log "所有服务检查完成!"
}

# =========================
# 实验主流程
# =========================

# 计算总实验数: 策略数 × (通用维度数+专用维度数) × 评分方式数
DIMENSIONS_PER_STRATEGY=$(( ${#GENERAL_DIMENSIONS[@]} + 1 ))  # 3通用 + 1专用 = 4
TOTAL_EXPS=$(( ${#STRATEGIES[@]} * DIMENSIONS_PER_STRATEGY * ${#SCORING_METHODS[@]} ))

log "============================================================"
log "Judge Prompt 实验 - 实验2"
log "============================================================"
log "训练数据: $TRAIN_DATA"
log "评估数据: $EVAL_DATA"
log "输出目录: $OUTPUT_DIR"
log "策略数: ${#STRATEGIES[@]} (${STRATEGIES[*]})"
log "每策略维度: $DIMENSIONS_PER_STRATEGY (3通用+1专用)"
log "评分方式: ${SCORING_METHODS[*]}"
log "总实验数: $TOTAL_EXPS"
log "每实验步数: $MAX_STEPS"
log "奖励权重: ASR=$ASR_WEIGHT, Judge=$JUDGE_WEIGHT, Format=$FORMAT_WEIGHT"
log "============================================================"

mkdir -p "$OUTPUT_DIR"

# 确保GPU1上的服务就绪
ensure_services_running

START_TIME=$(date +%s)
declare -a RESULTS

# ----------------------------------------------------------------
# 遍历所有实验组合
# ----------------------------------------------------------------
EXP_IDX=0
for strategy in "${STRATEGIES[@]}"; do

    # 该策略对应的所有judge维度: 3通用 + 1专用
    # 通用维度对所有策略都适用
    # 专用维度 = 策略本身的名称
    ALL_DIMENSIONS=("${GENERAL_DIMENSIONS[@]}" "$strategy")

    for dim in "${ALL_DIMENSIONS[@]}"; do
        for method in "${SCORING_METHODS[@]}"; do
            EXP_IDX=$((EXP_IDX + 1))

            # 判断是通用还是专用维度
            if [ "$dim" = "$strategy" ]; then
                dim_type="专用"
            else
                dim_type="通用"
            fi

            # 进度显示
            print_progress $EXP_IDX $TOTAL_EXPS
            log ""
            log "[$EXP_IDX/$TOTAL_EXPS] 实验: strategy=$strategy, judge_dim=$dim ($dim_type), scoring=$method"
            log "============================================================"

            EXP_OUTPUT="$OUTPUT_DIR/${strategy}_${dim}_${method}"
            mkdir -p "$EXP_OUTPUT"

            # GRPO训练
            log "[$EXP_IDX/$TOTAL_EXPS] 开始训练: $strategy / $dim / $method"

            python "$BASE_DIR/experiments/hybrid_reward_exp/hybrid_reward_grpo.py" \
                --experiment exp2 \
                --judge_prompt "$dim" \
                --scoring_method "$method" \
                --train_data "$TRAIN_DATA" \
                --policy_model "$POLICY_MODEL" \
                --target_model "$TARGET_MODEL" \
                --guard_model "$GUARD_MODEL" \
                --target_judge_port "$TARGET_JUDGE_PORT" \
                --guard_port "$GUARD_PORT" \
                --asr_weight "$ASR_WEIGHT" \
                --judge_weight "$JUDGE_WEIGHT" \
                --format_weight "$FORMAT_WEIGHT" \
                --max_steps "$MAX_STEPS" \
                --learning_rate "$LEARNING_RATE" \
                --num_generations "$NUM_GENERATIONS" \
                --beta "$BETA" \
                --output_dir "$EXP_OUTPUT" \
                --run_name "exp2_${strategy}_${dim}_${method}"

            # ASR评估
            log "[$EXP_IDX/$TOTAL_EXPS] 开始评估: $strategy / $dim / $method"

            EVAL_OUTPUT="$EXP_OUTPUT/eval_results"
            mkdir -p "$EVAL_OUTPUT"

            python "$BASE_DIR/scripts/eval.py" \
                --eval_path "$EVAL_DATA" \
                --lora_paths "$EXP_OUTPUT/final_lora" \
                --base_model_path "$POLICY_MODEL" \
                --target_model_path "$TARGET_MODEL" \
                --guard_model_path "$GUARD_MODEL" \
                --policy_port "$TARGET_JUDGE_PORT" \
                --target_port "$TARGET_JUDGE_PORT" \
                --guard_port "$GUARD_PORT" \
                --output_root "$EVAL_OUTPUT" \
                --run_name "eval_${strategy}_${dim}_${method}"

            RESULTS+=("${strategy}_${dim}_${method}")
            log "[$EXP_IDX/$TOTAL_EXPS] 完成: $strategy / $dim / $method"

        done
    done
done

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

log "\n结果汇总:"
for result in "${RESULTS[@]}"; do
    log "  - $result: 查看 $OUTPUT_DIR/$result/eval_results/"
done

log "\n提示: 可使用以下命令查看结果:"
log "  ls -la $OUTPUT_DIR/*/eval_results/"
log "  cat $OUTPUT_DIR/*/eval_results/*/summary.json | python -m json.tool"
