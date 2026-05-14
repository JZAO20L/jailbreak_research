#!/bin/bash
# 实验2 训练脚本 - 只负责训练，带checkpoint机制
#
# 用法:
#   bash experiments/hybrid_reward_exp/train_exp2.sh              # 训练全部 (默认single)
#   bash experiments/hybrid_reward_exp/train_exp2.sh --scoring_method tournament  # tournament打分
#   bash experiments/hybrid_reward_exp/train_exp2.sh --scoring_method all  # single+tournament
#   bash experiments/hybrid_reward_exp/train_exp2.sh --max_steps 1000  # 调整步数
#   bash experiments/hybrid_reward_exp/train_exp2.sh --strategy hypothetical_scenario  # 只训练指定策略
#   bash experiments/hybrid_reward_exp/train_exp2.sh --start_services  # 启动vLLM服务
#   bash experiments/hybrid_reward_exp/train_exp2.sh --reset  # 清空checkpoint重头开始

set -e

# =========================
# 配置区
# =========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 训练数据
TRAIN_DATA="${TRAIN_DATA:-$BASE_DIR/../data/dataset/processed/10k/train.jsonl}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/judge_prompt_exp_output}"

# 模型路径
POLICY_MODEL="${POLICY_MODEL:-/mnt/bn/chenxiong/mlx/users/jiazixiao/models/Qwen3-4B}"
TARGET_MODEL="${TARGET_MODEL:-/mnt/bn/chenxiong/mlx/users/jiazixiao/models/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/mnt/bn/chenxiong/mlx/users/jiazixiao/models/Qwen3Guard-Gen-4B}"

# 端口
TARGET_JUDGE_PORT=8001
GUARD_PORT=8002

# vLLM context length (tournament需要更长)
VLLM_MAX_MODEL_LEN_SINGLE=8192
VLLM_MAX_MODEL_LEN_TOURNAMENT=8192

# vLLM GPU memory utilization (提高以支持更多并发请求)
VLLM_GPU_UTIL_TARGET_SINGLE=0.45
VLLM_GPU_UTIL_TARGET_TOURNAMENT=0.55
VLLM_GPU_UTIL_GUARD=0.40

# 训练超参数
MAX_STEPS="${MAX_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
BETA="${BETA:-0.05}"

# 奖励权重
ASR_WEIGHT=0.5
JUDGE_WEIGHT=0.5

# =========================
# 策略 × 维度定义
# =========================
# 基于实验1 Top 3策略: hypothetical_scenario (30.8%), creative_writing (28.3%), role_playing (25.0%)
# 每个策略 4 个维度 (3通用 + 1专用) × single评分 = 4个实验/策略
STRATEGIES=(
    "hypothetical_scenario"
    "creative_writing"
    "role_playing"
)

GENERAL_DIMENSIONS=(
    "idea_preservation"
    "stealthiness"
    "naturalness"
)

# 默认评分方式 (可通过参数修改)
DEFAULT_SCORING_METHODS=("single")

# 解析命令行参数
SELECTED_STRATEGY=""
SELECTED_SCORING_METHOD=""
START_SERVICES=false
RESET_CKPT=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            SELECTED_STRATEGY="$2"
            shift 2
            ;;
        --scoring_method)
            SELECTED_SCORING_METHOD="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --start_services)
            START_SERVICES=true
            shift
            ;;
        --reset)
            RESET_CKPT=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --strategy NAME          只训练指定策略"
            echo "  --scoring_method METHOD  评分方式: single, tournament, all (默认single)"
            echo "  --max_steps N            训练步数 (默认500)"
            echo "  --start_services         启动Target和Guard vLLM服务"
            echo "  --reset                  清空checkpoint重头开始"
            echo "  --help                   显示帮助"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

if [ -n "$SELECTED_STRATEGY" ]; then
    STRATEGIES=("$SELECTED_STRATEGY")
fi

# 设置评分方式
if [ -n "$SELECTED_SCORING_METHOD" ]; then
    case "$SELECTED_SCORING_METHOD" in
        "single")
            SCORING_METHODS=("single")
            ;;
        "tournament")
            SCORING_METHODS=("tournament")
            ;;
        "all")
            SCORING_METHODS=("single" "tournament")
            ;;
        *)
            echo "错误: 未知的评分方式 '$SELECTED_SCORING_METHOD'"
            echo "可选: single, tournament, all"
            exit 1
            ;;
    esac
else
    SCORING_METHODS=("${DEFAULT_SCORING_METHODS[@]}")
fi

# =========================
# 辅助函数
# =========================
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

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

# =========================
# 服务管理函数
# =========================
check_port_active() {
    local port=$1
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q "200"
}

start_target_service() {
    log "启动Target模型服务 (端口: $TARGET_JUDGE_PORT)..."
    if check_port_active "$TARGET_JUDGE_PORT"; then
        log "Target服务已在运行,跳过启动"
        return 0
    fi

    # 根据评分方式选择context length和显存利用率
    local max_len=$VLLM_MAX_MODEL_LEN_SINGLE
    local gpu_util=$VLLM_GPU_UTIL_TARGET_SINGLE
    for method in "${SCORING_METHODS[@]}"; do
        if [ "$method" = "tournament" ]; then
            max_len=$VLLM_MAX_MODEL_LEN_TOURNAMENT
            gpu_util=$VLLM_GPU_UTIL_TARGET_TOURNAMENT
            break
        fi
    done
    log "使用 max-model-len=$max_len, gpu-memory-utilization=$gpu_util (评分方式: ${SCORING_METHODS[*]})"

    CUDA_VISIBLE_DEVICES=2 nohup vllm serve "$TARGET_MODEL" \
        --host 127.0.0.1 --port $TARGET_JUDGE_PORT \
        --max-model-len $max_len --gpu-memory-utilization $gpu_util \
        --served-model-name target \
        > "$OUTPUT_DIR/target_vllm.log" 2>&1 &

    log "等待Target服务启动..."
    for i in $(seq 1 120); do
        if check_port_active "$TARGET_JUDGE_PORT"; then
            log "Target服务启动成功!"
            return 0
        fi
        sleep 2
    done
    log "Target服务启动超时!"
    return 1
}

start_guard_service() {
    log "启动Guard模型服务 (端口: $GUARD_PORT)..."
    if check_port_active "$GUARD_PORT"; then
        log "Guard服务已在运行,跳过启动"
        return 0
    fi

    log "使用 gpu-memory-utilization=$VLLM_GPU_UTIL_GUARD"

    CUDA_VISIBLE_DEVICES=3 nohup vllm serve "$GUARD_MODEL" \
        --host 127.0.0.1 --port $GUARD_PORT \
        --max-model-len 8192 --gpu-memory-utilization $VLLM_GPU_UTIL_GUARD \
        --served-model-name guard \
        > "$OUTPUT_DIR/guard_vllm.log" 2>&1 &

    log "等待Guard服务启动..."
    for i in $(seq 1 120); do
        if check_port_active "$GUARD_PORT"; then
            log "Guard服务启动成功!"
            return 0
        fi
        sleep 2
    done
    log "Guard服务启动超时!"
    return 1
}

ensure_services_running() {
    log "============================================================"
    log "检查vLLM服务状态..."
    log "============================================================"

    local need_start=false
    local need_restart_target=false

    # 检查是否需要tournament context length
    local need_tournament_len=false
    for method in "${SCORING_METHODS[@]}"; do
        if [ "$method" = "tournament" ]; then
            need_tournament_len=true
            break
        fi
    done

    if ! check_port_active "$TARGET_JUDGE_PORT"; then
        log "Target服务未运行"
        need_start=true
    elif [ "$need_tournament_len" = true ]; then
        # Tournament实验需要8192 context length，可能需要重启
        # 检查当前服务的max_model_len (通过API查询)
        local current_max_len
        current_max_len=$(curl -s "http://127.0.0.1:$TARGET_JUDGE_PORT/v1/models" 2>/dev/null | \
            python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',[{}])[0].get('max_model_len',4096))" 2>/dev/null || echo "4096")

        if [ "$current_max_len" -lt 8000 ]; then
            log "Target服务当前 max_model_len=$current_max_len, tournament需要8192, 需要重启"
            need_restart_target=true
            # 先停止现有服务
            log "停止现有Target服务..."
            pkill -f "vllm.*port $TARGET_JUDGE_PORT" 2>/dev/null || true
            sleep 5
            need_start=true
        else
            log "Target服务已就绪 (端口 $TARGET_JUDGE_PORT, max_model_len=$current_max_len)"
        fi
    else
        log "Target服务已就绪 (端口 $TARGET_JUDGE_PORT)"
    fi

    if ! check_port_active "$GUARD_PORT"; then
        log "Guard服务未运行"
        need_start=true
    else
        log "Guard服务已就绪 (端口 $GUARD_PORT)"
    fi

    if [ "$need_start" = true ] || [ "$START_SERVICES" = true ]; then
        log "需要启动服务..."
        start_target_service
        start_guard_service
    fi

    log "所有服务检查完成!"
}

# =========================
# Checkpoint管理
# =========================
CKPT_FILE="$OUTPUT_DIR/train_checkpoint.json"

load_checkpoint() {
    if [ -f "$CKPT_FILE" ] && [ "$RESET_CKPT" = false ]; then
        python3 -c "
import json
with open('$CKPT_FILE') as f:
    ckpt = json.load(f)
print(' '.join(ckpt.get('completed', [])))
" 2>/dev/null
    fi
}

save_checkpoint() {
    local completed_list="$1"
    mkdir -p "$OUTPUT_DIR"
    python3 -c "
import json
completed = '$completed_list'.split() if '$completed_list' else []
ckpt = {'completed': completed, 'max_steps': $MAX_STEPS}
with open('$CKPT_FILE', 'w') as f:
    json.dump(ckpt, f, indent=2)
"
}

# 统计当前scoring_method的已完成数量
count_completed_for_scoring() {
    local all_completed="$1"
    local target_scoring="$2"
    local count=0

    if [ -z "$all_completed" ]; then
        echo 0
        return
    fi

    for exp in $all_completed; do
        # 检查是否以目标scoring_method结尾
        if [[ "$exp" == *"_${target_scoring}" ]]; then
            count=$((count + 1))
        fi
    done
    echo $count
}

# =========================
# 主流程
# =========================
DIMENSIONS_PER_STRATEGY=$(( ${#GENERAL_DIMENSIONS[@]} + 1 ))
TOTAL_EXPS=$(( ${#STRATEGIES[@]} * DIMENSIONS_PER_STRATEGY * ${#SCORING_METHODS[@]} ))

# 加载checkpoint
COMPLETED_STR=$(load_checkpoint)

# 计算当前scoring_method的已完成数量
COMPLETED_COUNT=0
for method in "${SCORING_METHODS[@]}"; do
    method_count=$(count_completed_for_scoring "$COMPLETED_STR" "$method")
    COMPLETED_COUNT=$((COMPLETED_COUNT + method_count))
done

if [ -n "$COMPLETED_STR" ] && [ "$COMPLETED_COUNT" -gt 0 ]; then
    log "检测到checkpoint: 当前评分方式已完成 $COMPLETED_COUNT 个训练，将跳过"
fi

if [ "$RESET_CKPT" = true ]; then
    log "重置checkpoint，将重新训练所有实验"
    COMPLETED_STR=""
    COMPLETED_COUNT=0
fi

REMAINING=$((TOTAL_EXPS - COMPLETED_COUNT))

log "============================================================"
log "实验2 训练脚本"
log "============================================================"
log "训练数据: $TRAIN_DATA"
log "输出目录: $OUTPUT_DIR"
log "策略数: ${#STRATEGIES[@]}"
log "每策略维度: $DIMENSIONS_PER_STRATEGY (3通用+1专用)"
log "评分方式: ${SCORING_METHODS[*]}"
log "总实验数: $TOTAL_EXPS"
log "已完成 (${SCORING_METHODS[*]}): $COMPLETED_COUNT"
log "剩余: $REMAINING"
log "每实验步数: $MAX_STEPS"
log "============================================================"

mkdir -p "$OUTPUT_DIR"

# 确保服务运行
ensure_services_running

# 遍历所有实验组合
EXP_IDX=0
COMPLETED_LIST=""
if [ -n "$COMPLETED_STR" ]; then
    COMPLETED_LIST="$COMPLETED_STR"
fi

for strategy in "${STRATEGIES[@]}"; do
    ALL_DIMENSIONS=("${GENERAL_DIMENSIONS[@]}" "$strategy")

    for dim in "${ALL_DIMENSIONS[@]}"; do
        for method in "${SCORING_METHODS[@]}"; do
            EXP_IDX=$((EXP_IDX + 1))

            EXP_KEY="${strategy}_${dim}_${method}"

            # 检查checkpoint中是否已完成
            if echo " $COMPLETED_LIST " | grep -q " $EXP_KEY "; then
                log "[跳过] $EXP_KEY (checkpoint标记已完成)"
                continue
            fi

            # 检查final_lora目录是否存在 (额外保障)
            LORA_PATH="$OUTPUT_DIR/${EXP_KEY}/final_lora"
            if [ -d "$LORA_PATH" ]; then
                log "[跳过] $EXP_KEY (final_lora已存在)"
                # 标记为已完成
                COMPLETED_LIST="$COMPLETED_LIST $EXP_KEY"
                save_checkpoint "$COMPLETED_LIST"
                continue
            fi

            # 判断维度类型
            if [ "$dim" = "$strategy" ]; then
                dim_type="专用"
            else
                dim_type="通用"
            fi

            # 进度显示
            print_progress $((COMPLETED_COUNT + 1)) $TOTAL_EXPS
            log ""
            log "[$EXP_IDX/$TOTAL_EXPS] 训练: strategy=$strategy, judge_dim=$dim ($dim_type), scoring=$method"
            log "============================================================"

            EXP_OUTPUT="$OUTPUT_DIR/${EXP_KEY}"
            mkdir -p "$EXP_OUTPUT"

            # GRPO训练
            CUDA_VISIBLE_DEVICES=0,1 python "$BASE_DIR/experiments/hybrid_reward_exp/hybrid_reward_grpo.py" \
                --experiment exp2 \
                --judge_prompt "$dim" \
                --scoring_method "$method" \
                --train_data "$TRAIN_DATA" \
                --policy_model "$POLICY_MODEL" \
                --target_judge_port "$TARGET_JUDGE_PORT" \
                --guard_port "$GUARD_PORT" \
                --asr_weight "$ASR_WEIGHT" \
                --judge_weight "$JUDGE_WEIGHT" \
                --max_steps "$MAX_STEPS" \
                --learning_rate "$LEARNING_RATE" \
                --num_generations "$NUM_GENERATIONS" \
                --beta "$BETA" \
                --output_dir "$EXP_OUTPUT" \
                --run_name "exp2_${EXP_KEY}"

            # 更新checkpoint
            COMPLETED_LIST="$COMPLETED_LIST $EXP_KEY"
            save_checkpoint "$COMPLETED_LIST"

            log "[$EXP_IDX/$TOTAL_EXPS] 完成: $EXP_KEY"
        done
    done
done

log ""
log "============================================================"
log "训练完成!"
log "============================================================"
log "结果保存在: $OUTPUT_DIR"
log "============================================================"