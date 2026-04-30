#!/bin/bash
# 实验2 训练脚本 - 只负责训练，带checkpoint机制
#
# 用法:
#   bash experiments/hybrid_reward_exp/train_exp2.sh              # 训练全部24个实验
#   bash experiments/hybrid_reward_exp/train_exp2.sh --max_steps 1000  # 调整步数
#   bash experiments/hybrid_reward_exp/train_exp2.sh --strategy hypothetical_scenario  # 只训练指定策略
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
POLICY_MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"

# 端口
TARGET_JUDGE_PORT=8001
GUARD_PORT=8002

# 训练超参数
MAX_STEPS="${MAX_STEPS:-500}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
BETA="${BETA:-0.05}"

# 奖励权重
ASR_WEIGHT=0.5
JUDGE_WEIGHT=0.5

# =========================
# 策略 × 维度定义
# =========================
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

SCORING_METHODS=(
    "single"
    "tournament"
)

# 解析命令行参数
SELECTED_STRATEGY=""
RESET_CKPT=false
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
        --reset)
            RESET_CKPT=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --strategy NAME      只训练指定策略"
            echo "  --max_steps N        训练步数 (默认500)"
            echo "  --reset              清空checkpoint重头开始"
            echo "  --help               显示帮助"
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

# Checkpoint管理
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

# =========================
# 主流程
# =========================
DIMENSIONS_PER_STRATEGY=$(( ${#GENERAL_DIMENSIONS[@]} + 1 ))
TOTAL_EXPS=$(( ${#STRATEGIES[@]} * DIMENSIONS_PER_STRATEGY * ${#SCORING_METHODS[@]} ))

# 加载checkpoint
COMPLETED_STR=$(load_checkpoint)
COMPLETED_COUNT=0
if [ -n "$COMPLETED_STR" ]; then
    COMPLETED_COUNT=$(echo "$COMPLETED_STR" | wc -w)
    log "检测到checkpoint: 已完成 $COMPLETED_COUNT 个训练，将跳过"
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
log "已完成: $COMPLETED_COUNT"
log "剩余: $REMAINING"
log "每实验步数: $MAX_STEPS"
log "============================================================"

mkdir -p "$OUTPUT_DIR"

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
            python "$BASE_DIR/experiments/hybrid_reward_exp/hybrid_reward_grpo.py" \
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
