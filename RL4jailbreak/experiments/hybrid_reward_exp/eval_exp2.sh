#!/bin/bash
# =============================================================================
# 实验2 评估脚本 - 重做版本
#
# 根据 TODO.md 实验重做 - 实验2:
# - 3攻击prompt × 5权重配置 = 15个评估实验
# - 每个训练实验对应一个评估实验
# - 使用test集进行ASR评估
#
# 用法:
#   bash eval_exp2.sh                    # 评估全部15个实验
#   bash eval_exp2.sh --attack_prompt hypothetical_scenario  # 只评估指定攻击prompt
#   bash eval_exp2.sh --weight_config intent_only            # 只评估指定权重配置
#   bash eval_exp2.sh --reset                                # 清空checkpoint重头开始
# =============================================================================

set -e

# =========================
# 配置区
# =========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 评估数据 - 使用test集进行最终ASR测试（与实验1保持一致）
EVAL_DATA="${EVAL_DATA:-$BASE_DIR/../data/dataset/processed/10k/test.jsonl}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"

# 模型路径
POLICY_MODEL="${POLICY_MODEL:-/home/tiger/models/Qwen/Qwen3-4B}"
TARGET_MODEL="${TARGET_MODEL:-/home/tiger/models/Qwen/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/home/tiger/models/Qwen/Qwen3Guard-Gen-4B}"

# 端口
POLICY_PORT=8003
TARGET_PORT=8001
GUARD_PORT=8002

# =========================
# 攻击prompt × 权重配置
# =========================
ATTACK_PROMPTS=(
    "creative_writing"
    "hypothetical_scenario"
    "role_playing"
)

WEIGHT_CONFIGS=(
    "intent_only"
    "stealth_only"
    "strategy_only"
    "potential_only"
    "uniform"
)

# 解析命令行参数
SELECTED_ATTACK=""
SELECTED_WEIGHT=""
RESET_CKPT=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --attack_prompt)
            SELECTED_ATTACK="$2"
            shift 2
            ;;
        --weight_config)
            SELECTED_WEIGHT="$2"
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
            echo "  --attack_prompt NAME    只评估指定攻击prompt"
            echo "  --weight_config NAME    只评估指定权重配置"
            echo "  --reset                 清空checkpoint重头开始"
            echo "  --help                  显示帮助"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

if [ -n "$SELECTED_ATTACK" ]; then
    ATTACK_PROMPTS=("$SELECTED_ATTACK")
fi

if [ -n "$SELECTED_WEIGHT" ]; then
    WEIGHT_CONFIGS=("$SELECTED_WEIGHT")
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
CKPT_FILE="$OUTPUT_DIR/eval_checkpoint.json"

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
ckpt = {'completed': completed}
with open('$CKPT_FILE', 'w') as f:
    json.dump(ckpt, f, indent=2)
"
}

check_port_active() {
    local port=$1
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q "200"
}

# 启动Target和Guard服务（只需启动一次）
start_target_guard() {
    log "启动Target模型服务 (端口: $TARGET_PORT, GPU2)..."
    if ! check_port_active "$TARGET_PORT"; then
        CUDA_VISIBLE_DEVICES=1 FLASHINFER_DISABLE_VERSION_CHECK=1 nohup vllm serve "$TARGET_MODEL" \
            --host 127.0.0.1 --port $TARGET_PORT \
            --max-model-len 8192 --gpu-memory-utilization 0.9 \
            --served-model-name target \
            > "$OUTPUT_DIR/target_vllm.log" 2>&1 &
        for i in $(seq 1 120); do
            if check_port_active "$TARGET_PORT"; then log "Target启动成功!"; break; fi
            sleep 2
        done
    else
        log "Target服务已在运行"
    fi

    log "启动Guard模型服务 (端口: $GUARD_PORT, GPU3)..."
    if ! check_port_active "$GUARD_PORT"; then
        CUDA_VISIBLE_DEVICES=2 FLASHINFER_DISABLE_VERSION_CHECK=1 nohup vllm serve "$GUARD_MODEL" \
            --host 127.0.0.1 --port $GUARD_PORT \
            --max-model-len 8192 --gpu-memory-utilization 0.9 \
            --served-model-name guard \
            > "$OUTPUT_DIR/guard_vllm.log" 2>&1 &
        for i in $(seq 1 120); do
            if check_port_active "$GUARD_PORT"; then log "Guard启动成功!"; break; fi
            sleep 2
        done
    else
        log "Guard服务已在运行"
    fi
}

# 启动带LoRA的Policy服务
start_policy_with_lora() {
    local lora_path=$1
    local strategy_name=$2

    log "关闭旧的Policy服务..."
    pkill -f "vllm.*8003" 2>/dev/null || true
    sleep 3

    log "启动带LoRA的Policy服务 (端口: $POLICY_PORT, GPU0)..."
    log "LoRA路径: $lora_path"
    log "策略名称: $strategy_name"

    CUDA_VISIBLE_DEVICES=0 FLASHINFER_DISABLE_VERSION_CHECK=1 nohup vllm serve "$POLICY_MODEL" \
        --host 127.0.0.1 --port $POLICY_PORT \
        --max-model-len 4096 --gpu-memory-utilization 0.9 \
        --served-model-name policy \
        --enable-lora \
        --lora-modules policy_lora="$lora_path" \
        --max-lora-rank 32 \
        > "$OUTPUT_DIR/policy_vllm.log" 2>&1 &

    for i in $(seq 1 120); do
        if check_port_active "$POLICY_PORT"; then log "Policy启动成功!"; return 0; fi
        sleep 2
    done
    log "Policy启动超时!"
    return 1
}

# 关闭Policy服务
stop_policy() {
    log "关闭Policy服务..."
    pkill -f "vllm.*8003" 2>/dev/null || true
    sleep 3
}

# =========================
# 主流程
# =========================
TOTAL_EXPS=$(( ${#ATTACK_PROMPTS[@]} * ${#WEIGHT_CONFIGS[@]} ))

# 加载checkpoint
COMPLETED_STR=$(load_checkpoint)
COMPLETED_COUNT=0
if [ -n "$COMPLETED_STR" ]; then
    COMPLETED_COUNT=$(echo "$COMPLETED_STR" | wc -w)
    log "检测到checkpoint: 已完成 $COMPLETED_COUNT 个评估"
fi

if [ "$RESET_CKPT" = true ]; then
    log "重置checkpoint，将重新评估所有实验"
    COMPLETED_STR=""
    COMPLETED_COUNT=0
fi

REMAINING=$((TOTAL_EXPS - COMPLETED_COUNT))

log "============================================================"
log "实验2 评估脚本 (重做版本)"
log "============================================================"
log "评估数据: $EVAL_DATA"
log "输出目录: $OUTPUT_DIR"
log "攻击prompt: ${ATTACK_PROMPTS[*]}"
log "权重配置: ${WEIGHT_CONFIGS[*]}"
log "总实验数: $TOTAL_EXPS"
log "已完成: $COMPLETED_COUNT"
log "剩余: $REMAINING"
log "============================================================"

mkdir -p "$OUTPUT_DIR"

# 启动Target和Guard（只需一次）
start_target_guard

# =========================
# 正式实验评估
# =========================
log ""
log "============================================================"
log "开始正式实验评估"
log "============================================================"

# 遍历所有实验组合
EXP_IDX=0
COMPLETED_LIST=""
if [ -n "$COMPLETED_STR" ]; then
    COMPLETED_LIST="$COMPLETED_STR"
fi

for attack in "${ATTACK_PROMPTS[@]}"; do
    for weight in "${WEIGHT_CONFIGS[@]}"; do
        EXP_IDX=$((EXP_IDX + 1))

        EXP_KEY="${attack}_${weight}"

        # 检查是否已完成
        if echo " $COMPLETED_LIST " | grep -q " $EXP_KEY "; then
            log "[跳过] $EXP_KEY (已评估)"
            continue
        fi

        # 检查训练结果是否存在
        EXP_OUTPUT="$OUTPUT_DIR/${EXP_KEY}"
        LORA_PATH="$EXP_OUTPUT/final_lora"
        if [ ! -d "$LORA_PATH" ]; then
            log "[跳过] $EXP_KEY (训练结果不存在: $LORA_PATH)"
            continue
        fi

        # 进度显示
        print_progress $((COMPLETED_COUNT + 1)) $TOTAL_EXPS
        log ""
        log "[$EXP_IDX/$TOTAL_EXPS] 评估: attack=$attack, weight=$weight"
        log "============================================================"

        # 启动带LoRA的Policy
        start_policy_with_lora "$LORA_PATH" "$attack"

        EVAL_OUTPUT="$EXP_OUTPUT/eval_results"
        mkdir -p "$EVAL_OUTPUT"

        # 使用eval.py进行评估，指定策略名称
        python "$BASE_DIR/scripts/eval.py" \
            --eval_path "$EVAL_DATA" \
            --lora_paths "$LORA_PATH" \
            --base_model_path "$POLICY_MODEL" \
            --target_model_path "$TARGET_MODEL" \
            --guard_model_path "$GUARD_MODEL" \
            --policy_port "$POLICY_PORT" \
            --target_port "$TARGET_PORT" \
            --guard_port "$GUARD_PORT" \
            --strategy_name "$attack" \
            --output_root "$EVAL_OUTPUT" \
            --run_name "eval_${EXP_KEY}"

        # 关闭Policy服务
        stop_policy

        # 更新checkpoint
        COMPLETED_LIST="$COMPLETED_LIST $EXP_KEY"
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
        save_checkpoint "$COMPLETED_LIST"

        log "[$EXP_IDX/$TOTAL_EXPS] 完成: $EXP_KEY"
    done
done

log ""
log "============================================================"
log "评估完成!"
log "============================================================"
log "结果保存在: $OUTPUT_DIR"
log "总完成评估数: $COMPLETED_COUNT"
log ""

# 生成汇总报告
log "生成汇总报告..."
if [ -f "$SCRIPT_DIR/summarize_results.py" ]; then
    python "$SCRIPT_DIR/summarize_results.py" --output_dir "$OUTPUT_DIR"
fi
log "============================================================"