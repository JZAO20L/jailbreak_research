#!/bin/bash
# =============================================================================
# 实验2 训练脚本 - 重做版本
#
# 根据 TODO.md 实验重做 - 实验2:
# - 3攻击prompt (来自实验1 top3) × 5权重配置 = 15个训练实验
# - 每个实验训练1000步
# - ASR reward 和 Judge reward 1:1混合
# - 使用单一多维度judge prompt
#
# 用法:
#   bash train_exp2.sh                    # 训练全部15个实验
#   bash train_exp2.sh --attack_prompt hypothetical_scenario  # 只训练指定攻击prompt
#   bash train_exp2.sh --weight_config intent_only            # 只训练指定权重配置
#   bash train_exp2.sh --max_steps 500                        # 调整步数
#   bash train_exp2.sh --start_services                       # 启动vLLM服务
#   bash train_exp2.sh --reset                                # 清空checkpoint重头开始
# =============================================================================

set -e

# =========================
# 配置区
# =========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 训练数据
TRAIN_DATA="${TRAIN_DATA:-$BASE_DIR/../data/dataset/processed/10k/train.jsonl}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"

# 模型路径
POLICY_MODEL="${POLICY_MODEL:-/home/tiger/models/Qwen/Qwen3-4B}"
TARGET_MODEL="${TARGET_MODEL:-/home/tiger/models/Qwen/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/home/tiger/models/Qwen/Qwen3Guard-Gen-4B}"

# 端口配置
VLLM_SERVER_PORT=8000    # Policy generation + Target + Judge (共用)
GUARD_PORT=8002          # Guard服务

# vLLM config
VLLM_MAX_MODEL_LEN_POLICY=4096   # Policy generation
VLLM_MAX_MODEL_LEN_GUARD=8192    # Guard
VLLM_GPU_UTIL_POLICY=0.9
VLLM_GPU_UTIL_GUARD=0.9

# 训练超参数 (固定)
MAX_STEPS="${MAX_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
BETA="${BETA:-0.05}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"  # 降低到2以减少内存压力
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"

# 奖励权重 (固定 1:1)
ASR_WEIGHT=0.5
JUDGE_WEIGHT=0.5

# =========================
# 攻击prompt × 权重配置
# =========================
# 根据实验1 top3结果
ATTACK_PROMPTS=(
    "creative_writing"      # ASR: 30.2%
    "hypothetical_scenario" # ASR: 30.2%
    "role_playing"          # ASR: 24.9%
)

# 权重配置: 4单一维度 + 1均匀维度
WEIGHT_CONFIGS=(
    "intent_only"           # [1.0, 0.0, 0.0, 0.0]
    "stealth_only"          # [0.0, 1.0, 0.0, 0.0]
    "strategy_only"         # [0.0, 0.0, 1.0, 0.0]
    "potential_only"        # [0.0, 0.0, 0.0, 1.0]
    "uniform"               # [0.25, 0.25, 0.25, 0.25]
)

# 默认: 全部实验
SELECTED_ATTACK=""
SELECTED_WEIGHT=""
START_SERVICES=false
RESET_CKPT=false

# 解析命令行参数
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
            echo "  --attack_prompt NAME    只训练指定攻击prompt"
            echo "                           可选: creative_writing, hypothetical_scenario, role_playing"
            echo "  --weight_config NAME    只训练指定权重配置"
            echo "                           可选: intent_only, stealth_only, strategy_only, potential_only, uniform"
            echo "  --max_steps N           训练步数 (默认1000)"
            echo "  --start_services        启动Target和Guard vLLM服务"
            echo "  --reset                 清空checkpoint重头开始"
            echo "  --help                  显示帮助"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# 设置攻击prompt
if [ -n "$SELECTED_ATTACK" ]; then
    ATTACK_PROMPTS=("$SELECTED_ATTACK")
fi

# 设置权重配置
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

# =========================
# 服务管理函数
# =========================
check_port_active() {
    local port=$1
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q "200"
}

start_policy_vllm_server() {
    log "启动vLLM Server (Policy+Target+Judge共用) (端口: $VLLM_SERVER_PORT)..."
    if check_port_active "$VLLM_SERVER_PORT"; then
        log "vLLM Server已在运行,跳过启动"
        return 0
    fi

    # 使用 trl vllm-serve 启动，支持 GRPO 的 weight update
    CUDA_VISIBLE_DEVICES=0 FLASHINFER_DISABLE_VERSION_CHECK=1 nohup trl vllm-serve \
        --model "$POLICY_MODEL" \
        --host 127.0.0.1 --port $VLLM_SERVER_PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization $VLLM_GPU_UTIL_POLICY \
        --max-model-len $VLLM_MAX_MODEL_LEN_POLICY \
        > "$OUTPUT_DIR/policy_vllm_server.log" 2>&1 &

    log "等待vLLM Server启动..."
    for i in $(seq 1 120); do
        if check_port_active "$VLLM_SERVER_PORT"; then
            log "vLLM Server启动成功!"
            return 0
        fi
        sleep 2
    done
    log "vLLM Server启动超时!"
    return 1
}

start_guard_service() {
    log "启动Guard模型服务 (端口: $GUARD_PORT)..."
    if check_port_active "$GUARD_PORT"; then
        log "Guard服务已在运行,跳过启动"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=2 FLASHINFER_DISABLE_VERSION_CHECK=1 nohup vllm serve "$GUARD_MODEL" \
        --host 127.0.0.1 --port $GUARD_PORT \
        --max-model-len $VLLM_MAX_MODEL_LEN_GUARD \
        --gpu-memory-utilization $VLLM_GPU_UTIL_GUARD \
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

    if ! check_port_active "$VLLM_SERVER_PORT"; then
        log "vLLM Server未运行"
        START_SERVICES=true
    else
        log "vLLM Server已就绪 (端口 $VLLM_SERVER_PORT)"
    fi

    if ! check_port_active "$GUARD_PORT"; then
        log "Guard服务未运行"
        START_SERVICES=true
    else
        log "Guard服务已就绪 (端口 $GUARD_PORT)"
    fi

    if [ "$START_SERVICES" = true ]; then
        log "需要启动服务..."
        start_policy_vllm_server
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

# =========================
# 主流程
# =========================
TOTAL_EXPS=$(( ${#ATTACK_PROMPTS[@]} * ${#WEIGHT_CONFIGS[@]} ))

# 加载checkpoint
COMPLETED_STR=$(load_checkpoint)
COMPLETED_COUNT=0
if [ -n "$COMPLETED_STR" ]; then
    COMPLETED_COUNT=$(echo "$COMPLETED_STR" | wc -w)
    log "检测到checkpoint: 已完成 $COMPLETED_COUNT 个训练"
fi

if [ "$RESET_CKPT" = true ]; then
    log "重置checkpoint，将重新训练所有实验"
    COMPLETED_STR=""
    COMPLETED_COUNT=0
fi

REMAINING=$((TOTAL_EXPS - COMPLETED_COUNT))

log "============================================================"
log "实验2 训练脚本 (重做版本)"
log "============================================================"
log "训练数据: $TRAIN_DATA"
log "输出目录: $OUTPUT_DIR"
log "攻击prompt数: ${#ATTACK_PROMPTS[@]} (${ATTACK_PROMPTS[*]})"
log "权重配置数: ${#WEIGHT_CONFIGS[@]} (${WEIGHT_CONFIGS[*]})"
log "总实验数: $TOTAL_EXPS"
log "已完成: $COMPLETED_COUNT"
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

for attack in "${ATTACK_PROMPTS[@]}"; do
    for weight in "${WEIGHT_CONFIGS[@]}"; do
        EXP_IDX=$((EXP_IDX + 1))

        EXP_KEY="${attack}_${weight}"

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

        # 进度显示
        print_progress $((COMPLETED_COUNT + 1)) $TOTAL_EXPS
        log ""
        log "[$EXP_IDX/$TOTAL_EXPS] 训练: attack=$attack, weight=$weight"
        log "============================================================"

        EXP_OUTPUT="$OUTPUT_DIR/${EXP_KEY}"
        mkdir -p "$EXP_OUTPUT"

        # GRPO训练 - 使用accelerate launch进行多GPU分布式训练
        # Policy使用GPU0&1 (2卡并发训练), Target在GPU2, Guard在GPU3
        CUDA_VISIBLE_DEVICES=0,1 FLASHINFER_DISABLE_VERSION_CHECK=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python WANDB_DISABLED=true accelerate launch \
            --num-processes 2 \
            --num-machines 1 \
            --machine-rank 0 \
            --main-process-port 29500 \
            "$BASE_DIR/experiments/hybrid_reward_exp/hybrid_reward_grpo.py" \
            --attack_prompt "$attack" \
            --weight_config "$weight" \
            --train_data "$TRAIN_DATA" \
            --policy_model "$POLICY_MODEL" \
            --target_judge_port "$TARGET_JUDGE_PORT" \
            --guard_port "$GUARD_PORT" \
            --max_steps "$MAX_STEPS" \
            --learning_rate "$LEARNING_RATE" \
            --num_generations "$NUM_GENERATIONS" \
            --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
            --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
            --beta "$BETA" \
            --output_dir "$EXP_OUTPUT" \
            --run_name "exp2_${EXP_KEY}"

        # 更新checkpoint
        COMPLETED_LIST="$COMPLETED_LIST $EXP_KEY"
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
        save_checkpoint "$COMPLETED_LIST"

        log "[$EXP_IDX/$TOTAL_EXPS] 完成: $EXP_KEY"
    done
done

log ""
log "============================================================"
log "训练完成!"
log "============================================================"
log "结果保存在: $OUTPUT_DIR"
log "总完成实验数: $COMPLETED_COUNT"
log "============================================================"