#!/bin/bash
# =============================================================================
# 实验2 训练脚本
#
# GPU分配:
# - GPU0&1: vLLM server (Qwen3-4B) - policy generation + target + judge
# - GPU2: Policy训练
# - GPU3: Guard server
#
# 用法:
#   bash run_exp2.sh                    # 训练全部15个实验
#   bash run_exp2.sh --start_services   # 启动服务并训练
#   bash run_exp2.sh --attack_prompt hypothetical_scenario
#   bash run_exp2.sh --weight_config intent_only
#   bash run_exp2.sh --reset            # 清空checkpoint重头开始
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Python解释器 (使用系统Python，torch安装在~/.local)
PYTHON="${PYTHON:-/usr/bin/python3}"

# =========================
# 配置
# =========================
TRAIN_DATA="${TRAIN_DATA:-$BASE_DIR/../data/dataset/processed/10k/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
POLICY_MODEL="${POLICY_MODEL:-/home/tiger/models/Qwen/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/home/tiger/models/Qwen/Qwen3Guard-Gen-4B}"

# 端口配置
INFERENCE_SERVER_PORT=8001  # 普通 vLLM: Target + Judge (reward函数使用)
GUARD_SERVER_PORT=8002      # Guard Server: ASR reward

# vLLM配置
INFERENCE_MAX_MODEL_LEN=8192  # Target + Judge
INFERENCE_GPU_MEMORY_UTIL=0.9
GUARD_MAX_MODEL_LEN=8192
GUARD_GPU_MEMORY_UTIL=0.9

# 训练超参数 (colocate mode下需要较小的batch以留出显存给vLLM)
MAX_STEPS="${MAX_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
BETA="${BETA:-0.05}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"

# 攻击prompt × 权重配置
ATTACK_PROMPTS=("creative_writing" "hypothetical_scenario" "role_playing")
WEIGHT_CONFIGS=("intent_only" "stealth_only" "strategy_only" "potential_only" "uniform")

# 参数解析
SELECTED_ATTACK=""
SELECTED_WEIGHT=""
START_SERVICES=false
RESET_CKPT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --attack_prompt) SELECTED_ATTACK="$2"; shift 2 ;;
        --weight_config) SELECTED_WEIGHT="$2"; shift 2 ;;
        --max_steps) MAX_STEPS="$2"; shift 2 ;;
        --start_services) START_SERVICES=true; shift ;;
        --reset) RESET_CKPT=true; shift ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --attack_prompt NAME    攻击prompt (creative_writing, hypothetical_scenario, role_playing)"
            echo "  --weight_config NAME    权重配置 (intent_only, stealth_only, strategy_only, potential_only, uniform)"
            echo "  --max_steps N           训练步数 (默认1000)"
            echo "  --start_services        启动vLLM服务"
            echo "  --reset                 清空checkpoint"
            echo "  --help                  显示帮助"
            exit 0
            ;;
        *) shift ;;
    esac
done

if [ -n "$SELECTED_ATTACK" ]; then ATTACK_PROMPTS=("$SELECTED_ATTACK"); fi
if [ -n "$SELECTED_WEIGHT" ]; then WEIGHT_CONFIGS=("$SELECTED_WEIGHT"); fi

# =========================
# 辅助函数
# =========================
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# 普通 vLLM 使用 /health (无结尾斜杠)
check_inference_server() {
    local port=$1
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q "200"
}

check_guard_server() {
    local port=$1
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q "200"
}

# =========================
# 服务管理
# =========================
start_inference_server() {
    log "启动Inference Server (端口: $INFERENCE_SERVER_PORT, GPU2)..."
    if check_inference_server "$INFERENCE_SERVER_PORT"; then
        log "Inference Server已在运行"
        return 0
    fi

    # Target + Judge 共用同一个 Qwen3-4B 实例 (单卡)
    CUDA_VISIBLE_DEVICES=2 \
    VLLM_USE_MODELSCOPE=true \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    nohup vllm serve "$POLICY_MODEL" \
        --host 127.0.0.1 --port $INFERENCE_SERVER_PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization $INFERENCE_GPU_MEMORY_UTIL \
        --max-model-len $INFERENCE_MAX_MODEL_LEN \
        --served-model-name inference \
        > "$OUTPUT_DIR/inference_server.log" 2>&1 &

    log "等待Inference Server启动..."
    for i in $(seq 1 120); do
        if check_inference_server "$INFERENCE_SERVER_PORT"; then
            log "Inference Server启动成功!"
            return 0
        fi
        sleep 2
    done
    log "Inference Server启动超时!"
    return 1
}

start_guard_server() {
    log "启动Guard Server (端口: $GUARD_SERVER_PORT, GPU3)..."
    if check_guard_server "$GUARD_SERVER_PORT"; then
        log "Guard Server已在运行"
        return 0
    fi

    # Guard 使用 Qwen3Guard-Gen-4B (单卡)
    CUDA_VISIBLE_DEVICES=3 \
    VLLM_USE_MODELSCOPE=true \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    nohup vllm serve "$GUARD_MODEL" \
        --host 127.0.0.1 --port $GUARD_SERVER_PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization $GUARD_GPU_MEMORY_UTIL \
        --max-model-len $GUARD_MAX_MODEL_LEN \
        --served-model-name guard \
        > "$OUTPUT_DIR/guard_server.log" 2>&1 &

    log "等待Guard Server启动..."
    for i in $(seq 1 120); do
        if check_guard_server "$GUARD_SERVER_PORT"; then
            log "Guard Server启动成功!"
            return 0
        fi
        sleep 2
    done
    log "Guard Server启动超时!"
    return 1
}

ensure_services() {
    log "检查服务状态..."

    if ! check_inference_server "$INFERENCE_SERVER_PORT"; then
        log "Inference Server未运行"
        START_SERVICES=true
    else
        log "Inference Server已就绪 (端口 $INFERENCE_SERVER_PORT)"
    fi

    if ! check_guard_server "$GUARD_SERVER_PORT"; then
        log "Guard Server未运行"
        START_SERVICES=true
    else
        log "Guard Server已就绪 (端口 $GUARD_SERVER_PORT)"
    fi

    if [ "$START_SERVICES" = true ]; then
        start_inference_server
        start_guard_server
    fi
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

COMPLETED_STR=$(load_checkpoint)
COMPLETED_COUNT=0
if [ -n "$COMPLETED_STR" ]; then
    COMPLETED_COUNT=$(echo "$COMPLETED_STR" | wc -w)
    log "检测到checkpoint: 已完成 $COMPLETED_COUNT 个训练"
fi

if [ "$RESET_CKPT" = true ]; then
    log "重置checkpoint"
    COMPLETED_STR=""
    COMPLETED_COUNT=0
fi

REMAINING=$((TOTAL_EXPS - COMPLETED_COUNT))

log "============================================================"
log "实验2 训练脚本"
log "============================================================"
log "训练数据: $TRAIN_DATA"
log "输出目录: $OUTPUT_DIR"
log "攻击prompt: ${ATTACK_PROMPTS[*]}"
log "权重配置: ${WEIGHT_CONFIGS[*]}"
log "总实验数: $TOTAL_EXPS, 已完成: $COMPLETED_COUNT, 剩余: $REMAINING"
log "每实验步数: $MAX_STEPS"
log "============================================================"

mkdir -p "$OUTPUT_DIR"
ensure_services

# 遍历实验组合
EXP_IDX=0
COMPLETED_LIST="$COMPLETED_STR"

for attack in "${ATTACK_PROMPTS[@]}"; do
    for weight in "${WEIGHT_CONFIGS[@]}"; do
        EXP_IDX=$((EXP_IDX + 1))
        EXP_KEY="${attack}_${weight}"

        # 检查checkpoint
        if echo " $COMPLETED_LIST " | grep -q " $EXP_KEY "; then
            log "[跳过] $EXP_KEY (checkpoint)"
            continue
        fi

        # 检查final_lora
        if [ -d "$OUTPUT_DIR/${EXP_KEY}/final_lora" ]; then
            log "[跳过] $EXP_KEY (final_lora已存在)"
            COMPLETED_LIST="$COMPLETED_LIST $EXP_KEY"
            save_checkpoint "$COMPLETED_LIST"
            continue
        fi

        log "[${EXP_IDX}/${TOTAL_EXPS}] 训练: $EXP_KEY"
        log "============================================================"

        EXP_OUTPUT="$OUTPUT_DIR/${EXP_KEY}"
        mkdir -p "$EXP_OUTPUT"

        # GRPO训练 - GPU0&1 (colocate mode: vLLM + 训练共置, 使用accelerate多卡并行)
        CUDA_VISIBLE_DEVICES=0,1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python WANDB_DISABLED=true \
        accelerate launch --num-processes 2 --num-machines 1 --main-process-port 29500 \
            "$SCRIPT_DIR/exp2_grpo.py" \
            --attack_prompt "$attack" \
            --weight_config "$weight" \
            --train_data "$TRAIN_DATA" \
            --policy_model "$POLICY_MODEL" \
            --guard_model "$GUARD_MODEL" \
            --inference_server_port $INFERENCE_SERVER_PORT \
            --guard_server_port $GUARD_SERVER_PORT \
            --vllm_gpu_memory_utilization 0.5 \
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

        log "[${EXP_IDX}/${TOTAL_EXPS}] 完成: $EXP_KEY"
    done
done

log ""
log "============================================================"
log "训练完成! 总完成: $COMPLETED_COUNT"
log "结果保存在: $OUTPUT_DIR"
log "============================================================"