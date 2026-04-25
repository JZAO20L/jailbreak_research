#!/bin/bash
# Reward Weight 实验脚本 - 实验3
# 对ASR reward和Judge reward不同权重比例进行混合奖励GRPO训练
#
# 用法:
#   bash experiments/hybrid_reward_exp/reward_weight_exp.sh              # 运行所有权重实验
#   bash experiments/hybrid_reward_exp/reward_weight_exp.sh 0.7 0.3      # 只运行指定权重
#   bash experiments/hybrid_reward_exp/reward_weight_exp.sh --judge_prompt stealthiness  # 指定judge prompt

set -e

# =========================
# 配置区 - 可根据需要修改
# =========================

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 训练数据
TRAIN_DATA="${TRAIN_DATA:-$BASE_DIR/data/dataset/processed/10k/train.jsonl}"

# 评估数据
EVAL_DATA="${EVAL_DATA:-$BASE_DIR/data/dataset/processed/10k/eval.jsonl}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/reward_weight_exp_output}"

# 模型路径
POLICY_MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
TARGET_MODEL="${TARGET_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B}"

# 端口
TARGET_JUDGE_PORT=8001
GUARD_PORT=8002

# 训练超参数
MAX_STEPS="${MAX_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
BETA="${BETA:-0.05}"

# Format reward权重 (固定)
FORMAT_WEIGHT=0.1

# 默认judge prompt和评分方式 (可通过命令行修改)
JUDGE_PROMPT="${JUDGE_PROMPT:-stealthiness}"
SCORING_METHOD="${SCORING_METHOD:-single}"

# =========================
# 权重比例列表 (Judge : ASR)
# =========================
# 实验2已做了5:5的情况
# 实验3测试: 2:8, 4:6, 6:4, 8:2
# 注意: 这里只存储judge_weight，asr_weight = 1 - judge_weight - format_weight
# 但由于format_weight=0.1，所以实际比例需要调整
#
# 简化处理: 我们直接设置judge_weight和asr_weight，让它们的和为0.9
# 这样加上format_weight=0.1后总和为1.0

# 如果传入了命令行参数，使用指定的权重
if [ $# -ge 2 ]; then
    JUDGE_WEIGHTS=("$1")
    ASR_WEIGHTS=("$2")
    echo "运行指定权重: judge_weight=$1, asr_weight=$2"
else
    # 默认权重列表 (judge_weight)
    # 对应的asr_weight = 0.9 - judge_weight
    JUDGE_WEIGHTS=(0.1 0.3 0.5 0.7 0.9)
fi

# =========================
# 辅助函数
# =========================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# 进度条显示函数
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
# GPU状态检查和模型加载函数 (和实验2相同)
# =============================================================================

# 检查GPU1状态
check_gpu1_status() {
    local gpu1_memory
    gpu1_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d '[:space:]')
    
    if [ -z "$gpu1_memory" ]; then
        echo "error"
        return 1
    fi
    
    if [ "$gpu1_memory" -lt 2000 ]; then
        echo "idle"
    elif [ "$gpu1_memory" -gt 15000 ]; then
        echo "loaded"
    else
        echo "busy"
    fi
}

# 检查指定端口是否有vLLM服务在运行
check_port_active() {
    local port=$1
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q "200"; then
        return 0
    else
        return 1
    fi
}

# 释放GPU1显存
release_gpu1_memory() {
    log "检查GPU1状态..."
    
    local status
    status=$(check_gpu1_status)
    log "GPU1当前状态: $status"
    
    case $status in
        "idle")
            log "GPU1空闲,无需释放"
            ;;
        "loaded")
            log "GPU1已加载模型,检查服务是否活跃..."
            
            local target_active=false
            local guard_active=false
            
            if check_port_active "$TARGET_JUDGE_PORT"; then
                target_active=true
                log "Target服务已在端口 $TARGET_JUDGE_PORT 运行"
            fi
            
            if check_port_active "$GUARD_PORT"; then
                guard_active=true
                log "Guard服务已在端口 $GUARD_PORT 运行"
            fi
            
            if $target_active && $guard_active; then
                log "所有服务已就绪,无需重新启动"
                return 0
            else
                log "部分服务未就绪,需要重新启动"
                return 1
            fi
            ;;
        "busy")
            log "GPU1非空闲但可能没加载目标模型"
            log "建议手动检查或重启服务"
            return 1
            ;;
        "error")
            log "无法获取GPU状态"
            return 1
            ;;
    esac
    
    return 0
}

# 启动Target模型服务 (GPU1)
start_target_service() {
    log "启动Target模型服务 (端口: $TARGET_JUDGE_PORT)..."
    
    if check_port_active "$TARGET_JUDGE_PORT"; then
        log "Target服务已在运行,跳过启动"
        return 0
    fi
    
    if [ -f "$BASE_DIR/scripts/start_target.sh" ]; then
        log "执行 start_target.sh..."
        bash "$BASE_DIR/scripts/start_target.sh" --port "$TARGET_JUDGE_PORT" --gpu 1 &
        
        log "等待Target服务启动..."
        for i in $(seq 1 60); do
            if check_port_active "$TARGET_JUDGE_PORT"; then
                log "Target服务启动成功!"
                return 0
            fi
            sleep 2
        done
        
        log "Target服务启动超时!"
        return 1
    else
        log "错误: 找不到 start_target.sh 脚本"
        log "请手动启动Target模型服务到端口 $TARGET_JUDGE_PORT"
        return 1
    fi
}

# 启动Guard模型服务 (GPU1)
start_guard_service() {
    log "启动Guard模型服务 (端口: $GUARD_PORT)..."
    
    if check_port_active "$GUARD_PORT"; then
        log "Guard服务已在运行,跳过启动"
        return 0
    fi
    
    if [ -f "$BASE_DIR/scripts/start_guard.sh" ]; then
        log "执行 start_guard.sh..."
        bash "$BASE_DIR/scripts/start_guard.sh" --port "$GUARD_PORT" --gpu 1 &
        
        log "等待Guard服务启动..."
        for i in $(seq 1 60); do
            if check_port_active "$GUARD_PORT"; then
                log "Guard服务启动成功!"
                return 0
            fi
            sleep 2
        done
        
        log "Guard服务启动超时!"
        return 1
    else
        log "错误: 找不到 start_guard.sh 脚本"
        log "请手动启动Guard模型服务到端口 $GUARD_PORT"
        return 1
    fi
}

# 确保所有需要的服务都在GPU1上运行
ensure_services_running() {
    log "============================================================"
    log "确保GPU1上的服务就绪..."
    log "============================================================"
    
    release_gpu1_memory
    start_target_service
    start_guard_service
    
    log "所有服务检查完成!"
}

# =============================================================================
# 实验主流程
# =============================================================================

# 计算总实验数
TOTAL_EXPS=${#JUDGE_WEIGHTS[@]}

log "============================================================"
log "Reward Weight 实验 - 实验3"
log "============================================================"
log "训练数据: $TRAIN_DATA"
log "评估数据: $EVAL_DATA"
log "输出目录: $OUTPUT_DIR"
log "Judge Prompt: $JUDGE_PROMPT"
log "评分方式: $SCORING_METHOD"
log "权重比例数: $TOTAL_EXPS"
log "每实验步数: $MAX_STEPS"
log "Format Weight: $FORMAT_WEIGHT (固定)"
log "============================================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ----------------------------------------------------------------
# Step 0: 确保GPU1上的服务就绪
# ----------------------------------------------------------------
ensure_services_running

# 记录实验开始时间
START_TIME=$(date +%s)

# 结果记录
declare -a RESULTS

# ----------------------------------------------------------------
# 遍历所有权重比例
# ----------------------------------------------------------------
EXP_IDX=0
for judge_weight in "${JUDGE_WEIGHTS[@]}"; do
    # 计算asr_weight (总和为0.9，剩下的0.1给format)
    asr_weight=$(echo "0.9 - $judge_weight" | bc)
    
    EXP_IDX=$((EXP_IDX + 1))

    # 进度显示
    print_progress $EXP_IDX $TOTAL_EXPS
    log ""
    log "[$EXP_IDX/$TOTAL_EXPS] 实验: judge_weight=$judge_weight, asr_weight=$asr_weight"
    log "============================================================"

    # 创建实验输出目录
    EXP_OUTPUT="$OUTPUT_DIR/judge${judge_weight}_asr${asr_weight}"
    mkdir -p "$EXP_OUTPUT"

    # ----------------------------------------------------------------
    # Step 1: GRPO训练
    # ----------------------------------------------------------------
    log "[$EXP_IDX/$TOTAL_EXPS] 开始训练: judge=$judge_weight, asr=$asr_weight"

    python "$BASE_DIR/experiments/hybrid_reward_exp/hybrid_reward_grpo.py" \
        --experiment exp3 \
        --judge_prompt "$JUDGE_PROMPT" \
        --scoring_method "$SCORING_METHOD" \
        --train_data "$TRAIN_DATA" \
        --policy_model "$POLICY_MODEL" \
        --target_model "$TARGET_MODEL" \
        --guard_model "$GUARD_MODEL" \
        --target_judge_port "$TARGET_JUDGE_PORT" \
        --guard_port "$GUARD_PORT" \
        --asr_weight "$asr_weight" \
        --judge_weight "$judge_weight" \
        --format_weight "$FORMAT_WEIGHT" \
        --max_steps "$MAX_STEPS" \
        --learning_rate "$LEARNING_RATE" \
        --num_generations "$NUM_GENERATIONS" \
        --beta "$BETA" \
        --output_dir "$EXP_OUTPUT" \
        --run_name "exp3_judge${judge_weight}_asr${asr_weight}"

    # ----------------------------------------------------------------
    # Step 2: 训练完成后进行ASR评估
    # ----------------------------------------------------------------
    log "[$EXP_IDX/$TOTAL_EXPS] 开始评估: judge=$judge_weight, asr=$asr_weight"

    EVAL_OUTPUT="$EXP_OUTPUT/eval_results"
    mkdir -p "$EVAL_OUTPUT"

    # 使用eval脚本进行评估
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
        --run_name "eval_judge${judge_weight}_asr${asr_weight}"

    # 记录结果
    RESULTS+=("judge${judge_weight}_asr${asr_weight}")

    log "[$EXP_IDX/$TOTAL_EXPS] 完成: judge=$judge_weight, asr=$asr_weight"

done

# 记录实验结束时间
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

# 打印结果汇总
log "\n结果汇总 (按Judge Weight排序):"
for result in "${RESULTS[@]}"; do
    log "  - $result: 查看 $OUTPUT_DIR/$result/eval_results/"
done

log "\n提示: 可使用以下命令查看结果:"
log "  ls -la $OUTPUT_DIR/*/eval_results/"
log "  cat $OUTPUT_DIR/*/eval_results/*/summary.json | python -m json.tool"
