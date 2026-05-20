#!/bin/bash
# 实验3 Baseline评估脚本 - base model + 3个最佳prompt组合
#
# 目的: 在实验3环境下补充baseline，用于与训练后的9个LoRA模型公平对比
# 评估: 未训练模型(base model)分别用3个最佳攻击策略重写prompt后的ASR
#
# 用法:
#   bash experiments/adaptive_hybrid_reward_exp/eval_baseline.sh              # 评估全部
#   bash experiments/adaptive_hybrid_reward_exp/eval_baseline.sh --reset     # 清空checkpoint重跑

set -e

# =========================
# 配置区（与实验3的exp.sh统一）
# =========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 评估数据 - test.jsonl（与exp.sh一致）
EVAL_DATA="${EVAL_DATA:-$BASE_DIR/../data/dataset/processed/10k/test.jsonl}"

# 输出目录 - 与exp.sh一致
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"

# 模型路径
POLICY_MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
TARGET_MODEL="${TARGET_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B}"

# 端口
POLICY_PORT=8003
TARGET_PORT=8001
GUARD_PORT=8002

# vLLM配置
VLLM_MAX_MODEL_LEN_POLICY=4096
VLLM_GPU_UTIL_POLICY=0.9
VLLM_MAX_MODEL_LEN_TARGET=8192
VLLM_GPU_UTIL_TARGET=0.4
VLLM_MAX_MODEL_LEN_GUARD=8192
VLLM_GPU_UTIL_GUARD=0.4

# 3个最佳prompt组合（来自实验2，与exp.sh一致）
COMBINATIONS=(
    "role_playing:role_playing"
    "hypothetical_scenario:naturalness"
    "creative_writing:stealthiness"
)

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
    local timeout=120
    log "Waiting for port $port..."
    for i in $(seq 1 $timeout); do
        if check_port_active "$port"; then
            log "Port $port is active!"
            return 0
        fi
        sleep 1
    done
    log "ERROR: Port $port timeout!"
    return 1
}

start_target_guard() {
    log "Starting Target model (port: $TARGET_PORT)..."
    if ! check_port_active "$TARGET_PORT"; then
        CUDA_VISIBLE_DEVICES=1 nohup vllm serve "$TARGET_MODEL" \
            --host 127.0.0.1 --port $TARGET_PORT \
            --max-model-len $VLLM_MAX_MODEL_LEN_TARGET \
            --gpu-memory-utilization $VLLM_GPU_UTIL_TARGET \
            --served-model-name target \
            > "$OUTPUT_DIR/target_vllm.log" 2>&1 &
        wait_for_port "$TARGET_PORT"
    else
        log "Target service already running on port $TARGET_PORT"
    fi

    log "Starting Guard model (port: $GUARD_PORT)..."
    if ! check_port_active "$GUARD_PORT"; then
        CUDA_VISIBLE_DEVICES=1 nohup vllm serve "$GUARD_MODEL" \
            --host 127.0.0.1 --port $GUARD_PORT \
            --max-model-len $VLLM_MAX_MODEL_LEN_GUARD \
            --gpu-memory-utilization $VLLM_GPU_UTIL_GUARD \
            --served-model-name guard \
            > "$OUTPUT_DIR/guard_vllm.log" 2>&1 &
        wait_for_port "$GUARD_PORT"
    else
        log "Guard service already running on port $GUARD_PORT"
    fi
}

start_policy() {
    log "Starting Policy (base model, no LoRA) on port $POLICY_PORT..."
    pkill -f "vllm.*$POLICY_PORT" 2>/dev/null || true
    sleep 3

    CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$POLICY_MODEL" \
        --host 127.0.0.1 --port $POLICY_PORT \
        --max-model-len $VLLM_MAX_MODEL_LEN_POLICY \
        --gpu-memory-utilization $VLLM_GPU_UTIL_POLICY \
        --served-model-name policy \
        > "$OUTPUT_DIR/policy_vllm.log" 2>&1 &
    wait_for_port "$POLICY_PORT"
}

stop_policy() {
    log "Stopping Policy service..."
    pkill -f "vllm.*$POLICY_PORT" 2>/dev/null || true
    sleep 3
}

# =========================
# 主流程
# =========================
TOTAL_EXPS=${#COMBINATIONS[@]}

log "============================================================"
log "Experiment 3 Baseline Evaluation"
log "Base model + 3 best prompt combinations (from Experiment 2)"
log "============================================================"
log "Evaluation data: $EVAL_DATA"
log "Output directory: $OUTPUT_DIR"
log "Model: $POLICY_MODEL (base model, no LoRA)"
log "Combinations:"
for combo in "${COMBINATIONS[@]}"; do
    log "  - $combo"
done
log "Total evaluations: $TOTAL_EXPS"
log "============================================================"

mkdir -p "$OUTPUT_DIR"

# 启动Target + Guard（只需一次）
start_target_guard

# 对每个组合进行评估
for combination in "${COMBINATIONS[@]}"; do
    attack_prompt="${combination%%:*}"
    judge_prompt="${combination##*:}"
    exp_key="baseline_${attack_prompt}_${judge_prompt}"
    eval_output="$OUTPUT_DIR/${exp_key}/eval_results"

    log ""
    log "============================================================"
    log "Evaluating: $exp_key"
    log "Attack: $attack_prompt, Judge: $judge_prompt"
    log "============================================================"

    mkdir -p "$eval_output"

    # 启动Policy（base model，不带LoRA）
    start_policy

    # 运行评估 - 与exp.sh的eval调用方式一致
    python "$BASE_DIR/scripts/eval.py" \
        --eval_path "$EVAL_DATA" \
        --base_model_path "$POLICY_MODEL" \
        --target_model_path "$TARGET_MODEL" \
        --guard_model_path "$GUARD_MODEL" \
        --policy_port "$POLICY_PORT" \
        --target_port "$TARGET_PORT" \
        --guard_port "$GUARD_PORT" \
        --output_root "$eval_output" \
        --run_name "eval_${exp_key}"

    # 关闭Policy
    stop_policy

    log "Complete: $exp_key"
done

log ""
log "============================================================"
log "All baseline evaluations complete!"
log "Results in: $OUTPUT_DIR"
log "============================================================"
