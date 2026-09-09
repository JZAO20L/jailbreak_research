#!/bin/bash
# =============================================================================
# Exp04: 10 轮 GRPO 训练 (M1 只GRPO / M3 RFT+GRPO 共用脚本)
# =============================================================================
#
# 10 轮实验矩阵主训练脚本 (见 docs/TODO.md 2026-09-01 节):
#   M1: MODEL 不设 (用 base),        bash scripts/exp04_grpo_10turn.sh
#   M3: MODEL=<rft_sft_conv merged>, MODEL_TAG=rft bash scripts/exp04_grpo_10turn.sh
#
# 关键点 (相对 exp03 的变更):
#   - max_turns 5 -> 10, num_generations 8 -> 16 (generation_batch_size=32)
#   - 显式 --loss_type grpo !! TRL 0.29 默认已是 dapo, 不设则 vanilla 对照被污染
#   - 300 步 (500 步边际收益低)
#   - 09-09 发散修复: 默认回 bf16 原生训练 (无 AMP/GradScaler, exp03 500 步稳定)。
#     fp16 下 GRPO 的 exp(ratio)/KL 项数值不稳定 (v5 首个非零奖励批 grad 270 ->
#     NaN, 09-09 记录), fp16+liger 仅作 10 轮长序列显存不足时的回退臂 (DTYPE=fp16)
#   - ⚠️ 10 轮序列 ~16k token, attention 显存随 seq² 膨胀 4x (相对 exp03):
#     per_device 2 + accum 4 首步 OOM 55.56GiB (09-07) → 改 per_device 1 +
#     accum 8 (有效 batch 不变) + gradient_checkpointing + expandable_segments
#     如仍 OOM 可再降 max_completion_length 2048->1024 (需同步评估口径)
#   - ⚠️ 二次 OOM 18.73GiB (09-08 step 2->3): 根因是 completion 段 ~15k token
#     全词表 logits 物化 (152k vocab, fwd+bwd ~19GB), 非 attention。swift 分块
#     路径 dynamic_num_samples 仅 per-turn 切分触发 → 用 liger-kernel
#     LigerFusedLinearGRPOLoss (fused linear+CE, 不物化 logits) 根治,
#     需 --use_liger_kernel true (V100 fp16 + triton 3.6 已验证可装)
#
# 前置 (训练用 4 卡布局: guard=GPU0/target=GPU1/rollout=GPU2/train=GPU3):
#   1) guard 8001 / target 8002 已就绪
#   2) swift rollout 服务就绪 (8004, GPU2), 模型 = $MODEL:
#      CUDA_VISIBLE_DEVICES=2 setsid swift rollout --model $MODEL \
#          --vllm_tensor_parallel_size 1 --port 8004 --vllm_max_model_len 8192 \
#          --vllm_gpu_memory_utilization 0.8 > output/logs/rollout_exp04.log 2>&1 &
#   3) MASTER_PORT 不要落在 29500-29522 段 (本机预占)
#
# DAPO 臂 (M4/M5, 暂缓): 打开 DAPO_FLAGS 即可
#   DAPO_FLAGS="--dynamic_sample true --max_resample_times 3 --epsilon_high 0.28"
#
# Usage:
#   bash scripts/exp04_grpo_10turn.sh                    # M1
#   MODEL=... MODEL_TAG=rft bash scripts/exp04_grpo_10turn.sh   # M3
# =============================================================================
NUM_GPUS=4

set -e
source "$(dirname "$0")/common.sh"

MODEL="${MODEL:-$BASE_MODEL}"
MODEL_TAG="${MODEL_TAG:-base}"
MAX_TURNS="${MAX_TURNS:-10}"   # 10=协议 5=恢复臂(09-09 plain-4B target + 5轮)
EXP_NAME="multi_turn_${MAX_TURNS}_agent_${MODEL_TAG}"
OUTPUT_SUBDIR="$OUTPUT_DIR/$EXP_NAME"

MAX_STEPS=300
LEARNING_RATE=1e-5
NUM_GENERATIONS=16
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
PLUGIN_PATH="$AGENTIC_DIR/src/plugin.py"
DAPO_FLAGS="${DAPO_FLAGS:-}"
# 09-09 发散修复: bf16 原生训练 (exp03 500 步稳定)。fp16+liger 仅 10 轮长序列
# 显存不足时回退 (DTYPE=fp16)。
DTYPE="${DTYPE:-bf16}"
if [ "$DTYPE" = "bf16" ]; then
    FP16_ARG="--fp16 false"
    BF16_ARG="--bf16 true"
    LIGER_ARG=""
else
    FP16_ARG="--fp16 true"
    BF16_ARG="--bf16 false"
    LIGER_ARG="--use_liger_kernel true"
fi

log_section "Exp04 GRPO: model_tag=$MODEL_TAG, turns=$MAX_TURNS, steps=$MAX_STEPS, num_gen=$NUM_GENERATIONS, dtype=$DTYPE"
log_info "初始权重: $MODEL"
log_info "输出: $OUTPUT_SUBDIR"

if [ "$MODEL" != "$BASE_MODEL" ] && [ ! -f "$MODEL/config.json" ]; then
    log_error "模型不存在: $MODEL"
    exit 1
fi
mkdir -p "$OUTPUT_SUBDIR"

# 服务检查 (guard/target 必需; rollout 8004 由外部 swift rollout 提供)
for p in $GUARD_PORT $TARGET_PORT; do
    if ! curl -s "http://127.0.0.1:$p/health" > /dev/null 2>&1; then
        log_error "服务 $p 未就绪 (start_servers_v100.sh)"
        exit 1
    fi
done
if ! curl -s "http://127.0.0.1:8004/health" > /dev/null 2>&1; then
    log_error "swift rollout (8004) 未就绪 —— 启动命令见本脚本头部注释"
    exit 1
fi

# 训练: GPU3 (4卡布局 TRAIN_GPUS=3)
TRAIN_GPUS="${TRAIN_GPUS:-3}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ROLLOUT_PORT=8004
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-43210}"  # 29500 段被系统预占, 避开

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    --dataset "$AGENTIC_DIR/output/grpo_data.jsonl" \
    --external_plugins "$PLUGIN_PATH" \
    --multi_turn_scheduler gym_scheduler \
    --gym_env jailbreak_env \
    --use_gym_env true \
    --max_turns $MAX_TURNS \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port $ROLLOUT_PORT \
    --vllm_server_timeout 1000 \
    --loss_type grpo \
    --per_device_train_batch_size $PER_DEVICE_BATCH \
    --generation_batch_size $((PER_DEVICE_BATCH * NUM_GENERATIONS)) \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_generations $NUM_GENERATIONS \
    --max_completion_length 2048 \
    $FP16_ARG \
    $BF16_ARG \
    --gradient_checkpointing true \
    $LIGER_ARG \
    --beta 0.05 \
    $DAPO_FLAGS \
    --output_dir "$OUTPUT_SUBDIR" \
    --report_to swanlab \
    --run_name "$EXP_NAME"

log_section "Exp04 完成 ($EXP_NAME)"
cat > "$OUTPUT_SUBDIR/summary.json" << EOF
{"experiment": "$EXP_NAME", "max_turns": $MAX_TURNS, "max_steps": $MAX_STEPS,
 "num_generations": $NUM_GENERATIONS, "loss_type": "grpo", "model": "$MODEL",
 "dtype": "$DTYPE", "dapo_flags": "$DAPO_FLAGS"}
EOF
log_info "下一步: merge LoRA -> vllm serve -> eval_conv.sh no_skill 1000 10 (单轨迹口径)"
