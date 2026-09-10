#!/bin/bash
# =============================================================================
# A800 服务布局 (2026-09-09 起, 4×A800-SXM4-80GB, 每卡一模型)
#   GPU0: Guard (Qwen3Guard-Gen-4B, port 8001)
#   GPU1: Target (Qwen3-4B plain, port 8002)      # 09-09 起 target 弃用 SafeRL
#   GPU2: Policy (Qwen3-4B, port 8003)            # GRPO 时换成 swift rollout 8004
#   GPU3: 预留训练卡 (SFT / GRPO)
#
# 相对 start_servers_v100.sh 的差异 (硬件换了, 不是协议换了):
#   - --dtype bfloat16: A800 (SM80) 原生支持 bf16。历史 V100 被迫 float16 是
#     09-09 GRPO fp16 发散(grad 270→NaN)的根因; 训练侧已定 bf16 原生,
#     推理服务同 dtype 才能保证 rollout 与 trainer 的 logprob 语义一致
#   - guard/target max-model-len 8192 → 16384: 10 轮攻击 prompt + plain-4B 长回复
#     在 8192 下会被静默截断(截断即改变 Guard 判定), 80GB 下放大无成本
#   - policy max-model-len 24576 → 32768: 沿用 C2 事故结论(10 轮全量累积 + 2048
#     生成会在 24576 上差 1 token 越界崩溃)
#   - 无需 FLASHINFER/FA2 相关妥协, 无需 xformers patch(SM<8 才生效)
#
# Usage: NUM_GPUS=4 bash scripts/start_servers_a800.sh
# =============================================================================
set -e
NUM_GPUS="${NUM_GPUS:-4}"
source "$(dirname "$0")/common.sh"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
VLLM_BIN="$PROJECT_ROOT/.venv/bin/vllm"
DTYPE="${DTYPE:-bfloat16}"
# 打包: guard+target 同住一卡时设 SVC_UTIL=0.42; 不需要评估用 policy 服务时 START_POLICY=0
SVC_UTIL="${SVC_UTIL:-0.9}"
START_POLICY="${START_POLICY:-1}"

log_section "启动 A800 服务 (guard=GPU$GUARD_GPU, target=GPU$TARGET_GPU, policy=GPU$ROLLOUT_GPUS, dtype=$DTYPE, util=$SVC_UTIL)"
log_info "guard 模型 : $GUARD_MODEL"
log_info "target 模型: $TARGET_MODEL"
log_info "policy 模型: $BASE_MODEL"

CUDA_VISIBLE_DEVICES=$GUARD_GPU "$VLLM_BIN" serve "$GUARD_MODEL" \
    --port $GUARD_PORT --max-model-len 16384 --gpu-memory-utilization $SVC_UTIL \
    --dtype $DTYPE --trust-remote-code \
    > "$LOG_DIR/guard_a800.log" 2>&1 &
log_info "guard: GPU$GUARD_GPU port $GUARD_PORT"
wait_for_server $GUARD_PORT "Guard" 1500

CUDA_VISIBLE_DEVICES=$TARGET_GPU "$VLLM_BIN" serve "$TARGET_MODEL" \
    --port $TARGET_PORT --max-model-len 16384 --gpu-memory-utilization $SVC_UTIL \
    --dtype $DTYPE --trust-remote-code \
    > "$LOG_DIR/target_a800.log" 2>&1 &
log_info "target: GPU$TARGET_GPU port $TARGET_PORT"
wait_for_server $TARGET_PORT "Target" 1500

if [ "$START_POLICY" = "1" ]; then
CUDA_VISIBLE_DEVICES=$ROLLOUT_GPUS "$VLLM_BIN" serve "$BASE_MODEL" \
    --port $ROLLOUT_PORT --max-model-len 32768 --gpu-memory-utilization $SVC_UTIL \
    --dtype $DTYPE --trust-remote-code \
    > "$LOG_DIR/policy_a800.log" 2>&1 &
log_info "policy: GPU$ROLLOUT_GPUS port $ROLLOUT_PORT"
wait_for_server $ROLLOUT_PORT "Policy" 1500
else
    log_info "跳过 policy 服务 (START_POLICY=0)"
fi

log_section "A800 服务就绪"
log_info "guard=$GUARD_PORT(GPU$GUARD_GPU) target=$TARGET_PORT(GPU$TARGET_GPU) policy=$ROLLOUT_PORT(GPU$ROLLOUT_GPUS) 训练卡=GPU$TRAIN_GPUS"
log_info "skills=$SKILLS_PATH"
