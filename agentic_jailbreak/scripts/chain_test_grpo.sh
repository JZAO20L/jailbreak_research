#!/bin/bash
# =============================================================================
# 小型 GRPO 训练链验证(冒烟测试)
# =============================================================================
#
# 验证 ms-swift 4.3.2 对话式 GRPO 全链路:
#   swift rollout(8003) ← 训练器 → gym env(jailbreak_env)→ target/guard
#
# 前置:conv 评估完成后,8003 已切换为 swift rollout
# Usage: bash scripts/chain_test_grpo.sh
# =============================================================================

# 4-GPU 机器:必须在 source common.sh 前声明(NUM_GPUS 决定 GPU 布局,默认 8)
NUM_GPUS=4

set -e

source "$(dirname "$0")/common.sh"

EXP_NAME="chain_test"
OUTPUT_SUBDIR="$OUTPUT_DIR/$EXP_NAME"
DATASET="$AGENTIC_DIR/output/grpo_chain_test.jsonl"
PLUGIN_PATH="$AGENTIC_DIR/src/plugin.py"

MAX_TURNS=3
MAX_STEPS=30
NUM_GENERATIONS=4
PER_DEVICE_BATCH=1
GRAD_ACCUM=1
TRAIN_GPUS="${TRAIN_GPUS:-3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

log_section "Chain test: 对话式 GRPO(冒烟)"
log_info "dataset: $DATASET (100 条)"
log_info "max_steps: $MAX_STEPS, turns: $MAX_TURNS, gen: $NUM_GENERATIONS"

# 检查 rollout(必须为 swift rollout 协议)
if ! curl -s "http://127.0.0.1:$ROLLOUT_PORT/health" > /dev/null 2>&1; then
    log_error "Rollout server (port $ROLLOUT_PORT) 未就绪"
    exit 1
fi

mkdir -p "$OUTPUT_SUBDIR"

# 失败的 trainer 可能卡死 rollout 服务(见 LOG 08-13)——每次训练前重启 rollout
restart_rollout() {
    log_info "重启 rollout server (port $ROLLOUT_PORT, GPU $ROLLOUT_GPUS)..."
    for pid in $(lsof -ti:$ROLLOUT_PORT 2>/dev/null); do kill -9 $pid 2>/dev/null || true; done
    pkill -9 -f "cli/rollout" 2>/dev/null || true
    sleep 3
    # 只清理 ROLLOUT GPU 上的残留进程(避免误杀 guard/target 引擎)
    for pid in $(nvidia-smi -i $ROLLOUT_GPUS --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        kill -9 $pid 2>/dev/null || true
    done
    sleep 4
    # 等待 ROLLOUT GPU 显存释放(最多 60s)
    for i in $(seq 1 12); do
        mem=$(nvidia-smi -i $ROLLOUT_GPUS --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ -z "$mem" ] || [ "$mem" -lt 1024 ]; then break; fi
        sleep 5
    done
    CUDA_VISIBLE_DEVICES=$ROLLOUT_GPUS $PROJECT_ROOT/.venv/bin/swift rollout \
        --model "$BASE_MODEL" --vllm_tensor_parallel_size 1 \
        --port $ROLLOUT_PORT --vllm_max_model_len 16384 \
        --vllm_gpu_memory_utilization 0.9 \
        > "$OUTPUT_DIR/logs/rollout_swift.log" 2>&1 &
    for i in $(seq 1 60); do
        if curl -s -m 3 "http://127.0.0.1:$ROLLOUT_PORT/health" > /dev/null 2>&1; then
            log_info "rollout 就绪"
            return 0
        fi
        sleep 5
    done
    log_error "rollout 启动超时"
    return 1
}

# NCCL init 在本机存在间歇性竞态(见 LOG 08-13),失败自动重试
for attempt in 1 2 3; do
    # rollout 健康则复用(避免无谓重启触发引擎崩溃);不健康才重启
    if ! curl -s -m 5 "http://127.0.0.1:$ROLLOUT_PORT/health" > /dev/null 2>&1; then
        restart_rollout || exit 1
    fi
    log_info "训练尝试 $attempt/3"
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS $PYTHON_BIN "$PROJECT_ROOT/.venv/lib/python3.12/site-packages/swift/cli/rlhf.py" \
        --rlhf_type grpo \
        --model "$BASE_MODEL" \
        --dataset "$DATASET" \
        --external_plugins "$PLUGIN_PATH" \
        --multi_turn_scheduler gym_scheduler \
        --gym_env jailbreak_env \
        --use_gym_env true \
        --max_turns $MAX_TURNS \
        --use_vllm true \
        --vllm_mode server \
        --vllm_server_host 127.0.0.1 \
        --vllm_server_port $ROLLOUT_PORT \
        --vllm_server_timeout 600 \
        --per_device_train_batch_size $PER_DEVICE_BATCH \
        --generation_batch_size $((PER_DEVICE_BATCH * NUM_GENERATIONS)) \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --max_steps $MAX_STEPS \
        --learning_rate 1e-5 \
        --num_generations $NUM_GENERATIONS \
        --max_completion_length 2048 \
        --bf16 true \
        --beta 0.05 \
        --output_dir "$OUTPUT_SUBDIR" \
        --report_to none \
        --run_name "$EXP_NAME" > "$OUTPUT_SUBDIR/train_attempt$attempt.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        log_info "训练成功(attempt $attempt)"
        break
    fi
    if ! grep -qE "no CUDA-capable|unhandled cuda" "$OUTPUT_SUBDIR/train_attempt$attempt.log"; then
        log_error "非 NCCL 错误,放弃重试"
        break
    fi
    log_info "NCCL 竞态,重试..."
    sleep 10
done

log_section "Chain test 完成"
log_info "Checkpoint: $OUTPUT_SUBDIR"