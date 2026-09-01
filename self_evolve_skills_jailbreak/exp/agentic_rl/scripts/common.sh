#!/bin/bash
# =============================================================================
# Agentic-RL 实验共享函数
# =============================================================================
#
# 所有实验脚本通过 source common.sh 使用这些函数
#
# =============================================================================

set -e

# =============================================================================
# 路径配置
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="/home/tiger/jailbreak_research"
SESS_DIR="$PROJECT_ROOT/self_evolve_skills_jailbreak"
AGENTIC_RL_DIR="$SESS_DIR/agentic_rl"

# =============================================================================
# 默认配置（可被实验脚本覆盖）
# =============================================================================

# GPU 配置
NUM_GPUS="${NUM_GPUS:-4}"

# 模型路径
BASE_MODEL="${BASE_MODEL:-/home/tiger/models/Qwen/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/home/tiger/models/Qwen/Qwen3Guard-Gen-4B}"
TARGET_MODEL="${TARGET_MODEL:-/home/tiger/models/Qwen/Qwen3-4B}"

# 端口
GUARD_PORT="${GUARD_PORT:-8001}"
TARGET_PORT="${TARGET_PORT:-8002}"
POLICY_PORT="${POLICY_PORT:-8003}"

# 根据 GPU 数量自动分配
if [ "$NUM_GPUS" -eq 4 ]; then
    GUARD_GPU="${GUARD_GPU:-0}"
    TARGET_GPU="${TARGET_GPU:-1}"
    TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
    TRAIN_TP=2
elif [ "$NUM_GPUS" -eq 8 ]; then
    GUARD_GPU="${GUARD_GPU:-0}"
    TARGET_GPU="${TARGET_GPU:-1}"
    TRAIN_GPUS="${TRAIN_GPUS:-2,3,4,5,6,7}"
    TRAIN_TP=4  # 保守选择，TP=6 也可以尝试
else
    echo "ERROR: NUM_GPUS must be 4 or 8, got $NUM_GPUS"
    exit 1
fi

# 数据路径
SKILL_LIBRARY="${SKILL_LIBRARY:-$SESS_DIR/exp/layer4/results/skills/skills_dan_data_medium_evo.json}"
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/dataset/processed/10k/train.jsonl}"

# RFT 和 GRPO 使用不同的数据范围（避免数据泄露）
# [0:1000]    SESS 已用（排除）
# [1000:4000] RFT 训练数据（3000 条）
# [4000:8000] GRPO 训练数据（4000 条）
RFT_DATA_START="${RFT_DATA_START:-1000}"
RFT_DATA_END="${RFT_DATA_END:-4000}"
GRPO_DATA_START="${GRPO_DATA_START:-4000}"
GRPO_DATA_END="${GRPO_DATA_END:-8000}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-$EXP_DIR/output}"

# =============================================================================
# 日志函数
# =============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_section() {
    echo ""
    echo "============================================================================="
    echo "$*"
    echo "============================================================================="
    echo ""
}

# =============================================================================
# GPU 清理
# =============================================================================

cleanup_gpus() {
    log_section "清理 GPU 进程"
    
    # 杀掉 vLLM 进程
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "vllm.api_server" 2>/dev/null || true
    sleep 2
    
    # 清理特定端口
    for port in 8001 8002 8003; do
        pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill -9 $pid 2>/dev/null || true
            log_info "清理端口 $port: PID $pid"
        fi
    done
    
    sleep 5
    log_info "GPU 状态:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "  (nvidia-smi 不可用)"
}

# =============================================================================
# vLLM Server 管理
# =============================================================================

start_guard_server() {
    local gpu="${1:-$GUARD_GPU}"
    local port="${2:-$GUARD_PORT}"
    local log_dir="${3:-$OUTPUT_DIR/logs}"
    
    mkdir -p "$log_dir"
    
    log_info "启动 Guard server: GPU $gpu, port $port"
    
    CUDA_VISIBLE_DEVICES=$gpu vllm serve "$GUARD_MODEL" \
        --port $port \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --dtype auto \
        > "$log_dir/guard.log" 2>&1 &
    
    GUARD_PID=$!
    wait_for_server $port "Guard"
}

start_target_server() {
    local gpu="${1:-$TARGET_GPU}"
    local port="${2:-$TARGET_PORT}"
    local model="${3:-$TARGET_MODEL}"
    local log_dir="${4:-$OUTPUT_DIR/logs}"
    
    mkdir -p "$log_dir"
    
    # 计算 TP
    local num_gpus=$(echo "$gpu" | tr ',' '\n' | wc -l)
    local tp=$num_gpus
    
    log_info "启动 Target server: GPU $gpu, port $port, TP=$tp, model=$model"
    
    CUDA_VISIBLE_DEVICES=$gpu vllm serve "$model" \
        --port $port \
        --tensor-parallel-size $tp \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --dtype auto \
        > "$log_dir/target.log" 2>&1 &
    
    TARGET_PID=$!
    wait_for_server $port "Target"
}

start_policy_server() {
    local gpu="${1:-$TRAIN_GPUS}"
    local port="${2:-$POLICY_PORT}"
    local model="${3:-$BASE_MODEL}"
    local lora_path="${4:-}"
    local log_dir="${5:-$OUTPUT_DIR/logs}"
    
    mkdir -p "$log_dir"
    
    local cmd="CUDA_VISIBLE_DEVICES=$gpu vllm serve $model --port $port --max-model-len 8192 --gpu-memory-utilization 0.9 --trust-remote-code --dtype auto"
    
    if [ -n "$lora_path" ] && [ -d "$lora_path" ]; then
        log_info "启动 Policy server (with LoRA): GPU $gpu, port $port, lora=$lora_path"
        cmd="$cmd --enable-lora --lora-module-names q_proj v_proj k_proj o_proj gate_proj up_proj down_proj"
    else
        log_info "启动 Policy server: GPU $gpu, port $port"
    fi
    
    eval $cmd > "$log_dir/policy.log" 2>&1 &
    
    POLICY_PID=$!
    wait_for_server $port "Policy"
}

wait_for_server() {
    local port=$1
    local name=$2
    local timeout=${3:-600}
    
    log_info "等待 $name server (port $port) 启动..."
    
    local start_time=$(date +%s)
    while true; do
        if curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            log_info "$name server 已就绪 (port $port)"
            return 0
        fi
        
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $timeout ]; then
            log_error "$name server 启动超时 (${timeout}s)"
            return 1
        fi
        
        sleep 5
    done
}

stop_all_servers() {
    log_section "停止所有 servers"
    
    [ -n "$GUARD_PID" ] && kill $GUARD_PID 2>/dev/null || true
    [ -n "$TARGET_PID" ] && kill $TARGET_PID 2>/dev/null || true
    [ -n "$POLICY_PID" ] && kill $POLICY_PID 2>/dev/null || true
    
    sleep 3
    
    # 强制清理
    for port in 8001 8002 8003; do
        pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill -9 $pid 2>/dev/null || true
        fi
    done
    
    log_info "所有 servers 已停止"
}

# =============================================================================
# 评估函数
# =============================================================================

eval_on_benchmark() {
    local benchmark=$1
    local lora_path=$2
    local output_subdir=$3
    local policy_port=${4:-$POLICY_PORT}
    local target_port=${5:-$TARGET_PORT}
    local guard_port=${6:-$GUARD_PORT}
    
    # 确定数据路径
    local data_path
    case $benchmark in
        default)
            data_path="$SESS_DIR/data/test_prompts.json"
            ;;
        advbench|harmbench_standard|harmbench_contextual|jailbreakBench)
            data_path="$PROJECT_ROOT/data/benchmark/${benchmark}.jsonl"
            ;;
        *)
            log_error "Unknown benchmark: $benchmark"
            return 1
            ;;
    esac
    
    log_info "评估 $benchmark: $data_path -> $output_subdir"
    
    mkdir -p "$output_subdir"
    
    python "$AGENTIC_RL_DIR/src/eval.py" \
        --model_path "$BASE_MODEL" \
        --lora_path "$lora_path" \
        --skill_library_path "$SKILL_LIBRARY" \
        --description_path "$OUTPUT_DIR/descriptions/skill_descriptions.json" \
        --test_data "$data_path" \
        --policy_port $policy_port \
        --target_port $target_port \
        --guard_port $guard_port \
        --output_dir "$output_subdir"
}

eval_all_benchmarks() {
    local lora_path=$1
    local output_base=$2
    
    local benchmarks=("default" "advbench" "harmbench_standard" "harmbench_contextual" "jailbreakBench")
    
    for benchmark in "${benchmarks[@]}"; do
        eval_on_benchmark "$benchmark" "$lora_path" "$output_base/$benchmark"
    done
    
    log_info "所有 benchmark 评估完成: $output_base"
}

# =============================================================================
# 训练函数
# =============================================================================

run_rft_training() {
    local train_data=$1
    local eval_data=$2
    local output_dir=$3
    
    log_section "RFT 训练"
    log_info "Train data: $train_data"
    log_info "Output: $output_dir"
    
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
        --num_processes $(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l) \
        "$AGENTIC_RL_DIR/src/sft_train.py" \
        --train_data_path "$train_data" \
        --eval_data_path "$eval_data" \
        --model_name_or_path "$BASE_MODEL" \
        --output_dir "$output_dir" \
        --num_epochs 3 \
        --learning_rate 2e-5 \
        --lora_r 16 \
        --lora_alpha 32 \
        --max_seq_length 2048 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --packing \
        --bf16
}

run_grpo_training() {
    local rft_model=$1
    local output_dir=$2
    local reward_type=$3  # asr, fixed, ema, ahr
    local extra_args="${4:-}"
    
    log_section "GRPO 训练: reward_type=$reward_type"
    log_info "RFT model: $rft_model"
    log_info "Output: $output_dir"
    log_info "GRPO data range: [$GRPO_DATA_START:$GRPO_DATA_END]"
    
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
        --num_processes $(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l) \
        "$AGENTIC_RL_DIR/src/grpo_train.py" \
        --rft_model_path "$rft_model" \
        --base_model_path "$BASE_MODEL" \
        --skill_library_path "$SKILL_LIBRARY" \
        --description_path "$OUTPUT_DIR/descriptions/skill_descriptions.json" \
        --output_dir "$output_dir" \
        --target_port $TARGET_PORT \
        --guard_port $GUARD_PORT \
        --reward_type "$reward_type" \
        --data_start $GRPO_DATA_START \
        --data_end $GRPO_DATA_END \
        --max_steps 500 \
        --learning_rate 1e-5 \
        --num_generations 8 \
        --max_completion_len 1024 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --vllm_tensor_parallel_size $TRAIN_TP \
        $extra_args
}

# =============================================================================
# 实验完成
# =============================================================================

experiment_complete() {
    local exp_name=$1
    local output_dir=$2
    
    log_section "实验完成: $exp_name"
    log_info "输出目录: $output_dir"
    
    if [ -f "$output_dir/summary.json" ]; then
        log_info "结果摘要:"
        cat "$output_dir/summary.json" | python -m json.tool 2>/dev/null || cat "$output_dir/summary.json"
    fi
}
