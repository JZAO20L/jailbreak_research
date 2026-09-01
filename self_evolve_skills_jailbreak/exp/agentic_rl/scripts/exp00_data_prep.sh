#!/bin/bash
# =============================================================================
# Exp00: 数据准备
# =============================================================================
#
# 三步：
#   1. 为 54 个 skills 生成 description
#   2. 生成 RFT 轨迹（基础模型 → N 候选 → 评估 → 过滤）
#   3. 构建训练数据（ChatML 格式）
#
# 数据划分（避免数据泄露）：
#   [0:1000]    SESS 已用（排除）
#   [1000:4000] RFT 训练数据（3000 条）
#   [4000:8000] GRPO 训练数据（4000 条）
#
# 需要：1 张 GPU（policy server）+ 1 张 GPU（target server）+ 1 张 GPU（guard server）
#
# Usage:
#   bash exp00_data_prep.sh                        # 默认：RFT 数据 [1000:4000]
#   bash exp00_data_prep.sh --start 4000 --end 8000  # GRPO 数据
#   bash exp00_data_prep.sh --max_samples 500         # 快速验证
#
# =============================================================================

set -e

# 加载共享函数
source "$(dirname "$0")/common.sh"

# =============================================================================
# 参数解析
# =============================================================================

# 数据范围（默认：RFT 数据）
DATA_START="${DATA_START:-1000}"
DATA_END="${DATA_END:-4000}"
MAX_SAMPLES="${MAX_SAMPLES:-3000}"
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
TOP_K_SKILLS="${TOP_K_SKILLS:-5}"
TOP_M_PER_PROMPT="${TOP_M_PER_PROMPT:-2}"
MIN_ASR="${MIN_ASR:-1.0}"

# 输出子目录后缀（用于区分 RFT/GRPO 数据）
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-rft}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --start) DATA_START="$2"; shift 2;;
        --end) DATA_END="$2"; shift 2;;
        --max_samples) MAX_SAMPLES="$2"; shift 2;;
        --num_candidates) NUM_CANDIDATES="$2"; shift 2;;
        --top_k) TOP_K_SKILLS="$2"; shift 2;;
        --suffix) OUTPUT_SUFFIX="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# 计算实际样本数
ACTUAL_SAMPLES=$((DATA_END - DATA_START))
if [ "$MAX_SAMPLES" -gt "$ACTUAL_SAMPLES" ]; then
    MAX_SAMPLES=$ACTUAL_SAMPLES
fi

# =============================================================================
# 输出目录
# =============================================================================

DESC_DIR="$OUTPUT_DIR/descriptions"
TRAJ_DIR="$OUTPUT_DIR/trajectories_${OUTPUT_SUFFIX}"
DATA_DIR="$OUTPUT_DIR/${OUTPUT_SUFFIX}_data"

mkdir -p "$DESC_DIR" "$TRAJ_DIR" "$DATA_DIR"

log_section "Exp00: 数据准备 ($OUTPUT_SUFFIX)"
log_info "Data range: [$DATA_START:$DATA_END] ($MAX_SAMPLES samples)"
log_info "Num candidates: $NUM_CANDIDATES"
log_info "Top-k skills: $TOP_K_SKILLS"
log_info "Skill library: $SKILL_LIBRARY"

# =============================================================================
# Step 1: 生成 Skill Descriptions
# =============================================================================

log_section "Step 1: 生成 Skill Descriptions"

if [ -f "$DESC_DIR/skill_descriptions.json" ]; then
    log_info "Descriptions 已存在，跳过生成"
else
    # 需要启动一个 policy server 来生成 descriptions
    log_info "启动 policy server 用于 description 生成..."
    
    cleanup_gpus
    
    # 使用 GPU 0 启动 policy server（用于生成 descriptions）
    CUDA_VISIBLE_DEVICES=0 vllm serve "$BASE_MODEL" \
        --port $POLICY_PORT \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --dtype auto \
        > "$OUTPUT_DIR/logs/policy_desc.log" 2>&1 &
    
    POLICY_PID=$!
    wait_for_server $POLICY_PORT "Policy"
    
    # 生成 descriptions
    python "$AGENTIC_RL_DIR/src/data_prep.py" \
        --step descriptions \
        --skill_path "$SKILL_LIBRARY" \
        --output_dir "$DESC_DIR" \
        --port $POLICY_PORT
    
    # 停止 policy server
    kill $POLICY_PID 2>/dev/null || true
    sleep 3
fi

log_info "Descriptions 已保存到: $DESC_DIR/skill_descriptions.json"

# =============================================================================
# Step 2: 生成 RFT 轨迹
# =============================================================================

log_section "Step 2: 生成 RFT 轨迹"

if [ -f "$TRAJ_DIR/rft_trajectories.jsonl" ]; then
    log_info "轨迹已存在，跳过生成"
else
    # 需要三个 server: policy, target, guard
    cleanup_gpus
    
    # 启动 servers（使用 3 张 GPU）
    CUDA_VISIBLE_DEVICES=0 vllm serve "$GUARD_MODEL" \
        --port $GUARD_PORT \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --dtype auto \
        > "$OUTPUT_DIR/logs/guard_traj.log" 2>&1 &
    GUARD_PID=$!
    
    CUDA_VISIBLE_DEVICES=1 vllm serve "$TARGET_MODEL" \
        --port $TARGET_PORT \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --dtype auto \
        > "$OUTPUT_DIR/logs/target_traj.log" 2>&1 &
    TARGET_PID=$!
    
    CUDA_VISIBLE_DEVICES=2 vllm serve "$BASE_MODEL" \
        --port $POLICY_PORT \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --dtype auto \
        > "$OUTPUT_DIR/logs/policy_traj.log" 2>&1 &
    POLICY_PID=$!
    
    wait_for_server $GUARD_PORT "Guard"
    wait_for_server $TARGET_PORT "Target"
    wait_for_server $POLICY_PORT "Policy"
    
    # 生成轨迹
    python "$AGENTIC_RL_DIR/src/data_prep.py" \
        --step trajectories \
        --skill_path "$SKILL_LIBRARY" \
        --description_path "$DESC_DIR/skill_descriptions.json" \
        --train_data "$TRAIN_DATA" \
        --output_dir "$TRAJ_DIR" \
        --policy_port $POLICY_PORT \
        --target_port $TARGET_PORT \
        --guard_port $GUARD_PORT \
        --num_candidates $NUM_CANDIDATES \
        --top_k_skills $TOP_K_SKILLS \
        --max_samples $MAX_SAMPLES \
        --min_asr $MIN_ASR \
        --top_m_per_prompt $TOP_M_PER_PROMPT \
        --data_start $DATA_START \
        --data_end $DATA_END
    
    # 停止 servers
    kill $GUARD_PID $TARGET_PID $POLICY_PID 2>/dev/null || true
    sleep 3
fi

log_info "轨迹已保存到: $TRAJ_DIR/rft_trajectories.jsonl"

# =============================================================================
# Step 3: 构建训练数据
# =============================================================================

log_section "Step 3: 构建训练数据"

python "$AGENTIC_RL_DIR/src/data_prep.py" \
    --step build \
    --trajectory_path "$TRAJ_DIR/rft_trajectories.jsonl" \
    --output_dir "$DATA_DIR"

log_info "训练数据已保存到: $DATA_DIR/"

# =============================================================================
# 统计
# =============================================================================

log_section "数据准备完成"

if [ -f "$DATA_DIR/data_stats.json" ]; then
    log_info "数据统计:"
    cat "$DATA_DIR/data_stats.json" | python -m json.tool
fi

log_info ""
log_info "输出文件:"
log_info "  Descriptions: $DESC_DIR/skill_descriptions.json"
log_info "  Trajectories: $TRAJ_DIR/rft_trajectories.jsonl"
log_info "  Train data:   $DATA_DIR/rft_train.jsonl"
log_info "  Val data:     $DATA_DIR/rft_val.jsonl"
log_info ""
log_info "下一步: bash exp01_rft_train.sh"
