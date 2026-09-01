#!/bin/bash
# =============================================================================
# RFT 冷启动 SFT (conv 协议原生数据 v3) — E2 臂 / E3 初始权重
#
# 数据: run_rft_collect_conv.sh 产出的全轨迹样本 (conv 协议, 映射为真;
#       swift default loss 对所有 assistant 轮算 loss, 每个动作恰好监督一次)
# 对比 v2 (rft_sft.sh, beam 协议来源, 负结果): 数据来源换了, 训练侧也改为
#   1-2 epochs 防分布坍缩 (v2 负结果候选原因 2, 见 exp/README.md §10.1)
#
# 前置: 单卡空闲即可 (SFT 不需要 guard/target 服务)
# Usage:
#   bash scripts/rft_sft_conv.sh            # 默认 2 epochs
#   EPOCHS=1 bash scripts/rft_sft_conv.sh   # 1 epoch
# =============================================================================
set -e
source "$(dirname "$0")/common.sh"

EPOCHS="${EPOCHS:-2}"
MAX_TURNS="${MAX_TURNS:-10}"
DATA="${DATA:-$OUTPUT_DIR/rft_data_conv_${MAX_TURNS}turn.jsonl}"
EXP_DIR="$OUTPUT_DIR/rft_sft_conv${MAX_TURNS}turn_e${EPOCHS}"
TRAIN_GPU="${TRAIN_GPU:-3}"
mkdir -p "$EXP_DIR"

if [ ! -s "$DATA" ]; then
    log_error "SFT 数据不存在或为空: $DATA (先运行 run_rft_collect_conv.sh)"
    exit 1
fi

log_section "RFT SFT (conv v3): epochs=$EPOCHS, data=$DATA, GPU $TRAIN_GPU"

CUDA_VISIBLE_DEVICES=$TRAIN_GPU swift sft \
    --model "$BASE_MODEL" \
    --dataset "$DATA" \
    --tuner_type lora \
    --max_length 4096 \
    --num_train_epochs $EPOCHS \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --fp16 true \
    --bf16 false \
    --save_strategy epoch \
    --output_dir "$EXP_DIR" \
    --report_to none

log_section "RFT SFT 完成, 输出: $EXP_DIR"
log_info "下一步: merge LoRA -> 评估 E2 臂; 或 merge 后作为 E3 (RFT+GRPO@10) 初始权重"
