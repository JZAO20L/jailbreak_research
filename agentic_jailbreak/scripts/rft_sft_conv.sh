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
# 6144 是按 09-04 那批 82 条(SafeRL target)实测定的; 换 target 后成功轨迹数与
# 长度分布都会变, 重采后务必重新量 p50/max 再定, 否则会静默截断长轨迹
MAX_LENGTH="${MAX_LENGTH:-6144}"
# 数值口径统一: 全项目已走 bf16 原生(服务 bf16 / GRPO DTYPE=bf16)。fp16 是 V100 无
# bf16 计算时的被迫选择, 也是 09-09 GRPO 发散的根因; SFT 权重是 M3 的初始权重,
# 不该再用另一套数值。要复现 V100 老口径时设 DTYPE=fp16。
DTYPE="${DTYPE:-bf16}"
# 有效 batch 固定 8 = PER_DEVICE × ACCUM (历史 V100 用 1×8)。A800-80GB 提 per_device
# 只是少几轮累积, 不改配方; 动这两个值请记录, 否则与历史 M2 不可比。
PER_DEVICE="${PER_DEVICE:-4}"
ACCUM="${ACCUM:-2}"
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
    --max_length $MAX_LENGTH \
    --num_train_epochs $EPOCHS \
    --learning_rate 1e-5 \
    --per_device_train_batch_size $PER_DEVICE \
    --gradient_accumulation_steps $ACCUM \
    --gradient_checkpointing true \
    --fp16 $([ "$DTYPE" = fp16 ] && echo true || echo false) \
    --bf16 $([ "$DTYPE" = fp16 ] && echo false || echo true) \
    --save_strategy epoch \
    --output_dir "$EXP_DIR" \
    --report_to none

log_section "RFT SFT 完成, 输出: $EXP_DIR"
log_info "下一步: merge LoRA -> 评估 E2 臂; 或 merge 后作为 E3 (RFT+GRPO@10) 初始权重"
