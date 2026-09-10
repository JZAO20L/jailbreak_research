#!/bin/bash
# =============================================================================
# Merge RFT LoRA -> 完整模型 rft_sft_merged (供 GRPO 复训/评估使用)
# Usage: bash scripts/merge_rft_lora.sh
# =============================================================================
set -e
source "$(dirname "$0")/common.sh"

# 默认沿用 08-26 v2 那版路径; 新 conv 版用 SFT_DIR / MERGED_DIR 覆盖
SFT_DIR="${SFT_DIR:-$AGENTIC_DIR/output/rft_sft_train1000_v2}"
MERGED_DIR="${MERGED_DIR:-$AGENTIC_DIR/output/rft_sft_train1000_v2_merged}"

# 找最新的 adapter checkpoint 目录 (swift sft 输出: vxxx-<ts>/checkpoint-xxx)
ADAPTER=$(ls -dt "$SFT_DIR"/v*/checkpoint-* 2>/dev/null | head -1)
if [ -z "$ADAPTER" ]; then
    ADAPTER=$(ls -dt "$SFT_DIR"/checkpoint-* 2>/dev/null | head -1)
fi
[ -z "$ADAPTER" ] && { log_error "未找到 SFT adapter: $SFT_DIR"; exit 1; }

log_section "合并 LoRA: $ADAPTER -> $MERGED_DIR"
CUDA_VISIBLE_DEVICES="${MERGE_GPU:-3}" swift export \
    --model "$BASE_MODEL" \
    --adapters "$ADAPTER" \
    --merge_lora true \
    --output_dir "$MERGED_DIR" \
    --max_length 4096

log_section "合并完成"
ls "$MERGED_DIR" | head