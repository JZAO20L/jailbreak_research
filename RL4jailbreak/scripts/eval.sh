#!/bin/bash
# 模型+prompt 评测脚本启动器
# 用法: bash scripts/eval.sh [参数...]
#   无参数: 使用默认配置 (val.jsonl + base model)
#   有参数: 传递给 eval.py

set -e

# =========================
# 用户可修改的默认配置
# =========================

# 评估数据集
EVAL_PATH="${EVAL_PATH:-data/dataset/processed/10k/val.jsonl}"

# 模型路径
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
TARGET_MODEL="${TARGET_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B}"

# LoRA 路径列表 (留空=base model)
# 示例: LORA_PATHS=("/path/to/lora1" "/path/to/lora2")
LORA_PATHS=()

# vLLM 端口
POLICY_PORT=8003
TARGET_PORT=8001
GUARD_PORT=8002

# 生成配置
K=1
REWRITE_TEMP=0.7
REWRITE_MAX_TOKENS=1024
MAX_MODEL_LEN=2048

# 输出
OUTPUT_ROOT="output/eval"

# =========================
# 构建命令行参数
# =========================
EVAL_ARGS=(
    "--eval_path" "${EVAL_PATH}"
    "--base_model_path" "${BASE_MODEL}"
    "--target_model_path" "${TARGET_MODEL}"
    "--guard_model_path" "${GUARD_MODEL}"
    "--policy_port" "${POLICY_PORT}"
    "--target_port" "${TARGET_PORT}"
    "--guard_port" "${GUARD_PORT}"
    "--k" "${K}"
    "--rewrite_temperature" "${REWRITE_TEMP}"
    "--rewrite_max_tokens" "${REWRITE_MAX_TOKENS}"
    "--max_model_len" "${MAX_MODEL_LEN}"
    "--output_root" "${OUTPUT_ROOT}"
)

# 添加 LoRA 路径
if [ ${#LORA_PATHS[@]} -gt 0 ]; then
    EVAL_ARGS+=("--lora_paths")
    for lp in "${LORA_PATHS[@]}"; do
        EVAL_ARGS+=("${lp}")
    done
fi

# 如果传入了命令行参数, 优先使用
if [ $# -gt 0 ]; then
    EVAL_ARGS=("$@")
fi

# =========================
# 执行评估
# =========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "============================================================"
echo "模型+prompt 评测脚本"
echo "============================================================"
echo "数据集: ${EVAL_PATH}"
echo "LoRA: ${#LORA_PATHS[@]} 个"
echo "输出: ${OUTPUT_ROOT}"
echo "============================================================"

python "${SCRIPT_DIR}/eval.py" "${EVAL_ARGS[@]}"
