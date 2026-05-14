#!/bin/bash
# =============================================================================
# 实验1 完整运行脚本 - Jailbreak Prompt 策略测试
# =============================================================================
#
# 根据 TODO.md "实验重做"：
#   - 测试24种jailbreak prompt重写策略
#   - 选取top3好的策略
#   - 用qwen3-max对比实验
#
# GPU配置: 3卡 (0:policy, 1:target, 2:guard)
# 模型路径: models/
#
# 用法:
#   cd /home/tiger/jailbreak_research/RL4jailbreak
#   bash experiments/jailbreak_prompt_exp/run_exp1.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "============================================================"
echo "实验1: Jailbreak Prompt 策略测试"
echo "============================================================"
echo ""

# =========================
# Step 1: 环境检查
# =========================
echo "[Step 1] 环境检查..."

# 检查模型是否存在
POLICY_MODEL="/home/tiger/models/Qwen3-4B"
TARGET_MODEL="/home/tiger/models/Qwen3-4B"
GUARD_MODEL="/home/tiger/models/Qwen3Guard-Gen-4B"

check_model() {
    local model_path="$1"
    local model_name="$2"
    if [ -d "$model_path" ]; then
        echo "  ✅ $model_name: $model_path"
    else
        echo "  ❌ $model_name 不存在: $model_path"
        echo "     请确保模型已下载或路径正确"
        return 1
    fi
}

echo "检查模型:"
check_model "$POLICY_MODEL" "Policy模型" || exit 1
check_model "$TARGET_MODEL" "Target模型" || exit 1
check_model "$GUARD_MODEL" "Guard模型" || exit 1
echo ""

# 检查数据集
TEST_SET="$BASE_DIR/../data/dataset/processed/10k/test.jsonl"
if [ -f "$TEST_SET" ]; then
    TEST_COUNT=$(wc -l < "$TEST_SET")
    echo "  ✅ 测试集: $TEST_SET ($TEST_COUNT 条)"
else
    echo "  ❌ 测试集不存在: $TEST_SET"
    exit 1
fi
echo ""

# 检查Python依赖
echo "检查Python依赖:"
python3 -c "import torch; print('  ✅ torch:', torch.__version__)" 2>/dev/null || { echo "  ❌ torch 未安装"; exit 1; }
python3 -c "import vllm; print('  ✅ vllm')" 2>/dev/null || { echo "  ❌ vllm 未安装"; exit 1; }
python3 -c "import trl; print('  ✅ trl')" 2>/dev/null || { echo "  ❌ trl 未安装"; exit 1; }
python3 -c "import transformers; print('  ✅ transformers')" 2>/dev/null || { echo "  ❌ transformers 未安装"; exit 1; }
echo ""

# 检查GPU
echo "检查GPU:"
python3 -c "import torch; n=torch.cuda.device_count(); print(f'  ✅ GPU数量: {n}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  ❌ CUDA 未可用"
    exit 1
fi
echo ""

# =========================
# Step 2: 运行实验
# =========================
echo "============================================================"
echo "[Step 2] 运行实验"
echo "============================================================"
echo ""

# 创建输出目录
OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

echo "运行命令:"
echo "  bash $SCRIPT_DIR/exp.sh --topk 3 --qwen3_max"
echo ""
echo "说明:"
echo "  --topk 3      : 只输出top3策略结果"
echo "  --qwen3_max   : 包含qwen3-max对比实验"
echo ""
echo "预计耗时: 约2-4小时 (取决于GPU和数据量)"
echo ""

# =========================
# 执行实验
# =========================
START_TIME=$(date +%s)

bash "$SCRIPT_DIR/exp.sh" --topk 3 --qwen3_max

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo "实验完成!"
echo "============================================================"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo ""

# =========================
# Step 3: 结果汇总
# =========================
echo "============================================================"
echo "[Step 3] 结果汇总"
echo "============================================================"
echo ""

if [ -f "$OUTPUT_DIR/experiment_summary.json" ]; then
    echo "=== Top 3 策略 ==="
    python3 -c "
import json
with open('$OUTPUT_DIR/experiment_summary.json') as f:
    summary = json.load(f)

results = summary.get('filtered_results', summary.get('all_results', {}))
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print('策略名                      ASR')
print('-' * 40)
for i, (name, asr) in enumerate(sorted_results[:3], 1):
    print(f'{i}. {name:<25} {asr:.4f}')
print('-' * 40)
print(f'基线ASR: {summary.get(\"baseline_asr\", 0):.4f}')
"
    echo ""

    # qwen3-max对比结果
    if [ -f "$OUTPUT_DIR/qwen3_max_comparison/comparison_summary.json" ]; then
        echo "=== qwen3-max vs 本地模型对比 ==="
        python3 -c "
import json
with open('$OUTPUT_DIR/qwen3_max_comparison/comparison_summary.json') as f:
    comp = json.load(f)

print('策略                        qwen3-4B    qwen3-max    差异')
print('-' * 60)
for s in comp['strategies']:
    sign = '+' if s['diff'] > 0 else ''
    print(f'{s[\"strategy\"]:<25} {s[\"local_asr\"]:.4f}      {s[\"qwen3max_asr\"]:.4f}      {sign}{s[\"diff\"]:.4f}')
print('-' * 60)
print(comp['analysis']['conclusion'])
"
    fi
else
    echo "⚠️ 未找到结果文件: $OUTPUT_DIR/experiment_summary.json"
fi

echo ""
echo "完整结果路径:"
echo "  - 本地模型: $OUTPUT_DIR/experiment_summary.json"
echo "  - qwen3-max对比: $OUTPUT_DIR/qwen3_max_comparison/comparison_summary.json"
echo ""
echo "查看详细结果:"
echo "  cat $OUTPUT_DIR/experiment_summary.json | python -m json.tool"
echo ""