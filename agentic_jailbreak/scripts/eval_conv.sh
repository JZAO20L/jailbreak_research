#!/bin/bash
# =============================================================================
# 对话式(Agentic)评估:主方法形态
# =============================================================================
#
# Usage:
#   bash scripts/eval_conv.sh no_skill 1000 3        # 变体 样本数 轮数
#   bash scripts/eval_conv.sh no_skill_beam 200 3 2  # beam-2
#   bash scripts/eval_conv.sh skill_decide_top3 200 3 2 3 1   # 变体 样本数 轮数 beam top_k work_memory(0/1)
#
# 变体:select_adapt / no_skill / no_skill_beam(默认 no_skill)
# 数据:data/dataset/processed/10k/test.jsonl(按需切分并行)
# =============================================================================

set -e

source "$(dirname "$0")/common.sh"

VARIANT="${1:-no_skill}"
MAX_SAMPLES="${2:-200}"
# 默认值 = 09-02 定稿协议(主结果口径 10 轮 + 精选 10 skill 全量候选 top_k=10);
# 漏传参数不再会静默跑成 3 轮/top_k=5 的旧口径
MAX_TURNS="${3:-10}"
BEAM_WIDTH="${4:-2}"
TOP_K="${5:-10}"
WORK_MEMORY="${6:-0}"
# A 轴各臂(M0/M1/M2/M3)评的是不同权重, 必须带标签, 否则共用目录互相覆盖:
#   RUN_TAG=m0 / m1 / m2 / m3
RUN_TAG="${RUN_TAG:-}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/dataset/processed/10k/test.jsonl}"
MEM_TAG=""
if [ "$WORK_MEMORY" = "1" ]; then
    MEM_TAG="_mem"
fi
OUT_BASE="$OUTPUT_DIR/eval_results/conv_${VARIANT}_top${TOP_K}${MEM_TAG}_${MAX_TURNS}turn${RUN_TAG:+_$RUN_TAG}"

log_section "对话式评估: variant=$VARIANT, samples=$MAX_SAMPLES, turns=$MAX_TURNS, top_k=$TOP_K, work_memory=$WORK_MEMORY"

# 检查 servers(评估用标准 vllm policy)
if ! curl -s "http://127.0.0.1:$ROLLOUT_PORT/health" > /dev/null 2>&1; then
    log_error "Policy server (port $ROLLOUT_PORT) 未就绪"
    exit 1
fi

# 切分数据(并行分片数: EVAL_WORKERS, 默认 4 与历史口径一致; A800 单卡可开 12+)
EVAL_WORKERS="${EVAL_WORKERS:-4}"
SLICE_DIR="$OUTPUT_DIR/slices/conv_${VARIANT}_top${TOP_K}${MEM_TAG}_${MAX_SAMPLES}_w${EVAL_WORKERS}${RUN_TAG:+_$RUN_TAG}"
# 派生目录: 上一轮遗留的多余 part* 会被下面的合并步骤重复统计, 先清掉
rm -rf "$SLICE_DIR"/part*.jsonl "$OUT_BASE"/part*
mkdir -p "$SLICE_DIR"
$PYTHON_BIN - "$DATA_PATH" "$MAX_SAMPLES" "$SLICE_DIR" "$EVAL_WORKERS" << 'EOF'
import json, sys
from pathlib import Path
data_path, max_samples, out_dir = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
want = int(sys.argv[4])
lines = [l for l in open(data_path) if l.strip()][:max_samples]
n = len(lines)
workers = max(1, min(want, n))
size = max(1, n // workers)
for i in range(workers):
    chunk = lines[i*size:(i+1)*size] if i < workers-1 else lines[i*size:]
    (out_dir / f"part{i}.jsonl").write_text("".join(chunk))
print(f"sliced {n} into {workers} parts")
EOF

# 并行评估
PIDS=()
for part in "$SLICE_DIR"/part*.jsonl; do
    name=$(basename "$part" .jsonl)
    OUT_SUB="$OUT_BASE/$name"
    ARGS="--test_data $part --skills_path $SKILLS_PATH --policy_port $ROLLOUT_PORT \
          --target_port $TARGET_PORT --guard_port $GUARD_PORT \
          --max_turns $MAX_TURNS --max_samples $MAX_SAMPLES \
          --variant $VARIANT --mode conversational --top_k_skills $TOP_K"
    if [ "$WORK_MEMORY" = "1" ]; then
        ARGS="$ARGS --work_memory"
    fi
    if [ "$VARIANT" = "no_skill_beam" ]; then
        ARGS="$ARGS --beam_width $BEAM_WIDTH"
    fi
    $PYTHON_BIN "$AGENTIC_DIR/src/eval.py" $ARGS \
        --output_dir "$OUT_SUB" > "$OUTPUT_DIR/logs/conv_${VARIANT}_${name}.log" 2>&1 &
    PIDS+=($!)
    log_info "启动 part: $name (PID $!)"
done

for pid in "${PIDS[@]}"; do
    wait $pid
done

log_section "合并结果"
$PYTHON_BIN - "$OUT_BASE" "$VARIANT" "$MAX_TURNS" << 'EOF'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
total = success = 0
turns_sum = 0
rows = []
for part in sorted(base.glob("part*/summary.json")):
    s = json.load(open(part))
    total += s["total"]; success += s["success"]; turns_sum += s["avg_turns"]*s["total"]
    rows.extend(json.loads(l) for l in open(part.parent/"results.jsonl"))
asr = success / total if total else 0.0
with (base/"results.jsonl").open("w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
json.dump({"total": total, "success": success, "asr": asr,
           "avg_turns": turns_sum/total if total else 0.0,
           "max_turns": int(sys.argv[3]), "variant": sys.argv[2],
           "mode": "conversational"}, open(base/"summary.json", "w"), ensure_ascii=False, indent=2)
print(f"[conv/{sys.argv[2]}] ASR={asr:.2%} ({success}/{total}), avg_turns={turns_sum/total:.2f}")
EOF