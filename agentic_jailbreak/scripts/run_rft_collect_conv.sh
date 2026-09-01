#!/bin/bash
# =============================================================================
# RFT 数据采集 (conv 协议原生, v3) — E2/E3 的 SFT 数据来源
#
# 流程:
#   1) split A 种子 (train[1000:2000], rft_seed_train1000.json) -> jsonl 转换
#   2) base 模型 + conv_eval (no_skill, 单轨迹, max_turns=10) 全量采集
#   3) build_rft_data_conv.py 前缀式展开 -> SFT jsonl
#
# 数据隔离 (保持不变): A(本脚本采集 RFT) / B(GRPO 训练 train[0:1000]) / C(test 评估)
#
# 前置: guard/target/policy(base) 三服务已就绪 (start_servers*.sh, NUM_GPUS 先声明)
# Usage:
#   NUM_GPUS=4 bash scripts/run_rft_collect_conv.sh
# =============================================================================
set -e
source "$(dirname "$0")/common.sh"

MAX_TURNS="${MAX_TURNS:-10}"
SEED_JSON="${SEED_JSON:-$OUTPUT_DIR/rft_seed_train1000.json}"
COLLECT_DIR="$OUTPUT_DIR/rft_collect_conv_${MAX_TURNS}turn"
RFT_DATA="$OUTPUT_DIR/rft_data_conv_${MAX_TURNS}turn.jsonl"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

log_section "RFT conv 采集: turns=$MAX_TURNS, seed=$SEED_JSON"

# --- 1. 种子数据: json 数组 -> jsonl {id, prompt} ---
SEED_JSONL="$OUTPUT_DIR/rft_seed_train1000.jsonl"
$PYTHON_BIN - "$SEED_JSON" "$SEED_JSONL" << 'EOF'
import json, sys
prompts = json.load(open(sys.argv[1]))
assert len(prompts) == 1000, f"split A 应为 1000 条, 实际 {len(prompts)}"
with open(sys.argv[2], "w") as f:
    for i, p in enumerate(prompts):
        f.write(json.dumps({"id": i, "prompt": p}, ensure_ascii=False) + "\n")
print(f"seed jsonl: {len(prompts)} prompts -> {sys.argv[2]}")
EOF

# --- 2. 服务检查 (policy 必须是 base 模型, 未训练) ---
for p in $GUARD_PORT $TARGET_PORT $ROLLOUT_PORT; do
    if ! curl -s "http://127.0.0.1:$p/health" > /dev/null 2>&1; then
        log_error "服务 $p 未就绪 (先用 start_servers*.sh 启动)"
        exit 1
    fi
done

# --- 3. 切分并行采集 (同 eval_conv.sh 模式) ---
SLICE_DIR="$OUTPUT_DIR/slices/rft_collect_${MAX_TURNS}turn"
mkdir -p "$SLICE_DIR" "$COLLECT_DIR"
$PYTHON_BIN - "$SEED_JSONL" "$SLICE_DIR" << 'EOF'
import json, sys
from pathlib import Path
lines = [l for l in open(sys.argv[1]) if l.strip()]
n = len(lines)
workers = 4
size = max(1, n // workers)
for i in range(workers):
    chunk = lines[i*size:(i+1)*size] if i < workers-1 else lines[i*size:]
    (Path(sys.argv[2]) / f"part{i}.jsonl").write_text("".join(chunk))
print(f"sliced {n} into {workers} parts")
EOF

PIDS=()
for part in "$SLICE_DIR"/part*.jsonl; do
    name=$(basename "$part" .jsonl)
    $PYTHON_BIN "$AGENTIC_DIR/src/eval.py" \
        --test_data "$part" --skills_path "$SKILLS_PATH" \
        --policy_port $ROLLOUT_PORT --target_port $TARGET_PORT --guard_port $GUARD_PORT \
        --max_turns $MAX_TURNS --max_samples 1000 \
        --variant no_skill --mode conversational \
        --output_dir "$COLLECT_DIR/$name" \
        > "$OUTPUT_DIR/logs/rft_collect_conv_${name}.log" 2>&1 &
    PIDS+=($!)
    log_info "启动采集 part: $name (PID $!)"
done
for pid in "${PIDS[@]}"; do wait $pid; done

# --- 4. 合并 ---
$PYTHON_BIN - "$COLLECT_DIR" << 'EOF'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
total = success = 0
rows = []
for part in sorted(base.glob("part*/summary.json")):
    s = json.load(open(part))
    total += s["total"]; success += s["success"]
    rows.extend(json.loads(l) for l in open(part.parent / "results.jsonl"))
asr = success / total if total else 0.0
with (base / "results.jsonl").open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"total": total, "success": success, "asr": asr, "mode": "rft_collect_conv"},
          open(base / "summary.json", "w"), ensure_ascii=False, indent=2)
print(f"[rft_collect] base@采集 ASR={asr:.2%} ({success}/{total})")
EOF

# --- 5. 前缀式展开 -> SFT 数据 ---
log_section "构建 SFT 数据"
$PYTHON_BIN "$AGENTIC_DIR/scripts/build_rft_data_conv.py" \
    --results "$COLLECT_DIR/results.jsonl" --out "$RFT_DATA"

log_section "RFT conv 采集完成"
log_info "采集结果: $COLLECT_DIR"
log_info "SFT 数据: $RFT_DATA (下一步: bash scripts/rft_sft_conv.sh)"
