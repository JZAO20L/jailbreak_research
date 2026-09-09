#!/bin/bash
# =============================================================================
# B 轴 C2 臂崩溃片重跑 (2026-09-04): C2 work_memory 每轮额外插入总结消息,
# 上下文增速超 C1/C3, 24576 下 input 22529 + max_tokens 2048 = 24577 越界崩溃。
# 重启 policy @32768 后重跑 C2 part0/part1, 最后汇总 B 轴三臂全表。
# =============================================================================
set -e
cd /home/tiger/jailbreak_research/agentic_jailbreak
source "$(dirname "$0")/common.sh"
PYTHON_BIN=/home/tiger/jailbreak_research/.venv/bin/python
AGENTIC_DIR=/home/tiger/jailbreak_research/agentic_jailbreak
SLICE_DIR="$OUTPUT_DIR/slices/baxis_test300"
BASE_DIR="$OUTPUT_DIR/baxis_ctx"
SKILLS_TOP10="$AGENTIC_DIR/exp/skill_asr_sweep/seed_skills_top10.json"
LOG_DIR="$AGENTIC_DIR/output/logs"

# --- 1. 重启 policy @32768 ---
echo "[rerun-c2] $(date) 重启 policy server (8003, max-model-len 32768)"
for pid in $(pgrep -f "port 8003"); do kill "$pid" 2>/dev/null || true; done
sleep 10
for pid in $(pgrep -f "port 8003"); do kill -9 "$pid" 2>/dev/null || true; done
sleep 5
CUDA_VISIBLE_DEVICES=2 /home/tiger/jailbreak_research/.venv/bin/vllm serve \
    /home/tiger/models/Qwen/Qwen3-4B \
    --port 8003 --max-model-len 32768 --gpu-memory-utilization 0.9 \
    --dtype float16 --trust-remote-code \
    > "$LOG_DIR/policy_v100_32768.log" 2>&1 &
until curl -s http://127.0.0.1:8003/health > /dev/null 2>&1; do sleep 10; done
echo "[rerun-c2] $(date) policy 就绪"

# --- 2. 重跑 C2 part0/part1 ---
PIDS=()
for j in 0 1; do
    $PYTHON_BIN "$AGENTIC_DIR/src/eval.py" \
        --test_data "$SLICE_DIR/part$j.jsonl" --skills_path "$SKILLS_TOP10" \
        --policy_port 8003 --target_port 8002 --guard_port 8001 \
        --max_turns 10 --max_samples 1000 \
        --variant skill_decide --top_k_skills 10 --mode conversational \
        --work_memory \
        --output_dir "$BASE_DIR/C2_workmem/part$j" \
        > "$BASE_DIR/C2_workmem/part$j.log" 2>&1 &
    PIDS+=("$!")
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
[ "$FAIL" -ne 0 ] && { log_error "C2 重跑仍有失败"; exit 1; }

# --- 3. B 轴三臂全表汇总 ---
$PYTHON_BIN - "$BASE_DIR" << 'EOF'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
print("\n===== B 轴上下文消融汇总 (base policy, test C 300) =====")
print(f"{'config':<14} {'ASR':>7} {'succ/total':>11} {'avg_turns':>10}")
summary = {}
for cfg_dir in sorted(base.iterdir()):
    if not cfg_dir.is_dir():
        continue
    total = success = 0
    turns = 0.0
    parts = 0
    for s in cfg_dir.glob("part*/summary.json"):
        d = json.load(open(s))
        total += d["total"]; success += d["success"]
        turns += d.get("avg_turns", 0) * d["total"]
        parts += 1
    if total == 0:
        continue
    assert parts == 3 and total == 300, f"{cfg_dir.name} 不完整: {parts} 片 {total} 条"
    summary[cfg_dir.name] = {"asr": success/total, "success": success, "total": total,
                             "avg_turns": turns/total}
    print(f"{cfg_dir.name:<14} {success/total:>7.2%} {success:>6}/{total:<4} {turns/total:>10.2f}")
json.dump(summary, open(base / "summary.json", "w"), ensure_ascii=False, indent=2)
print(f"\nsaved -> {base}/summary.json")
EOF

echo "[rerun-c2] $(date) B 轴三臂全部完成"
