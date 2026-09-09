#!/bin/bash
# =============================================================================
# B 轴上下文维护机制消融 (2026-09-03, 基座 policy, 推理侧)
#   C1_full     全量消息累积 (现状协议, 采集/M2 数据口径)
#   C2_workmem  分层工作记忆 (前轮总结 + 末轮完整反馈, --work_memory)
#   C3_window3  滑动窗口 (system+初始user + 最近 3 轮, --ctx_window 3)
# 统一口径: test C 前 300 条 (test_prompts.json, 标准 1000 条评估的子集),
#           variant=skill_decide, top_k=10, max_turns=10, 每 config 3 并行片
# 产出: output/baxis_ctx/<config>/part*/summary.json + 汇总表
# Usage: bash scripts/run_baxis_ctx_ablation.sh
# =============================================================================
set -e
cd /home/tiger/jailbreak_research/agentic_jailbreak
source "$(dirname "$0")/common.sh"

MAX_TURNS=10
VARIANT=skill_decide
TOP_K_SKILLS=10
N_SAMPLES=300
N_PARTS=3
SKILLS_TOP10="$AGENTIC_DIR/exp/skill_asr_sweep/seed_skills_top10.json"
TEST_SRC="/home/tiger/jailbreak_research/self_evolve_skills_jailbreak/data/test_prompts.json"
BASE_DIR="$OUTPUT_DIR/baxis_ctx"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
WINDOW="${WINDOW:-3}"

log_section "B 轴上下文消融: C1_full / C2_workmem / C3_window${WINDOW}, ${N_SAMPLES} 样本 × 3 片"

# --- 1. 构造 test C 前 300 条切片 ---
SLICE_DIR="$OUTPUT_DIR/slices/baxis_test${N_SAMPLES}"
mkdir -p "$SLICE_DIR"
$PYTHON_BIN - "$TEST_SRC" "$SLICE_DIR" "$N_SAMPLES" "$N_PARTS" << 'EOF'
import json, sys, os
src, slice_dir, n, n_parts = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
prompts = json.load(open(src))
rows = [{"id": i, "prompt": p} for i, p in enumerate(prompts[:n])]
out = os.path.join(slice_dir, "test300.jsonl")
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
size = (len(rows) + n_parts - 1) // n_parts
for j in range(n_parts):
    part = rows[j * size:(j + 1) * size]
    with open(os.path.join(slice_dir, f"part{j}.jsonl"), "w") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"part{j}: {len(part)} samples")
print(f"total: {len(rows)} -> {out}")
EOF

# --- 2. 服务检查 ---
for p in $GUARD_PORT $TARGET_PORT $ROLLOUT_PORT; do
    if ! curl -s "http://127.0.0.1:$p/health" > /dev/null 2>&1; then
        log_error "服务 $p 未就绪 (先用 start_servers*.sh 启动)"
        exit 1
    fi
done

# --- 3. 三配置 × 3 片 并行启动 ---
PIDS=()
run_config() {
    local cfg="$1"; shift
    local out_dir="$BASE_DIR/$cfg"
    mkdir -p "$out_dir"
    for j in $(seq 0 $((N_PARTS - 1))); do
        local extra_args=("$@")
        $PYTHON_BIN "$AGENTIC_DIR/src/eval.py" \
            --test_data "$SLICE_DIR/part$j.jsonl" --skills_path "$SKILLS_TOP10" \
            --policy_port $ROLLOUT_PORT --target_port $TARGET_PORT --guard_port $GUARD_PORT \
            --max_turns $MAX_TURNS --max_samples $N_SAMPLES \
            --variant $VARIANT --top_k_skills $TOP_K_SKILLS --mode conversational \
            "${extra_args[@]}" \
            --output_dir "$out_dir/part$j" \
            > "$out_dir/part$j.log" 2>&1 &
        PIDS+=("$!")
    done
}

run_config C1_full
run_config C2_workmem --work_memory
run_config C3_window$WINDOW --ctx_window "$WINDOW"
echo "launched ${#PIDS[@]} eval processes"

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done
if [ "$FAIL" -ne 0 ]; then
    log_error "存在失败的 eval 进程, 检查 $BASE_DIR/*.log"
    exit 1
fi

# --- 4. 汇总 ---
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
    for s in cfg_dir.glob("part*/summary.json"):
        d = json.load(open(s))
        total += d["total"]; success += d["success"]
        turns += d.get("avg_turns", 0) * d["total"]
    if total == 0:
        continue
    asr = success / total
    summary[cfg_dir.name] = {"asr": asr, "success": success, "total": total,
                             "avg_turns": turns / total}
    print(f"{cfg_dir.name:<14} {asr:>7.2%} {success:>6}/{total:<4} {turns/total:>10.2f}")
json.dump(summary, open(base / "summary.json", "w"), ensure_ascii=False, indent=2)
print(f"\nsaved -> {base}/summary.json")
EOF

log_section "B 轴消融完成"
