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

MAX_TURNS="${MAX_TURNS:-10}"
VARIANT=skill_decide
TOP_K_SKILLS=10
N_SAMPLES="${N_SAMPLES:-300}"
N_PARTS="${N_PARTS:-3}"
SKILLS_TOP10="$AGENTIC_DIR/exp/skill_asr_sweep/seed_skills_top10.json"
# split C: 由 scripts/build_splits.py 从 data/dataset/processed/10k/test.jsonl 重建
TEST_SRC="${TEST_SRC:-$OUTPUT_DIR/test_prompts.json}"
BASE_DIR="${BASE_DIR:-$OUTPUT_DIR/baxis_ctx}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
WINDOW="${WINDOW:-3}"

log_section "B 轴上下文消融: ${ARMS:-C1_full,C2_workmem,C3_window,C4_c4000,C4_c8000} | ${N_SAMPLES} 样本 × ${N_PARTS} 片"

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
    if [ "${RESUME:-0}" != "1" ]; then
        rm -rf "$out_dir"
    fi
    mkdir -p "$out_dir"
    local resume_flag=()
    if [ "${RESUME:-0}" = "1" ]; then resume_flag=(--resume); fi
    for j in $(seq 0 $((N_PARTS - 1))); do
        local extra_args=("$@" "${resume_flag[@]}")
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

# 臂列表可裁剪(逗号分隔): ARMS=C1_full,C3_window3,C4_c4000 bash ...
# 阈值依据 09-10 实测: C1 十轮累计输入 token 中位 6.7k / p90 8.7k / max 10.0k,
# 所以阈值必须 <10k 才会真正触发压缩(4k 约第 6-7 轮起折, 8k 只在最长尾部轨迹触发)
ARMS="${ARMS:-C1_full,C3_window,C4_c2500,C4_c4000}"
for arm in ${ARMS//,/ }; do
    case "$arm" in
        C1_full)      run_config C1_full ;;
        C2_workmem)   run_config C2_workmem --work_memory ;;
        C3_window)    run_config "C3_window$WINDOW" --ctx_window "$WINDOW" ;;
        C4_c2500)     run_config C4_c2500 --ctx_compress 2500 --ctx_keep 3 ;;
        C4_c4000)     run_config C4_c4000 --ctx_compress 4000 --ctx_keep 3 ;;
        C4_c8000)     run_config C4_c8000 --ctx_compress 8000 --ctx_keep 3 ;;
        *) log_error "未知臂: $arm"; exit 1 ;;
    esac
done
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
hdr = f"{'config':<14} {'ASR':>7} {'succ/total':>11} {'avg_turns':>10} {'ctx_max':>8} {'full_max':>9} {'folds':>6}"
print(hdr)
summary = {}
for cfg_dir in sorted(base.iterdir()):
    if not cfg_dir.is_dir():
        continue
    total = success = folds = 0
    turns = 0.0
    ctx_max = full_max = 0
    for s in cfg_dir.glob("part*/summary.json"):
        d = json.load(open(s))
        total += d["total"]; success += d["success"]
        turns += d.get("avg_turns", 0) * d["total"]
        cs = d.get("compress_stats") or {}
        folds += cs.get("folds", 0)
        ctx_max = max(ctx_max, cs.get("max_view", 0))
        full_max = max(full_max, cs.get("max_full", 0))
    if total == 0:
        continue
    asr = success / total
    summary[cfg_dir.name] = {"asr": asr, "success": success, "total": total,
                             "avg_turns": turns / total, "ctx_max_tokens": ctx_max,
                             "full_max_tokens": full_max, "folds": folds}
    c = f"{ctx_max:>8}" if ctx_max else f"{'-':>8}"
    fm = f"{full_max:>9}" if full_max else f"{'-':>9}"
    fd = f"{folds:>6}" if folds else f"{'-':>6}"
    print(f"{cfg_dir.name:<14} {asr:>7.2%} {success:>6}/{total:<4} {turns/total:>10.2f} {c} {fm} {fd}")
json.dump(summary, open(base / "summary.json", "w"), ensure_ascii=False, indent=2)
print(f"\nsaved -> {base}/summary.json")
print("注: ctx_max = 实际喂给 policy 的最大输入 token; full_max = 同批轨迹若不压缩会到的最大 token")
EOF

log_section "B 轴消融完成"
