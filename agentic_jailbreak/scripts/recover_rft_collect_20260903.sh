#!/bin/bash
# =============================================================================
# RFT 采集崩溃恢复 (2026-09-03 凌晨)
#
# 背景: 12 路并发下 4 个 part (0/1/8/11) 因 client timeout=60s 过短崩溃
#       (APITimeoutError, 结果落盘仅在结束 → 全损需重跑); 8 路存活继续跑。
#       client 超时已修 60→600 (conv_eval.py / env.py), 仅对新起进程生效。
#
# 本脚本流程 (全自动, 后台运行):
#   1) 等待现存 8 路 eval.py 全部退出
#   2) 将 4 个崩溃 part 的 slice 各重切 3 份 -> 12 个子任务, 并行补采
#      (输出到 part{i}_r{j}/, 合并脚本 glob part*/summary.json 可覆盖)
#   3) 等待补采退出 -> 合并 12+12 part -> 构建 SFT 数据
# =============================================================================
set -e
cd /home/tiger/jailbreak_research/agentic_jailbreak
PYTHON_BIN="${PYTHON_BIN:-/home/tiger/jailbreak_research/.venv/bin/python}"
AGENTIC_DIR=/home/tiger/jailbreak_research/agentic_jailbreak
COLLECT_DIR="$AGENTIC_DIR/output/rft_collect_conv_10turn"
SLICE_DIR="$AGENTIC_DIR/output/slices/rft_collect_10turn"
RFT_DATA="$AGENTIC_DIR/output/rft_data_conv_10turn.jsonl"
SKILLS_TOP10="$AGENTIC_DIR/exp/skill_asr_sweep/seed_skills_top10.json"
MAX_TURNS=10

echo "[recover] $(date) 等待存活采集进程退出..."
while pgrep -f "src/eval.py" > /dev/null; do sleep 60; done
echo "[recover] $(date) 存活进程已全部退出"

# --- 2. 重切 4 个崩溃 part -> 12 子任务 ---
$PYTHON_BIN - "$SLICE_DIR" << 'EOF'
import sys
from pathlib import Path
slice_dir = Path(sys.argv[1])
n_sub = 3
for p in [0, 1, 8, 11]:
    lines = (slice_dir / f"part{p}.jsonl").read_text().splitlines(keepends=True)
    size = max(1, len(lines) // n_sub)
    for j in range(n_sub):
        chunk = lines[j*size:(j+1)*size] if j < n_sub-1 else lines[j*size:]
        out = slice_dir / f"rerun_{p}_{j}.jsonl"
        out.write_text("".join(chunk))
        print(f"rerun_{p}_{j}: {len(chunk)} prompts")
EOF

# --- 3. 并行补采 12 子任务 ---
PIDS=()
for sub in "$SLICE_DIR"/rerun_*.jsonl; do
    name=$(basename "$sub" .jsonl)   # e.g. rerun_0_0
    p=${name#rerun_}; p=${p%%_*}
    $PYTHON_BIN "$AGENTIC_DIR/src/eval.py" \
        --test_data "$sub" --skills_path "$SKILLS_TOP10" \
        --policy_port 8003 --target_port 8002 --guard_port 8001 \
        --max_turns $MAX_TURNS --max_samples 1000 \
        --variant skill_decide --top_k_skills 10 --mode conversational \
        --output_dir "$COLLECT_DIR/${name}" \
        > "$AGENTIC_DIR/output/logs/rft_collect_conv_${name}.log" 2>&1 &
    PIDS+=($!)
    echo "[recover] $(date) 启动补采: $name (PID $!)"
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
if [ "$FAIL" -ne 0 ]; then echo "[recover] $(date) 有补采子任务失败!"; exit 1; fi
echo "[recover] $(date) 补采完成"

# --- 4. 合并 + 构建 SFT 数据 (与 run_rft_collect_conv.sh 步骤 4/5 一致) ---
$PYTHON_BIN - "$COLLECT_DIR" << 'EOF'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
total = success = 0
rows = []
parts = sorted(base.glob("part*/summary.json"))
print(f"merging {len(parts)} parts")
for s in parts:
    d = json.load(open(s))
    total += d["total"]; success += d["success"]
    rows.extend(json.loads(l) for l in open(s.parent / "results.jsonl"))
assert total == 1000, f"合并总数 {total} != 1000, 有 part 缺失!"
asr = success / total
with (base / "results.jsonl").open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"total": total, "success": success, "asr": asr, "mode": "rft_collect_conv"},
          open(base / "summary.json", "w"), ensure_ascii=False, indent=2)
print(f"[rft_collect] base@采集 ASR={asr:.2%} ({success}/{total})")
EOF

$PYTHON_BIN "$AGENTIC_DIR/scripts/build_rft_data_conv.py" \
    --results "$COLLECT_DIR/results.jsonl" --out "$RFT_DATA" \
    --variant skill_decide --skills-path "$SKILLS_TOP10" --top-k-skills 10

echo "[recover] $(date) 全部完成: SFT 数据 = $RFT_DATA"
