#!/bin/bash
# =============================================================================
# 采集收尾 (2026-09-03): 重启 policy 服务(24576 上下文) -> 补跑越界崩溃的
# rerun_0_0(28条) -> 合并 1000 条 -> 构建 SFT 数据
# =============================================================================
set -e
cd /home/tiger/jailbreak_research/agentic_jailbreak
PYTHON_BIN="${PYTHON_BIN:-/home/tiger/jailbreak_research/.venv/bin/python}"
AGENTIC_DIR=/home/tiger/jailbreak_research/agentic_jailbreak
COLLECT_DIR="$AGENTIC_DIR/output/rft_collect_conv_10turn"
SLICE_DIR="$AGENTIC_DIR/output/slices/rft_collect_10turn"
RFT_DATA="$AGENTIC_DIR/output/rft_data_conv_10turn.jsonl"
SKILLS_TOP10="$AGENTIC_DIR/exp/skill_asr_sweep/seed_skills_top10.json"
LOG_DIR="$AGENTIC_DIR/output/logs"

echo "[finalize] $(date) 等待在跑采集进程退出..."
while pgrep -f "src/eval.py" > /dev/null; do sleep 30; done

# --- 1. 重启 policy server (max-model-len 16384 -> 24576) ---
echo "[finalize] $(date) 重启 policy server (8003, max-model-len 24576)"
for pid in $(pgrep -f "port 8003"); do kill "$pid" 2>/dev/null || true; done
sleep 10
for pid in $(pgrep -f "port 8003"); do kill -9 "$pid" 2>/dev/null || true; done
sleep 5
VLLM_BIN=/home/tiger/jailbreak_research/.venv/bin/vllm
BASE_MODEL=/home/tiger/models/Qwen/Qwen3-4B
CUDA_VISIBLE_DEVICES=2 "$VLLM_BIN" serve "$BASE_MODEL" \
    --port 8003 --max-model-len 24576 --gpu-memory-utilization 0.9 \
    --dtype float16 --trust-remote-code \
    > "$LOG_DIR/policy_v100_24576.log" 2>&1 &
echo "[finalize] $(date) policy server 启动中, 等待就绪..."
until curl -s http://127.0.0.1:8003/health > /dev/null 2>&1; do sleep 10; done
echo "[finalize] $(date) policy server 就绪"

# --- 2. 补跑 rerun_0_0 (28 条, 因 input+max_tokens 越界崩溃) ---
$PYTHON_BIN "$AGENTIC_DIR/src/eval.py" \
    --test_data "$SLICE_DIR/rerun_0_0.jsonl" --skills_path "$SKILLS_TOP10" \
    --policy_port 8003 --target_port 8002 --guard_port 8001 \
    --max_turns 10 --max_samples 1000 \
    --variant skill_decide --top_k_skills 10 --mode conversational \
    --output_dir "$COLLECT_DIR/rerun_0_0" \
    > "$LOG_DIR/rft_collect_conv_rerun_0_0_retry.log" 2>&1
echo "[finalize] $(date) rerun_0_0 完成"

# --- 3. 合并 + 构建 SFT ---
$PYTHON_BIN - "$COLLECT_DIR" << 'EOF'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
total = success = 0
rows = []
# 幸存原始分片(part*) + 崩溃重采目录(rerun_*) 都要计入
dirs = sorted([d for d in base.iterdir()
               if d.is_dir() and (d.name.startswith("part") or d.name.startswith("rerun"))
               and (d / "summary.json").exists()])
print(f"merging {len(dirs)} result dirs")
assert len(dirs) == 20, f"应有 20 个结果目录(8 幸存 part + 12 rerun), 实际 {len(dirs)}"
for s in [d / "summary.json" for d in dirs]:
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

echo "[finalize] $(date) 全部完成: SFT 数据 = $RFT_DATA"
