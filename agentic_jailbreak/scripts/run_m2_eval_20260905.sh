#!/bin/bash
# =============================================================================
# A 轴 M2 臂: RFT LoRA merge -> 部署 -> C1 协议评估 (test C 前 300, 与 B 轴同口径)
#   merge: checkpoint-22 (epoch 2) -> output/rft_sft_conv10turn_e2_merged
#   部署:  merged 权重替换 GPU2:8003 的 base policy (B 轴已结束)
#   评估:  skill_decide@10skills, 10 轮, 全量累积(C1), 3 片并行
# 对比基线: B 轴 C1_full = 9.33% (28/300)
# Usage: bash scripts/run_m2_eval_20260905.sh
# =============================================================================
set -e
cd /home/tiger/jailbreak_research/agentic_jailbreak
source "$(dirname "$0")/common.sh"
PYTHON_BIN=/home/tiger/jailbreak_research/.venv/bin/python
AGENTIC_DIR=/home/tiger/jailbreak_research/agentic_jailbreak
SFT_DIR="$AGENTIC_DIR/output/rft_sft_conv10turn_e2"
MERGED_DIR="$AGENTIC_DIR/output/rft_sft_conv10turn_e2_merged"
SLICE_DIR="$OUTPUT_DIR/slices/baxis_test300"
BASE_DIR="$OUTPUT_DIR/ataxis_M2_rft"
SKILLS_TOP10="$AGENTIC_DIR/exp/skill_asr_sweep/seed_skills_top10.json"
LOG_DIR="$AGENTIC_DIR/output/logs"
mkdir -p "$BASE_DIR"

# --- 1. merge LoRA (GPU3, checkpoint-22 = epoch 2) ---
ADAPTER="$SFT_DIR/v1-20260905-064326/checkpoint-22"
[ -d "$ADAPTER" ] || ADAPTER=$(ls -dt "$SFT_DIR"/v*/checkpoint-* | head -1)
if [ ! -d "$MERGED_DIR" ] || [ -z "$(ls "$MERGED_DIR"/*.safetensors 2>/dev/null)" ]; then
    log_section "merge LoRA: $ADAPTER -> $MERGED_DIR"
    CUDA_VISIBLE_DEVICES=3 /home/tiger/jailbreak_research/.venv/bin/swift export \
        --model /home/tiger/models/Qwen/Qwen3-4B \
        --adapters "$ADAPTER" \
        --merge_lora true \
        --output_dir "$MERGED_DIR" \
        --max_length 6144
fi
MERGED_MODEL=$(ls -dt "$MERGED_DIR"/*/ 2>/dev/null | head -1)
[ -z "$MERGED_MODEL" ] && MERGED_MODEL="$MERGED_DIR"
log_info "merged 模型: $MERGED_MODEL"

# --- 2. 部署 merged 权重 (GPU2:8003, 替换 base policy) ---
log_section "重启 policy: merged M2 @8003 (max-model-len 24576)"
for pid in $(pgrep -f "port 8003"); do kill "$pid" 2>/dev/null || true; done
sleep 10
for pid in $(pgrep -f "port 8003"); do kill -9 "$pid" 2>/dev/null || true; done
sleep 5
CUDA_VISIBLE_DEVICES=2 /home/tiger/jailbreak_research/.venv/bin/vllm serve "$MERGED_MODEL" \
    --port 8003 --max-model-len 24576 --gpu-memory-utilization 0.9 \
    --dtype float16 --trust-remote-code --served-model-name policy-m2 \
    > "$LOG_DIR/policy_m2_merged.log" 2>&1 &
until curl -s http://127.0.0.1:8003/health > /dev/null 2>&1; do sleep 10; done
log_info "M2 policy 就绪"

# --- 3. C1 协议评估 (test C 前 300, 3 片并行, 无 work_memory/无 ctx_window) ---
PIDS=()
for j in 0 1 2; do
    $PYTHON_BIN "$AGENTIC_DIR/src/eval.py" \
        --test_data "$SLICE_DIR/part$j.jsonl" --skills_path "$SKILLS_TOP10" \
        --policy_port 8003 --target_port 8002 --guard_port 8001 \
        --max_turns 10 --max_samples 1000 \
        --variant skill_decide --top_k_skills 10 --mode conversational \
        --output_dir "$BASE_DIR/part$j" \
        > "$BASE_DIR/part$j.log" 2>&1 &
    PIDS+=("$!")
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
[ "$FAIL" -ne 0 ] && { log_error "M2 评估存在失败进程"; exit 1; }

# --- 4. 汇总 ---
$PYTHON_BIN - "$BASE_DIR" << 'EOF'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
total = success = 0; turns = 0.0; parts = 0
for s in sorted(base.glob("part*/summary.json")):
    d = json.load(open(s))
    total += d["total"]; success += d["success"]; turns += d["avg_turns"] * d["total"]; parts += 1
assert parts == 3 and total == 300, f"M2 评估不完整: {parts} 片 {total} 条"
asr = success / total
print(f"\n===== A 轴 M2 (RFT SFT) vs base (C1 协议, test C 300) =====")
print(f"base    : 9.33% (28/300, avg 9.37 轮)  [B 轴 C1_full 臂]")
print(f"M2  RFT : {asr:.2%} ({success}/{total}, avg {turns/total:.2f} 轮)")
json.dump({"asr": asr, "success": success, "total": total, "avg_turns": turns/total},
          open(base / "summary.json", "w"), ensure_ascii=False, indent=2)
EOF

log_section "M2 臂评估完成"
