#!/bin/bash
# =============================================================================
# M1 收工后的自动接力链 (09-10 晚, 本机仅执行)
#   1) 等 exp04 (M1 GRPO) 进程退出
#   2) M1 merge -> output/m1_10turn_merged
#   3) M1 评估 300 条 (RUN_TAG=m1, GPU3 起 policy 8003)
#   4) M2 评估 1000 条 (RUN_TAG=m2, 换 8003 为 M2 merged)
# 停在 M3 之前: M3 是 ~20h 的连续占用, 留人工确认
# 用法: setsid nohup bash scripts/chain_after_m1.sh > output/logs/chain_after_m1.log 2>&1 &
# =============================================================================
set -e
cd /home/tiger/jailbreak_research/agentic_jailbreak
source "$(dirname "$0")/common.sh"
V="$PROJECT_ROOT/.venv/bin"
PY="$V/python"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

log "接力链启动, 等待 M1 trainer 退出..."
while pgrep -f "exp04_grpo_10turn" >/dev/null 2>&1; do sleep 60; done
log "M1 trainer 已退出, 等 60s 让显存释放"
sleep 60

# --- 1) M1 merge (GPU3) ---
CKPT=$(ls -dt output/multi_turn_10_agent_base/*/checkpoint-* 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then log "ERROR: 找不到 M1 checkpoint"; exit 1; fi
log "M1 merge: $CKPT -> output/m1_10turn_merged"
CUDA_VISIBLE_DEVICES=3 "$V/swift" export \
    --model "$BASE_MODEL" --adapters "$CKPT" --merge_lora true \
    --output_dir output/m1_10turn_merged --max_length 4096 2>&1 | tail -2 || true
[ -f output/m1_10turn_merged/config.json ] || { log "ERROR: M1 merge 失败"; exit 1; }

# --- 2) M1 评估 300 (policy 8003 = M1 merged, GPU3) ---
log "起 M1 merged policy 8003 (GPU3)"
CUDA_VISIBLE_DEVICES=3 setsid nohup "$V/vllm" serve output/m1_10turn_merged \
    --port 8003 --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --dtype bfloat16 --trust-remote-code > output/logs/policy_m1_eval.log 2>&1 &
for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:8003/health)" = "200" ] && break
    sleep 2
done
log "RUN_TAG=m1 评估 300 启动"
RUN_TAG=m1 EVAL_WORKERS=12 bash scripts/eval_conv.sh skill_decide 300 10 2 10 2>&1 | tail -2

# --- 3) M2 评估 1000 (换 8003 为 M2 merged) ---
kill $(lsof -ti:8003 2>/dev/null) 2>/dev/null || true
sleep 15
nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read p; do kill "$p" 2>/dev/null || true; done
sleep 10
log "起 M2 merged policy 8003 (GPU3)"
CUDA_VISIBLE_DEVICES=3 setsid nohup "$V/vllm" serve output/rft_sft_conv10turn_e2_merged \
    --port 8003 --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --dtype bfloat16 --trust-remote-code > output/logs/policy_m2_1000.log 2>&1 &
for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:8003/health)" = "200" ] && break
    sleep 2
done
log "RUN_TAG=m2 评估 1000 启动"
RUN_TAG=m2 EVAL_WORKERS=12 bash scripts/eval_conv.sh skill_decide 1000 10 2 10 2>&1 | tail -2

log "接力链完成: M1 评估 300 + M2 评估 1000 均已落盘"
log "下一步待人工确认: M3 (MODEL=output/rft_sft_conv10turn_e2_merged MODEL_TAG=rft exp04)"