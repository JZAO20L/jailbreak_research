#!/bin/bash
# =============================================================================
# 09-11: 先补 A 轴同口径 (base@1000, M1@1000), 再启动 M3 (RFT+GRPO, ~20h)
# 顺序: 备份 300 子集 → base@1000 → M1@1000 → 换 rollout 为 M2 merged → M3 训练
# 任何一步失败即退出(set -e), 不会带病跑 M3
# 用法: setsid nohup bash scripts/chain_eval1000_then_m3.sh > output/logs/chain_eval_m3.log 2>&1 &
# =============================================================================
set -e
cd /home/tiger/jailbreak_research/agentic_jailbreak
source "$(dirname "$0")/common.sh"
V="$PROJECT_ROOT/.venv/bin"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# --- 0) 备份 300 子集结果(1000 条版会覆盖同名目录) ---
for t in m0 m1; do
    d="output/eval_results/conv_skill_decide_top10_10turn_${t}"
    if [ -d "$d" ] && [ ! -d "${d}_300subset" ]; then
        cp -a "$d" "${d}_300subset"
        log "备份 $d -> ${d}_300subset"
    fi
done

kill_port(){ # $1=端口 $2=GPU
    fuser -k "$1/tcp" 2>/dev/null || true
    sleep 8
    P=$(nvidia-smi -i "$2" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ')
    [ -n "$P" ] && kill $P 2>/dev/null || true
    sleep 10
}

serve_policy(){ # $1=模型 $2=日志
    log "起 policy 8003 = $1 (GPU3)"
    CUDA_VISIBLE_DEVICES=3 setsid nohup "$V/vllm" serve "$1" --port 8003 \
        --max-model-len 32768 --gpu-memory-utilization 0.85 \
        --dtype bfloat16 --trust-remote-code > "$2" 2>&1 &
    for i in $(seq 1 600); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:8003/health)" = "200" ] && { log "8003 就绪 (${i}s)"; return 0; }
        sleep 2
    done
    log "ERROR: 8003 起不来: $1"; exit 1
}

# --- 1) base@1000 ---
kill_port 8003 3
serve_policy "$BASE_MODEL" output/logs/policy_base_1000.log
log "RUN_TAG=m0 @ 1000 启动"
RUN_TAG=m0 EVAL_WORKERS=12 bash scripts/eval_conv.sh skill_decide 1000 10 2 10 2>&1 | tail -1

# --- 2) M1@1000 ---
kill_port 8003 3
serve_policy "output/m1_10turn_merged" output/logs/policy_m1_1000.log
log "RUN_TAG=m1 @ 1000 启动"
RUN_TAG=m1 EVAL_WORKERS=12 bash scripts/eval_conv.sh skill_decide 1000 10 2 10 2>&1 | tail -1

# --- 3) M3: rollout 8004 换 M2 merged (GPU2), trainer GPU3 ---
kill_port 8003 3
kill_port 8004 2
log "起 rollout 8004 = M2 merged (GPU2)"
CUDA_VISIBLE_DEVICES=2 setsid nohup "$V/swift" rollout \
    --model output/rft_sft_conv10turn_e2_merged --vllm_tensor_parallel_size 1 \
    --port 8004 --vllm_max_model_len 32768 --vllm_gpu_memory_utilization 0.8 \
    --torch_dtype bfloat16 > output/logs/rollout_m3.log 2>&1 &
code=000
for i in $(seq 1 900); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:8004/health 2>/dev/null || echo 000)
    [ "$code" = "200" ] && break
    [ "$code" != "000" ] && [ "$code" != "200" ] && break   # swift rollout 常回 307, 服即活
    sleep 2
done
log "rollout 8004 health=$code"
log "启动 M3: MODEL=<M2 merged> MODEL_TAG=rft, MAX_TURNS=10, bf16 (GPU3)"
MODEL="$(pwd)/output/rft_sft_conv10turn_e2_merged" MODEL_TAG=rft MAX_TURNS=10 DTYPE=bf16 \
    setsid nohup bash scripts/exp04_grpo_10turn.sh > output/logs/m3_20260911.log 2>&1 &
log "M3 已启动 (pid $!), 日志 output/logs/m3_20260911.log"
log "M3 早期健康检查: 前 10-20 步 grad_norm 若复现 M1 的 0.9-1.2 震荡, 应提前停"