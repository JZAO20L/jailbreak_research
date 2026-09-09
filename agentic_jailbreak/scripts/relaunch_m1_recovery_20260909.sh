#!/bin/bash
# =============================================================================
# M1 恢复臂 relaunch (09-09): plain-4B target + 5 轮 + bf16 原生训练
#   - 发散修复: DTYPE=bf16 (exp03 证明 500 步稳定; fp16 下 GRPO loss 数值爆炸)
#   - 轮次降到 5: 缩短序列 -> 减轻显存 + 训练更稳
# 前置: swap_target_plain4b.sh 已执行 (target=plain Qwen3-4B @8002)
# =============================================================================
cd /home/tiger/jailbreak_research/agentic_jailbreak
export MAX_TURNS=5
export DTYPE=bf16
setsid nohup bash scripts/exp04_grpo_10turn.sh \
    > output/logs/m1_recovery_20260909_5turn_bf16.log 2>&1 < /dev/null &
echo "recovery relaunched pid $!"
date
