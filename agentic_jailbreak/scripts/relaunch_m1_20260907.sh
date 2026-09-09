#!/bin/bash
# M1 重启 wrapper v3 (09-08 liger + xformers SDPA patch 后)
cd /home/tiger/jailbreak_research/agentic_jailbreak
setsid nohup bash scripts/exp04_grpo_10turn.sh \
    > output/logs/m1_train_20260908_xformers.log 2>&1 < /dev/null &
echo "exp04 relaunched pid $!"
date
