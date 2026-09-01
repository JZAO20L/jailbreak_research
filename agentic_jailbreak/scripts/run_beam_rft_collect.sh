#!/bin/bash
# RFT 数据收集 v2: train 种子集 3 变体 x 2 独立 rollout(拒绝采样)
# 服务: policy=8003(GPU2) target=8002(GPU1) guard=8001(GPU0)
set -e
cd /home/tiger/jailbreak_research
PY=/home/tiger/jailbreak_research/.venv/bin/python
DATA=/home/tiger/jailbreak_research/agentic_jailbreak/output/rft_seed_train1000.json
OUT=/home/tiger/jailbreak_research/agentic_jailbreak/exp/beam_pilot
SKILLS=/home/tiger/jailbreak_research/agentic_jailbreak/exp/skill_asr_sweep/seed_skills_top10.json

RUN() {
    echo "===== $1 ====="
    shift
    "$PY" -B /home/tiger/jailbreak_research/agentic_jailbreak/scripts/run_beam_pilot.py \
        --n 1000 --seed 42 --beams 1 --max_depth 4 --workers 12 --rollouts 2 \
        --policy_port 8003 --data "$DATA" --outdir "$OUT" "$@"
}

RUN no_skill    --no_skills --tag noskill_train_seed_r2
RUN skills_step --skills "$SKILLS" --selection_mode step --tag skills_step_train_seed_r2
RUN skills_traj --skills "$SKILLS" --selection_mode trajectory --tag skills_traj_train_seed_r2
echo "ALL DONE"