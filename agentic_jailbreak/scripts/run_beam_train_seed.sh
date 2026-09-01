#!/bin/bash
# 在 train 种子集(rft_seed_train1000.json)上重跑三个单流 beam 实验(全量1000)
# 全部使用绝对路径,避免后台环境相对路径问题
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
        --n 1000 --seed 42 --beams 1 --max_depth 4 --workers 16 \
        --policy_port 8003 --data "$DATA" --outdir "$OUT" "$@"
}

RUN no_skill    --no_skills --tag noskill_train_seed
RUN skills_step --skills "$SKILLS" --selection_mode step --tag skills_step_train_seed
RUN skills_traj --skills "$SKILLS" --selection_mode trajectory --tag skills_traj_train_seed
echo "ALL DONE"