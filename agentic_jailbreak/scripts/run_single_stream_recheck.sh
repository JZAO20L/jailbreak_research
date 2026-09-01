#!/bin/bash
# 单流(beams=1)三变体全量重测: no_skill / skills_step / skills_trajectory
# 与 beam_pilot 双流全量同源同 seed,便于对比
set -e

source "$(dirname "$0")/common.sh"

cd "$PROJECT_ROOT"
OUT="$AGENTIC_DIR/exp/beam_pilot"
SKILLS="$AGENTIC_DIR/exp/skill_asr_sweep/seed_skills_top10.json"
PY="$PROJECT_ROOT/.venv/bin/python"

RUN() {
    log_section "运行: $1"
    shift
    "$PY" agentic_jailbreak/scripts/run_beam_pilot.py \
        --n 1000 --seed 42 --beams 1 --max_depth 4 --workers 16 \
        --policy_port 8003,8004 --outdir "$OUT" "$@"
}

RUN "no_skill 单流"       --no_skills --tag noskill_single
RUN "skills step 单流"    --skills "$SKILLS" --selection_mode step --tag skills_single_step
RUN "skills traj 单流"    --skills "$SKILLS" --selection_mode trajectory --tag skills_single_trajectory

log_section "三个单流变体全部完成"