#!/bin/bash
# 全量 1000 条 ASR 评测: Crescendo + TAP
# 数据: self_evolve_skills_jailbreak/data/test_prompts.json (1000 条, 与 agent 实验同源)
# 服务: target=8002 (Qwen3-4B-SafeRL) guard=8001 (Qwen3Guard-Gen-4B)
set -e
cd /home/tiger/jailbreak_research
source agentic_jailbreak/scripts/common.sh

RUN() {
    local name=$1; shift
    local outdir=$1; shift
    log_section "运行 $name 全量评测"
    .venv/bin/python -B -m baselines.cli batch \
        --input self_evolve_skills_jailbreak/data/test_prompts.json \
        --strategy "$name" \
        --output "$outdir" \
        --limit 1000 --evaluate --max_workers 8 \
        --target-port 8002 --guard-port 8001 "$@"
}

RUN crescendo baselines/experiments/crescendo_asr --max-turns 8
RUN tap baselines/experiments/tap_asr --max-turns 6 --branching-factor 2 --prune-top-k 1

log_section "两个基线全量评测完成"