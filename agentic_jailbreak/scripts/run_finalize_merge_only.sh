#!/bin/bash
# 执行 finalize 的 3) merge + 4) build SFT 步骤(1/2 步已完成: policy 重启 + rerun_0_0 补跑)
# 被抽取片段依赖 finalize 脚本前部定义的变量, 在此显式导出
set -e
cd /home/tiger/jailbreak_research/agentic_jailbreak
export PYTHON_BIN=/home/tiger/jailbreak_research/.venv/bin/python
export AGENTIC_DIR=/home/tiger/jailbreak_research/agentic_jailbreak
export COLLECT_DIR="$AGENTIC_DIR/output/rft_collect_conv_10turn"
export RFT_DATA="$AGENTIC_DIR/output/rft_data_conv_10turn.jsonl"
export SKILLS_TOP10="$AGENTIC_DIR/exp/skill_asr_sweep/seed_skills_top10.json"
sed -n "/--- 3\. 合并/,/全部完成/p" scripts/finalize_rft_collect_20260903.sh | bash
