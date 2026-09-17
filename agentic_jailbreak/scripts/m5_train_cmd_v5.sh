#!/bin/bash
# M5b v5 (09-16 00:15, 显存红线预案激活): 从 checkpoint-50 续跑, PDB=1 安全模式
# 背景: v3.3 (PDB=2) 于 00:06 触发 68G 红线告警 (74.1G/80G, 3min +10G), 按预案止损
#   → ckpt-50 (bar step 50 = 100 条) 经 swift export merge 进 m3_10turn_merged = 新初始
# 口径: 累计 2000 条 = M3 300 + ckpt-50 段 100 + 本 run 1600 (PDB=1 → 1600 步)
# 09-17 目标修订(分段裁决): 先至 1.0ep(step600, cum1000) → 评估 ckpt-500/600 看趋势 → 上升争取 1.4-2.0 / 平降提前收转 C 轴
# 显存策略: PDB=1 (每调用 1 行, old/ref logps logits 尖峰约减半) + liger + 诊断补丁保留
# 必须 setsid 启动!
set -e
cd /opt/tiger/JudgeForge/jailbreak_research/agentic_jailbreak
export PATH=/opt/tiger/JudgeForge/jailbreak_research/.venv/bin:$PATH
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ROLLOUT_PORT=8004
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=43212

INIT_MODEL=$PWD/output/m5v5_init_ckpt50
OUT=output/multi_turn_10_agent_rft_long
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=3 swift rlhf \
    --rlhf_type grpo \
    --model "$INIT_MODEL" \
    --dataset "$PWD/output/grpo_data.jsonl" \
    --external_plugins "$PWD/src/plugin.py" \
    --multi_turn_scheduler gym_scheduler \
    --gym_env jailbreak_env \
    --use_gym_env true \
    --max_turns 10 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8004 \
    --vllm_server_timeout 1000 \
    --loss_type grpo \
    --per_device_train_batch_size 1 \
    --generation_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --max_steps 1600 \
    --save_steps 50 \
    --learning_rate 1e-5 \
    --num_generations 8 \
    --max_completion_length 2048 \
    --fp16 false --bf16 true \
    --gradient_checkpointing true \
    --use_liger_kernel true \
    --beta 0.05 \
    --output_dir "$OUT" \
    --report_to none \
    --run_name multi_turn_10_agent_rft_long