#!/bin/bash
# GRPO 启动包装: 自动选择空闲 MASTER_PORT, 避免训练退出后的端口残留导致 EADDRINUSE
set -e
cd /home/tiger/jailbreak_research

for port in 43210 43211 43212 43213 43214 43215; do
    if .venv/bin/python -B -c "import socket; s=socket.socket(); s.bind(('0.0.0.0', $port)); s.close()" 2>/dev/null; then
        MASTER_PORT=$port
        break
    fi
done
echo "[launch] MASTER_PORT=$MASTER_PORT"

export ROLLOUT_PORT=8004
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$MASTER_PORT
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=WARN

exec env CUDA_VISIBLE_DEVICES=3 .venv/bin/python -B -m swift.cli.rlhf \
    --rlhf_type grpo \
    --model agentic_jailbreak/output/rft_sft_train1000_v2_merged \
    --dataset agentic_jailbreak/output/grpo_data_64.jsonl \
    --external_plugins agentic_jailbreak/src/plugin.py \
    --multi_turn_scheduler gym_scheduler \
    --gym_env jailbreak_env \
    --use_gym_env true \
    --max_turns 5 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8004 \
    --vllm_server_timeout 3600 \
    --per_device_train_batch_size 2 \
    --generation_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 200 \
    --save_strategy steps \
    --save_steps 50 \
    --learning_rate 1e-5 \
    --num_generations 8 \
    --max_completion_length 512 \
    --fp16 false \
    --tuner_type lora \
    --bf16 false \
    --torch_dtype float32 \
    --beta 0.05 \
    --output_dir /home/tiger/jailbreak_research/agentic_jailbreak/output/multi_turn_5_agent_rft_v2 \
    --report_to none