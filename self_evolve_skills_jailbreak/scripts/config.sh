#!/bin/bash
# Self Evolve Skills 实验配置
#
# 服务配置：
# - Guard (GPU0): 端口 8002, Qwen3Guard-Gen-4B
# - Target (GPU1-2): 端口 8001, Qwen3-4B

export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_MODELSCOPE=true

# Guard 模型
GUARD_PORT=8002
GUARD_MODEL_PATH="/home/tiger/models/Qwen/Qwen3Guard-Gen-4B"

# Target 模型
TARGET_PORT=8001
TARGET_MODEL_PATH="/home/tiger/models/Qwen/Qwen3-4B"

# Skill 库路径
SKILL_LIBRARY_PATH="self_evolve_skills_jailbreak/skills/skills_library.json"

# 实验参数
MAX_ITERATIONS=10
RETRIEVE_TOP_K=3
NUM_EPOCHS=3