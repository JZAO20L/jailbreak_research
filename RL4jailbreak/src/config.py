# -*- coding: utf-8 -*-
"""
全局配置文件
统一管理模型路径、数据集路径等配置
"""

import os

# =============================================================================
# 路径配置 (相对于此文件的父目录)
# =============================================================================

# 项目根目录 (RL4jailbreak)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 统一数据集路径 (jailbreak/data/dataset/processed/10k/)
DATA_DIR = os.path.join(PROJECT_DIR, "../data/dataset/processed/10k")

# 训练/验证/测试集
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH = os.path.join(DATA_DIR, "val.jsonl")
TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

# =============================================================================
# 模型路径 (服务器环境)
# =============================================================================

# Base Model (用于改写的策略模型)
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3-4B"

# Target Model (被攻击的目标模型)
TARGET_MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3-4B"

# Guard Model (安全分类模型)
GUARD_MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B"

# =============================================================================
# GPU 配置
# =============================================================================

DEFAULT_GPU_IDS = "0,1"
DEFAULT_GPU_MEM_UTIL = 0.7
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_MAX_COMPLETION_LEN = 2048

# =============================================================================
# 服务配置
# =============================================================================

DEFAULT_TARGET_PORT = 8001
DEFAULT_GUARD_PORT = 8002
DEFAULT_REWRITE_PORT = 8003
