# -*- coding: utf-8 -*-
"""
全局配置文件
统一管理模型路径、数据集路径等配置
"""

import os

# =============================================================================
# 模型路径
# =============================================================================

# Base Model (用于改写的策略模型)
BASE_MODEL_PATH = "/home/jiazixiao.jzx/models/Qwen/Qwen3-4B"

# Target Model (被攻击的目标模型)
TARGET_MODEL_PATH = "/home/jiazixiao.jzx/models/Qwen/Qwen3-4B"

# Guard Model (安全分类模型)
GUARD_MODEL_PATH = "/dev/shm/models/Qwen/Qwen3Guard-Gen-4B"

# =============================================================================
# 数据集路径
# =============================================================================

DATA_DIR = "/home/jiazixiao.jzx/.nanobot/TSRL4jailbreak/data/dataset/processed"

# 统一数据集 (10k)
TRAIN_PATH = os.path.join(DATA_DIR, "10k/train.jsonl")
VAL_PATH = os.path.join(DATA_DIR, "10k/val.jsonl")
TEST_PATH = os.path.join(DATA_DIR, "10k/test.jsonl")

# =============================================================================
# 输出路径
# =============================================================================

OUTPUT_DIR = "/home/jiazixiao.jzx/.nanobot/TSRL4jailbreak/output"

# =============================================================================
# GPU 配置
# =============================================================================

DEFAULT_GPU_IDS = "0,1"
DEFAULT_GPU_MEM_UTIL = 0.7
DEFAULT_MAX_MODEL_LEN = 4096

# =============================================================================
# 服务配置
# =============================================================================

DEFAULT_TARGET_PORT = 8102
DEFAULT_GUARD_PORT = 8103
DEFAULT_REWRITE_PORT = 8104