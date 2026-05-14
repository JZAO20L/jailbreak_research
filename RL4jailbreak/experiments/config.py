# -*- coding: utf-8 -*-
"""
统一实验配置 - 根据 TODO.md "实验重做"

基础配置：
- 数据集: jailbreak_research/data
- qwen3-4b和guard模型: /mnt/bn/chenxiong/mlx/users/jiazixiao/models

上下文长度：
- policy: 4k (4096)
- target, judge, guard: 8k (8192)

RL实验统一训练：1000步

GPU配置：
- eval时：GPU0:policy, GPU1:target, GPU2:guard (3卡)
- train时：GPU0&1:policy并发训练, GPU2:target, GPU3:guard (4卡)
"""

# =============================================================================
# 模型路径
# =============================================================================
MODEL_BASE = "/mnt/bn/chenxiong/mlx/users/jiazixiao/models"

POLICY_MODEL = f"{MODEL_BASE}/Qwen3-4B"
TARGET_MODEL = f"{MODEL_BASE}/Qwen3-4B"
GUARD_MODEL = f"{MODEL_BASE}/Qwen3Guard-Gen-4B"

# =============================================================================
# 上下文长度
# =============================================================================
POLICY_MAX_MODEL_LEN = 4096
TARGET_MAX_MODEL_LEN = 8192
GUARD_MAX_MODEL_LEN = 8192

# =============================================================================
# 训练配置
# =============================================================================
MAX_STEPS = 1000  # RL实验统一训练1000步

# =============================================================================
# GPU配置 - 端口
# =============================================================================
POLICY_PORT = 8003
TARGET_PORT = 8001
GUARD_PORT = 8002

# =============================================================================
# GPU配置 - 显存利用率
# =============================================================================
POLICY_GPU_UTIL = 0.9
TARGET_GPU_UTIL = 0.4
GUARD_GPU_UTIL = 0.4

# =============================================================================
# 数据集路径
# =============================================================================
DATA_BASE = "jailbreak_research/data/dataset/processed/10k"
TRAIN_DATA = f"{DATA_BASE}/train.jsonl"
VAL_DATA = f"{DATA_BASE}/val.jsonl"
TEST_DATA = f"{DATA_BASE}/test.jsonl"

# =============================================================================
# qwen3-max API配置
# =============================================================================
QWEN3_MAX_API_KEY = "sk-sp-eb50d67ca64a451b820cc4ab87ef8e6c"
QWEN3_MAX_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
QWEN3_MAX_MODEL = "qwen3-max-2026-01-23"


def get_gpu_config(mode: str) -> dict:
    """
    获取GPU配置
    
    Args:
        mode: "eval" 或 "train"
    
    Returns:
        dict: GPU配置信息
    """
    if mode == "eval":
        return {
            "num_gpus": 3,
            "policy_gpu": "0",
            "target_gpu": "1",
            "guard_gpu": "2",
            "description": "eval时使用3卡：0:policy, 1:target, 2:guard",
        }
    elif mode == "train":
        return {
            "num_gpus": 4,
            "policy_gpu": "0,1",  # 并发训练
            "target_gpu": "2",
            "guard_gpu": "3",
            "description": "train时使用4卡：0&1:policy并发训练, 2:target, 3:guard",
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")


def print_config():
    """打印配置信息"""
    print("=" * 60)
    print("实验统一配置 (根据 TODO.md '实验重做')")
    print("=" * 60)
    print(f"模型路径:")
    print(f"  Policy: {POLICY_MODEL}")
    print(f"  Target: {TARGET_MODEL}")
    print(f"  Guard:  {GUARD_MODEL}")
    print("-" * 60)
    print(f"上下文长度:")
    print(f"  Policy: {POLICY_MAX_MODEL_LEN}")
    print(f"  Target: {TARGET_MAX_MODEL_LEN}")
    print(f"  Guard:  {GUARD_MAX_MODEL_LEN}")
    print("-" * 60)
    print(f"训练步数: {MAX_STEPS}")
    print("-" * 60)
    print("GPU配置:")
    print(f"  eval: {get_gpu_config('eval')['description']}")
    print(f"  train: {get_gpu_config('train')['description']}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()