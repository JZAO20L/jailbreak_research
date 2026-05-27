"""
配置文件
"""

import os

# 项目根目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# 模型配置
# =============================================================================

MODEL_CONFIG = {
    "target_model": "/home/tiger/models/Qwen/Qwen3-4B",
    "guard_model": "/home/tiger/models/Qwen/Qwen3Guard-Gen-4B",
    "target_port": 8001,
    "guard_port": 8002,
    "max_tokens": 512,
    "temperature": 0.7,
}

# =============================================================================
# Skill配置
# =============================================================================

SKILL_CONFIG = {
    "storage_path": os.path.join(PROJECT_DIR, "skills/skills_library.json"),
    "max_skills": 100,
    "max_skill_length": 500,
    "min_success_rate": 0.1,
    "min_usage": 10,
    "similarity_threshold": 0.75,
    "maintenance_interval": 50,  # 每50次攻击后维护
}

# =============================================================================
# 攻击配置
# =============================================================================

ATTACK_CONFIG = {
    "pair_max_iterations": 10,
    "autodan_generations": 5,
    "autodan_population_size": 4,
    "genetic_mutation_rate": 0.3,
}

# =============================================================================
# 日志配置
# =============================================================================

LOG_CONFIG = {
    "log_dir": os.path.join(PROJECT_DIR, "logs"),
    "log_level": "INFO",
}