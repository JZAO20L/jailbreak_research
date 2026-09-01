"""
ms-swift Plugin for Agentic-RL

注册 JailbreakEnv 和 reward functions 到 ms-swift 框架。

使用方式：
    swift rlhf --rlhf_type grpo \
        --external_plugins plugin.py \
        --multi_turn_scheduler gym_scheduler \
        --env jailbreak_env \
        ...
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from self_evolve_skills_jailbreak.agentic_rl.src.jailbreak_env import JailbreakEnv
from self_evolve_skills_jailbreak.agentic_rl.src.reward_functions import (
    ASRReward,
    ProcessReward,
    EfficiencyReward,
    FormatReward,
    register_rewards,
)


# =============================================================================
# 注册 JailbreakEnv
# =============================================================================

def register_env():
    """注册 JailbreakEnv 到 ms-swift"""
    try:
        from swift.rollout.multi_turn import envs
        
        class JailbreakEnvWrapper:
            """Wrapper for JailbreakEnv to match ms-swift GYM interface"""
            
            def __init__(self, **kwargs):
                self.env = JailbreakEnv(**kwargs)
            
            def reset(self, prompt: str):
                """重置环境"""
                return self.env.reset(prompt)
            
            def step(self, action):
                """执行一步"""
                return self.env.step(action)
            
            def close(self):
                """关闭环境"""
                self.env.close()
        
        envs['jailbreak_env'] = JailbreakEnvWrapper
        print("Registered environment: jailbreak_env")
        
    except ImportError:
        print("Warning: swift.rollout.multi_turn not available. Environment not registered.")


# =============================================================================
# 注册 Reward Functions
# =============================================================================

def register_all():
    """注册所有组件"""
    register_env()
    register_rewards()


# 自动注册
register_all()


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Agentic-RL ms-swift Plugin")
    print("=" * 60)
    print()
    print("Registered components:")
    print("  - Environment: jailbreak_env")
    print("  - Rewards: asr_reward, process_reward, efficiency_reward, format_reward")
    print()
    print("Usage example:")
    print()
    print("  swift rlhf --rlhf_type grpo \\")
    print("      --external_plugins plugin.py \\")
    print("      --multi_turn_scheduler gym_scheduler \\")
    print("      --env jailbreak_env \\")
    print("      --env_config '{")
    print("          \"skill_library_path\": \"path/to/skills.json\",")
    print("          \"description_path\": \"path/to/descriptions.json\",")
    print("          \"target_port\": 8002,")
    print("          \"guard_port\": 8001,")
    print("          \"max_turns\": 5")
    print("      }' \\")
    print("      --reward_funcs asr_reward format_reward \\")
    print("      --reward_weights 1.0 0.1 \\")
    print("      --max_turns 5 \\")
    print("      --model /home/tiger/models/Qwen/Qwen3-4B \\")
    print("      --use_vllm true \\")
    print("      --vllm_mode server \\")
    print("      --vllm_server_host 127.0.0.1 \\")
    print("      --vllm_server_port 8003")
    print()
