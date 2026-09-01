"""
训练脚本

使用 ms-swift 的多轮 GRPO 训练。

使用方式：
    bash scripts/exp01_single_turn.sh
    bash scripts/exp02_multi_turn_3.sh
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("""
Agentic Jailbreak 训练脚本

请使用 ms-swift 命令行工具进行训练：

    swift rlhf --rlhf_type grpo \\
        --external_plugins src/plugin.py \\
        --multi_turn_scheduler gym_scheduler \\
        --env jailbreak_env \\
        --env_config '{
            "skills_path": "data/skills.json",
            "target_port": 8002,
            "guard_port": 8001,
            "max_turns": 5
        }' \\
        --reward_funcs asr_reward format_reward \\
        --reward_weights 1.0 0.1 \\
        --max_turns 5 \\
        --model /home/tiger/models/Qwen/Qwen3-4B \\
        --use_vllm true \\
        --vllm_mode server \\
        --vllm_server_host 127.0.0.1 \\
        --vllm_server_port 8003

或者使用实验脚本：

    bash scripts/exp01_single_turn.sh
    bash scripts/exp02_multi_turn_3.sh
    bash scripts/exp03_multi_turn_5.sh
""")
