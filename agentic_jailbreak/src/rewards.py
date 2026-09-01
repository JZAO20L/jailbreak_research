"""
奖励函数

支持多种奖励函数：
1. ASR Reward: 最终攻击成功率
2. Process Reward: 每轮尝试的过程奖励
3. Efficiency Reward: 攻击效率（轮数越少越好）
4. Format Reward: 输出格式正确性
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ASRReward:
    """
    ASR Reward: 最终攻击成功率
    
    从 rollout_infos 中获取最终的 ASR 结果
    """
    
    def __init__(self, success_reward: float = 1.0, failure_reward: float = 0.0):
        self.success_reward = success_reward
        self.failure_reward = failure_reward
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        计算 ASR reward
        
        Args:
            completions: 模型生成的文本（未使用，从 rollout_infos 获取）
            **kwargs: 包含 rollout_infos
        
        Returns:
            rewards: 每个 completion 的奖励
        """
        rollout_infos = kwargs.get("rollout_infos", {})
        
        rewards = []
        for info in rollout_infos:
            success = info.get("success", False)
            if success:
                rewards.append(self.success_reward)
            else:
                rewards.append(self.failure_reward)
        
        return rewards


class ProcessReward:
    """
    Process Reward: 每轮尝试的过程奖励
    
    鼓励模型多尝试，即使最终失败也给予部分奖励
    """
    
    def __init__(
        self,
        per_turn_reward: float = 0.1,
        success_bonus: float = 1.0,
        max_turns: int = 5,
    ):
        self.per_turn_reward = per_turn_reward
        self.success_bonus = success_bonus
        self.max_turns = max_turns
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        计算 process reward
        
        reward = per_turn_reward * num_turns + success_bonus * success
        """
        rollout_infos = kwargs.get("rollout_infos", {})
        
        rewards = []
        for info in rollout_infos:
            total_turns = info.get("total_turns", 1)
            success = info.get("success", False)
            
            reward = self.per_turn_reward * total_turns
            if success:
                reward += self.success_bonus
            
            rewards.append(reward)
        
        return rewards


class EfficiencyReward:
    """
    Efficiency Reward: 攻击效率
    
    鼓励模型用更少的轮数成功攻击
    reward = success_bonus * (1 - (turns - 1) / max_turns)
    """
    
    def __init__(
        self,
        success_bonus: float = 1.0,
        max_turns: int = 5,
    ):
        self.success_bonus = success_bonus
        self.max_turns = max_turns
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        计算 efficiency reward
        """
        rollout_infos = kwargs.get("rollout_infos", {})
        
        rewards = []
        for info in rollout_infos:
            total_turns = info.get("total_turns", 1)
            success = info.get("success", False)
            
            if success:
                # 成功：轮数越少，奖励越高
                efficiency = 1.0 - (total_turns - 1) / self.max_turns
                reward = self.success_bonus * efficiency
            else:
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards


class FormatReward:
    """
    Format Reward: 输出格式正确性
    
    检查模型输出是否包含 "Selection: Skill [index]" 和 "Adapted Strategy:"
    """
    
    def __init__(self, correct_reward: float = 0.1, incorrect_reward: float = 0.0):
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        检查格式正确性
        """
        import re
        
        rewards = []
        for completion in completions:
            has_selection = bool(re.search(r'Selection:\s*Skill\s*\[?\d+\]?', completion))
            has_adapted = bool(re.search(r'Adapted Strategy:', completion))
            
            if has_selection and has_adapted:
                rewards.append(self.correct_reward)
            else:
                rewards.append(self.incorrect_reward)
        
        return rewards


class CombinedReward:
    """
    Combined Reward: 组合多个奖励函数
    """
    
    def __init__(
        self,
        reward_functions: List[Any],
        weights: List[float],
    ):
        """
        Args:
            reward_functions: 奖励函数列表
            weights: 对应的权重
        """
        self.reward_functions = reward_functions
        self.weights = weights
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        计算组合奖励
        """
        all_rewards = []
        for reward_fn in self.reward_functions:
            rewards = reward_fn(completions, **kwargs)
            all_rewards.append(rewards)
        
        # 加权求和
        combined = []
        for i in range(len(completions)):
            total = sum(
                w * all_rewards[j][i]
                for j, w in enumerate(self.weights)
            )
            combined.append(total)
        
        return combined


# =============================================================================
# ms-swift 插件注册
# =============================================================================

def register_rewards():
    """注册所有奖励函数到 ms-swift"""
    try:
        from swift.rewards import ORM, orms
        
        # ASR Reward
        class ASRRewardORM(ORM):
            def __init__(self):
                self.reward_fn = ASRReward()
            
            def __call__(self, completions, **kwargs):
                return self.reward_fn(completions, **kwargs)
        
        orms['asr_reward'] = ASRRewardORM
        
        # Process Reward
        class ProcessRewardORM(ORM):
            def __init__(self):
                self.reward_fn = ProcessReward()
            
            def __call__(self, completions, **kwargs):
                return self.reward_fn(completions, **kwargs)
        
        orms['process_reward'] = ProcessRewardORM
        
        # Efficiency Reward
        class EfficiencyRewardORM(ORM):
            def __init__(self):
                self.reward_fn = EfficiencyReward()
            
            def __call__(self, completions, **kwargs):
                return self.reward_fn(completions, **kwargs)
        
        orms['efficiency_reward'] = EfficiencyRewardORM
        
        # Format Reward
        class FormatRewardORM(ORM):
            def __init__(self):
                self.reward_fn = FormatReward()
            
            def __call__(self, completions, **kwargs):
                return self.reward_fn(completions, **kwargs)
        
        orms['format_reward'] = FormatRewardORM
        
        print("Registered reward functions: asr_reward, process_reward, efficiency_reward, format_reward")
        
    except ImportError:
        print("Warning: swift.rewards not available. Reward functions not registered.")


# 自动注册
if __name__ == "__main__":
    register_rewards()
