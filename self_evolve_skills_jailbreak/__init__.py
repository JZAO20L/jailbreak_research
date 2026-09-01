"""
Self-Evolving Skills Jailbreak

基于自进化 Skills 的 Jailbreak Prompt 生成系统

核心思想：Skills = 多轮迭代攻击中的"捷径"

三阶段流程：
1. 冷启动：攻击种子数据 → 总结 skills
2. 进化：攻击 → 反思 → 更新 skills
3. 测试：固定 skills → 统计 ASR

消融实验参数：
- skill_call_mode: single_call | every_iteration
- skill_extraction_mode: final_prompt | trajectory
- update_strategy: success_only | failure_only | both | statistical
"""

from .src.skill import Skill, DEFAULT_SKILLS
from .src.skill_library import SkillLibrary
from .src.attacker import SkillGuidedAttacker, AttackResult
from .src.reflector import SkillReflector, SkillUpdater, ReflectionResult

__version__ = "0.1.0"
__all__ = [
    "Skill",
    "DEFAULT_SKILLS",
    "SkillLibrary",
    "SkillGuidedAttacker",
    "AttackResult",
    "SkillReflector",
    "SkillUpdater",
    "ReflectionResult",
]