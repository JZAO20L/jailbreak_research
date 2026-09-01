"""
Skill库管理

Skill的存储、检索、聚类等核心操作
"""

import os
import json
import uuid
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np

from .skill import Skill, DEFAULT_SKILLS


class SkillLibrary:
    """
    Skill库管理类

    负责：
    - Skill存储和加载
    - Skill检索
    - Skill聚类
    - Skill合并

    线程安全：使用锁保护文件写入操作
    """

    def __init__(
        self,
        storage_path: str = "skills/skills_library.json",
        max_skills: int = 100,
        similarity_threshold: float = 0.75,
        max_skill_length: int = 500,
    ):
        self.storage_path = storage_path
        self.max_skills = max_skills
        self.similarity_threshold = similarity_threshold
        self.max_skill_length = max_skill_length
        self._lock = threading.Lock()  # 线程安全锁

        self.skills: Dict[str, Skill] = {}
        self._load()

    # =========================================================================
    # 存储
    # =========================================================================

    def _load(self):
        """加载Skill库"""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for skill_data in data.get("skills", []):
                    skill = Skill.from_dict(skill_data)
                    if skill.is_valid():
                        self.skills[skill.skill_id] = skill
            print(f"[SkillLibrary] Loaded {len(self.skills)} skills")
        else:
            # 初始化默认Skills
            for skill in DEFAULT_SKILLS:
                self.skills[skill.skill_id] = skill
            self._save()
            print(f"[SkillLibrary] Initialized with {len(DEFAULT_SKILLS)} default skills")

    def _save(self):
        """保存Skill库"""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "skills": [s.to_dict() for s in self.skills.values()],
            "metadata": {
                "total_skills": len(self.skills),
                "last_updated": datetime.now().isoformat(),
            }
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, source_path: str) -> int:
        """
        从指定文件加载Skills（用于Transfer实验）

        Args:
            source_path: 源文件路径（如 skills_library.json）

        Returns:
            加载的Skill数量
        """
        if not os.path.exists(source_path):
            print(f"[SkillLibrary] Source file not found: {source_path}")
            return 0

        loaded_count = 0
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for skill_data in data.get("skills", []):
                skill = Skill.from_dict(skill_data)
                if skill.is_valid():
                    self.skills[skill.skill_id] = skill
                    loaded_count += 1

        print(f"[SkillLibrary] Loaded {loaded_count} skills from {source_path}")
        return loaded_count

    # =========================================================================
    # 基本操作
    # =========================================================================

    def add_skill(self, skill: Skill) -> bool:
        """
        添加新Skill（线程安全）

        Args:
            skill: 要添加的Skill

        Returns:
            是否成功添加
        """
        with self._lock:
            if not skill.is_valid():
                return False

            # 检查是否已存在相似Skill
            for existing in self.skills.values():
                if self._compute_similarity(skill.content, existing.content) > 0.9:
                    # 太相似，不添加
                    return False

            self.skills[skill.skill_id] = skill
            self._save()
            return True

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """获取指定Skill"""
        return self.skills.get(skill_id)

    def get_skill_by_name(self, name: str) -> Optional[Skill]:
        """按名称查找Skill"""
        for skill in self.skills.values():
            if skill.name == name:
                return skill
        return None

    def update_skill(self, skill: Skill):
        """更新Skill（线程安全）"""
        with self._lock:
            if skill.skill_id in self.skills:
                self.skills[skill.skill_id] = skill
                self._save()

    def remove_skill(self, skill_id: str) -> bool:
        """删除Skill（线程安全）"""
        with self._lock:
            if skill_id in self.skills:
                del self.skills[skill_id]
                self._save()
                return True
            return False

    def list_skills(self) -> List[Skill]:
        """列出所有Skills"""
        return list(self.skills.values())

    def count(self) -> int:
        """获取Skill数量"""
        return len(self.skills)

    # =========================================================================
    # 检索
    # =========================================================================

    def retrieve(self, prompt: str, top_k: int = 1) -> List[Skill]:
        """
        检索最匹配的Skills

        Args:
            prompt: 原始prompt
            top_k: 返回top_k个

        Returns:
            匹配的Skills列表
        """
        if not self.skills:
            return []

        # 提取prompt特征
        prompt_features = self._extract_prompt_features(prompt)

        # 评分所有Skills
        scored_skills = []
        for skill in self.skills.values():
            score = self._score_skill_match(skill, prompt_features)
            scored_skills.append((skill, score))

        # 排序
        scored_skills.sort(key=lambda x: x[1], reverse=True)

        # 返回top_k
        return [s[0] for s in scored_skills[:top_k]]

    def _extract_prompt_features(self, prompt: str) -> Dict:
        """提取prompt特征"""
        prompt_lower = prompt.lower()
        return {
            "length": len(prompt),
            "keywords": self._extract_keywords(prompt),
            "harm_type": self._classify_harm_type(prompt_lower),
            "has_question": "?" in prompt,
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = []
        keyword_patterns = [
            "how to", "guide", "instructions", "create", "make",
            "build", "explain", "write", "help", "provide",
            "expert", "professional", "research", "study",
        ]
        text_lower = text.lower()
        for kw in keyword_patterns:
            if kw in text_lower:
                keywords.append(kw)
        return keywords

    def _classify_harm_type(self, prompt_lower: str) -> str:
        """分类伤害类型"""
        type_keywords = {
            "violence": ["weapon", "gun", "bomb", "explosive", "attack"],
            "dangerous_substance": ["drug", "poison", "chemical", "toxic"],
            "cybersecurity": ["hack", "exploit", "malware", "virus"],
            "financial": ["steal", "fraud", "scam", "money"],
            "social": ["manipulate", "deceive", "trick"],
        }

        for harm_type, kws in type_keywords.items():
            if any(kw in prompt_lower for kw in kws):
                return harm_type

        return "general"

    def _score_skill_match(self, skill: Skill, features: Dict) -> float:
        """计算Skill匹配得分"""
        # 基础得分 = 质量
        score = skill.quality_score

        # 匹配关键词加分
        prompt_keywords = features["keywords"]
        matched = sum(1 for p in skill.applicable_patterns if p in prompt_keywords)
        if matched > 0:
            score *= (1 + matched * 0.3)

        # 伤害类型匹配加分
        if features["harm_type"] in skill.applicable_patterns:
            score *= 1.5

        return score

    # =========================================================================
    # 聚类和合并
    # =========================================================================

    def cluster_skills(self) -> Dict[str, List[str]]:
        """
        聚类相似的Skills

        Returns:
            cluster_id -> skill_ids 映射
        """
        skill_ids = list(self.skills.keys())
        n = len(skill_ids)

        if n == 0:
            return {}

        # 计算相似度矩阵
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._compute_similarity(
                    self.skills[skill_ids[i]].content,
                    self.skills[skill_ids[j]].content,
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        # 简单聚类
        clusters = defaultdict(list)
        visited = set()

        for i in range(n):
            if skill_ids[i] in visited:
                continue

            cluster_id = f"cluster_{i}"
            clusters[cluster_id].append(skill_ids[i])
            visited.add(skill_ids[i])
            self.skills[skill_ids[i]].cluster_id = cluster_id

            for j in range(i + 1, n):
                if skill_ids[j] not in visited and similarity_matrix[i, j] > self.similarity_threshold:
                    clusters[cluster_id].append(skill_ids[j])
                    visited.add(skill_ids[j])
                    self.skills[skill_ids[j]].cluster_id = cluster_id

        return dict(clusters)

    def merge_cluster(self, cluster_id: str, skill_ids: List[str]) -> Optional[Skill]:
        """
        合并一个聚类中的Skills（线程安全）

        策略：保留质量最高的Skill，合并统计信息

        Args:
            cluster_id: 聚类ID
            skill_ids: 聚类中的Skill IDs

        Returns:
            合并后的Skill
        """
        with self._lock:
            if len(skill_ids) <= 1:
                return None

            # 找到质量最高的Skill作为主Skill
            skills = [self.skills[sid] for sid in skill_ids if sid in self.skills]
            if len(skills) <= 1:
                return None

            best_skill = max(skills, key=lambda s: s.quality_score)

            # 合并其他Skill的统计信息
            for skill in skills:
                if skill.skill_id != best_skill.skill_id:
                    best_skill.usage_count += skill.usage_count
                    best_skill.success_count += skill.success_count
                    # 合并适用模式
                    for p in skill.applicable_patterns:
                        if p not in best_skill.applicable_patterns:
                            best_skill.applicable_patterns.append(p)
                    # 合并父IDs
                    for pid in skill.parent_ids:
                        if pid not in best_skill.parent_ids:
                            best_skill.parent_ids.append(pid)
                    # 删除被合并的Skill
                    if skill.skill_id in self.skills:
                        del self.skills[skill.skill_id]

            best_skill.source = "merged"
            best_skill._update_quality()
            self._save()

            return best_skill

    # =========================================================================
    # 维护
    # =========================================================================

    def prune_low_quality(self, min_success_rate: float = 0.1, min_usage: int = 10) -> int:
        """
        清理低效Skills（线程安全）

        Returns:
            删除的数量
        """
        with self._lock:
            to_remove = []
            for skill_id, skill in self.skills.items():
                if skill.is_low_quality(min_success_rate, min_usage):
                    to_remove.append(skill_id)

            for skill_id in to_remove:
                del self.skills[skill_id]

            if to_remove:
                self._save()

            return len(to_remove)

    def trim_long_skills(self) -> int:
        """
        裁剪过长的Skills（线程安全）

        Returns:
            裁剪的数量
        """
        with self._lock:
            trimmed = 0
            for skill in self.skills.values():
                if len(skill.content) > self.max_skill_length:
                    skill.content = skill.content[:self.max_skill_length]
                    skill.length = len(skill.content)
                    trimmed += 1

            if trimmed:
                self._save()

            return trimmed

    def limit_count(self) -> int:
        """
        限制Skills数量（线程安全）

        Returns:
            删除的数量
        """
        with self._lock:
            if len(self.skills) > self.max_skills:
                # 按质量排序，保留top
                sorted_skills = sorted(
                    self.skills.values(),
                    key=lambda s: s.quality_score,
                    reverse=True
                )
                keep_ids = {s.skill_id for s in sorted_skills[:self.max_skills]}
                to_remove = [sid for sid in self.skills if sid not in keep_ids]
                for sid in to_remove:
                    del self.skills[sid]
                self._save()
                return len(to_remove)

            return 0

    def run_maintenance(self, min_success_rate: float = 0.1, min_usage: int = 10) -> Dict:
        """
        执行完整维护流程

        Returns:
            维护统计信息
        """
        stats = {
            "pruned": self.prune_low_quality(min_success_rate, min_usage),
            "trimmed": self.trim_long_skills(),
            "clusters": 0,
            "merged": 0,
            "limited": self.limit_count(),
        }

        # 聚类和合并
        clusters = self.cluster_skills()
        stats["clusters"] = len(clusters)

        for cluster_id, skill_ids in clusters.items():
            if len(skill_ids) > 1:
                merged = self.merge_cluster(cluster_id, skill_ids)
                if merged:
                    stats["merged"] += 1

        stats["final_count"] = len(self.skills)

        return stats

    # =========================================================================
    # 相似度计算
    # =========================================================================

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度（Jaccard）
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    # =========================================================================
    # 统计
    # =========================================================================

    def get_stats(self) -> Dict:
        """获取Skill库统计信息"""
        skills = list(self.skills.values())

        if not skills:
            return {
                "count": 0,
                "avg_quality": 0,
                "avg_success_rate": 0,
                "avg_usage": 0,
            }

        return {
            "count": len(skills),
            "avg_quality": np.mean([s.quality_score for s in skills]),
            "avg_success_rate": np.mean([s.success_rate for s in skills]),
            "avg_usage": np.mean([s.usage_count for s in skills]),
            "total_usage": sum(s.usage_count for s in skills),
            "total_success": sum(s.success_count for s in skills),
            "sources": dict(defaultdict(int, [(s.source, 1) for s in skills])),
        }

    # =========================================================================
    # 导出
    # =========================================================================

    def export_to_skill_md(
        self,
        output_dir: str,
        min_quality: float = 0.0,
        min_usage: int = 0,
        top_k: Optional[int] = None,
        skill_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        导出 skills 为 SKILL.md 格式

        业内标准格式：YAML frontmatter + Markdown body
        便于论文展示和方法认可

        Args:
            output_dir: 输出目录
            min_quality: 最小质量分数阈值
            min_usage: 最小使用次数阈值
            top_k: 只导出 top-k 个高质量 skills
            skill_ids: 指定导出的 skill IDs

        Returns:
            导出统计信息
        """
        # 使用延迟导入避免循环依赖
        try:
            from ..utils.skill_exporter import SkillExporter
        except ImportError:
            from self_evolve_skills_jailbreak.utils.skill_exporter import SkillExporter

        return SkillExporter.export_library(
            library=self,
            output_dir=output_dir,
            min_quality=min_quality,
            min_usage=min_usage,
            top_k=top_k,
            skill_ids=skill_ids,
        )