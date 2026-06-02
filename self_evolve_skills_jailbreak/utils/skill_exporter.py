"""
Skill 导出工具

将 JSON 格式的 Skill 导出为业内标准的 SKILL.md + YAML frontmatter 格式
"""

import os
import yaml
from typing import List, Optional, Callable
from datetime import datetime

# 使用绝对导入，避免相对导入问题
try:
    from ..core.skill import Skill
    from ..core.skill_library import SkillLibrary
except ImportError:
    # 当作为独立模块运行时
    from self_evolve_skills_jailbreak.core.skill import Skill
    from self_evolve_skills_jailbreak.core.skill_library import SkillLibrary


class SkillExporter:
    """
    Skill 导出工具类

    将 JSON 存储的 skills 导出为 SKILL.md 格式，便于：
    - 论文展示方法实现
    - 符合业内标准格式
    - 人类 review 和理解
    """

    @staticmethod
    def _generate_description(skill: Skill) -> str:
        """
        根据 skill 信息生成 description

        用于 SKILL.md 的 YAML frontmatter
        """
        patterns = skill.applicable_patterns or []
        pattern_str = ", ".join(patterns[:3]) if patterns else "general"

        # 根据来源类型生成不同的描述
        if skill.source == "evolved":
            desc = f"Self-evolved skill for {pattern_str} scenarios. "
        elif skill.source == "merged":
            desc = f"Merged skill combining multiple effective patterns for {pattern_str}. "
        elif skill.source == "extracted":
            desc = f"Extracted from successful attacks, applicable to {pattern_str}. "
        else:
            desc = f"Initial skill template for {pattern_str} scenarios. "

        # 添加质量信息
        if skill.quality_score > 0:
            desc += f"Quality score: {skill.quality_score:.3f}, success rate: {skill.success_rate:.2%}."

        return desc

    @staticmethod
    def skill_to_md(skill: Skill) -> str:
        """
        将 Skill 转换为 SKILL.md 格式字符串

        格式：
        - YAML frontmatter (name, description, metadata)
        - Markdown body (content, statistics)

        Args:
            skill: 要转换的 Skill 对象

        Returns:
            SKILL.md 格式的字符串
        """
        # YAML frontmatter
        frontmatter = {
            "name": skill.name,
            "description": SkillExporter._generate_description(skill),
            "skill_id": skill.skill_id,
            "source": skill.source,
            "quality_score": round(skill.quality_score, 3),
            "success_rate": round(skill.success_rate, 3),
            "applicable_patterns": skill.applicable_patterns,
        }

        # YAML 格式化
        yaml_str = yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False, sort_keys=False)

        # Markdown body
        md_body = f"""# {skill.name}

{skill.content}

## Statistics

- **Usage Count**: {skill.usage_count}
- **Success Count**: {skill.success_count}
- **Success Rate**: {skill.success_rate:.2%}
- **Quality Score**: {skill.quality_score:.3f}
- **Last Used**: {skill.last_used or 'N/A'}

## Metadata

- **Source**: {skill.source}
- **Created At**: {skill.created_at}
- **Parent IDs**: {skill.parent_ids if skill.parent_ids else 'None'}
- **Cluster ID**: {skill.cluster_id or 'None'}
- **Content Length**: {skill.length} characters
"""

        # 组合完整 SKILL.md
        return f"---\n{yaml_str}---\n\n{md_body}"

    @staticmethod
    def export_skill(skill: Skill, output_dir: str, include_stats: bool = True) -> str:
        """
        导出单个 skill 到指定目录

        目录结构：
        output_dir/
        └── {skill.name}/
            └── SKILL.md

        Args:
            skill: 要导出的 Skill
            output_dir: 输出根目录
            include_stats: 是否包含统计信息

        Returns:
            生成的 SKILL.md 文件路径
        """
        # 创建 skill 目录（使用 name，处理特殊字符）
        safe_name = skill.name.replace("/", "_").replace(" ", "_")
        skill_dir = os.path.join(output_dir, safe_name)
        os.makedirs(skill_dir, exist_ok=True)

        # 生成 SKILL.md
        md_content = SkillExporter.skill_to_md(skill)
        skill_md_path = os.path.join(skill_dir, "SKILL.md")

        with open(skill_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return skill_md_path

    @staticmethod
    def export_library(
        library: SkillLibrary,
        output_dir: str,
        filter_func: Optional[Callable[[Skill], bool]] = None,
        min_quality: float = 0.0,
        min_usage: int = 0,
        top_k: Optional[int] = None,
        skill_ids: Optional[List[str]] = None,
    ) -> dict:
        """
        批量导出 SkillLibrary 中的 skills

        Args:
            library: SkillLibrary 对象
            output_dir: 输出根目录
            filter_func: 自定义过滤函数
            min_quality: 最小质量分数阈值
            min_usage: 最小使用次数阈值
            top_k: 只导出 top-k 个高质量 skills
            skill_ids: 指定导出的 skill IDs

        Returns:
            导出统计信息
        """
        # 获取要导出的 skills
        skills = library.list_skills()

        # 按 skill_ids 筛选
        if skill_ids:
            skills = [s for s in skills if s.skill_id in skill_ids]

        # 按阈值筛选
        skills = [s for s in skills if s.quality_score >= min_quality]
        skills = [s for s in skills if s.usage_count >= min_usage]

        # 自定义过滤
        if filter_func:
            skills = [s for s in skills if filter_func(s)]

        # 按质量排序
        skills.sort(key=lambda s: s.quality_score, reverse=True)

        # top-k 限制
        if top_k:
            skills = skills[:top_k]

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 导出每个 skill
        exported_paths = []
        for skill in skills:
            path = SkillExporter.export_skill(skill, output_dir)
            exported_paths.append(path)

        # 生成索引文件
        index_path = os.path.join(output_dir, "INDEX.md")
        SkillExporter._generate_index(skills, index_path)

        # 返回统计信息
        return {
            "total_skills": len(library.skills),
            "exported_count": len(skills),
            "exported_paths": exported_paths,
            "output_dir": output_dir,
            "index_path": index_path,
        }

    @staticmethod
    def _generate_index(skills: List[Skill], index_path: str) -> str:
        """
        生成 skills 索引文件

        Args:
            skills: 导出的 skills 列表
            index_path: 索引文件路径

        Returns:
            索引文件路径
        """
        # 按质量排序
        sorted_skills = sorted(skills, key=lambda s: s.quality_score, reverse=True)

        index_content = f"""# Skills Library Index

Exported at: {datetime.now().isoformat()}

Total skills: {len(skills)}

## Skills by Quality Score

| Rank | Name | Quality | Success Rate | Usage | Source |
|------|------|---------|--------------|-------|--------|
"""

        for i, skill in enumerate(sorted_skills, 1):
            safe_name = skill.name.replace("/", "_").replace(" ", "_")
            index_content += f"| {i} | [{skill.name}](./{safe_name}/SKILL.md) | {skill.quality_score:.3f} | {skill.success_rate:.2%} | {skill.usage_count} | {skill.source} |\n"

        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_content)

        return index_path