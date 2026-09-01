#!/usr/bin/env python3
"""
Skills 导出脚本

将 JSON 格式的 skills_library 导出为业内标准的 SKILL.md + YAML frontmatter 格式

Usage:
    python export_skills.py --output-dir skills_exported
    python export_skills.py --output-dir skills_exported --min-quality 0.3 --top-k 20
    python export_skills.py --output-dir skills_exported --skill-ids <id1> <id2>
"""

import argparse
import sys
import os

# 获取 jailbreak_research 项目根目录（脚本位于 self_evolve_skills_jailbreak/scripts/）
script_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(script_dir)  # self_evolve_skills_jailbreak
project_root = os.path.dirname(package_dir)  # jailbreak_research

# 添加项目路径
sys.path.insert(0, project_root)

from self_evolve_skills_jailbreak.src.skill_library import SkillLibrary
from self_evolve_skills_jailbreak.utils.skill_exporter import SkillExporter


def main():
    parser = argparse.ArgumentParser(
        description="Export skills from JSON library to SKILL.md format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all skills
  python export_skills.py --output-dir skills_exported

  # Export only high-quality skills
  python export_skills.py --output-dir skills_exported --min-quality 0.5

  # Export top 20 skills
  python export_skills.py --output-dir skills_exported --top-k 20

  # Export specific skills by ID
  python export_skills.py --output-dir skills_exported --skill-ids abc123 def456
        """,
    )

    parser.add_argument(
        "--library-path",
        type=str,
        default="skills/skills_library.json",
        help="Path to skills_library.json (default: skills/skills_library.json)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="skills_exported",
        help="Output directory for SKILL.md files (default: skills_exported)",
    )

    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="Minimum quality score threshold (default: 0.0)",
    )

    parser.add_argument(
        "--min-usage",
        type=int,
        default=0,
        help="Minimum usage count threshold (default: 0)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Export only top-k skills by quality score",
    )

    parser.add_argument(
        "--skill-ids",
        type=str,
        nargs="+",
        default=None,
        help="Export specific skills by ID",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed export information",
    )

    args = parser.parse_args()

    # 确定库路径（相对于项目根目录 jailbreak_research）
    library_path = args.library_path
    if not os.path.isabs(library_path):
        library_path = os.path.join(project_root, library_path)

    # 检查库文件是否存在
    if not os.path.exists(library_path):
        print(f"Error: Skills library not found at {library_path}")
        print("Please specify correct --library-path or ensure skills_library.json exists")
        sys.exit(1)

    print(f"Loading skills library from: {library_path}")

    # 加载 skill library
    library = SkillLibrary(storage_path=library_path)

    print(f"Total skills in library: {library.count()}")

    # 设置输出目录（相对于项目根目录）
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)

    print(f"Exporting to: {output_dir}")

    # 执行导出
    result = library.export_to_skill_md(
        output_dir=output_dir,
        min_quality=args.min_quality,
        min_usage=args.min_usage,
        top_k=args.top_k,
        skill_ids=args.skill_ids,
    )

    # 打印结果
    print(f"\nExport completed!")
    print(f"  - Total skills in library: {result['total_skills']}")
    print(f"  - Skills exported: {result['exported_count']}")
    print(f"  - Output directory: {result['output_dir']}")
    print(f"  - Index file: {result['index_path']}")

    if args.verbose and result['exported_paths']:
        print(f"\nExported files:")
        for path in result['exported_paths']:
            print(f"  - {path}")

    # 打印筛选条件摘要
    if args.min_quality > 0:
        print(f"\nFilter: min_quality >= {args.min_quality}")
    if args.min_usage > 0:
        print(f"Filter: min_usage >= {args.min_usage}")
    if args.top_k:
        print(f"Filter: top_k = {args.top_k}")
    if args.skill_ids:
        print(f"Filter: skill_ids = {args.skill_ids}")


if __name__ == "__main__":
    main()