#!/bin/bash
# =============================================================================
# 准备数据：从 SESS 复制有效 skills
# =============================================================================
#
# 从 SESS 第二章的实验结果中提取有效 skills，复制到 agentic_jailbreak/data/
#
# Usage:
#   bash scripts/prepare_data.sh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="/home/tiger/jailbreak_research"

# SESS skills 路径
SESS_SKILLS="$PROJECT_ROOT/self_evolve_skills_jailbreak/exp/layer4/results/skills/skills_dan_data_medium_evo.json"

# 目标路径
TARGET_DIR="$EXP_DIR/data"
TARGET_SKILLS="$TARGET_DIR/skills.json"

echo "============================================================================="
echo "准备数据：从 SESS 复制有效 skills"
echo "============================================================================="
echo ""

# 检查 SESS skills 是否存在
if [ ! -f "$SESS_SKILLS" ]; then
    echo "ERROR: SESS skills 不存在: $SESS_SKILLS"
    echo "请先完成第二章 SESS 实验"
    exit 1
fi

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 复制 skills
echo "复制 skills: $SESS_SKILLS -> $TARGET_SKILLS"
cp "$SESS_SKILLS" "$TARGET_SKILLS"

# 统计信息
SKILL_COUNT=$(python3 -c "import json; print(len(json.load(open('$TARGET_SKILLS'))['skills']))")
echo ""
echo "Skills 数量: $SKILL_COUNT"
echo ""
echo "============================================================================="
echo "数据准备完成"
echo "============================================================================="
echo ""
echo "输出文件: $TARGET_SKILLS"
echo ""
echo "下一步: bash scripts/start_servers.sh"
