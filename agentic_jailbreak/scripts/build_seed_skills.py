#!/usr/bin/env python3
"""
从 ASR sweep 结果构建 top-N 种子技能库（确定性、可复现）

选择规则:
  1. 按实测 ASR 降序排序; 平局按 skill_id 升序（skill_id 全局唯一, 保证全确定性）
  2. 取 top-N
统计字段来自实测: usage_count=100, success_count=实测, success_rate=实测ASR
content 等原文从 54-skill 源文件按 skill_id 原样复制, 不做任何改写

用法:
  python agentic_jailbreak/scripts/build_seed_skills.py --top_n 10
"""

import json
import argparse
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 家族标注: 8 位 skill_id 前缀 -> 策略家族（仅用于论文分析, 不影响选择规则）
FAMILY_BY_PREFIX = {
    "25c00385": "subtle_framing",
    "025f9775": "metaphor_encoding",
    "0dd3721b": "subtle_framing",
    "6f0a98eb": "role_mode",
    "6dd9e311": "dan_conversation",
    "18dc6dba": "dan_fictional_world",
    "4b543a2b": "testimonial_framing",
    "32eb8f32": "writing_realism",
    "c027bb9c": "role_mode",
    "ef60ef90": "role_mode",
    "54d86378": "mockery_dialogue",
    "9cda2249": "authority_fabrication",
}

# 重命名: 源库中 8 个技能同名 skill_extracted_19, 结果记录只存 name 会导致无法归属
RENAME_BY_PREFIX = {
    "25c00385": "subtle_framing_document",
    "025f9775": "metaphor_dialogue",
    "0dd3721b": "subtle_framing_stereotype",
    "6f0a98eb": "role_mode_dual_response",
    "18dc6dba": "dan_fictional_world",
    "32eb8f32": "writing_realism_genre",
    "4b543a2b": "testimonial_product_claim",
    "6dd9e311": "dan_conversation",
    "c027bb9c": "role_mode_narrative",
    "54d86378": "mockery_dialogue",
    "ef60ef90": "role_mode_scenario",
    "9cda2249": "authority_fabricated_study",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per_skill", default=os.path.join(
        PROJECT_ROOT, "agentic_jailbreak/exp/skill_asr_sweep/per_skill_asr.json"))
    p.add_argument("--source", default=os.path.join(
        PROJECT_ROOT, "self_evolve_skills_jailbreak/exp/layer4/results/skills/skills_dan_data_medium_evo.json"))
    p.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "agentic_jailbreak/exp/skill_asr_sweep/seed_skills_top10.json"))
    p.add_argument("--top_n", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.per_skill) as f:
        sweep = json.load(f)
    with open(args.source) as f:
        lib = json.load(f)

    by_id = {s["skill_id"]: s for s in lib["skills"]}
    assert len(by_id) == len(lib["skills"]), "source library has duplicate skill_id"
    assert sweep["n_samples"] == 100

    # 确定性排序: -asr, skill_id
    ranked = sorted(sweep["ranked"], key=lambda r: (-r["asr"], r["skill_id"]))
    top = ranked[: args.top_n]
    if len(top) < args.top_n:
        print(f"[warn] only {len(top)} skills available")

    seeds = []
    families = {}
    for r in top:
        s = by_id[r["skill_id"]]
        seed = dict(s)  # 原样复制 content 等字段
        # 覆写统计字段为实测值
        seed["usage_count"] = r["total"]
        seed["success_count"] = r["success"]
        seed["success_rate"] = r["asr"]
        seed["quality_score"] = r["asr"] * (r["total"] ** 0.5)  # 镜像 skill.py 的 quality 公式
        seeds.append(seed)
        seed["name"] = RENAME_BY_PREFIX.get(r["skill_id"][:8], seed["name"])
        families[r["skill_id"]] = FAMILY_BY_PREFIX.get(r["skill_id"][:8], "unlabeled")

    out = {
        "skills": seeds,
        "families": families,
        "metadata": {
            "built_by": "build_seed_skills.py",
            "selection_rule": "top-N by measured single-call ASR on SafeRL @8002, tie-break by skill_id",
            "eval_set": "eval_set_100_seed42.json (100 random from SESS test_prompts, seed=42)",
            "n_samples_per_skill": sweep["n_samples"],
            "n_skills": len(seeds),
            "built_at": datetime.now().isoformat(),
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[build] seed library saved: {args.out} ({len(seeds)} skills)")
    print(f"{'asr':>5} {'skill_id':<10} {'name':<18} {'source':<12} family")
    for r, s in zip(top, seeds):
        print(f"{r['asr']:5.3f} {s['skill_id'][:8]:<10} {s['name']:<18} {s['source']:<12} {families[s['skill_id']]}")


if __name__ == "__main__":
    main()