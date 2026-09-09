#!/usr/bin/env python3
"""
RFT (Rejection Sampling Fine-tuning) 数据构建 — conv 协议原生版 (v3)

与 build_rft_data.py (v2, beam 协议来源) 的区别:
    v2: beam 协议轨迹 -> 人工拼装成对话式消息 -> 协议失配 (负结果, 见 exp/README.md §10.1)
    v3: conv_eval (对话式协议) 直接落盘的成功轨迹 -> 同协议全轨迹样本, 上下文->动作映射为真

输入: conv_eval 的 results.jsonl (每行 {id, prompt, success, turns, trajectory})
    trajectory[i] = {turn, actions: [{skill_idx, adapted_content}], evals: [{..., target_response, guard_label, success, refused}]}
输出: ms-swift SFT 格式 jsonl, 每行 {"messages": [...]}, 每条成功轨迹一个样本

监督语义说明 (重要):
    swift 的 default loss_scale 对 messages 中所有 assistant 轮计算 loss,
    因此全轨迹样本下每个动作恰好被监督一次, 且其上下文(causal attention)
    就是真实前缀 —— 等价于"前缀式展开"的意图。
    v2 的前缀式展开 + default loss 会把早期动作重复监督 T-i+1 次
    (过加权早期轮), 这不是期望的 RFT credit assignment, v3 已修正。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_jailbreak.src.conv_eval import SYSTEM_PROMPT, build_feedback, build_initial_message

MIN_ACTION_LEN = 10  # 动作文本最小长度(空壳/解析失败过滤)


def _has_format_error(trajectory):
    """轨迹是否含格式错误轮(空动作/环境错误)。"""
    for rec in trajectory:
        actions = rec.get("actions") or []
        if len(actions) != 1:
            # 采集必须用单轨迹(beam_width=1), 多候选说明来源不对
            return True
        action = str(actions[0].get("adapted_content") or "").strip()
        if len(action) < MIN_ACTION_LEN:
            return True
        resp = str((rec.get("evals") or [{}])[0].get("target_response") or "")
        if resp.startswith("[Error]"):
            return True
    return False


def load_candidates(skills_path, top_k):
    """复刻 env.JailbreakEnv._load_skills 的质量池逻辑, 保证 skill_idx 与采集时一致。

    env: 按 name 去重(保留 quality 最高) -> 丢弃 quality<=0 -> 按 quality 降序 -> 取 top_k。
    """
    data = json.loads(Path(skills_path).read_text(encoding="utf-8"))
    pool = {}
    for s in data.get("skills", []):
        q = float(s.get("quality_score", 0.0) or 0.0)
        if q <= 0:
            continue
        name = s["name"]
        if name not in pool or q > pool[name]["quality_score"]:
            pool[name] = {
                "name": name,
                "description": s.get("description", f"攻击策略: {name}"),
                "quality_score": q,
            }
    ranked = sorted(pool.values(), key=lambda x: x["quality_score"], reverse=True)
    return ranked[:top_k]


def build_sample(prompt, trajectory, variant, skill_library):
    """一条 conv 成功轨迹 -> 一个全轨迹 SFT 样本。

    消息结构与 conv_eval.evaluate_conversational 逐字一致:
    system(SYSTEM_PROMPT) -> user(build_initial_message) -> assistant/feedback 交替,
    最后一个 assistant 即成功轮动作。default loss 下每个动作监督一次。

    assistant 内容用 actions[0]["raw"](模型原始输出, 含 Selection/Adapted Strategy 标记):
    skill_decide 协议下监督目标必须是原始动作文本, 剥掉标记会导致协议漂移。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_message(prompt, skill_library, variant)},
    ]
    for i, rec in enumerate(trajectory):
        action = str(rec["actions"][0].get("raw") or rec["actions"][0].get("adapted_content") or "").strip()
        messages.append({"role": "assistant", "content": action})
        if i < len(trajectory) - 1:
            ev = rec["evals"][0]
            feedback = build_feedback([{
                "success": ev["success"],
                "guard_label": ev["guard_label"],
                "refused": ev["refused"],
                "target_response": ev["target_response"],
            }])
            messages.append({"role": "user", "content": feedback})
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=str, required=True,
                        help="逗号分隔的 conv results.jsonl 文件列表(支持 part 合并后的单文件)")
    parser.add_argument("--out", type=str, required=True, help="输出 SFT jsonl 路径")
    parser.add_argument("--variant", type=str, default="skill_decide",
                        help="采集时的 conv variant(须与轨迹来源一致)")
    parser.add_argument("--skills-path", type=str,
                        default="/home/tiger/jailbreak_research/agentic_jailbreak/exp/skill_asr_sweep/seed_skills_top10.json",
                        help="skills 库路径(skill_decide/select_adapt 系 variant 需要)")
    parser.add_argument("--top-k-skills", type=int, default=10,
                        help="候选 skill 数(须与采集时 --top_k_skills 一致)")
    parser.add_argument("--max-traj-per-prompt", type=int, default=1,
                        help="每条 prompt 最多取几条成功轨迹(单轨迹采集下天然为 1, 为未来多次采样保留)")
    args = parser.parse_args()

    skill_library = None
    if args.variant not in ("no_skill", "no_skill_beam"):
        skill_library = load_candidates(args.skills_path, args.top_k_skills)
        print(f"candidates for {args.variant}: {len(skill_library)} skills "
              f"({[s['name'] for s in skill_library]})")

    files = [f.strip() for f in args.results.split(",") if f.strip()]
    per_prompt = Counter()

    n_total = n_success = n_dropped = n_used = 0
    turn_dist = Counter()
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as out:
        for rel in files:
            with open(rel, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    n_total += 1
                    if not d.get("success"):
                        continue
                    n_success += 1
                    traj = d["trajectory"]
                    if _has_format_error(traj) or per_prompt[d["prompt"]] >= args.max_traj_per_prompt:
                        n_dropped += 1
                        continue
                    per_prompt[d["prompt"]] += 1
                    out.write(json.dumps(build_sample(d["prompt"], traj, args.variant, skill_library), ensure_ascii=False) + "\n")
                    n_used += 1
                    turn_dist[len(traj)] += 1

    print(f"trajectories: total={n_total}, success={n_success}, "
          f"dropped(format_error/over_cap)={n_dropped}, used={n_used}")
    print("used trajectory turn distribution:", dict(sorted(turn_dist.items())))
    print(f"sft samples (full trajectory): {n_used}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
