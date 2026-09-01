#!/usr/bin/env python3
"""
Beam agent pilot runner

用法:
  python agentic_jailbreak/scripts/run_beam_pilot.py --skills agentic_jailbreak/exp/skill_asr_sweep/seed_skills_top10.json --n 50
  python agentic_jailbreak/scripts/run_beam_pilot.py --no_skills --n 50
"""

import sys
import os
import json
import argparse
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.vllm_client import VLLMClient  # noqa: E402
from agentic_jailbreak.src.beam_agent import BeamAgent  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beams", type=int, default=2)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--selection_mode", type=str, default="step", choices=["step", "trajectory"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--policy_port", type=str, default="8003",
                   help="policy vLLM 端口,逗号分隔可传多个(负载均衡)")
    p.add_argument("--target_port", type=int, default=8002)
    p.add_argument("--guard_port", type=int, default=8001)
    p.add_argument("--policy_temp", type=float, default=0.8)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--skills", type=str, default=None, help="技能库 json 路径")
    g.add_argument("--no_skills", action="store_true", help="无技能条件")
    p.add_argument("--outdir", default=os.path.join(PROJECT_ROOT, "agentic_jailbreak/exp/beam_pilot"))
    p.add_argument("--tag", type=str, default=None, help="结果文件标签(默认自动)")
    p.add_argument("--data", type=str, default=None,
                   help="输入 prompt 文件(json 字符串数组, 默认 self_evolve test_prompts.json)")
    p.add_argument("--rollouts", type=int, default=1,
                   help="每个 prompt 的独立攻击次数(拒绝采样, 供 RFT 收集成功轨迹)")
    return p.parse_args()


def main():
    args = parse_args()

    skills = None
    if args.skills:
        lib = json.load(open(args.skills))
        skills = lib["skills"]
        print(f"[pilot] skills mode: {len(skills)} skills")

    data_path = args.data or os.path.join(PROJECT_ROOT, "self_evolve_skills_jailbreak/data/test_prompts.json")
    with open(data_path) as f:
        data = json.load(f)
    rng = random.Random(args.seed)
    sample = rng.sample(data, args.n)
    if args.rollouts > 1:
        sample = [p for p in sample for _ in range(args.rollouts)]
    print(f"[pilot] {args.n} prompts x {args.rollouts} rollouts = {len(sample)} attacks (seed={args.seed})")

    os.makedirs(args.outdir, exist_ok=True)
    tag = args.tag or f"{'skills' if skills else 'noskill'}_{args.selection_mode}"
    results_path = os.path.join(args.outdir, f"pilot_{tag}_b{args.beams}_d{args.max_depth}.jsonl")

    target = VLLMClient(port=args.target_port)
    guard = VLLMClient(port=args.guard_port)
    policy_ports = [int(x) for x in args.policy_port.split(",") if x.strip()]
    policies = [VLLMClient(port=p) for p in policy_ports]
    print(f"[pilot] target={target.base_url_root} guard={guard.base_url_root} "
          f"policy={[p.base_url_root for p in policies]}")
    print(f"[pilot] health: target={target.health_check()} guard={guard.health_check()} "
          f"policy={[p.health_check() for p in policies]}")

    agent = BeamAgent(
        target=target,
        guard=guard,
        policy=policies if len(policies) > 1 else policies[0],
        skills=skills,
        num_beams=args.beams,
        max_depth=args.max_depth,
        selection_mode=args.selection_mode,
        policy_temp=args.policy_temp,
        verbose=False,
    )

    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(agent.attack, req): i for i, req in enumerate(sample)}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 10 == 0 or done == args.n:
                asr = sum(1 for x in results if x["success"]) / len(results)
                print(f"[pilot] {done}/{args.n} asr={asr:.3f} elapsed={time.time()-start:.0f}s")

    results.sort(key=lambda x: sample.index(x["original_prompt"]))
    with open(results_path, "w") as f:
        for r in results:
            compact = {
                "original_prompt": r["original_prompt"],
                "success": r["success"],
                "min_success_depth": r["min_success_depth"],
                "success_beams": r["success_beams"],
                "time_cost": r["time_cost"],
                "beams": [
                    {
                        "beam_idx": b["beam_idx"],
                        "history": b["history"],
                    }
                    for b in r["beams"]
                ],
            }
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")

    # summary
    n = len(results)
    succ = sum(1 for r in results if r["success"])
    depths = [r["min_success_depth"] for r in results if r["success"]]
    skill_counter = Counter()
    for r in results:
        for b in r["beams"]:
            for h in b["history"]:
                skill_counter[h["skill"]] += 1

    summary = {
        "mode": tag,
        "n": n,
        "beams": args.beams,
        "max_depth": args.max_depth,
        "asr": succ / n,
        "success": succ,
        "avg_min_success_depth": round(sum(depths) / len(depths), 2) if depths else None,
        "skill_usage": dict(skill_counter.most_common(15)),
        "time_cost_s": round(time.time() - start, 1),
    }
    with open(os.path.join(args.outdir, f"summary_{tag}_b{args.beams}_d{args.max_depth}.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== {tag.upper()} pilot (B={args.beams}, D={args.max_depth}) ===")
    print(f"ASR: {succ}/{n} = {succ/n:.3f}")
    print(f"avg min success depth: {summary['avg_min_success_depth']}")
    print(f"top skill usage: {summary['skill_usage']}")
    print(f"results: {results_path}")


if __name__ == "__main__":
    main()