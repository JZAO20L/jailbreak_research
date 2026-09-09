"""
评估脚本

在多个 benchmark 上测试 ASR。
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RL4jailbreak"))

from src.vllm_client import VLLMClient
from src.utils import extract_label_categories_refusal
from src.prompts import GUARD_PROMPT
from agentic_jailbreak.src.env import JailbreakEnv
from agentic_jailbreak.src.agent import Agent
from agentic_jailbreak.src.agent_variants import AGENT_VARIANTS
from agentic_jailbreak.src.memory import Memory


def parse_args():
    parser = argparse.ArgumentParser(description="Agentic Jailbreak Evaluation")
    parser.add_argument("--test_data", type=str, required=True, help="Test data path (JSONL)")
    parser.add_argument("--skills_path", type=str, required=True, help="Skills JSON path")
    parser.add_argument("--policy_port", type=int, default=8003, help="Policy model port")
    parser.add_argument("--target_port", type=int, default=8002, help="Target model port")
    parser.add_argument("--guard_port", type=int, default=8001, help="Guard model port")
    parser.add_argument("--max_turns", type=int, default=5, help="Max turns per attack")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="output/eval_results", help="Output directory")
    parser.add_argument("--use_memory", action="store_true", help="Use memory mechanism")
    parser.add_argument("--variant", type=str, default="select_adapt",
                        choices=list(AGENT_VARIANTS.keys()) + ["no_skill_beam", "skill_content", "skill_prefix",
                                                                "skill_once", "skill_every_turn", "skill_decide",
                                                                "skill_decide_top1", "skill_decide_top3"],
                        help="Agent variant: select_adapt / no_skill / select_only / beam / skill_content / skill_prefix / skill_once / skill_every_turn / skill_decide / skill_decide_top1 / skill_decide_top3")
    parser.add_argument("--retrieval_mode", type=str, default="quality",
                        choices=["quality", "similarity"],
                        help="Skill retrieval mode")
    parser.add_argument("--top_k_skills", type=int, default=5, help="Number of skill candidates (e.g., 1 or 3)")
    parser.add_argument("--work_memory", action="store_true",
                        help="Use hierarchical working memory (previous-turn summaries + last-turn full feedback)")
    parser.add_argument("--ctx_window", type=int, default=0,
                        help="C3 sliding-window context: keep system+initial user + last N turns (0 = full accumulation, C1)")
    parser.add_argument("--beam_width", type=int, default=2, help="Beam width (variant=beam)")
    parser.add_argument("--mode", type=str, default="conversational",
                        choices=["stateless", "conversational"],
                        help="Eval harness mode: stateless (rebuild prompt) or conversational (message accumulation)")
    return parser.parse_args()


def load_test_data(path: str, max_samples: int) -> List[Dict]:
    """加载测试数据"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data.append({"id": obj.get("id", len(data)), "prompt": obj["prompt"]})
            if len(data) >= max_samples:
                break
    return data


def evaluate_single(
    prompt: str,
    env: JailbreakEnv,
    agent: Agent,
    memory: Memory = None,
) -> Dict[str, Any]:
    """
    评估单个 prompt
    
    Returns:
        result: {
            "success": bool,
            "turns": int,
            "history": List[Dict],
        }
    """
    # 重置环境
    obs = env.reset(prompt)
    
    # 多轮交互
    for turn in range(env.max_turns):
        # Agent 分析
        analysis = agent.analyze(obs, memory)
        
        # Agent 选择动作
        action = agent.select_action(obs, analysis, memory)
        
        # 环境执行
        next_obs, reward, done, info = env.step(action)
        
        # 更新记忆(只记"用了哪个 skill + 结果",不存完整 action)
        if memory is not None:
            last_hist = env.state.history[-1] if env.state.history else {}
            memory.update({
                "prompt": prompt,
                "turn": turn + 1,
                "skill_idx": action.get("skill_idx"),
                "skill_name": last_hist.get("skill_name", ""),
                "success": info["success"],
                "summary": (
                    f"Skill[{action.get('skill_idx')}]{last_hist.get('skill_name', '')} "
                    f"{'成功' if info['success'] else '失败'} "
                    f"(Guard: {info.get('guard_label', '?')})"
                ),
            })
        
        # 检查是否结束
        if done:
            return {
                "success": info["success"],
                "turns": info["total_turns"],
                "history": env.state.history,
            }
        
        obs = next_obs
    
    # 达到最大轮数
    return {
        "success": False,
        "turns": env.max_turns,
        "history": env.state.history,
    }


def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载测试数据
    test_data = load_test_data(args.test_data, args.max_samples)
    print(f"Loaded {len(test_data)} test samples")
    
    # 初始化环境(两种模式共用:target/guard 评估)
    env = JailbreakEnv(
        skills_path=args.skills_path,
        target_port=args.target_port,
        guard_port=args.guard_port,
        max_turns=args.max_turns,
        retrieval_mode=args.retrieval_mode,
        top_k_skills=args.top_k_skills,
    )
    
    # 对话式(Agentic 核心方法形态):消息累积,每轮 1 次 policy 调用
    if args.mode == "conversational":
        # 首次 reset 绑定 state(build_initial_message 需要 skill_library)
        test_prompt = test_data[0]["prompt"] if test_data else ""
        env.reset(test_prompt)
        from agentic_jailbreak.src.conv_eval import run_conversational_eval
        run_conversational_eval(args, test_data, env, output_dir)
        return
    
    # 无状态重建模式(legacy)
    # 初始化 Agent(按变体)
    agent_cls = AGENT_VARIANTS[args.variant]
    if args.variant == "beam":
        agent = agent_cls(policy_port=args.policy_port, beam_width=args.beam_width)
    else:
        agent = agent_cls(policy_port=args.policy_port)
    print(f"[{args.variant}] agent ready (retrieval={args.retrieval_mode})")
    
    # 初始化记忆（可选）
    memory = None
    if args.use_memory:
        memory = Memory(memory_path=str(output_dir / "memory.json"))
    
    # 评估
    results = []
    success_count = 0
    
    for item in tqdm(test_data, desc="Evaluating"):
        result = evaluate_single(item["prompt"], env, agent, memory)
        
        if result["success"]:
            success_count += 1
        
        results.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "success": result["success"],
            "turns": result["turns"],
        })
    
    # 计算 ASR
    asr = success_count / len(test_data) if test_data else 0.0
    
    # 保存结果
    results_path = output_dir / "results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # 保存摘要
    summary = {
        "total": len(test_data),
        "success": success_count,
        "asr": asr,
        "avg_turns": sum(r["turns"] for r in results) / len(results) if results else 0.0,
        "max_turns": args.max_turns,
        "use_memory": args.use_memory,
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"ASR: {asr:.2%} ({success_count}/{len(test_data)})")
    print(f"Avg turns: {summary['avg_turns']:.2f}")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    
    # 关闭
    env.close()
    agent.close()


if __name__ == "__main__":
    main()
