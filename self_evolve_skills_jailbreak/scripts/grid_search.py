"""
Grid Search 实验脚本

自动遍历消融点的参数组合，运行实验并汇总结果

分层实验设计：
- Layer 1: 核心方法消融（A, B, C） - 16 种组合，默认数据配置
- Layer 2: 数据消融（D, E） - 在最佳方法上验证数据策略

消融点 A: skill_call_mode - single_call | every_iteration
消融点 B: skill_extraction_mode - final_prompt | trajectory
消融点 C: update_strategy - success_only | failure_only | both | statistical
消融点 D: data_size - small | medium | large
消融点 E: split_strategy - early_focus | balanced | evolution_focus
"""

import os
import sys
import json
import time
import subprocess
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 消融参数定义
# =============================================================================

ABLATION_PARAMS = {
    # Layer 1: 核心方法消融
    "skill_call_mode": ["single_call", "every_iteration"],         # 2 种
    "skill_extraction_mode": ["final_prompt", "trajectory"],       # 2 种
    "update_strategy": ["success_only", "failure_only", "both", "statistical"],  # 4 种

    # Layer 2: 数据消融（可选）
    "data_size": ["small", "medium", "large"],                     # 3 种
    "split_strategy": ["early_focus", "balanced", "evolution_focus"],  # 3 种
}


def generate_method_combinations() -> List[Dict[str, str]]:
    """
    生成 Layer 1 组合（核心方法消融）

    共 2 × 2 × 4 = 16 种
    """
    keys = ["skill_call_mode", "skill_extraction_mode", "update_strategy"]
    values = [ABLATION_PARAMS[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def generate_data_combinations() -> List[Dict[str, str]]:
    """
    生成 Layer 2 组合（数据消融）

    共 3 × 3 = 9 种
    """
    keys = ["data_size", "split_strategy"]
    values = [ABLATION_PARAMS[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def generate_full_combinations() -> List[Dict[str, str]]:
    """
    生成全部组合（包含所有消融点）

    共 2 × 2 × 4 × 3 × 3 = 72 种（谨慎使用）
    """
    keys = list(ABLATION_PARAMS.keys())
    values = [ABLATION_PARAMS[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def run_single_experiment(
    combo: Dict[str, str],
    base_args: Dict[str, Any],
    experiment_id: int,
    output_dir: str,
) -> Dict:
    """
    运行单个实验

    Args:
        combo: 参数组合
        base_args: 基础参数（端口、数据路径等）
        experiment_id: 实验编号
        output_dir: 输出目录

    Returns:
        实验结果
    """
    print(f"\n{'='*60}")
    print(f"Experiment {experiment_id}/16")
    print(f"{'='*60}")
    print(f"skill_call_mode: {combo['skill_call_mode']}")
    print(f"skill_extraction_mode: {combo['skill_extraction_mode']}")
    print(f"update_strategy: {combo['update_strategy']}")

    # 构建命令
    cmd = [
        "python",
        str(PROJECT_ROOT / "self_evolve_skills_jailbreak/scripts/pipeline.py"),
        "--mode", "full",
        "--skill_call_mode", combo["skill_call_mode"],
        "--skill_extraction_mode", combo["skill_extraction_mode"],
        "--update_strategy", combo["update_strategy"],
        "--skip_launch",  # 连接已有服务
    ]

    # 添加基础参数
    for key, value in base_args.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])

    # 运行实验（实时输出）
    start_time = time.time()

    try:
        # 使用 Popen 实时显示输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # 行缓冲
        )

        # 实时读取并显示输出
        stdout_lines = []
        for line in process.stdout:
            print(line, end='')  # 实时显示
            stdout_lines.append(line)

        process.wait(timeout=3600 * 2)  # 2小时超时

        success = process.returncode == 0
        stdout = ''.join(stdout_lines)
        stderr = ""

    except subprocess.TimeoutExpired:
        process.kill()
        success = False
        stdout = ""
        stderr = "Timeout expired"
    except Exception as e:
        success = False
        stdout = ""
        stderr = str(e)

    elapsed_time = time.time() - start_time

    # 读取结果文件（优先从 output_dir，其次从 PROJECT_ROOT）
    results_filename = f"result_{combo['skill_call_mode']}_{combo['skill_extraction_mode']}_{combo['update_strategy']}.json"
    results_file = Path(output_dir) / results_filename

    # 如果 output_dir 没有找到，尝试 PROJECT_ROOT
    if not results_file.exists():
        results_file = PROJECT_ROOT / f"results_{combo['skill_call_mode']}_{combo['skill_extraction_mode']}_{combo['update_strategy']}.json"

    experiment_result = {
        "experiment_id": experiment_id,
        "combination": combo,
        "success": success,
        "elapsed_time": elapsed_time,
        "timestamp": datetime.now().isoformat(),
    }

    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
                experiment_result["results"] = data

                # 提取关键指标
                if "test" in data:
                    test_data = data["test"]
                    experiment_result["asr"] = test_data.get("asr", 0)
                    experiment_result["avg_iterations"] = test_data.get("avg_iterations", 0)

                if "cold_start" in data:
                    experiment_result["cold_start_skills"] = data["cold_start"].get("final_skill_count", 0)

                if "evolution" in data:
                    experiment_result["evolution_skills"] = data["evolution"].get("final_skill_count", 0)
        except Exception as e:
            experiment_result["parse_error"] = str(e)

    # 保存单独实验结果（使用 result_ 前缀以便 summarize_layer1.py 解析）
    experiment_file = Path(output_dir) / f"result_{combo['skill_call_mode']}_{combo['skill_extraction_mode']}_{combo['update_strategy']}.json"
    with open(experiment_file, "w") as f:
        json.dump(experiment_result, f, indent=2, ensure_ascii=False)

    # 打印结果摘要
    if success:
        print(f"\n✓ Experiment completed")
        print(f"  ASR: {experiment_result.get('asr', 'N/A')*100:.1f}%")
        print(f"  Avg iterations: {experiment_result.get('avg_iterations', 'N/A'):.1f}")
        print(f"  Final skills: {experiment_result.get('evolution_skills', 'N/A')}")
        print(f"  Time: {elapsed_time:.1f}s")
    else:
        print(f"\n✗ Experiment failed")
        print(f"  Error: {stderr[:200]}")

    return experiment_result


def run_grid_search(
    base_args: Dict[str, Any] = None,
    output_dir: str = None,
    resume_from: int = 0,
) -> Dict:
    """
    运行 Grid Search

    Args:
        base_args: 基础参数
        output_dir: 输出目录
        resume_from: 从第几个实验开始（用于断点续跑）

    Returns:
        所有实验结果汇总
    """
    if base_args is None:
        base_args = {}

    if output_dir is None:
        output_dir = PROJECT_ROOT / "experiments" / f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(output_dir, exist_ok=True)

    # 生成所有组合
    combinations = generate_method_combinations()
    total = len(combinations)

    print("="*60)
    print("Grid Search for Self Evolve Skills")
    print("="*60)
    print(f"Total combinations: {total}")
    print(f"Output directory: {output_dir}")
    print(f"Ablation points:")
    print(f"  A: skill_call_mode (2 options)")
    print(f"  B: skill_extraction_mode (2 options)")
    print(f"  C: update_strategy (4 options)")

    # 运行所有实验
    all_results = []

    for i, combo in enumerate(combinations, 1):
        if i < resume_from:
            print(f"Skipping experiment {i} (resume from {resume_from})")
            continue

        result = run_single_experiment(
            combo=combo,
            base_args=base_args,
            experiment_id=i,
            output_dir=output_dir,
        )
        all_results.append(result)

        # 每个实验后保存汇总
        save_summary(all_results, output_dir)

    # 最终汇总
    final_summary = create_final_summary(all_results, output_dir)

    return final_summary


def save_summary(results: List[Dict], output_dir: str):
    """保存当前汇总"""
    summary_file = Path(output_dir) / "summary_latest.json"

    summary = {
        "total_experiments": len(results),
        "completed": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def create_final_summary(results: List[Dict], output_dir: str) -> Dict:
    """创建最终汇总报告"""

    # 按消融点分组分析
    analysis = {
        "by_skill_call_mode": {},
        "by_extraction_mode": {},
        "by_update_strategy": {},
        "best_combination": None,
        "worst_combination": None,
    }

    # 收集成功实验的指标
    successful = [r for r in results if r["success"] and "asr" in r]

    if successful:
        # 按 skill_call_mode 分组
        for mode in ABLATION_PARAMS["skill_call_mode"]:
            mode_results = [r for r in successful if r["combination"]["skill_call_mode"] == mode]
            if mode_results:
                analysis["by_skill_call_mode"][mode] = {
                    "count": len(mode_results),
                    "avg_asr": sum(r["asr"] for r in mode_results) / len(mode_results),
                    "avg_iterations": sum(r.get("avg_iterations", 0) for r in mode_results) / len(mode_results),
                }

        # 按 extraction_mode 分组
        for mode in ABLATION_PARAMS["skill_extraction_mode"]:
            mode_results = [r for r in successful if r["combination"]["skill_extraction_mode"] == mode]
            if mode_results:
                analysis["by_extraction_mode"][mode] = {
                    "count": len(mode_results),
                    "avg_asr": sum(r["asr"] for r in mode_results) / len(mode_results),
                    "avg_iterations": sum(r.get("avg_iterations", 0) for r in mode_results) / len(mode_results),
                }

        # 按 update_strategy 分组
        for mode in ABLATION_PARAMS["update_strategy"]:
            mode_results = [r for r in successful if r["combination"]["update_strategy"] == mode]
            if mode_results:
                analysis["by_update_strategy"][mode] = {
                    "count": len(mode_results),
                    "avg_asr": sum(r["asr"] for r in mode_results) / len(mode_results),
                    "avg_iterations": sum(r.get("avg_iterations", 0) for r in mode_results) / len(mode_results),
                }

        # 最佳和最差组合
        sorted_by_asr = sorted(successful, key=lambda r: r["asr"], reverse=True)
        analysis["best_combination"] = sorted_by_asr[0]["combination"] if sorted_by_asr else None
        analysis["worst_combination"] = sorted_by_asr[-1]["combination"] if sorted_by_asr else None

    # 创建完整汇总
    final_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "results": results,
        "analysis": analysis,
    }

    # 保存最终汇总
    final_file = Path(output_dir) / "summary_final.json"
    with open(final_file, "w") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    # 打印分析结果
    print("\n" + "="*60)
    print("Grid Search Analysis Results")
    print("="*60)

    if successful:
        print("\nBy skill_call_mode:")
        for mode, stats in analysis["by_skill_call_mode"].items():
            print(f"  {mode}: ASR={stats['avg_asr']*100:.1f}%, Iter={stats['avg_iterations']:.1f}")

        print("\nBy extraction_mode:")
        for mode, stats in analysis["by_extraction_mode"].items():
            print(f"  {mode}: ASR={stats['avg_asr']*100:.1f}%, Iter={stats['avg_iterations']:.1f}")

        print("\nBy update_strategy:")
        for mode, stats in analysis["by_update_strategy"].items():
            print(f"  {mode}: ASR={stats['avg_asr']*100:.1f}%, Iter={stats['avg_iterations']:.1f}")

        if analysis["best_combination"]:
            print(f"\nBest combination: {analysis['best_combination']}")
            best_result = sorted_by_asr[0]
            print(f"  ASR: {best_result['asr']*100:.1f}%")
            print(f"  Avg iterations: {best_result.get('avg_iterations', 0):.1f}")

        if analysis["worst_combination"]:
            print(f"\nWorst combination: {analysis['worst_combination']}")
            worst_result = sorted_by_asr[-1]
            print(f"  ASR: {worst_result['asr']*100:.1f}%")

    print(f"\nSummary saved to: {output_dir}")

    return final_summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Grid Search for Self Evolve Skills")

    # 基础参数
    parser.add_argument("--seed_limit", type=int, default=20, help="种子数据限制")
    parser.add_argument("--test_limit", type=int, default=10, help="测试数据限制")
    parser.add_argument("--eval_limit", type=int, default=100, help="中间评估数据数量")
    parser.add_argument("--num_epochs", type=int, default=2, help="进化轮数")
    parser.add_argument("--max_iterations", type=int, default=5, help="最大攻击迭代次数")

    # 服务参数
    parser.add_argument("--guard_port", type=int, default=8002)
    parser.add_argument("--target_port", type=int, default=8001)

    # 输出参数
    parser.add_argument("--output_dir", type=str, default=None, help="实验结果输出目录")
    parser.add_argument("--resume_from", type=int, default=0, help="从第几个实验开始（断点续跑）")

    # 单独运行某个组合（调试用）
    parser.add_argument("--single", type=str, nargs=3, metavar=("CALL_MODE", "EXTRACTION_MODE", "UPDATE_STRATEGY"),
                        help="只运行单个组合，如: --single single_call final_prompt both")

    args = parser.parse_args()

    base_args = {
        "seed_limit": args.seed_limit,
        "test_limit": args.test_limit,
        "eval_limit": args.eval_limit,
        "num_epochs": args.num_epochs,
        "max_iterations": args.max_iterations,
        "guard_port": args.guard_port,
        "target_port": args.target_port,
    }

    if args.single:
        # 单独运行
        combo = {
            "skill_call_mode": args.single[0],
            "skill_extraction_mode": args.single[1],
            "update_strategy": args.single[2],
        }
        print("Running single experiment:")
        print(f"  {combo}")

        result = run_single_experiment(
            combo=combo,
            base_args=base_args,
            experiment_id=1,
            output_dir=args.output_dir or "experiments/single",
        )
        return

    # 运行完整 Grid Search
    run_grid_search(
        base_args=base_args,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()