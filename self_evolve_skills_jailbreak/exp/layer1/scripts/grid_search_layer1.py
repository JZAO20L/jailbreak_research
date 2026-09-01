"""
Layer 1 Grid Search 实验脚本

基于 Layer 1 的核心方法消融实验设计：
- 消融点 A: skill_call_mode (single_call | every_iteration)
- 消融点 B: skill_extraction_mode (final_prompt | trajectory)
- 消融点 C: update_strategy (success_only | failure_only | both | statistical)

组合数: 2 × 2 × 4 = 16 种

数据配置（预抽取文件）：
- Cold Start: cold_start_prompts.json (200 条)
- Evolution: evolution_prompts.json (800 条)
- Test: test_prompts.json (1000 条)
"""

import os
import sys
import json
import time
import subprocess
import itertools
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent  # jailbreak_research
EXP_ROOT = PROJECT_ROOT / "self_evolve_skills_jailbreak"
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Layer 1 参数定义
# =============================================================================

# 消融参数
ABLATION_PARAMS = {
    "skill_call_mode": ["single_call", "every_iteration"],
    "skill_extraction_mode": ["final_prompt", "trajectory"],
    "update_strategy": ["success_only", "failure_only", "both", "statistical"],
}

# 数据配置（预抽取文件路径）
DATA_FILES = {
    "cold_start": EXP_ROOT / "data/cold_start_prompts.json",
    "evolution": EXP_ROOT / "data/evolution_prompts.json",
    "test": EXP_ROOT / "data/test_prompts.json",
}


def generate_layer1_combinations() -> List[Dict[str, str]]:
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
        base_args: 基础参数
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
        str(EXP_ROOT / "scripts/pipeline.py"),
        "--mode", "full",
        "--skill_call_mode", combo["skill_call_mode"],
        "--skill_extraction_mode", combo["skill_extraction_mode"],
        "--update_strategy", combo["update_strategy"],
        "--output_dir", output_dir,
        "--skip_launch",
    ]

    # 添加基础参数
    for key, value in base_args.items():
        if value is None:
            continue
        cmd.extend([f"--{key}", str(value)])

    # 运行实验（实时输出）
    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        stdout_lines = []
        for line in process.stdout:
            print(line, end='')
            stdout_lines.append(line)

        process.wait(timeout=3600 * 2)

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

    # 读取结果文件
    exp_name = f"{combo['skill_call_mode']}_{combo['skill_extraction_mode']}_{combo['update_strategy']}"
    results_file = Path(output_dir) / f"result_{exp_name}.json"

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
                    experiment_result["test_success"] = test_data.get("success", 0)
                    experiment_result["test_total"] = test_data.get("total", 0)

                if "cold_start" in data:
                    experiment_result["cold_start_skills"] = data["cold_start"].get("final_skill_count", 0)

                if "evolution" in data:
                    experiment_result["evolution_skills"] = data["evolution"].get("final_skill_count", 0)
                    experiment_result["final_skill_count"] = data["evolution"].get("final_skill_count", 0)
        except Exception as e:
            experiment_result["parse_error"] = str(e)

    # 保存单独实验结果
    experiment_file = Path(output_dir) / f"result_{exp_name}.json"
    with open(experiment_file, "w") as f:
        json.dump(experiment_result, f, indent=2, ensure_ascii=False)

    # 打印结果摘要
    if success:
        asr = experiment_result.get('asr')
        avg_iter = experiment_result.get('avg_iterations')
        evo_skills = experiment_result.get('evolution_skills')

        print(f"\n✓ Experiment completed")
        if asr is not None:
            print(f"  ASR: {asr*100:.1f}%")
        else:
            print(f"  ASR: N/A")
        if avg_iter is not None:
            print(f"  Avg iterations: {avg_iter:.1f}")
        else:
            print(f"  Avg iterations: N/A")
        print(f"  Final skills: {evo_skills if evo_skills is not None else 'N/A'}")
        print(f"  Time: {elapsed_time:.1f}s")
    else:
        print(f"\n✗ Experiment failed")
        print(f"  Error: {stderr[:200]}")

    return experiment_result


def run_layer1_grid_search(
    base_args: Dict[str, Any] = None,
    output_dir: str = None,
    resume_from: int = 0,
) -> Dict:
    """
    运行 Layer 1 Grid Search

    Args:
        base_args: 基础参数
        output_dir: 输出目录
        resume_from: 从第几个实验开始

    Returns:
        所有实验结果汇总
    """
    if base_args is None:
        base_args = {}

    if output_dir is None:
        output_dir = PROJECT_ROOT / "experiments" / f"layer1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "skills"), exist_ok=True)

    # 生成所有组合
    combinations = generate_layer1_combinations()
    total = len(combinations)

    print("="*60)
    print("Layer 1 Grid Search for Self Evolve Skills")
    print("="*60)
    print(f"Total combinations: {total}")
    print(f"Output directory: {output_dir}")
    print(f"Data files:")
    print(f"  Cold Start: {DATA_FILES['cold_start']}")
    print(f"  Evolution: {DATA_FILES['evolution']}")
    print(f"  Test: {DATA_FILES['test']}")
    print(f"Ablation points:")
    print(f"  A: skill_call_mode (single_call | every_iteration)")
    print(f"  B: skill_extraction_mode (final_prompt | trajectory)")
    print(f"  C: update_strategy (success_only | failure_only | both | statistical)")
    print("="*60)

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
        "results": {r["experiment_id"]: r for r in results},
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def create_final_summary(results: List[Dict], output_dir: str) -> Dict:
    """创建最终汇总报告"""

    successful = [r for r in results if r["success"] and "asr" in r]

    # 按消融点分组分析
    analysis = {
        "by_skill_call_mode": {},
        "by_extraction_mode": {},
        "by_update_strategy": {},
        "best_combination": None,
        "worst_combination": None,
    }

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
        "results": {r["experiment_id"]: r for r in results},
        "analysis": analysis,
    }

    # 保存最终汇总
    final_file = Path(output_dir) / "summary_final.json"
    with open(final_file, "w") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    # 打印分析结果
    print("\n" + "="*60)
    print("Layer 1 Grid Search Analysis Results")
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
    parser = argparse.ArgumentParser(description="Layer 1 Grid Search for Self Evolve Skills")

    # 基础参数
    parser.add_argument("--eval_limit", type=int, default=100, help="中间评估数据数量")
    parser.add_argument("--num_epochs", type=int, default=1, help="进化轮数")
    parser.add_argument("--max_iterations", type=int, default=10, help="最大攻击迭代次数")
    parser.add_argument("--max_workers", type=int, default=8, help="轨迹级并发数")
    parser.add_argument("--min_success_rate", type=float, default=0.7, help="低效 skill 清理阈值")
    parser.add_argument("--maintenance_interval", type=int, default=100, help="维护间隔步数")

    # 服务参数
    parser.add_argument("--guard_port", type=int, default=8002)
    parser.add_argument("--target_port", type=int, default=8001)

    # 输出参数
    parser.add_argument("--output_dir", type=str, default=None, help="实验结果输出目录")
    parser.add_argument("--resume_from", type=int, default=0, help="从第几个实验开始")

    # 单独运行某个组合（调试用）
    parser.add_argument("--single", type=str, nargs=3,
                        metavar=("CALL_MODE", "EXTRACTION_MODE", "UPDATE_STRATEGY"),
                        help="只运行单个组合，如: --single single_call trajectory both")

    # Test 限制（可选）
    parser.add_argument("--test_limit", type=int, default=None, help="Test 数据量限制")

    args = parser.parse_args()

    base_args = {
        "eval_limit": args.eval_limit,
        "num_epochs": args.num_epochs,
        "max_iterations": args.max_iterations,
        "max_workers": args.max_workers,
        "min_success_rate": args.min_success_rate,
        "maintenance_interval": args.maintenance_interval,
        "guard_port": args.guard_port,
        "target_port": args.target_port,
        "test_limit": args.test_limit,
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
            output_dir=args.output_dir or "experiments/layer1_single",
        )
        return

    # 运行完整 Grid Search
    run_layer1_grid_search(
        base_args=base_args,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()