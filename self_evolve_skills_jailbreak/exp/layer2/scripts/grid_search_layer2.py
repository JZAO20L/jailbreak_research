"""
Layer 2 Grid Search 实验脚本

基于 Layer 1 确定的最佳方法组合，验证数据量和分配策略的影响

实验组合: 4 × 3 × 3 = 36 组
- 方法组合 (来自 Layer 1): 4 种
- 数据量: small/medium/large (300/500/1000)
- 配比: early/balanced/evo (30%/20%/10%)
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent  # jailbreak_research
EXP_ROOT = PROJECT_ROOT / "self_evolve_skills_jailbreak"
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Layer 2 参数定义
# =============================================================================

# 来自 Layer 1 的最佳方法组合
METHOD_COMBINATIONS = [
    {"skill_call_mode": "single_call", "skill_extraction_mode": "trajectory", "update_strategy": "statistical"},
    {"skill_call_mode": "single_call", "skill_extraction_mode": "trajectory", "update_strategy": "success_only"},
    {"skill_call_mode": "single_call", "skill_extraction_mode": "final_prompt", "update_strategy": "success_only"},
    {"skill_call_mode": "single_call", "skill_extraction_mode": "final_prompt", "update_strategy": "statistical"},
]

# 数据量配置
DATA_SIZES = {
    "small": 300,
    "medium": 500,
    "large": 1000,
}

# 配比配置
CS_RATIOS = {
    "early": 0.30,      # 30% Cold Start, 70% Evolution
    "balanced": 0.20,   # 20% Cold Start, 80% Evolution
    "evo": 0.10,        # 10% Cold Start, 90% Evolution
}


def generate_layer2_combinations() -> List[Dict[str, Any]]:
    """
    生成 Layer 2 组合

    共 4 × 3 × 3 = 36 种
    """
    combinations = []
    exp_id = 1

    for method in METHOD_COMBINATIONS:
        for data_size_name, data_size in DATA_SIZES.items():
            for ratio_name, ratio in CS_RATIOS.items():
                cold_start = int(data_size * ratio)
                evolution = data_size - cold_start

                combo = {
                    "exp_id": exp_id,
                    "method": method,
                    "data_size_name": data_size_name,
                    "data_size": data_size,
                    "ratio_name": ratio_name,
                    "ratio": ratio,
                    "cold_start_size": cold_start,
                    "evolution_size": evolution,
                }
                combinations.append(combo)
                exp_id += 1

    return combinations


def run_single_experiment(
    combo: Dict[str, Any],
    base_args: Dict[str, Any],
    output_dir: str,
) -> Dict:
    """
    运行单个实验
    """
    method = combo["method"]
    data_size = combo["data_size"]
    ratio = combo["ratio"]
    exp_id = combo["exp_id"]

    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}/36")
    print(f"{'='*60}")
    print(f"Method: {method['skill_call_mode']} + {method['skill_extraction_mode']} + {method['update_strategy']}")
    print(f"Data: {combo['data_size_name']} ({data_size}), Ratio: {combo['ratio_name']} ({ratio*100:.0f}%)")
    print(f"Cold Start: {combo['cold_start_size']}, Evolution: {combo['evolution_size']}")

    # 构建命令
    cmd = [
        "python",
        str(EXP_ROOT / "scripts/pipeline.py"),
        "--mode", "full",
        "--skill_call_mode", method["skill_call_mode"],
        "--skill_extraction_mode", method["skill_extraction_mode"],
        "--update_strategy", method["update_strategy"],
        "--output_dir", output_dir,
        "--skip_launch",
    ]

    # 添加数据参数
    cmd.extend([
        "--train_limit", str(data_size),
        "--cs_ratio", str(ratio),
    ])

    # 添加基础参数
    for key, value in base_args.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])

    # 运行实验
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
    exp_name = f"{method['skill_call_mode']}_{method['skill_extraction_mode']}_{method['update_strategy']}_{combo['data_size_name']}_{combo['ratio_name']}"
    results_file = Path(output_dir) / f"result_{exp_name}.json"

    experiment_result = {
        "experiment_id": exp_id,
        "method": method,
        "data_size": combo["data_size_name"],
        "data_size_value": data_size,
        "ratio": combo["ratio_name"],
        "ratio_value": ratio,
        "cold_start_size": combo["cold_start_size"],
        "evolution_size": combo["evolution_size"],
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
                    experiment_result["avg_iterations"] = test_data.get("total_iterations", 0) / test_data.get("total", 1) if test_data.get("total", 0) > 0 else 0
                    experiment_result["test_success"] = test_data.get("success", 0)
                    experiment_result["test_total"] = test_data.get("total", 0)

                if "cold_start" in data:
                    experiment_result["cold_start_skills"] = data["cold_start"].get("final_skill_count", 0)

                if "evolution" in data:
                    experiment_result["evolution_skills"] = data["evolution"].get("final_skill_count", 0)
        except Exception as e:
            experiment_result["parse_error"] = str(e)

    # 保存单独实验结果
    meta_file = Path(output_dir) / f"meta_{exp_name}.json"
    with open(meta_file, "w") as f:
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
            print(f"  Avg iterations: {avg_iter:.2f}")
        else:
            print(f"  Avg iterations: N/A")
        print(f"  Final skills: {evo_skills if evo_skills is not None else 'N/A'}")
        print(f"  Time: {elapsed_time:.1f}s")
    else:
        print(f"\n✗ Experiment failed")
        print(f"  Error: {stderr[:200]}")

    return experiment_result


def run_layer2_grid_search(
    base_args: Dict[str, Any] = None,
    output_dir: str = None,
    resume_from: int = 0,
) -> Dict:
    """
    运行 Layer 2 Grid Search
    """
    if base_args is None:
        base_args = {}

    if output_dir is None:
        output_dir = PROJECT_ROOT / "experiments" / f"layer2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "skills"), exist_ok=True)

    # 生成所有组合
    combinations = generate_layer2_combinations()
    total = len(combinations)

    print("="*60)
    print("Layer 2 Grid Search for Self Evolve Skills")
    print("="*60)
    print(f"Total combinations: {total}")
    print(f"Output directory: {output_dir}")
    print(f"Method combinations: 4 (from Layer 1)")
    print(f"Data sizes: small(300), medium(500), large(1000)")
    print(f"CS ratios: early(30%), balanced(20%), evo(10%)")
    print("="*60)

    # 运行所有实验
    all_results = []

    for combo in combinations:
        exp_id = combo["exp_id"]

        if exp_id < resume_from:
            print(f"Skipping experiment {exp_id} (resume from {resume_from})")
            continue

        result = run_single_experiment(
            combo=combo,
            base_args=base_args,
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

    successful = [r for r in results if r["success"] and "asr" in r]

    # 按数据量分组
    by_data_size = {}
    for size_name in DATA_SIZES.keys():
        size_results = [r for r in successful if r.get("data_size") == size_name]
        if size_results:
            by_data_size[size_name] = {
                "count": len(size_results),
                "avg_asr": sum(r["asr"] for r in size_results) / len(size_results),
                "avg_iter": sum(r.get("avg_iterations", 0) for r in size_results) / len(size_results),
            }

    # 按配比分组
    by_ratio = {}
    for ratio_name in CS_RATIOS.keys():
        ratio_results = [r for r in successful if r.get("ratio") == ratio_name]
        if ratio_results:
            by_ratio[ratio_name] = {
                "count": len(ratio_results),
                "avg_asr": sum(r["asr"] for r in ratio_results) / len(ratio_results),
                "avg_iter": sum(r.get("avg_iterations", 0) for r in ratio_results) / len(ratio_results),
            }

    # 按方法分组
    by_method = {}
    for method in METHOD_COMBINATIONS:
        method_name = f"{method['skill_call_mode']}_{method['skill_extraction_mode']}_{method['update_strategy']}"
        method_results = [r for r in successful if
            r.get("method", {}).get("skill_call_mode") == method["skill_call_mode"] and
            r.get("method", {}).get("skill_extraction_mode") == method["skill_extraction_mode"] and
            r.get("method", {}).get("update_strategy") == method["update_strategy"]]
        if method_results:
            by_method[method_name] = {
                "count": len(method_results),
                "avg_asr": sum(r["asr"] for r in method_results) / len(method_results),
                "avg_iter": sum(r.get("avg_iterations", 0) for r in method_results) / len(method_results),
            }

    # 最佳组合
    sorted_by_asr = sorted(successful, key=lambda r: r["asr"], reverse=True)

    final_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "results": results,
        "analysis": {
            "by_data_size": by_data_size,
            "by_ratio": by_ratio,
            "by_method": by_method,
            "best_combination": sorted_by_asr[0] if sorted_by_asr else None,
        }
    }

    # 保存最终汇总
    final_file = Path(output_dir) / "summary_final.json"
    with open(final_file, "w") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    # 打印分析结果
    print("\n" + "="*60)
    print("Layer 2 Grid Search Analysis Results")
    print("="*60)

    if successful:
        print("\nBy data_size:")
        for size, stats in by_data_size.items():
            print(f"  {size}: ASR={stats['avg_asr']*100:.1f}%, Iter={stats['avg_iter']:.2f}")

        print("\nBy ratio:")
        for ratio, stats in by_ratio.items():
            print(f"  {ratio}: ASR={stats['avg_asr']*100:.1f}%, Iter={stats['avg_iter']:.2f}")

        print("\nBy method:")
        for method, stats in by_method.items():
            print(f"  {method}: ASR={stats['avg_asr']*100:.1f}%, Iter={stats['avg_iter']:.2f}")

        if sorted_by_asr:
            best = sorted_by_asr[0]
            print(f"\nBest combination:")
            print(f"  Method: {best.get('method')}")
            print(f"  Data: {best.get('data_size')}, Ratio: {best.get('ratio')}")
            print(f"  ASR: {best['asr']*100:.1f}%")

    print(f"\nSummary saved to: {output_dir}")

    return final_summary


def main():
    parser = argparse.ArgumentParser(description="Layer 2 Grid Search for Self Evolve Skills")

    # 基础参数
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--min_success_rate", type=float, default=0.7)
    parser.add_argument("--maintenance_interval", type=int, default=100)

    # 服务参数
    parser.add_argument("--guard_port", type=int, default=8002)
    parser.add_argument("--target_port", type=int, default=8001)

    # 输出参数
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=int, default=0)

    # 单独运行某个组合
    parser.add_argument("--single", type=str, nargs=5,
                        metavar=("CALL_MODE", "EXTRACTION", "UPDATE", "DATA_SIZE", "RATIO"),
                        help="只运行单个组合")

    args = parser.parse_args()

    base_args = {
        "num_epochs": args.num_epochs,
        "max_iterations": args.max_iterations,
        "max_workers": args.max_workers,
        "min_success_rate": args.min_success_rate,
        "maintenance_interval": args.maintenance_interval,
        "guard_port": args.guard_port,
        "target_port": args.target_port,
    }

    if args.single:
        # 单独运行
        call_mode, extraction, update, data_size, ratio = args.single

        # 验证参数
        if data_size not in DATA_SIZES:
            print(f"Error: data_size must be one of {list(DATA_SIZES.keys())}")
            return
        if ratio not in CS_RATIOS:
            print(f"Error: ratio must be one of {list(CS_RATIOS.keys())}")
            return

        combo = {
            "exp_id": 1,
            "method": {
                "skill_call_mode": call_mode,
                "skill_extraction_mode": extraction,
                "update_strategy": update,
            },
            "data_size_name": data_size,
            "data_size": DATA_SIZES[data_size],
            "ratio_name": ratio,
            "ratio": CS_RATIOS[ratio],
            "cold_start_size": int(DATA_SIZES[data_size] * CS_RATIOS[ratio]),
            "evolution_size": DATA_SIZES[data_size] - int(DATA_SIZES[data_size] * CS_RATIOS[ratio]),
        }

        print("Running single experiment:")
        print(f"  {combo}")

        run_single_experiment(
            combo=combo,
            base_args=base_args,
            output_dir=args.output_dir or "experiments/layer2_single",
        )
        return

    # 运行完整 Grid Search
    run_layer2_grid_search(
        base_args=base_args,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()