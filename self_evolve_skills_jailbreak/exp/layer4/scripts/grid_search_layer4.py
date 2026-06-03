"""
Layer 4 Grid Search 实验脚本 - AutoDAN 数据策略验证

基于 Layer 3.1 最佳配置，验证数据量和配比的影响

固定配置（来自 Layer 3.1 最佳结果）：
- skill_call_mode: every_iteration
- update_strategy: success_only
- skill_source: dan_templates

Grid 变量：
- data_size: small/medium/large (300/500/1000)
- cs_ratio: full_evolve/early/balanced/evo (0%/30%/20%/10%)

实验总数：3 × 4 = 12 组
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
# Layer 4 参数定义 - 数据策略 Grid
# =============================================================================

# 固定配置（Layer 3.1 最佳）
FIXED_CONFIG = {
    "skill_call_mode": "every_iteration",
    "skill_extraction_mode": "trajectory",
    "update_strategy": "success_only",
    "skill_source": "dan_templates",
}

# 数据量配置
DATA_SIZES = {
    "small": 300,
    "medium": 500,
    "large": 1000,
}

# 配比配置
CS_RATIOS = {
    "full_evolve": 0.00,  # 无 Cold Start，直接进化
    "early": 0.30,        # 30% Cold Start, 70% Evolution
    "balanced": 0.20,     # 20% Cold Start, 80% Evolution
    "evo": 0.10,          # 10% Cold Start, 90% Evolution
}


def generate_layer4_combinations() -> List[Dict[str, Any]]:
    """
    生成 Layer 4 组合

    共 3 × 4 = 12 种
    - data_size: small/medium/large (3)
    - cs_ratio: full_evolve/early/balanced/evo (4)
    """
    combinations = []
    exp_id = 1

    for data_size_name, data_size in DATA_SIZES.items():
        for ratio_name, ratio in CS_RATIOS.items():
            cold_start = int(data_size * ratio)
            evolution = data_size - cold_start

            combo = {
                "exp_id": exp_id,
                "fixed_config": FIXED_CONFIG,
                "data_size_name": data_size_name,
                "data_size": data_size,
                "ratio_name": ratio_name,
                "ratio": ratio,
                "cold_start_size": cold_start,
                "evolution_size": evolution,
                "skip_cold_start": ratio == 0.0,
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
    运行单个 Layer 4 实验

    Args:
        combo: 实验组合配置
        base_args: 基础参数
        output_dir: 输出目录

    Returns:
        实验结果
    """
    data_size = combo["data_size"]
    ratio = combo["ratio"]
    ratio_name = combo["ratio_name"]
    data_size_name = combo["data_size_name"]
    skip_cold_start = combo["skip_cold_start"]
    cold_start_size = combo["cold_start_size"]
    evolution_size = combo["evolution_size"]
    exp_id = combo["exp_id"]

    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}/12")
    print(f"{'='*60}")
    print(f"数据量: {data_size_name} ({data_size})")
    print(f"配比: {ratio_name} ({ratio*100:.0f}% CS)")
    print(f"Cold Start: {cold_start_size}, Evolution: {evolution_size}")
    print(f"固定配置: every_iteration + success_only + DAN模板")

    # 构建实验名称
    exp_name = f"dan_data_{data_size_name}_{ratio_name}"

    # 构建命令
    cmd = [
        "python",
        str(EXP_ROOT / "scripts/pipeline.py"),
        "--mode", "full",
        "--skill_call_mode", FIXED_CONFIG["skill_call_mode"],
        "--skill_extraction_mode", FIXED_CONFIG["skill_extraction_mode"],
        "--update_strategy", FIXED_CONFIG["update_strategy"],
        "--output_dir", output_dir,
        "--exp_name", exp_name,
        "--skip_launch",
        "--skill_source", FIXED_CONFIG["skill_source"],
        "--train_limit", str(data_size),
        "--cs_ratio", str(ratio),
    ]

    # full_evolve 模式：跳过 Cold Start
    if skip_cold_start:
        cmd.append("--skip_cold_start")

    # 基础参数
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
    results_file = Path(output_dir) / f"result_{exp_name}.json"

    experiment_result = {
        "experiment_id": exp_id,
        "exp_name": exp_name,
        "fixed_config": FIXED_CONFIG,
        "data_size": data_size_name,
        "data_size_value": data_size,
        "ratio": ratio_name,
        "ratio_value": ratio,
        "cold_start_size": cold_start_size,
        "evolution_size": evolution_size,
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
                    cs_total = data["cold_start"].get("total", 0)
                    cs_success = data["cold_start"].get("success", 0)
                    experiment_result["cs_asr"] = cs_success / cs_total if cs_total > 0 else 0
                    experiment_result["cs_skills_extracted"] = data["cold_start"].get("skills_extracted", 0)

                if "evolution" in data:
                    evo_total = data["evolution"].get("total_attempts", 0)
                    evo_success = data["evolution"].get("success", 0)
                    experiment_result["evo_asr"] = evo_success / evo_total if evo_total > 0 else 0
                    experiment_result["evo_skills_added"] = data["evolution"].get("skills_added", 0)
                    experiment_result["final_skill_count"] = data["evolution"].get("final_skill_count", 0)
        except Exception as e:
            experiment_result["parse_error"] = str(e)

    # 保存元数据
    meta_file = Path(output_dir) / f"meta_{exp_name}.json"
    meta_data = {
        "exp_id": exp_id,
        "fixed_config": FIXED_CONFIG,
        "data_size": data_size_name,
        "data_size_value": data_size,
        "ratio": ratio_name,
        "ratio_value": ratio,
        "cold_start_size": cold_start_size,
        "evolution_size": evolution_size,
        "skill_source": "dan_templates",
        "success": success,
        "elapsed_time": elapsed_time,
        "timestamp": datetime.now().isoformat(),
    }

    with open(meta_file, "w") as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)

    # 打印结果摘要
    if success:
        asr = experiment_result.get('asr')
        final_skills = experiment_result.get('final_skill_count', 'N/A')

        print(f"\n✓ Experiment completed")
        if asr is not None:
            print(f"  ASR: {asr*100:.1f}%")
        else:
            print(f"  ASR: N/A")
        print(f"  Final skills: {final_skills}")
        print(f"  Time: {elapsed_time:.1f}s")
    else:
        print(f"\n✗ Experiment failed")
        print(f"  Error: {stderr[:200]}")

    return experiment_result


def main():
    parser = argparse.ArgumentParser(description="Layer 4 Grid Search - Data Strategy")

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--min_success_rate", type=float, default=0.7)
    parser.add_argument("--maintenance_interval", type=int, default=100)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--guard_port", type=int, default=8002)
    parser.add_argument("--target_port", type=int, default=8001)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--single", type=str, nargs=2, help="Run single experiment: data_size ratio")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "skills").mkdir(parents=True, exist_ok=True)

    # 生成组合
    combinations = generate_layer4_combinations()

    print("=" * 60)
    print("Layer 4: AutoDAN 数据策略 Grid")
    print("=" * 60)
    print(f"固定配置（Layer 3.1 最佳）:")
    print(f"  - skill_call_mode: every_iteration")
    print(f"  - update_strategy: success_only")
    print(f"  - skill_source: DAN模板 (6个)")
    print(f"\nGrid 变量:")
    print(f"  - data_size: {list(DATA_SIZES.keys())}")
    print(f"  - cs_ratio: {list(CS_RATIOS.keys())}")
    print(f"\n实验总数: {len(combinations)}")
    print("=" * 60)

    # 基础参数
    base_args = {
        "num_epochs": args.num_epochs,
        "max_iterations": args.max_iterations,
        "min_success_rate": args.min_success_rate,
        "maintenance_interval": args.maintenance_interval,
        "guard_port": args.guard_port,
        "target_port": args.target_port,
        "max_workers": args.max_workers,
    }

    # 单实验模式
    if args.single:
        data_size_name, ratio_name = args.single

        # 找到对应组合
        combo = None
        for c in combinations:
            if c["data_size_name"] == data_size_name and c["ratio_name"] == ratio_name:
                combo = c
                break

        if combo is None:
            print(f"错误: 未找到组合 {data_size_name} / {ratio_name}")
            return

        print(f"\n运行单个实验: {data_size_name} / {ratio_name}")
        run_single_experiment(combo, base_args, str(output_dir))
        return

    # 断点续跑
    start_idx = args.resume_from
    if start_idx > 0:
        print(f"\n断点续跑: 从实验 {start_idx} 开始")
        combinations = combinations[start_idx-1:]

    # 运行所有实验（串行）
    results = []

    for i, combo in enumerate(combinations):
        print(f"\n进度: {start_idx + i + 1}/12")

        try:
            result = run_single_experiment(combo, base_args, str(output_dir))
            results.append(result)
        except Exception as e:
            print(f"错误: 实验 {combo['exp_id']} 失败: {e}")
            results.append({
                "exp_id": combo["exp_id"],
                "error": str(e),
                "success": False,
            })

    # 保存汇总
    summary_path = output_dir / f"layer4_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # 计算统计
    successful_results = [r for r in results if r.get("success", True) and "asr" in r]

    avg_by_data_size = {}
    avg_by_ratio = {}

    for r in successful_results:
        ds = r.get("data_size", "unknown")
        rt = r.get("ratio", "unknown")
        asr = r.get("asr", 0)

        if ds not in avg_by_data_size:
            avg_by_data_size[ds] = []
        avg_by_data_size[ds].append(asr)

        if rt not in avg_by_ratio:
            avg_by_ratio[rt] = []
        avg_by_ratio[rt].append(asr)

    # 计算平均值
    avg_by_data_size = {k: sum(v)/len(v) for k, v in avg_by_data_size.items()}
    avg_by_ratio = {k: sum(v)/len(v) for k, v in avg_by_ratio.items()}

    # 最佳配置
    best_result = max(successful_results, key=lambda x: x.get("asr", 0)) if successful_results else None

    summary = {
        "layer": 4,
        "experiment_type": "data_strategy",
        "total_experiments": len(results),
        "successful": len(successful_results),
        "failed": len([r for r in results if not r.get("success", True)]),
        "avg_by_data_size": avg_by_data_size,
        "avg_by_ratio": avg_by_ratio,
        "best_config": {
            "data_size": best_result.get("data_size") if best_result else None,
            "ratio": best_result.get("ratio") if best_result else None,
            "asr": best_result.get("asr") if best_result else None,
            "final_skill_count": best_result.get("final_skill_count") if best_result else None,
        } if best_result else None,
        "results": results,
        "config": {
            "fixed_config": FIXED_CONFIG,
            "data_sizes": DATA_SIZES,
            "cs_ratios": CS_RATIOS,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Layer 4 Grid Search 完成")
    print("=" * 60)
    print(f"总实验数: {len(results)}")
    print(f"成功: {len(successful_results)}")
    print(f"失败: {len([r for r in results if not r.get('success', True)])}")
    print(f"汇总文件: {summary_path}")

    if best_result:
        print(f"\n最佳结果:")
        print(f"  数据量: {best_result.get('data_size')}")
        print(f"  配比: {best_result.get('ratio')}")
        print(f"  ASR: {best_result.get('asr', 0)*100:.1f}%")
        print(f"  Final skills: {best_result.get('final_skill_count', 'N/A')}")

    # 打印分组统计
    print(f"\n按数据量平均 ASR:")
    for ds, avg in sorted(avg_by_data_size.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ds}: {avg*100:.1f}%")

    print(f"\n按配比平均 ASR:")
    for rt, avg in sorted(avg_by_ratio.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rt}: {avg*100:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()