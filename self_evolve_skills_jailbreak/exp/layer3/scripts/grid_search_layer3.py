"""
Layer 3 Grid Search 实验脚本

基于AutoDAN的DAN模板作为初始Skills，探索数据策略对Skills进化的影响

实验组合: 3 × 4 = 12组
- 数据量: small/medium/large (300/500/1000)
- 配比: full_evolve/early/balanced/evo (0%/30%/20%/10%)

固定配置:
- Skills来源: DAN模板 (6个)
- 更新策略: statistical
- 检索模式: single_call
- Extraction: trajectory
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
# Layer 3 参数定义
# =============================================================================

# 固定配置 (来自 Layer 1/2 验证最优)
FIXED_CONFIG = {
    "skill_call_mode": "single_call",
    "skill_extraction_mode": "trajectory",
}

# 更新策略配置（新增变量）
UPDATE_STRATEGIES = {
    "statistical": "statistical",  # 统计驱动，不实时添加
    "success_only": "success_only",  # 仅成功时添加
}

# 数据量配置
DATA_SIZES = {
    "small": 300,
    "medium": 500,
    "large": 1000,
}

# 配比配置
CS_RATIOS = {
    "full_evolve": 0.00,  # 无冷启动，直接进化
    "early": 0.30,        # 30% Cold Start, 70% Evolution
    "balanced": 0.20,     # 20% Cold Start, 80% Evolution
    "evo": 0.10,          # 10% Cold Start, 90% Evolution
}


def generate_layer3_combinations() -> List[Dict[str, Any]]:
    """
    生成 Layer 3 组合

    共 3 × 4 × 2 = 24 种
    - 数据量: small/medium/large (3)
    - 配比: full_evolve/early/balanced/evo (4)
    - 更新策略: statistical/success_only (2)
    """
    combinations = []
    exp_id = 1

    for data_size_name, data_size in DATA_SIZES.items():
        for ratio_name, ratio in CS_RATIOS.items():
            for strategy_name, strategy in UPDATE_STRATEGIES.items():
                cold_start = int(data_size * ratio)
                evolution = data_size - cold_start

                combo = {
                    "exp_id": exp_id,
                    "fixed_config": {**FIXED_CONFIG, "update_strategy": strategy},
                    "data_size_name": data_size_name,
                    "data_size": data_size,
                    "ratio_name": ratio_name,
                    "ratio": ratio,
                    "update_strategy_name": strategy_name,
                    "update_strategy": strategy,
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
    运行单个 Layer 3 实验

    Args:
        combo: 实验组合配置
        base_args: 基础参数
        output_dir: 输出目录
        total_experiments: 总实验数（用于显示进度）

    Returns:
        实验结果
    """
    data_size = combo["data_size"]
    ratio = combo["ratio"]
    ratio_name = combo["ratio_name"]
    data_size_name = combo["data_size_name"]
    update_strategy = combo["update_strategy"]
    update_strategy_name = combo["update_strategy_name"]
    exp_id = combo["exp_id"]

    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}")
    print(f"{'='*60}")
    print(f"数据量: {data_size_name} ({data_size})")
    print(f"配比: {ratio_name} ({ratio*100:.0f}% CS)")
    print(f"更新策略: {update_strategy_name}")
    print(f"Cold Start: {combo['cold_start_size']}, Evolution: {combo['evolution_size']}")

    # 构建实验名称（包含 update_strategy，避免覆盖已有结果）
    exp_name = f"dan_start_{data_size_name}_{ratio_name}_{update_strategy_name}"

    # 构建命令
    cmd = [
        "python",
        str(EXP_ROOT / "scripts/pipeline.py"),
        "--mode", "full",
        "--skill_call_mode", FIXED_CONFIG["skill_call_mode"],
        "--skill_extraction_mode", FIXED_CONFIG["skill_extraction_mode"],
        "--update_strategy", update_strategy,
        "--output_dir", output_dir,
        "--exp_name", exp_name,  # 使用实验名称作为结果文件名
        "--skip_launch",
        # Layer 3 新参数
        "--skill_source", "dan_templates",
    ]

    # full_evove模式：跳过Cold Start
    if ratio_name == "full_evolve":
        cmd.append("--skip_cold_start")

    # 数据参数
    cmd.extend([
        "--train_limit", str(data_size),
        "--cs_ratio", str(ratio),
    ])

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
        "fixed_config": combo["fixed_config"],
        "data_size": data_size_name,
        "data_size_value": data_size,
        "ratio": ratio_name,
        "ratio_value": ratio,
        "update_strategy": update_strategy_name,
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

    # 保存元数据
    meta_file = Path(output_dir) / f"meta_{exp_name}.json"
    meta_data = {
        "exp_id": exp_id,
        "skill_source": "dan_templates",
        "data_size": data_size_name,
        "data_size_value": data_size,
        "ratio": ratio_name,
        "ratio_value": ratio,
        "update_strategy": update_strategy_name,
        "cold_start_size": combo["cold_start_size"],
        "evolution_size": combo["evolution_size"],
        "success": success,
        "elapsed_time": elapsed_time,
        "timestamp": datetime.now().isoformat(),
    }

    with open(meta_file, "w") as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)

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


def main():
    parser = argparse.ArgumentParser(description="Layer 3 Grid Search")

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--min_success_rate", type=float, default=0.7)
    parser.add_argument("--maintenance_interval", type=int, default=100)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--guard_port", type=int, default=8002)
    parser.add_argument("--target_port", type=int, default=8001)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--single", type=str, nargs=3, help="Run single experiment: data_size ratio update_strategy")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "skills").mkdir(parents=True, exist_ok=True)

    # 生成组合
    combinations = generate_layer3_combinations()

    print("=" * 60)
    print("Layer 3: AutoDAN起点Skills实验")
    print("=" * 60)
    print(f"固定配置:")
    print(f"  - Skills来源: DAN模板 (6个)")
    print(f"  - 检索模式: {FIXED_CONFIG['skill_call_mode']}")
    print(f"  - Extraction: {FIXED_CONFIG['skill_extraction_mode']}")
    print(f"\n消融变量:")
    print(f"  - 数据量: {list(DATA_SIZES.keys())}")
    print(f"  - 配比: {list(CS_RATIOS.keys())}")
    print(f"  - 更新策略: {list(UPDATE_STRATEGIES.keys())}")
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
        data_size_name, ratio_name, strategy_name = args.single

        # 找到对应组合
        combo = None
        for c in combinations:
            if c["data_size_name"] == data_size_name and c["ratio_name"] == ratio_name and c["update_strategy_name"] == strategy_name:
                combo = c
                break

        if combo is None:
            print(f"错误: 未找到组合 {data_size_name} / {ratio_name} / {strategy_name}")
            return

        print(f"\n运行单个实验: {data_size_name} / {ratio_name} / {strategy_name}")
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
        print(f"\n进度: {i+1}/{len(combinations)}")

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
    summary_path = output_dir / f"layer3_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # 计算统计
    successful_results = [r for r in results if r.get("success", True) and "asr" in r]

    avg_by_data_size = {}
    avg_by_ratio = {}
    avg_by_update_strategy = {}

    for r in successful_results:
        ds = r.get("data_size", "unknown")
        ratio_name = r.get("ratio", "unknown")
        strategy_name = r.get("update_strategy", "unknown")
        asr = r.get("asr", 0)

        if ds not in avg_by_data_size:
            avg_by_data_size[ds] = []
        avg_by_data_size[ds].append(asr)

        if ratio_name not in avg_by_ratio:
            avg_by_ratio[ratio_name] = []
        avg_by_ratio[ratio_name].append(asr)

        if strategy_name not in avg_by_update_strategy:
            avg_by_update_strategy[strategy_name] = []
        avg_by_update_strategy[strategy_name].append(asr)

    # 计算平均值
    avg_by_data_size = {k: sum(v)/len(v) for k, v in avg_by_data_size.items()}
    avg_by_ratio = {k: sum(v)/len(v) for k, v in avg_by_ratio.items()}
    avg_by_update_strategy = {k: sum(v)/len(v) for k, v in avg_by_update_strategy.items()}

    # 最佳配置
    best_result = max(successful_results, key=lambda x: x.get("asr", 0)) if successful_results else None

    summary = {
        "layer": 3,
        "total_experiments": len(results),
        "successful": len(successful_results),
        "failed": len([r for r in results if not r.get("success", True)]),
        "avg_by_data_size": avg_by_data_size,
        "avg_by_ratio": avg_by_ratio,
        "avg_by_update_strategy": avg_by_update_strategy,
        "best_config": {
            "data_size": best_result.get("data_size") if best_result else None,
            "ratio": best_result.get("ratio") if best_result else None,
            "update_strategy": best_result.get("update_strategy") if best_result else None,
            "asr": best_result.get("asr") if best_result else None,
        } if best_result else None,
        "results": results,
        "config": {
            "fixed_config": FIXED_CONFIG,
            "data_sizes": DATA_SIZES,
            "cs_ratios": CS_RATIOS,
            "update_strategies": UPDATE_STRATEGIES,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Layer 3 Grid Search 完成")
    print("=" * 60)
    print(f"总实验数: {len(results)}")
    print(f"成功: {len(successful_results)}")
    print(f"失败: {len([r for r in results if not r.get('success', True)])}")
    print(f"汇总文件: {summary_path}")

    if best_result:
        print(f"\n最佳结果:")
        print(f"  数据量: {best_result.get('data_size')}")
        print(f"  配比: {best_result.get('ratio')}")
        print(f"  更新策略: {best_result.get('update_strategy')}")
        print(f"  ASR: {best_result.get('asr', 0)*100:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()