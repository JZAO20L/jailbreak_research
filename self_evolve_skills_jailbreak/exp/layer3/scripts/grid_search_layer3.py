"""
Layer 3 Grid Search 实验脚本 - 核心机制验证

基于AutoDAN的DAN模板作为初始Skills，验证最优 skills 管理机制

实验设计：
- Grid: skill_call_mode × update_strategy = 2 × 5 = 10 组
- 固定: data_size = large (1000), ratio = full_evolve

目标：
对比 pure（不修改）vs statistical（维护）vs success_only（添加）等策略
确定 AutoDAN 场景下的最优配置
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
# Layer 3.1 参数定义 - 核心机制 Grid
# =============================================================================

# Grid 变量
SKILL_CALL_MODES = {
    "single_call": "single_call",      # 固定一个 skill
    "every_iteration": "every_iteration",  # 每轮动态切换
}

UPDATE_STRATEGIES = {
    "success_only": "success_only",    # 仅成功时添加
    "failure_only": "failure_only",    # 仅失败时添加
    "both": "both",                    # 成功和失败都添加
    "statistical": "statistical",      # 不添加，只维护
    "pure": "pure",                    # 完全不修改（新增）
}

# 固定配置
FIXED_CONFIG = {
    "skill_extraction_mode": "trajectory",
    "skill_source": "dan_templates",
    "data_size": "large",
    "data_size_value": 1000,
    "ratio": "full_evolve",
    "ratio_value": 0.0,
    "skip_cold_start": True,
}


def generate_layer3_combinations() -> List[Dict[str, Any]]:
    """
    生成 Layer 3.1 组合

    共 2 × 5 = 10 种
    - skill_call_mode: single_call / every_iteration (2)
    - update_strategy: success_only/failure_only/both/statistical/pure (5)
    """
    combinations = []
    exp_id = 1

    for call_mode_name, call_mode in SKILL_CALL_MODES.items():
        for strategy_name, strategy in UPDATE_STRATEGIES.items():
            combo = {
                "exp_id": exp_id,
                "skill_call_mode_name": call_mode_name,
                "skill_call_mode": call_mode,
                "update_strategy_name": strategy_name,
                "update_strategy": strategy,
                "fixed_config": FIXED_CONFIG,
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
    运行单个 Layer 3.1 实验

    Args:
        combo: 实验组合配置
        base_args: 基础参数
        output_dir: 输出目录

    Returns:
        实验结果
    """
    call_mode = combo["skill_call_mode"]
    call_mode_name = combo["skill_call_mode_name"]
    strategy = combo["update_strategy"]
    strategy_name = combo["update_strategy_name"]
    exp_id = combo["exp_id"]

    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}/10")
    print(f"{'='*60}")
    print(f"skill_call_mode: {call_mode_name}")
    print(f"update_strategy: {strategy_name}")
    print(f"data_size: large (1000)")
    print(f"ratio: full_evolve (无 Cold Start)")

    # 构建实验名称
    exp_name = f"dan_{call_mode_name}_{strategy_name}_large_full_evolve"

    # 构建命令
    cmd = [
        "python",
        str(EXP_ROOT / "scripts/pipeline.py"),
        "--mode", "full",
        "--skill_call_mode", call_mode,
        "--skill_extraction_mode", FIXED_CONFIG["skill_extraction_mode"],
        "--update_strategy", strategy,
        "--output_dir", output_dir,
        "--exp_name", exp_name,
        "--skip_launch",
        "--skill_source", FIXED_CONFIG["skill_source"],
        "--skip_cold_start",
        "--train_limit", str(FIXED_CONFIG["data_size_value"]),
        "--cs_ratio", str(FIXED_CONFIG["ratio_value"]),
    ]

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
        "skill_call_mode": call_mode_name,
        "update_strategy": strategy_name,
        "data_size": "large",
        "data_size_value": 1000,
        "ratio": "full_evolve",
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

                if "evolution" in data:
                    experiment_result["evolution_skills"] = data["evolution"].get("final_skill_count", 0)
                    experiment_result["skills_added"] = data["evolution"].get("skills_added", 0)

                # 记录初始 DAN skills 保留情况
                if "config" in data and "skills_path" in data["config"]:
                    skills_path = data["config"]["skills_path"]
                    if Path(skills_path).exists():
                        with open(skills_path, "r") as sf:
                            skills_data = json.load(sf)
                            dan_skills = [s for s in skills_data.get("skills", []) if s.get("source") == "dan_template"]
                            experiment_result["dan_skills_preserved"] = len(dan_skills)
        except Exception as e:
            experiment_result["parse_error"] = str(e)

    # 保存元数据
    meta_file = Path(output_dir) / f"meta_{exp_name}.json"
    meta_data = {
        "exp_id": exp_id,
        "skill_call_mode": call_mode_name,
        "update_strategy": strategy_name,
        "data_size": "large",
        "data_size_value": 1000,
        "ratio": "full_evolve",
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
        dan_preserved = experiment_result.get('dan_skills_preserved', 'N/A')

        print(f"\n✓ Experiment completed")
        if asr is not None:
            print(f"  ASR: {asr*100:.1f}%")
        else:
            print(f"  ASR: N/A")
        print(f"  DAN skills preserved: {dan_preserved}")
        print(f"  Time: {elapsed_time:.1f}s")
    else:
        print(f"\n✗ Experiment failed")
        print(f"  Error: {stderr[:200]}")

    return experiment_result


def main():
    parser = argparse.ArgumentParser(description="Layer 3.1 Grid Search - Core Mechanism")

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--min_success_rate", type=float, default=0.7)
    parser.add_argument("--maintenance_interval", type=int, default=100)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--guard_port", type=int, default=8002)
    parser.add_argument("--target_port", type=int, default=8001)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--single", type=str, nargs=2, help="Run single experiment: skill_call_mode update_strategy")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "skills").mkdir(parents=True, exist_ok=True)

    # 生成组合
    combinations = generate_layer3_combinations()

    print("=" * 60)
    print("Layer 3.1: AutoDAN Skills Core Mechanism Grid")
    print("=" * 60)
    print(f"Grid 变量:")
    print(f"  - skill_call_mode: {list(SKILL_CALL_MODES.keys())}")
    print(f"  - update_strategy: {list(UPDATE_STRATEGIES.keys())}")
    print(f"\n固定配置:")
    print(f"  - data_size: large (1000)")
    print(f"  - ratio: full_evolve (无 Cold Start)")
    print(f"  - skill_source: DAN模板 (6个)")
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
        call_mode_name, strategy_name = args.single

        # 找到对应组合
        combo = None
        for c in combinations:
            if c["skill_call_mode_name"] == call_mode_name and c["update_strategy_name"] == strategy_name:
                combo = c
                break

        if combo is None:
            print(f"错误: 未找到组合 {call_mode_name} / {strategy_name}")
            return

        print(f"\n运行单个实验: {call_mode_name} / {strategy_name}")
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
        print(f"\n进度: {start_idx + i + 1}/10")

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
    summary_path = output_dir / f"layer3_core_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # 计算统计
    successful_results = [r for r in results if r.get("success", True) and "asr" in r]

    avg_by_call_mode = {}
    avg_by_strategy = {}

    for r in successful_results:
        cm = r.get("skill_call_mode", "unknown")
        st = r.get("update_strategy", "unknown")
        asr = r.get("asr", 0)

        if cm not in avg_by_call_mode:
            avg_by_call_mode[cm] = []
        avg_by_call_mode[cm].append(asr)

        if st not in avg_by_strategy:
            avg_by_strategy[st] = []
        avg_by_strategy[st].append(asr)

    # 计算平均值
    avg_by_call_mode = {k: sum(v)/len(v) for k, v in avg_by_call_mode.items()}
    avg_by_strategy = {k: sum(v)/len(v) for k, v in avg_by_strategy.items()}

    # 最佳配置
    best_result = max(successful_results, key=lambda x: x.get("asr", 0)) if successful_results else None

    summary = {
        "layer": "3.1",
        "experiment_type": "core_mechanism",
        "total_experiments": len(results),
        "successful": len(successful_results),
        "failed": len([r for r in results if not r.get("success", True)]),
        "avg_by_call_mode": avg_by_call_mode,
        "avg_by_strategy": avg_by_strategy,
        "best_config": {
            "skill_call_mode": best_result.get("skill_call_mode") if best_result else None,
            "update_strategy": best_result.get("update_strategy") if best_result else None,
            "asr": best_result.get("asr") if best_result else None,
            "dan_skills_preserved": best_result.get("dan_skills_preserved") if best_result else None,
        } if best_result else None,
        "results": results,
        "config": {
            "grid_variables": {
                "skill_call_modes": SKILL_CALL_MODES,
                "update_strategies": UPDATE_STRATEGIES,
            },
            "fixed_config": FIXED_CONFIG,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Layer 3.1 Grid Search 完成")
    print("=" * 60)
    print(f"总实验数: {len(results)}")
    print(f"成功: {len(successful_results)}")
    print(f"失败: {len([r for r in results if not r.get('success', True)])}")
    print(f"汇总文件: {summary_path}")

    if best_result:
        print(f"\n最佳结果:")
        print(f"  skill_call_mode: {best_result.get('skill_call_mode')}")
        print(f"  update_strategy: {best_result.get('update_strategy')}")
        print(f"  ASR: {best_result.get('asr', 0)*100:.1f}%")
        print(f"  DAN skills preserved: {best_result.get('dan_skills_preserved', 'N/A')}")

    # 打印策略对比
    print(f"\n按策略平均 ASR:")
    for st, avg in sorted(avg_by_strategy.items(), key=lambda x: x[1], reverse=True):
        print(f"  {st}: {avg*100:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()