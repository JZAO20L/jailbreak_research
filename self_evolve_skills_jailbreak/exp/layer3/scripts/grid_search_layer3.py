"""
Layer 3 Grid Search 实验脚本 - 核心机制验证

基于AutoDAN的DAN模板作为初始Skills，验证最优 skills 管理机制

实验设计：
- Grid: skill_call_mode × update_strategy × cs_ratio = 2 × 4 × 2 = 16 组
- 固定: data_size = large (1000)

目标：
对比 full_evolve (无CS) vs 30% CS 的效果
对比不同 update_strategy 的效果
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
}

CS_RATIOS = {
    "full_evolve": 0.0,    # 无 Cold Start
    "early": 0.3,         # 30% Cold Start
}

# 固定配置
FIXED_CONFIG = {
    "skill_extraction_mode": "trajectory",
    "skill_source": "dan_templates",
    "data_size": "large",
    "data_size_value": 1000,
}


def generate_layer3_combinations() -> List[Dict[str, Any]]:
    """
    生成 Layer 3.1 组合

    共 2 × 4 × 2 = 16 种
    - skill_call_mode: single_call / every_iteration (2)
    - update_strategy: success_only/failure_only/both/statistical (4)
    - cs_ratio: full_evolve(0%) / early(30%) (2)
    """
    combinations = []
    exp_id = 1

    for call_mode_name, call_mode in SKILL_CALL_MODES.items():
        for strategy_name, strategy in UPDATE_STRATEGIES.items():
            for ratio_name, ratio in CS_RATIOS.items():
                cold_start = int(FIXED_CONFIG["data_size_value"] * ratio)
                evolution = FIXED_CONFIG["data_size_value"] - cold_start

                combo = {
                    "exp_id": exp_id,
                    "skill_call_mode_name": call_mode_name,
                    "skill_call_mode": call_mode,
                    "update_strategy_name": strategy_name,
                    "update_strategy": strategy,
                    "ratio_name": ratio_name,
                    "ratio": ratio,
                    "cold_start_size": cold_start,
                    "evolution_size": evolution,
                    "skip_cold_start": ratio == 0.0,
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
    ratio_name = combo["ratio_name"]
    ratio = combo["ratio"]
    skip_cold_start = combo["skip_cold_start"]
    cold_start_size = combo["cold_start_size"]
    evolution_size = combo["evolution_size"]
    exp_id = combo["exp_id"]

    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}/16")
    print(f"{'='*60}")
    print(f"skill_call_mode: {call_mode_name}")
    print(f"update_strategy: {strategy_name}")
    print(f"cs_ratio: {ratio_name} ({ratio*100:.0f}% CS)")
    print(f"Cold Start: {cold_start_size}, Evolution: {evolution_size}")

    # 构建实验名称
    exp_name = f"dan_{call_mode_name}_{strategy_name}_large_{ratio_name}"

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
        "--train_limit", str(FIXED_CONFIG["data_size_value"]),
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
        "skill_call_mode": call_mode_name,
        "update_strategy": strategy_name,
        "cs_ratio": ratio_name,
        "cs_ratio_value": ratio,
        "cold_start_size": cold_start_size,
        "evolution_size": evolution_size,
        "data_size": "large",
        "data_size_value": 1000,
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
                    experiment_result["cs_asr"] = data["cold_start"].get("success", 0) / data["cold_start"].get("total", 1) if data["cold_start"].get("total", 0) > 0 else 0

                if "evolution" in data:
                    experiment_result["evo_asr"] = data["evolution"].get("success", 0) / data["evolution"].get("total_attempts", 1) if data["evolution"].get("total_attempts", 0) > 0 else 0
                    experiment_result["evo_skills"] = data["evolution"].get("final_skill_count", 0)
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
        "cs_ratio": ratio_name,
        "cs_ratio_value": ratio,
        "cold_start_size": cold_start_size,
        "evolution_size": evolution_size,
        "data_size": "large",
        "data_size_value": 1000,
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
        skills_added = experiment_result.get('skills_added', 'N/A')

        print(f"\n✓ Experiment completed")
        if asr is not None:
            print(f"  ASR: {asr*100:.1f}%")
        else:
            print(f"  ASR: N/A")
        print(f"  DAN skills preserved: {dan_preserved}")
        print(f"  Skills added: {skills_added}")
        print(f"  Time: {elapsed_time:.1f}s")
    else:
        print(f"\n✗ Experiment failed")
        print(f"  Error: {stderr[:200]}")

    return experiment_result


