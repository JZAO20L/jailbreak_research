#!/usr/bin/env python3
"""
Jailbreak Prompt 实验脚本 - 实验1

对所有jailbreak_prompts中的prompt策略在test集上进行ASR测试。
支持：
- 原始prompt基线测试
- Top-K 筛选
- Gap Threshold 筛选
- 进度条显示

使用方法:
    python jailbreak_prompt_exp.py                               # 运行所有策略
    python jailbreak_prompt_exp.py --topk 5                      # 只输出top5策略
    python jailbreak_prompt_exp.py --gap_threshold 0.05          # gap>5%时舍弃
    python jailbreak_prompt_exp.py --strategies s1 s2            # 指定策略
    python jailbreak_prompt_exp.py --test_set test.jsonl         # 指定测试集
"""

import os
import sys
import json
import argparse
import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from experiments.jailbreak_prompt_exp.jailbreak_prompts import (
    JAILBREAK_PROMPTS,
    get_all_strategy_names,
    get_strategy_template,
    get_strategy_info,
)

# =========================
# 配置
# =========================
DEFAULT_CONFIG = {
    # 测试集路径 (默认使用test.jsonl)
    "test_set": os.path.join(BASE_DIR, "../data/dataset/processed/10k/test.jsonl"),
    
    # 输出目录
    "output_root": os.path.join(BASE_DIR, "experiments/jailbreak_prompt_exp/output"),
    
    # 模型路径
    "base_model": "/root/autodl-tmp/models/Qwen/Qwen3-4B",
    "target_model": "/root/autodl-tmp/models/Qwen/Qwen3-4B",
    "guard_model": "/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B",
    
    # vLLM端口
    "policy_port": 8003,
    "target_port": 8001,
    "guard_port": 8002,
    
    # 生成配置
    "k": 1,
    "rewrite_temperature": 0.7,
    "rewrite_max_tokens": 1024,
    "max_model_len": 2048,
    
    # 其他
    "sleep_between_evals": 10,  # 每次评估之间等待的秒数
}


def extract_asr_from_summary(summary_dir: str) -> float:
    """
    从eval输出的summary.json中提取ASR
    
    Returns:
        float: ASR值，如果找不到则返回0.0
    """
    import glob
    
    # 查找summary.json文件
    pattern = os.path.join(summary_dir, "*", "summary.json")
    files = glob.glob(pattern)
    
    if not files:
        # 尝试直接在目录中查找
        pattern = os.path.join(summary_dir, "summary.json")
        files = glob.glob(pattern)
    
    if not files:
        return 0.0
    
    try:
        with open(files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 尝试多种可能的路径
        if 'combinations' in data:
            return data['combinations'][0].get('asr', 0.0)
        elif 'overall' in data:
            return data['overall'].get('asr', 0.0)
        elif 'asr' in data:
            return data['asr']
        else:
            return 0.0
    except Exception as e:
        print(f"  [警告] 无法解析summary文件: {e}")
        return 0.0


def run_eval_for_strategy(
    strategy_name: str,
    strategy_template: Optional[str] = None,
    test_set: str = "",
    output_dir: str = "",
    config: dict = None,
    is_baseline: bool = False,
) -> float:
    """
    对单个策略运行评估
    
    Args:
        strategy_name: 策略名称
        strategy_template: 策略模板（None表示使用原始prompt）
        test_set: 测试集路径
        output_dir: 输出目录
        config: 配置字典
        is_baseline: 是否是基线测试
    
    Returns:
        float: ASR值
    """
    if config is None:
        config = {}
    
    # 导入eval脚本
    sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))
    from eval import main as eval_main
    
    # 构建参数
    run_name = "baseline_original" if is_baseline else f"strategy_{strategy_name}"
    
    eval_args = [
        "--eval_path", test_set,
        "--base_model_path", config.get("base_model", DEFAULT_CONFIG["base_model"]),
        "--target_model_path", config.get("target_model", DEFAULT_CONFIG["target_model"]),
        "--guard_model_path", config.get("guard_model", DEFAULT_CONFIG["guard_model"]),
        "--policy_port", str(config.get("policy_port", DEFAULT_CONFIG["policy_port"])),
        "--target_port", str(config.get("target_port", DEFAULT_CONFIG["target_port"])),
        "--guard_port", str(config.get("guard_port", DEFAULT_CONFIG["guard_port"])),
        "--k", str(config.get("k", DEFAULT_CONFIG["k"])),
        "--rewrite_temperature", str(config.get("rewrite_temperature", DEFAULT_CONFIG["rewrite_temperature"])),
        "--rewrite_max_tokens", str(config.get("rewrite_max_tokens", DEFAULT_CONFIG["rewrite_max_tokens"])),
        "--max_model_len", str(config.get("max_model_len", DEFAULT_CONFIG["max_model_len"])),
        "--output_root", output_dir,
        "--run_name", run_name,
    ]
    
    if is_baseline:
        # 基线测试：使用原始prompt (不重写)
        eval_args.extend(["--prompt_ids", "original"])
    else:
        # 策略测试：使用策略名
        eval_args.extend(["--strategy_name", strategy_name])
    
    print(f"  启动评估: {strategy_name}")
    print(f"  输出目录: {output_dir}")
    
    # 执行评估
    try:
        eval_main(eval_args)
    except Exception as e:
        print(f"  [错误] 评估失败: {e}")
        return 0.0
    
    # 提取ASR
    asr = extract_asr_from_summary(output_dir)
    return asr


def filter_results(
    results: List[Tuple[str, float]],
    baseline_asr: float,
    topk: Optional[int] = None,
    gap_threshold: Optional[float] = None,
) -> List[Tuple[str, float]]:
    """
    筛选结果
    
    Args:
        results: [(策略名, ASR)] 列表，已按ASR降序排序
        baseline_asr: 基线ASR
        topk: 只保留前K个
        gap_threshold: 差距阈值
    
    Returns:
        筛选后的结果列表
    """
    if topk is not None:
        return results[:topk]
    
    if gap_threshold is not None:
        filtered = []
        prev_asr = None
        for name, asr in results:
            if prev_asr is None or (prev_asr - asr) <= gap_threshold:
                filtered.append((name, asr))
                prev_asr = asr
            else:
                print(f"\n  [Gap筛选] {name} 与上一策略差距为 {prev_asr - asr:.4f} > {gap_threshold}，舍弃及后续策略")
                break
        return filtered
    
    # 没有筛选条件，返回全部
    return results


def print_results_table(
    results: List[Tuple[str, float]],
    baseline_asr: float,
    filtered_results: Optional[List[Tuple[str, float]]] = None,
):
    """打印结果表格"""
    print("\n" + "="*80)
    print("实验1 结果汇总 (按ASR降序)")
    print("="*80)
    print(f"{'排名':<6} {'策略':<30} {'ASR':<10} {'vs基线':<10}")
    print("-"*80)
    
    for rank, (name, asr) in enumerate(results, 1):
        vs_baseline = asr - baseline_asr
        sign = "+" if vs_baseline >= 0 else ""
        marker = " *" if filtered_results and (name, asr) not in filtered_results else ""
        print(f"{rank:<6} {name:<30} {asr:<10.4f} {sign}{vs_baseline:.4f}{marker}")
    
    print("="*80)
    if filtered_results:
        print("\n标记 * 的策略被筛选条件舍弃")
        print(f"\n筛选后保留的策略 ({len(filtered_results)}个):")
        for rank, (name, asr) in enumerate(filtered_results, 1):
            print(f"  {rank}. {name}: {asr:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Jailbreak Prompt 实验脚本 - 实验1")
    
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="要评估的策略列表 (默认: 所有策略)",
    )
    parser.add_argument(
        "--test_set",
        type=str,
        default=None,
        help="测试集路径 (默认: data/dataset/processed/10k/test.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="只输出表现最好的K个策略",
    )
    parser.add_argument(
        "--gap_threshold",
        type=float,
        default=None,
        help="当策略间ASR差距大于此值时，舍弃后续策略 (如 0.05 表示5%)",
    )
    parser.add_argument(
        "--sleep_between_evals",
        type=int,
        default=None,
        help="两次评估之间的等待时间 (秒)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印配置，不实际执行",
    )
    
    args = parser.parse_args()
    
    # 合并配置
    config = dict(DEFAULT_CONFIG)
    if args.test_set:
        config["test_set"] = args.test_set
    if args.output_dir:
        config["output_root"] = args.output_dir
    if args.sleep_between_evals is not None:
        config["sleep_between_evals"] = args.sleep_between_evals
    
    # 确定要评估的策略
    if args.strategies:
        strategies = args.strategies
    else:
        strategies = get_all_strategy_names()
    
    # 验证策略名称
    for s in strategies:
        if s not in JAILBREAK_PROMPTS:
            raise ValueError(f"未知策略: {s}")
    
    # 打印实验配置
    print("="*80)
    print("Jailbreak Prompt 实验 - 实验1")
    print("="*80)
    print(f"测试集: {config['test_set']}")
    print(f"策略数量: {len(strategies)}")
    print(f"总评估次数: {len(strategies) + 1} (含原始基线)")
    print(f"输出目录: {config['output_root']}")
    if args.topk:
        print(f"Top-K筛选: 只保留前 {args.topk} 个策略")
    if args.gap_threshold:
        print(f"Gap筛选: 差距阈值 = {args.gap_threshold}")
    print("="*80)
    
    # 如果是dry run，只打印配置
    if args.dry_run:
        print("\n[Dry Run] 配置已验证，未实际执行评估")
        print(f"将会创建 {len(strategies) + 1} 个评估任务")
        print("  - baseline_original (基线)")
        for s in strategies:
            print(f"  - {s}")
        return
    
    # 创建输出目录
    os.makedirs(config["output_root"], exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # =================================================================
    # Step 1: 测试原始prompt的ASR (基线)
    # =================================================================
    print("\n" + "="*80)
    print("Step 1/2: 评估基线 - 原始prompt (无重写)")
    print("="*80)
    
    baseline_output_dir = os.path.join(config["output_root"], "baseline_original")
    os.makedirs(baseline_output_dir, exist_ok=True)
    
    baseline_asr = run_eval_for_strategy(
        strategy_name="baseline_original",
        test_set=config["test_set"],
        output_dir=baseline_output_dir,
        config=config,
        is_baseline=True,
    )
    
    print(f"\n  基线ASR: {baseline_asr:.4f}")
    
    # 等待
    if config["sleep_between_evals"] > 0:
        print(f"  等待 {config['sleep_between_evals']} 秒...")
        time.sleep(config["sleep_between_evals"])
    
    # =================================================================
    # Step 2: 依次评估各策略
    # =================================================================
    print("\n" + "="*80)
    print(f"Step 2/2: 评估 {len(strategies)} 个策略")
    print("="*80)
    
    results = []
    sleep_time = config["sleep_between_evals"]
    
    # 使用tqdm显示进度
    for idx, strategy_name in enumerate(tqdm(strategies, desc="评估策略", unit="策略")):
        strategy_output_dir = os.path.join(config["output_root"], strategy_name)
        os.makedirs(strategy_output_dir, exist_ok=True)
        
        asr = run_eval_for_strategy(
            strategy_name=strategy_name,
            test_set=config["test_set"],
            output_dir=strategy_output_dir,
            config=config,
            is_baseline=False,
        )
        
        results.append((strategy_name, asr))
        
        # 等待
        if idx < len(strategies) - 1 and sleep_time > 0:
            time.sleep(sleep_time)
    
    # 记录结束时间
    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # =================================================================
    # Step 3: 结果筛选与汇总
    # =================================================================
    print("\n" + "="*80)
    print("Step 3/3: 结果筛选与汇总")
    print("="*80)
    
    # 按ASR降序排序
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # 应用筛选逻辑
    filtered_results = filter_results(
        sorted_results,
        baseline_asr,
        topk=args.topk,
        gap_threshold=args.gap_threshold,
    )
    
    # 打印结果表格
    print_results_table(sorted_results, baseline_asr, filtered_results)
    
    # 保存汇总结果
    summary = {
        "experiment": "jailbreak_prompt_exp_1",
        "timestamp": datetime.datetime.now().isoformat(),
        "test_set": config["test_set"],
        "baseline_asr": baseline_asr,
        "all_results": {name: asr for name, asr in sorted_results},
        "filtered_results": {name: asr for name, asr in filtered_results},
        "filter_criteria": {
            "topk": args.topk,
            "gap_threshold": args.gap_threshold,
        },
        "elapsed_time": {
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds,
        },
    }
    
    summary_path = os.path.join(config["output_root"], "experiment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n汇总结果已保存: {summary_path}")
    print(f"\n实验完成! 总耗时: {hours}小时 {minutes}分钟 {seconds}秒")


if __name__ == "__main__":
    main()
