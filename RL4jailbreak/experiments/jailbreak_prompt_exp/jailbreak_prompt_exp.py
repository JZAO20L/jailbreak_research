#!/usr/bin/env python3
"""
Jailbreak Prompt 实验脚本 - 实验1

对所有jailbreak_prompts中的prompt策略在test集上进行ASR测试。

特点:
- 不启动/关闭模型，连接由sh脚本启动的已有服务
- 三个模型只连接一次，所有策略共享
- Top-K / Gap Threshold 筛选

使用方法:
    # 推荐: 通过 exp.sh 运行 (会自动启动/关闭模型)
    bash experiments/jailbreak_prompt_exp/exp.sh

    # 手动运行 (需要确保模型服务已在对应端口运行)
    python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py
"""

import os
import sys
import json
import gc
import argparse
import datetime
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.vllm_client import VLLMClient
from src.generate import rewrite_prompts_k
from src.test import run_asr_test_serial
from experiments.jailbreak_prompt_exp.jailbreak_prompts import (
    JAILBREAK_PROMPTS,
    get_all_strategy_names,
    get_strategy_template,
    get_strategy_info,
)

# =========================
# 配置 (默认值)
# =========================
DEFAULT_CONFIG = {
    # 测试集路径
    "test_set": os.path.join(BASE_DIR, "../data/dataset/processed/10k/test.jsonl"),

    # 输出目录
    "output_root": os.path.join(BASE_DIR, "experiments/jailbreak_prompt_exp/output"),

    # 模型路径
    "policy_model": "/root/autodl-tmp/models/Qwen/Qwen3-4B",
    "target_model": "/root/autodl-tmp/models/Qwen/Qwen3-4B",
    "guard_model": "/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B",

    # 端口 (连接已有服务，不启动新服务)
    "policy_port": 8003,
    "target_port": 8001,
    "guard_port": 8002,

    # 生成配置
    "k": 1,
    "rewrite_temperature": 0.7,
    "rewrite_max_tokens": 2048,
    "max_model_len": 4096,

    # ASR测试配置
    "test_batch_size": 64,
    "test_max_workers": 16,
    "target_max_tokens": 512,
    "target_temperature": 0.0,
    "guard_max_tokens": 256,
    "guard_temperature": 0.0,

    # 其他
    "sleep_between_evals": 5,
}


def load_checkpoint(output_root: str) -> Dict:
    """加载检查点"""
    ckpt_path = os.path.join(output_root, "checkpoint.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "results": [], "baseline_asr": None}


def save_checkpoint(output_root: str, ckpt: Dict) -> None:
    """保存检查点"""
    ckpt_path = os.path.join(output_root, "checkpoint.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


def compute_confusion_matrix(baseline_raw_path: str, strategy_raw_path: str) -> Dict:
    """计算原始prompt vs 新prompt的混淆矩阵"""
    if not os.path.exists(baseline_raw_path) or not os.path.exists(strategy_raw_path):
        return {}

    with open(baseline_raw_path, "r", encoding="utf-8") as f:
        baseline_results = json.load(f)
    with open(strategy_raw_path, "r", encoding="utf-8") as f:
        strategy_results = json.load(f)

    baseline_map = {r["id"]: r for r in baseline_results}

    confusion = {
        "fail_to_success": 0,
        "success_to_success": 0,
        "fail_to_fail": 0,
        "success_to_fail": 0,
        "details": [],
    }

    for sr in strategy_results:
        sid = sr["id"]
        br = baseline_map.get(sid)
        if not br:
            continue

        orig_success = br.get("guard_label") == "success"
        new_success = sr.get("guard_label") == "success"

        if not orig_success and new_success:
            confusion["fail_to_success"] += 1
            case = "fail_to_success"
        elif orig_success and new_success:
            confusion["success_to_success"] += 1
            case = "success_to_success"
        elif not orig_success and not new_success:
            confusion["fail_to_fail"] += 1
            case = "fail_to_fail"
        else:
            confusion["success_to_fail"] += 1
            case = "success_to_fail"

        confusion["details"].append({
            "id": sid,
            "original_prompt": sr.get("original_prompt", ""),
            "rewritten_prompt": sr.get("prompt", ""),
            "original_success": orig_success,
            "new_success": new_success,
            "case": case,
        })

    confusion["total"] = len(confusion["details"])
    return confusion


def print_confusion_matrix(confusion: Dict):
    """打印混淆矩阵"""
    if not confusion:
        return
    total = confusion.get("total", 0)
    if total == 0:
        return

    f2s = confusion["fail_to_success"]
    s2s = confusion["success_to_success"]
    f2f = confusion["fail_to_fail"]
    s2f = confusion["success_to_fail"]
    lift = f2s - s2f

    print("\n" + "="*60)
    print("混淆矩阵 (原始 vs 新prompt)")
    print("="*60)
    print(f"{'':>20} {'新prompt成功':>15} {'新prompt失败':>15}")
    print(f"{'原始prompt成功':>20} {s2s:>10} ({s2s/total*100:.1f}%) {s2f:>10} ({s2f/total*100:.1f}%)")
    print(f"{'原始prompt失败':>20} {f2s:>10} ({f2s/total*100:.1f}%) {f2f:>10} ({f2f/total*100:.1f}%)")
    print("-"*60)
    print(f"总样本数: {total}")
    print(f"Lift: {lift:+d} ({lift/total*100:+.1f}%)")
    print("="*60)


def load_test_set(path: str) -> List[Dict]:
    """加载测试集"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def run_rewrite(
    policy_client: VLLMClient,
    originals: List[str],
    prompt_template: Optional[str] = None,
    k: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> List[List[str]]:
    """对测试集进行重写"""
    if prompt_template is None:
        # 基线：直接使用原始prompt
        return [[o] for o in originals]

    # 使用jailbreak模板重写
    prompts = [prompt_template.format(original_prompt=o) for o in originals]

    rewritten = rewrite_prompts_k(
        client=policy_client,
        prompts=prompts,
        k=k,
        temperature=temperature,
        max_tokens=max_tokens,
        require_tag=False,
        show_progress=True,
        tqdm_desc="rewrite",
    )
    return rewritten


def run_asr_test(
    target_client: VLLMClient,
    guard_client: VLLMClient,
    originals: List[str],
    rewritten_buckets: List[List[str]],
    test_batch_size: int = 64,
    test_max_workers: int = 16,
    target_max_tokens: int = 512,
    target_temperature: float = 0.0,
    guard_max_tokens: int = 256,
    guard_temperature: float = 0.0,
    save_raw_results: bool = False,
    raw_output_path: Optional[str] = None,
) -> Dict:
    """对重写后的prompt进行ASR测试"""
    # 构建测试数据
    test_data = []
    for orig, bucket in zip(originals, rewritten_buckets):
        for j, new_prompt in enumerate(bucket):
            new_prompt = (new_prompt or "").strip()
            if not new_prompt:
                continue
            test_data.append({
                "prompt": new_prompt,
                "original_prompt": orig,
                "id": f"{len(test_data)}_{j}",
            })

    if not test_data:
        return {"asr": 0.0, "refusal_rate": 0.0, "partial_rate": 0.0, "success_rate": 0.0, "total": 0, "valid": 0}

    # 临时保存测试数据
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        temp_path = f.name

    try:
        target_cfg = {
            "model_name": "target",
            "model_path": "",
            "host": "127.0.0.1",
            "port": target_client.port,
            "gpu_id": "0",
            "timeout": 900,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
        }
        guard_cfg = {
            "model_name": "guard",
            "model_path": "",
            "host": "127.0.0.1",
            "port": guard_client.port,
            "gpu_id": "0",
            "timeout": 900,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
        }

        metrics = run_asr_test_serial(
            prompt_path=temp_path,
            target_client_config=target_cfg,
            guard_client_config=guard_cfg,
            output_path=None,
            batch_size=test_batch_size,
            max_workers=test_max_workers,
            target_max_tokens=target_max_tokens,
            target_temperature=target_temperature,
            guard_max_tokens=guard_max_tokens,
            guard_temperature=guard_temperature,
            show_progress=True,
            sleep_s_between_stage=0.0,
            save_raw_results=save_raw_results,  # 控制是否保存原始结果
        )

        # 如果需要保存原始生成内容
        if save_raw_results and raw_output_path and "results" in metrics:
            with open(raw_output_path, "w", encoding="utf-8") as f:
                json.dump(metrics["results"], f, ensure_ascii=False, indent=2)
            print(f"\n  原始生成内容已保存: {raw_output_path}")

        # 返回精简后的指标（不含完整results）
        summary = {k: v for k, v in metrics.items() if k != "results"}
        return summary
    finally:
        os.unlink(temp_path)


def filter_results(
    results: List[Tuple[str, float]],
    baseline_asr: float,
    topk: Optional[int] = None,
    gap_threshold: Optional[float] = None,
) -> List[Tuple[str, float]]:
    """筛选结果"""
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

    parser.add_argument("--strategies", nargs="*", default=None,
                        help="要评估的策略列表 (默认: 所有策略)")
    parser.add_argument("--test_set", type=str, default=None,
                        help="测试集路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--topk", type=int, default=None,
                        help="只输出表现最好的K个策略")
    parser.add_argument("--gap_threshold", type=float, default=None,
                        help="当策略间ASR差距大于此值时，舍弃后续策略")
    parser.add_argument("--sleep_between_evals", type=int, default=None,
                        help="两次评估之间的等待时间 (秒)")
    parser.add_argument("--policy_port", type=int, default=None,
                        help="Policy模型端口 (默认8003)")
    parser.add_argument("--target_port", type=int, default=None,
                        help="Target模型端口 (默认8001)")
    parser.add_argument("--guard_port", type=int, default=None,
                        help="Guard模型端口 (默认8002)")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印配置，不实际执行")

    args = parser.parse_args()

    # 合并配置
    config = dict(DEFAULT_CONFIG)
    if args.test_set:
        config["test_set"] = args.test_set
    if args.output_dir:
        config["output_root"] = args.output_dir
    if args.sleep_between_evals is not None:
        config["sleep_between_evals"] = args.sleep_between_evals
    if args.policy_port is not None:
        config["policy_port"] = args.policy_port
    if args.target_port is not None:
        config["target_port"] = args.target_port
    if args.guard_port is not None:
        config["guard_port"] = args.guard_port

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
    print(f"Policy端口: {config['policy_port']}")
    print(f"Target端口: {config['target_port']}")
    print(f"Guard端口: {config['guard_port']}")
    if args.topk:
        print(f"Top-K筛选: 只保留前 {args.topk} 个策略")
    if args.gap_threshold:
        print(f"Gap筛选: 差距阈值 = {args.gap_threshold}")
    print("="*80)

    if args.dry_run:
        print("\n[Dry Run] 配置已验证，未实际执行评估")
        print(f"将会创建 {len(strategies) + 1} 个评估任务")
        print("  - baseline_original (基线)")
        for s in strategies:
            print(f"  - {s}")
        return

    # 创建输出目录
    os.makedirs(config["output_root"], exist_ok=True)

    start_time = time.time()

    # =================================================================
    # Step 1: 连接已启动的模型服务 (由sh脚本启动)
    # =================================================================
    print("\n" + "="*80)
    print("Step 1/3: 连接模型服务 (由sh脚本已启动)")
    print("="*80)

    print(f"  连接Policy (GPU0:{config['policy_port']})...")
    policy_client = VLLMClient(
        model_name="policy",
        model_path=config["policy_model"],
        host="127.0.0.1",
        port=config["policy_port"],
        launch_server=False,
        timeout=900,
    )
    print("  Policy连接成功!")

    print(f"  连接Target (GPU1:{config['target_port']})...")
    target_client = VLLMClient(
        model_name="target",
        model_path=config["target_model"],
        host="127.0.0.1",
        port=config["target_port"],
        launch_server=False,
        timeout=900,
    )
    print("  Target连接成功!")

    print(f"  连接Guard (GPU1:{config['guard_port']})...")
    guard_client = VLLMClient(
        model_name="guard",
        model_path=config["guard_model"],
        host="127.0.0.1",
        port=config["guard_port"],
        launch_server=False,
        timeout=900,
    )
    print("  Guard连接成功!")

    # 加载测试集
    test_items = load_test_set(config["test_set"])
    test_originals = [it.get("prompt", "") for it in test_items]
    print(f"\n  加载测试集: {len(test_items)} 条数据")

    # 加载检查点
    ckpt = load_checkpoint(config["output_root"])
    completed_strategies = set(ckpt.get("completed", []))
    results = list(ckpt.get("results", []))
    baseline_asr = ckpt.get("baseline_asr")

    if completed_strategies:
        print(f"  检测到检查点: 已完成 {len(completed_strategies)} 个策略")
        print(f"  将跳过已完成的策略，继续未完成的部分")

    # =================================================================
    # Step 2: 评估基线 (原始prompt)
    # =================================================================
    print("\n" + "="*80)
    print("Step 2/3: 评估基线 - 原始prompt (无重写)")
    print("="*80)

    baseline_output_dir = os.path.join(config["output_root"], "baseline_original")
    os.makedirs(baseline_output_dir, exist_ok=True)

    if baseline_asr is None:
        baseline_rewritten = [[o] for o in test_originals]
        baseline_raw_path = os.path.join(baseline_output_dir, "raw_results.json")

        baseline_metrics = run_asr_test(
            target_client=target_client,
            guard_client=guard_client,
            originals=test_originals,
            rewritten_buckets=baseline_rewritten,
            test_batch_size=config.get("test_batch_size", 64),
            test_max_workers=config.get("test_max_workers", 16),
            target_max_tokens=config.get("target_max_tokens", 512),
            target_temperature=config.get("target_temperature", 0.0),
            guard_max_tokens=config.get("guard_max_tokens", 256),
            guard_temperature=config.get("guard_temperature", 0.0),
            save_raw_results=True,
            raw_output_path=baseline_raw_path,
        )

        baseline_asr = baseline_metrics.get("overall", {}).get("asr", 0.0)
        print(f"\n  基线ASR: {baseline_asr:.4f}")

        baseline_result = {
            "strategy": "baseline_original",
            "asr": baseline_asr,
            "metrics": baseline_metrics,
        }
        with open(os.path.join(baseline_output_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(baseline_result, f, ensure_ascii=False, indent=2)

        # 保存检查点
        ckpt["baseline_asr"] = baseline_asr
        save_checkpoint(config["output_root"], ckpt)
    else:
        print(f"  基线已完成 (ASR: {baseline_asr:.4f})，跳过")

    # =================================================================
    # Step 3: 依次评估各策略 (跳过已完成的)
    # =================================================================
    remaining = [s for s in strategies if s not in completed_strategies]
    print("\n" + "="*80)
    print(f"Step 3/3: 评估 {len(remaining)} 个策略 (跳过 {len(completed_strategies)} 个已完成)")
    print("="*80)

    sleep_time = config["sleep_between_evals"]

    for idx, strategy_name in enumerate(tqdm(remaining, desc="评估策略", unit="策略")):
        strategy_output_dir = os.path.join(config["output_root"], strategy_name)
        os.makedirs(strategy_output_dir, exist_ok=True)

        strategy_template = get_strategy_template(strategy_name)

        # 重写
        rewritten = run_rewrite(
            policy_client=policy_client,
            originals=test_originals,
            prompt_template=strategy_template,
            k=config.get("k", 1),
            temperature=config.get("rewrite_temperature", 0.7),
            max_tokens=config.get("rewrite_max_tokens", 2048),
        )

        # ASR测试
        raw_output_path = os.path.join(strategy_output_dir, "raw_results.json")
        metrics = run_asr_test(
            target_client=target_client,
            guard_client=guard_client,
            originals=test_originals,
            rewritten_buckets=rewritten,
            test_batch_size=config.get("test_batch_size", 64),
            test_max_workers=config.get("test_max_workers", 16),
            target_max_tokens=config.get("target_max_tokens", 512),
            target_temperature=config.get("target_temperature", 0.0),
            guard_max_tokens=config.get("guard_max_tokens", 256),
            guard_temperature=config.get("guard_temperature", 0.0),
            save_raw_results=True,
            raw_output_path=raw_output_path,
        )

        asr = metrics.get("overall", {}).get("asr", 0.0)
        results.append((strategy_name, asr))

        strategy_result = {
            "strategy": strategy_name,
            "asr": asr,
            "metrics": metrics,
        }
        with open(os.path.join(strategy_output_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(strategy_result, f, ensure_ascii=False, indent=2)

        # 计算混淆矩阵
        baseline_raw_path = os.path.join(config["output_root"], "baseline_original", "raw_results.json")
        confusion = compute_confusion_matrix(baseline_raw_path, raw_output_path)
        if confusion:
            print_confusion_matrix(confusion)
            # 保存混淆矩阵
            confusion_summary = {
                "fail_to_success": confusion["fail_to_success"],
                "success_to_success": confusion["success_to_success"],
                "fail_to_fail": confusion["fail_to_fail"],
                "success_to_fail": confusion["success_to_fail"],
                "total": confusion["total"],
                "lift": confusion["fail_to_success"] - confusion["success_to_fail"],
            }
            with open(os.path.join(strategy_output_dir, "confusion_matrix.json"), "w", encoding="utf-8") as f:
                json.dump(confusion_summary, f, ensure_ascii=False, indent=2)

        # 更新检查点
        completed_strategies.add(strategy_name)
        ckpt["completed"] = list(completed_strategies)
        ckpt["results"] = results
        save_checkpoint(config["output_root"], ckpt)

        if idx < len(remaining) - 1 and sleep_time > 0:
            time.sleep(sleep_time)

    # 记录结束时间
    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    # =================================================================
    # Step 4: 结果筛选与汇总 (不关闭服务，由sh脚本负责)
    # =================================================================
    print("\n" + "="*80)
    print("结果筛选与汇总")
    print("="*80)

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    filtered_results = filter_results(
        sorted_results,
        baseline_asr,
        topk=args.topk,
        gap_threshold=args.gap_threshold,
    )

    print_results_table(sorted_results, baseline_asr, filtered_results)

    # 收集所有策略的混淆矩阵
    confusion_all = {}
    for name, _ in sorted_results:
        cm_path = os.path.join(config["output_root"], name, "confusion_matrix.json")
        if os.path.exists(cm_path):
            with open(cm_path, "r", encoding="utf-8") as f:
                confusion_all[name] = json.load(f)

    summary = {
        "experiment": "jailbreak_prompt_exp_1",
        "timestamp": datetime.datetime.now().isoformat(),
        "test_set": config["test_set"],
        "baseline_asr": baseline_asr,
        "all_results": {name: asr for name, asr in sorted_results},
        "filtered_results": {name: asr for name, asr in filtered_results},
        "confusion_matrices": confusion_all,
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
    print("注意: 模型服务未关闭，请由调用方 (sh脚本) 负责关闭")


if __name__ == "__main__":
    main()
