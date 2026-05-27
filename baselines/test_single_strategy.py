"""
Test Single Strategy Script (Concurrent Version)

用于 run_baseline_experiments.sh 调用，测试单个 baseline 策略并保存结果。

支持并发执行多个攻击流程：
- 单个攻击流程内部保持串行（如 PAIR 的 10 次迭代）
- 多个攻击流程之间并发执行

Usage:
    python baselines/test_single_strategy.py \
        --strategy pair \
        --test_path data/dataset/processed/10k/test.jsonl \
        --target_port 8001 \
        --guard_port 8002 \
        --max_iterations 10 \
        --workers 8 \
        --output_dir experiments/baseline_asr
"""

import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines import get_attacker, STRATEGIES
from RL4jailbreak.src.vllm_client import VLLMClient


def load_test_prompts(test_path: str, limit: int = None) -> List[str]:
    """加载测试 prompts"""
    prompts = []

    if test_path.endswith('.jsonl'):
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                prompt = data.get('prompt', data.get('question', data.get('text', '')))
                if prompt:
                    prompts.append(prompt)
    else:
        with open(test_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict):
                    prompts.append(item.get('prompt', item.get('question', '')))

    if limit:
        prompts = prompts[:limit]

    return prompts


class ConcurrentTestRunner:
    """
    并发测试执行器

    - 多个攻击流程并发执行
    - 单个流程内部保持串行
    - 使用共享的 client 实例
    """

    def __init__(
        self,
        strategy: str,
        target_client: VLLMClient,
        guard_client: VLLMClient,
        max_iterations: int,
        verbose: bool = False,
    ):
        self.strategy = strategy
        self.target_client = target_client
        self.guard_client = guard_client
        self.max_iterations = max_iterations
        self.verbose = verbose

        # 创建共享的 attacker（client 是共享的）
        self.attacker = get_attacker(
            strategy,
            target_client=target_client,
            guard_client=guard_client,
            max_iterations=max_iterations,
            verbose=False,  # 并发时关闭 verbose，避免输出混乱
        )

        # 线程安全的统计
        self.lock = threading.Lock()
        self.stats = {
            "strategy": strategy,
            "total": 0,
            "success": 0,
            "failure": 0,
            "total_iterations": 0,
            "total_time": 0.0,
            "results": [],
        }

    def attack_single(self, prompt: str, index: int) -> Dict:
        """
        执行单个攻击流程（内部串行）

        Args:
            prompt: 原始 prompt
            index: prompt 索引（用于结果排序）

        Returns:
            单个攻击结果
        """
        start_time = time.time()

        try:
            # 执行攻击（内部串行）
            result = self.attacker.attack(prompt, evaluate=True)

            elapsed = time.time() - start_time

            single_result = {
                "index": index,
                "original_prompt": prompt[:200],
                "attack_prompt": result.attack_prompt[:200],
                "is_success": result.is_success,
                "iterations": result.iterations,
                "time": elapsed,
                "guard_label": result.guard_label,
            }

            # 更新统计（线程安全）
            with self.lock:
                self.stats["total"] += 1
                self.stats["total_time"] += elapsed
                self.stats["total_iterations"] += result.iterations

                if result.is_success:
                    self.stats["success"] += 1
                else:
                    self.stats["failure"] += 1

                self.stats["results"].append(single_result)

            return single_result

        except Exception as e:
            elapsed = time.time() - start_time

            single_result = {
                "index": index,
                "original_prompt": prompt[:200],
                "is_success": False,
                "error": str(e),
                "time": elapsed,
            }

            with self.lock:
                self.stats["total"] += 1
                self.stats["failure"] += 1
                self.stats["total_time"] += elapsed
                self.stats["results"].append(single_result)

            return single_result

    def run_concurrent(
        self,
        test_prompts: List[str],
        max_workers: int = 8,
    ) -> Dict:
        """
        并发执行所有攻击流程

        Args:
            test_prompts: 测试 prompt 列表
            max_workers: 并发数

        Returns:
            统计结果
        """
        n = len(test_prompts)
        self.stats["total"] = n  # 预设总数

        print(f"\n并发配置:")
        print(f"  Workers: {max_workers}")
        print(f"  Prompts: {n}")
        print(f"  单流程内部: 串行执行")
        print(f"  多流程之间: 并发执行")

        # 使用 ThreadPoolExecutor 并发执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self.attack_single, prompt, i): i
                for i, prompt in enumerate(test_prompts)
            }

            # 使用 tqdm 显示进度
            with tqdm(total=n, desc=f"[{self.strategy.upper()}]") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        pbar.update(1)

                        if self.verbose:
                            status = "SUCCESS" if result.get("is_success") else "FAILURE"
                            iter_str = f"Iter: {result.get('iterations', 'N/A')}"
                            time_str = f"Time: {result.get('time', 0):.1f}s"
                            pbar.write(f"  [{result['index']}] {status} | {iter_str} | {time_str}")

                    except Exception as e:
                        pbar.update(1)
                        if self.verbose:
                            pbar.write(f"  ERROR: {str(e)[:50]}")

        # 计算指标
        self.stats["asr"] = self.stats["success"] / self.stats["total"] if self.stats["total"] > 0 else 0
        self.stats["avg_iterations"] = self.stats["total_iterations"] / self.stats["total"] if self.stats["total"] > 0 else 0
        self.stats["avg_time"] = self.stats["total_time"] / self.stats["total"] if self.stats["total"] > 0 else 0

        # 按索引排序结果
        self.stats["results"] = sorted(self.stats["results"], key=lambda x: x.get("index", 0))

        return self.stats


def test_strategy(
    strategy: str,
    target_port: int,
    guard_port: int,
    test_prompts: List[str],
    max_iterations: int,
    max_workers: int = 8,
    verbose: bool = False,
) -> Dict:
    """
    测试单个策略（并发版本）

    Returns:
        统计结果字典
    """
    print(f"\n{'='*60}")
    print(f"Testing Strategy: {strategy.upper()}")
    print(f"{'='*60}")
    print(f"Test prompts: {len(test_prompts)}")
    print(f"Max iterations: {max_iterations}")

    # 连接服务
    print(f"\n连接服务...")
    print(f"  Guard: 端口 {guard_port}")
    print(f"  Target: 端口 {target_port}")

    guard_client = VLLMClient(port=guard_port, launch_server=False)
    target_client = VLLMClient(port=target_port, launch_server=False)

    print(f"  ✓ Guard 已连接: {guard_client.model_name}")
    print(f"  ✓ Target 已连接: {target_client.model_name}")

    # 创建并发执行器
    runner = ConcurrentTestRunner(
        strategy=strategy,
        target_client=target_client,
        guard_client=guard_client,
        max_iterations=max_iterations,
        verbose=verbose,
    )

    # 并发执行
    stats = runner.run_concurrent(test_prompts, max_workers=max_workers)

    # 关闭客户端
    guard_client.close()
    target_client.close()

    # 打印结果
    print(f"\n{strategy.upper()} Results:")
    print(f"  ASR: {stats['asr']*100:.1f}% ({stats['success']}/{stats['total']})")
    print(f"  Avg iterations: {stats['avg_iterations']:.1f}")
    print(f"  Avg time: {stats['avg_time']:.1f}s")
    print(f"  Total time: {stats['total_time']:.1f}s")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Test Single Baseline Strategy (Concurrent)")

    parser.add_argument("--strategy", type=str, required=True,
                        choices=list(STRATEGIES.keys()),
                        help="要测试的策略")
    parser.add_argument("--test_path", type=str, required=True,
                        help="测试数据路径")
    parser.add_argument("--target_port", type=int, default=8001,
                        help="Target 服务端口")
    parser.add_argument("--guard_port", type=int, default=8002,
                        help="Guard 服务端口")
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="最大迭代次数")
    parser.add_argument("--test_limit", type=int, default=None,
                        help="测试数量限制")
    parser.add_argument("--workers", type=int, default=8,
                        help="并发数（同时执行的攻击流程数）")
    parser.add_argument("--output_dir", type=str, default="experiments/baseline_asr",
                        help="输出目录")
    parser.add_argument("--verbose", action="store_true",
                        help="打印详细信息")

    args = parser.parse_args()

    # 加载测试数据
    test_prompts = load_test_prompts(args.test_path, args.test_limit)
    print(f"加载测试数据: {len(test_prompts)} prompts from {args.test_path}")

    # 运行测试
    stats = test_strategy(
        strategy=args.strategy,
        target_port=args.target_port,
        guard_port=args.guard_port,
        test_prompts=test_prompts,
        max_iterations=args.max_iterations,
        max_workers=args.workers,
        verbose=args.verbose,
    )

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"{args.strategy}_asr_{timestamp}.json")
    os.makedirs(args.output_dir, exist_ok=True)

    # 添加并发配置到结果
    stats["config"] = {
        "max_iterations": args.max_iterations,
        "max_workers": args.workers,
        "test_path": args.test_path,
        "test_limit": args.test_limit,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()