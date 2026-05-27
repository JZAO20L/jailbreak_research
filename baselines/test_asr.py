"""
Baseline ASR 测试脚本

在 test 集上测试 PAIR 和 AutoDAN 的 ASR

支持两种模式：
1. --auto_launch: 自动启动 vLLM 服务后测试
2. 默认模式: 连接已启动的服务进行测试

配置：
- Guard: GPU 2, 端口 8002, Qwen3Guard-Gen-4B
- Target: GPU 1, 端口 8001, Qwen3-4B
- 轮次上限: 10
- Test 集: data/dataset/processed/10k/test.jsonl
"""

import os
import sys
import json
import time
import signal
import argparse
import subprocess
import httpx
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from datetime import datetime

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines import PAIRAttacker, AutoDANAttacker, get_attacker
from RL4jailbreak.src.vllm_client import VLLMClient


# =============================================================================
# 服务配置
# =============================================================================

SERVICE_CONFIG = {
    "guard": {
        "port": 8002,
        "gpu": 0,
        "model_path": "/home/tiger/models/Qwen/Qwen3Guard-Gen-4B",
        "model_name": "Qwen3Guard-Gen-4B",
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
    },
    "target": {
        "port": 8001,
        "gpu": "1,2",
        "model_path": "/home/tiger/models/Qwen/Qwen3-4B",
        "model_name": "Qwen3-4B",
        "max_model_len": 8192,
        "tensor_parallel_size": 2,
    },
}


# =============================================================================
# 服务启动/停止
# =============================================================================

class ServiceManager:
    """vLLM 服务管理器"""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}

    def start_service(self, name: str, config: Dict) -> bool:
        """
        启动单个 vLLM 服务

        Args:
            name: "guard" | "target"
            config: 服务配置

        Returns:
            是否成功启动
        """
        print(f"\n启动 {name} 服务...")
        print(f"  GPU: {config['gpu']}")
        print(f"  Port: {config['port']}")
        print(f"  Model: {config['model_path']}")

        # 构建命令
        env_vars = f"CUDA_VISIBLE_DEVICES={config['gpu']} VLLM_USE_MODELSCOPE=true FLASHINFER_DISABLE_VERSION_CHECK=1"
        tp_size = config.get('tensor_parallel_size', 1)

        cmd = f"""
{env_vars} vllm serve {config['model_path']} \
    --port {config['port']} \
    --max-model-len {config['max_model_len']} \
    --tensor-parallel-size {tp_size} \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0
"""

        # 启动进程
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # 创建新进程组，便于终止
            )
            self.processes[name] = process
            print(f"  ✓ 进程已启动 (PID: {process.pid})")

            # 等待服务就绪
            if self._wait_for_ready(name, config['port'], timeout=300):
                print(f"  ✓ {name} 服务就绪")
                return True
            else:
                print(f"  ✗ {name} 服务启动超时")
                return False

        except Exception as e:
            print(f"  ✗ 启动失败: {str(e)}")
            return False

    def _wait_for_ready(self, name: str, port: int, timeout: int = 300) -> bool:
        """等待服务就绪"""
        url = f"http://127.0.0.1:{port}/health"
        start_time = time.time()

        print(f"  等待 {name} 服务就绪 (timeout: {timeout}s)...")

        with httpx.Client(timeout=5.0) as client:
            while time.time() - start_time < timeout:
                try:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        return True
                except Exception:
                    pass
                time.sleep(2)

        return False

    def stop_service(self, name: str):
        """停止单个服务"""
        if name in self.processes:
            process = self.processes[name]
            print(f"\n停止 {name} 服务 (PID: {process.pid})...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
                print(f"  ✓ {name} 服务已停止")
            except Exception as e:
                print(f"  ⚠ 停止时出错: {str(e)}")
                try:
                    process.kill()
                except:
                    pass
            del self.processes[name]

    def stop_all(self):
        """停止所有服务"""
        for name in list(self.processes.keys()):
            self.stop_service(name)

    def check_running(self, port: int) -> bool:
        """检查服务是否已运行"""
        url = f"http://127.0.0.1:{port}/health"
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url)
                return resp.status_code == 200
        except:
            return False


# =============================================================================
# 数据加载
# =============================================================================

def load_test_prompts(test_path: str, limit: int = None) -> List[str]:
    """加载 test prompts"""
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


# =============================================================================
# 测试逻辑
# =============================================================================

def connect_clients(guard_port: int, target_port: int):
    """连接已启动的服务"""
    print(f"\n连接服务...")
    print(f"  Guard: 端口 {guard_port}")
    print(f"  Target: 端口 {target_port}")

    guard_client = VLLMClient(port=guard_port, launch_server=False)
    target_client = VLLMClient(port=target_port, launch_server=False)

    print(f"  ✓ Guard 已连接: {guard_client.model_name}")
    print(f"  ✓ Target 已连接: {target_client.model_name}")

    return guard_client, target_client


def test_single_method(
    method: str,
    target_client,
    guard_client,
    test_prompts: List[str],
    max_iterations: int = 10,
    verbose: bool = False,
) -> Dict:
    """测试单个 baseline 方法"""
    print(f"\n{'='*60}")
    print(f"Testing {method.upper()}")
    print(f"{'='*60}")
    print(f"Test prompts: {len(test_prompts)}")
    print(f"Max iterations: {max_iterations}")

    # 创建 attacker
    if method == "pair":
        attacker = PAIRAttacker(
            target_client=target_client,
            guard_client=guard_client,
            max_iterations=max_iterations,
            verbose=verbose,
        )
    elif method == "autodan":
        attacker = AutoDANAttacker(
            target_client=target_client,
            guard_client=guard_client,
            max_generations=max_iterations,
            population_size=6,
            verbose=verbose,
        )
    else:
        attacker = get_attacker(method)(
            target_client=target_client,
            guard_client=guard_client,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    # 统计
    stats = {
        "method": method,
        "total": len(test_prompts),
        "success": 0,
        "failure": 0,
        "total_iterations": 0,
        "total_time": 0,
        "results": [],
    }

    # 逐个测试
    for prompt in tqdm(test_prompts, desc=f"[{method.upper()}]"):
        start_time = time.time()

        try:
            result = attacker.attack(prompt, evaluate=True)

            elapsed = time.time() - start_time
            stats["total_time"] += elapsed
            stats["total_iterations"] += result.iterations

            if result.is_success:
                stats["success"] += 1
            else:
                stats["failure"] += 1

            stats["results"].append({
                "original_prompt": prompt[:100],
                "attack_prompt": result.attack_prompt[:100],
                "is_success": result.is_success,
                "iterations": result.iterations,
                "time": elapsed,
                "guard_label": result.guard_label,
            })

            if verbose:
                status = "✓ SUCCESS" if result.is_success else "✗ FAILURE"
                print(f"  {status} | Iterations: {result.iterations} | Time: {elapsed:.1f}s")

        except Exception as e:
            stats["failure"] += 1
            stats["results"].append({
                "original_prompt": prompt[:100],
                "is_success": False,
                "error": str(e),
            })
            if verbose:
                print(f"  ✗ ERROR: {str(e)[:50]}")

    # 计算指标
    stats["asr"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0
    stats["avg_iterations"] = stats["total_iterations"] / stats["total"] if stats["total"] > 0 else 0
    stats["avg_time"] = stats["total_time"] / stats["total"] if stats["total"] > 0 else 0

    # 打印结果
    print(f"\n{method.upper()} Results:")
    print(f"  ASR: {stats['asr']*100:.1f}% ({stats['success']}/{stats['total']})")
    print(f"  Avg iterations: {stats['avg_iterations']:.1f}")
    print(f"  Avg time: {stats['avg_time']:.1f}s")
    print(f"  Total time: {stats['total_time']:.1f}s")

    return stats


def run_all_baselines(
    methods: List[str],
    guard_port: int,
    target_port: int,
    test_path: str,
    test_limit: int,
    max_iterations: int,
    output_dir: str,
    verbose: bool = False,
    auto_launch: bool = False,
    auto_stop: bool = True,
):
    """运行所有 baseline 测试"""

    service_manager = ServiceManager()

    try:
        if auto_launch:
            # 自动启动服务
            print("\n" + "="*60)
            print("Auto Launch Mode")
            print("="*60)

            # 检查是否已有服务运行
            guard_running = service_manager.check_running(guard_port)
            target_running = service_manager.check_running(target_port)

            if guard_running:
                print(f"\nGuard 服务已在端口 {guard_port} 运行，跳过启动")
            else:
                if not service_manager.start_service("guard", SERVICE_CONFIG["guard"]):
                    print("✗ Guard 服务启动失败，退出")
                    return None

            if target_running:
                print(f"\nTarget 服务已在端口 {target_port} 运行，跳过启动")
            else:
                if not service_manager.start_service("target", SERVICE_CONFIG["target"]):
                    print("✗ Target 服务启动失败，退出")
                    service_manager.stop_all()
                    return None

        # 连接服务
        guard_client, target_client = connect_clients(guard_port, target_port)

        # 加载测试数据
        test_prompts = load_test_prompts(test_path, test_limit)
        print(f"\n加载测试数据: {len(test_prompts)} prompts from {test_path}")

        # 运行测试
        all_stats = {}

        for method in methods:
            stats = test_single_method(
                method=method,
                target_client=target_client,
                guard_client=guard_client,
                test_prompts=test_prompts,
                max_iterations=max_iterations,
                verbose=verbose,
            )
            all_stats[method] = stats

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"baseline_asr_{timestamp}.json")
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"All Baselines Summary")
        print(f"{'='*60}")
        print(f"{'Method':<12} {'ASR':>10} {'Avg Iter':>10} {'Avg Time':>10}")
        print(f"{'-'*42}")
        for method, stats in all_stats.items():
            print(f"{method:<12} {stats['asr']*100:>9.1f}% {stats['avg_iterations']:>10.1f} {stats['avg_time']:>10.1f}s")
        print(f"\n结果已保存: {output_file}")

        # 关闭客户端
        guard_client.close()
        target_client.close()

        return all_stats

    finally:
        if auto_launch and auto_stop:
            # 自动停止服务
            service_manager.stop_all()


def main():
    parser = argparse.ArgumentParser(description="Baseline ASR Testing")

    # 方法选择
    parser.add_argument("--methods", type=str, nargs='+', default=["pair", "autodan"],
                        choices=["pair", "autodan", "genetic", "deepinception"],
                        help="要测试的方法")

    # 启动模式
    parser.add_argument("--auto_launch", action="store_true",
                        help="自动启动 vLLM 服务后测试")
    parser.add_argument("--no_auto_stop", action="store_true",
                        help="测试结束后不自动停止服务（保留服务继续使用）")

    # 服务端口（用于非自动启动模式）
    parser.add_argument("--guard_port", type=int, default=8002, help="Guard 服务端口")
    parser.add_argument("--target_port", type=int, default=8001, help="Target 服务端口")

    # 数据配置
    parser.add_argument("--test_path", type=str,
                        default="data/dataset/processed/10k/test.jsonl",
                        help="Test 数据路径")
    parser.add_argument("--test_limit", type=int, default=None, help="测试数量限制")

    # 迭代配置
    parser.add_argument("--max_iterations", type=int, default=10, help="最大迭代次数")

    # 输出
    parser.add_argument("--output_dir", type=str, default="experiments/baseline_asr")
    parser.add_argument("--verbose", action="store_true", help="打印详细信息")

    args = parser.parse_args()

    run_all_baselines(
        methods=args.methods,
        guard_port=args.guard_port,
        target_port=args.target_port,
        test_path=args.test_path,
        test_limit=args.test_limit,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        verbose=args.verbose,
        auto_launch=args.auto_launch,
        auto_stop=not args.no_auto_stop,
    )


if __name__ == "__main__":
    main()