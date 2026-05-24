#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器端批量重写脚本（并发版本）

并发处理多个prompts，显著提升性能
"""

import os
import sys
import json
import argparse
import time
import threading
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root and RL4jailbreak to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_DIR = os.path.join(BASE_DIR, "RL4jailbreak")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, RL_DIR)

from src.vllm_client import VLLMClient
from baselines import get_attacker


# =============================================================================
# Thread-safe File Writing
# =============================================================================
file_lock = threading.Lock()

def save_result_threadsafe(result: Dict, output_file: str):
    """线程安全的保存结果到JSONL文件"""
    with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


# =============================================================================
# VLLM Client Wrapper
# =============================================================================
class VLLMClientWrapper:
    def __init__(self, port: int, host: str = "127.0.0.1", temperature: float = 0.7, timeout: float = 120.0):
        self.port = port
        self.host = host
        self.base_url = f"http://{host}:{port}"
        self.base_url_v1 = f"{self.base_url}/v1"
        self.temperature = temperature
        self.timeout = timeout
        self._http_client = httpx.Client(timeout=self.timeout)
        self._wait_for_server()
        self.model_name = self._get_model_name()
        from openai import OpenAI
        self.openai_client = OpenAI(api_key="EMPTY", base_url=self.base_url_v1, http_client=self._http_client)

    def _wait_for_server(self, timeout_s: float = 60.0):
        print(f"[VLLMClientWrapper] Waiting for server on port {self.port}...")
        start_time = time.time()
        while True:
            try:
                r = self._http_client.get(f"{self.base_url}/health")
                if r.status_code == 200:
                    print(f"[VLLMClientWrapper] Server ready on port {self.port}!")
                    return
            except Exception:
                pass
            if time.time() - start_time > timeout_s:
                raise RuntimeError(f"Server not ready on port {self.port} within {timeout_s}s")
            time.sleep(2)

    def _get_model_name(self) -> str:
        try:
            r = self._http_client.get(f"{self.base_url_v1}/models")
            if r.status_code == 200:
                models_data = r.json()
                if "data" in models_data and len(models_data["data"]) > 0:
                    return models_data["data"][0].get("id", "")
        except Exception:
            pass
        return "default"

    def llm_call(self, prompt: Optional[str] = None, messages: Optional[List[Dict]] = None,
                 system_prompt: Optional[str] = None, max_tokens: int = 1024,
                 temperature: Optional[float] = None) -> str:
        if messages is not None:
            final_messages = messages
        elif prompt is not None:
            final_messages = []
            if system_prompt:
                final_messages.append({"role": "system", "content": system_prompt})
            final_messages.append({"role": "user", "content": prompt})
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        try:
            resp = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=final_messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"API调用失败: {e}")
            return ""

    def close(self):
        try:
            self._http_client.close()
        except Exception:
            pass


# =============================================================================
# Load Functions
# =============================================================================
def load_prompts(input_path: str) -> List[Dict[str, Any]]:
    prompts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                prompts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return prompts


# =============================================================================
# Concurrent Processing
# =============================================================================
def process_single_prompt(
    prompt_data: Dict,
    attacker,
    output_file: str,
    strategy: str,
) -> Dict:
    """处理单个prompt"""
    original_prompt = prompt_data.get("prompt", "")
    try:
        result_obj = attacker.attack(original_prompt, evaluate=False)
        result = {
            "id": prompt_data.get("id", ""),
            "original_prompt": original_prompt,
            "attack_prompt": result_obj.attack_prompt,
            "strategy": strategy,
            "iterations": result_obj.iterations,
            "time_cost": result_obj.time_cost,
        }
        save_result_threadsafe(result, output_file)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e), "prompt_id": prompt_data.get("id", "")}


# =============================================================================
# Batch Rewrite (Concurrent)
# =============================================================================
def batch_rewrite_concurrent(
    input_path: str,
    output_dir: str,
    strategies: List[str],
    limit: Optional[int] = None,
    rewrite_port: int = 8001,
    guard_port: int = 8002,
    max_iterations: int = 5,
    max_workers: int = 8,
):
    os.makedirs(output_dir, exist_ok=True)
    prompts = load_prompts(input_path)
    if limit:
        prompts = prompts[:limit]

    print(f"\n{'='*70}")
    print(f"并发批量重写脚本")
    print(f"{'='*70}")
    print(f"输入: {input_path} ({len(prompts)} prompts)")
    print(f"输出: {output_dir}")
    print(f"策略: {strategies}")
    print(f"并发数: {max_workers}")
    print(f"{'='*70}\n")

    rewrite_client = VLLMClientWrapper(port=rewrite_port)
    guard_client = VLLMClientWrapper(port=guard_port)
    target_client = rewrite_client

    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"策略: {strategy}")
        print(f"{'='*70}")
        output_file = os.path.join(output_dir, f"{strategy}.jsonl")

        # 创建attacker实例池（每个线程一个，避免状态冲突）
        if strategy == "deepinception":
            attacker_pool = [get_attacker(strategy)]
        elif strategy == "multilingual":
            attacker_pool = [get_attacker(strategy, translate_client=rewrite_client, verbose=False) 
                            for _ in range(max_workers)]
        elif strategy == "genetic":
            attacker_pool = [get_attacker(strategy, target_client=target_client, guard_client=guard_client,
                                          max_iterations=max_iterations, max_generations=max_iterations,
                                          population_size=5, verbose=False) for _ in range(max_workers)]
        elif strategy == "pair":
            attacker_pool = [get_attacker(strategy, target_client=target_client, guard_client=guard_client,
                                          max_iterations=max_iterations, verbose=False) for _ in range(max_workers)]
        else:
            print(f"不支持: {strategy}")
            continue

        # 并发处理
        success_count = 0
        total_time = 0.0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, prompt_data in enumerate(prompts):
                attacker = attacker_pool[i % len(attacker_pool)]
                future = executor.submit(process_single_prompt, prompt_data, attacker, output_file, strategy)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"[{strategy}]"):
                result = future.result()
                if result["success"]:
                    success_count += 1
                    total_time += result["result"].get("time_cost", 0)

        avg_time = total_time / success_count if success_count > 0 else 0
        print(f"\n完成: {success_count}/{len(prompts)}, 平均耗时: {avg_time:.2f}s")

        summary = {"strategy": strategy, "total": len(prompts), "success": success_count, 
                   "avg_time": avg_time, "input": input_path, "output": output_file}
        with open(os.path.join(output_dir, f"{strategy}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    rewrite_client.close()
    guard_client.close()
    print(f"\n{'='*70}\n完成!\n{'='*70}\n")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="并发批量重写脚本")
    parser.add_argument("--input", "-i", default="data/dataset/processed/10k/test.jsonl")
    parser.add_argument("--output", "-o", default="baselines/output")
    parser.add_argument("--strategies", "-s", nargs="+", default=["deepinception", "multilingual", "pair", "genetic"])
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--rewrite-port", type=int, default=8001)
    parser.add_argument("--guard-port", type=int, default=8002)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=8, help="并发线程数")
    args = parser.parse_args()

    batch_rewrite_concurrent(
        input_path=args.input,
        output_dir=args.output,
        strategies=args.strategies,
        limit=args.limit,
        rewrite_port=args.rewrite_port,
        guard_port=args.guard_port,
        max_iterations=args.max_iterations,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()