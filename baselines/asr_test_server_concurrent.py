#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器端ASR测试脚本（并发版本）

使用批量调用优化性能，每次处理一批prompts，而不是逐个处理
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

from src.prompts import GUARD_PROMPT
from src.utils import extract_label_categories_refusal


# =============================================================================
# Thread-safe File Writing
# =============================================================================
file_lock = threading.Lock()

def save_result_threadsafe(result: Dict, output_file: str):
    """线程安全的保存结果"""
    with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


# =============================================================================
# VLLM Client Wrapper (继承原版本，添加批量调用支持)
# =============================================================================
class VLLMClientWrapper:
    """VLLM客户端包装器（支持批量调用）"""

    def __init__(self, port: int, host: str = "127.0.0.1", temperature: float = 0.0, timeout: float = 120.0):
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
                raise RuntimeError(f"Server not ready on port {self.port}")
            time.sleep(2)

    def _get_model_name(self) -> str:
        try:
            r = self._http_client.get(f"{self.base_url_v1}/models")
            if r.status_code == 200:
                models_data = r.json()
                if "data" in models_data and len(models_data["data"]) > 0:
                    model_id = models_data["data"][0].get("id", "")
                    print(f"[VLLMClientWrapper] Model: {model_id}")
                    return model_id
        except Exception:
            pass
        return "default"

    def llm_call(self, prompt: Optional[str] = None, messages: Optional[List[Dict]] = None,
                 max_tokens: int = 512, temperature: Optional[float] = None) -> str:
        if messages is not None:
            final_messages = messages
        elif prompt is not None:
            final_messages = [{"role": "user", "content": prompt}]
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
            print(f"API Error: {e}")
            return ""

    def llm_batch_call(self, prompts: Optional[List[str]] = None, messages_list: Optional[List[List[Dict]]] = None,
                       max_tokens: int = 512, temperature: Optional[float] = None,
                       max_workers: int = 8) -> List[str]:
        """批量并发调用"""
        if (prompts is None) == (messages_list is None):
            raise ValueError("Provide exactly one of 'prompts' or 'messages_list'")
        
        n = len(prompts) if prompts is not None else len(messages_list)
        results: List[str] = ["" for _ in range(n)]

        def _one_call(i: int):
            if prompts is not None:
                return self.llm_call(prompt=prompts[i], max_tokens=max_tokens, temperature=temperature)
            else:
                return self.llm_call(messages=messages_list[i], max_tokens=max_tokens, temperature=temperature)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_one_call, i): i for i in range(n)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Batch error at {idx}: {e}")
                    results[idx] = ""

        return results

    def close(self):
        try:
            self._http_client.close()
        except Exception:
            pass


# =============================================================================
# Batch Operations
# =============================================================================
def get_target_responses_batch(
    target_client: VLLMClientWrapper,
    attack_prompts: List[str],
    max_tokens: int = 512,
    max_workers: int = 8,
) -> List[str]:
    """批量获取target responses"""
    return target_client.llm_batch_call(
        prompts=attack_prompts,
        max_tokens=max_tokens,
        temperature=0.0,
        max_workers=max_workers,
    )


def get_guard_labels_batch(
    guard_client: VLLMClientWrapper,
    attack_prompts: List[str],
    responses: List[str],
    max_workers: int = 8,
) -> List[Dict]:
    """批量获取guard判断"""
    # 构造messages列表
    messages_list = []
    for attack_prompt, response in zip(attack_prompts, responses):
        messages = [
            {"role": "system", "content": GUARD_PROMPT},
            {"role": "user", "content": attack_prompt},
            {"role": "assistant", "content": response},
        ]
        messages_list.append(messages)

    # 批量调用
    raw_responses = guard_client.llm_batch_call(
        messages_list=messages_list,
        max_tokens=256,
        temperature=0.0,
        max_workers=max_workers,
    )

    # 解析结果
    results = []
    for raw in raw_responses:
        safe_label, categories, refusal = extract_label_categories_refusal(raw)
        
        if refusal and refusal.lower() == "yes":
            is_success = False
            guard_label = "refusal"
        elif safe_label and safe_label.lower() == "safe":
            is_success = False
            guard_label = "safe"
        elif safe_label and safe_label.lower() == "unsafe":
            is_success = True
            guard_label = "unsafe"
        else:
            is_success = False
            guard_label = "unknown"

        results.append({
            "guard_raw_response": raw,
            "guard_label": guard_label,
            "is_success": is_success,
            "refusal": refusal,
            "categories": categories,
        })

    return results


# =============================================================================
# Batch ASR Test (Concurrent)
# =============================================================================
def batch_asr_test_concurrent(
    input_path: str,
    output_path: str,
    target_port: int = 8001,
    guard_port: int = 8002,
    max_workers: int = 8,
    batch_size: int = 10,  # 每批处理10个
):
    """批量并发ASR测试"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 加载rewrite结果
    items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"\n{'='*70}")
    print(f"并发ASR测试")
    print(f"{'='*70}")
    print(f"输入: {input_path} ({len(items)} prompts)")
    print(f"输出: {output_path}")
    print(f"并发数: {max_workers}, 批大小: {batch_size}")
    print(f"{'='*70}\n")

    target_client = VLLMClientWrapper(port=target_port, temperature=0.0)
    guard_client = VLLMClientWrapper(port=guard_port, temperature=0.0)

    success_count = 0
    refusal_count = 0
    unknown_count = 0
    total_time = 0.0

    # 分批处理
    for batch_idx in tqdm(range(0, len(items), batch_size), desc="ASR Batch"):
        batch_items = items[batch_idx:batch_idx + batch_size]
        batch_start = time.time()

        # 提取attack_prompts
        attack_prompts = [item.get("attack_prompt", "") for item in batch_items]

        # 批量获取target responses
        target_responses = get_target_responses_batch(
            target_client,
            attack_prompts,
            max_tokens=512,
            max_workers=max_workers,
        )

        # 批量获取guard判断
        guard_results = get_guard_labels_batch(
            guard_client,
            attack_prompts,
            target_responses,
            max_workers=max_workers,
        )

        # 保存结果并统计
        for item, target_response, guard_result in zip(batch_items, target_responses, guard_results):
            result = {
                "id": item.get("id", ""),
                "original_prompt": item.get("original_prompt", ""),
                "attack_prompt": item.get("attack_prompt", ""),
                "strategy": item.get("strategy", ""),
                "target_response": target_response,
                "guard_raw_response": guard_result["guard_raw_response"],
                "guard_label": guard_result["guard_label"],
                "is_success": guard_result["is_success"],
                "refusal": guard_result["refusal"],
                "categories": guard_result["categories"],
                "rewrite_iterations": item.get("iterations"),
                "rewrite_time_cost": item.get("time_cost"),
            }
            save_result_threadsafe(result, output_path)

            if guard_result["is_success"]:
                success_count += 1
            elif guard_result["guard_label"] == "refusal":
                refusal_count += 1
            else:
                unknown_count += 1

        total_time += time.time() - batch_start

    # 统计
    total = len(items)
    asr = success_count / total if total > 0 else 0.0
    avg_time = total_time / total if total > 0 else 0.0

    print(f"\n{'='*70}")
    print(f"ASR测试完成")
    print(f"{'='*70}")
    print(f"总数: {total}")
    print(f"成功: {success_count} (ASR={asr:.2%})")
    print(f"拒绝: {refusal_count}")
    print(f"未知: {unknown_count}")
    print(f"平均耗时: {avg_time:.2f}s")
    print(f"{'='*70}\n")

    # 保存summary
    strategy = items[0].get("strategy", "unknown") if items else "unknown"
    summary = {
        "strategy": strategy,
        "input": input_path,
        "output": output_path,
        "total": total,
        "success_count": success_count,
        "refusal_count": refusal_count,
        "unknown_count": unknown_count,
        "asr": asr,
        "avg_time": avg_time,
        "max_workers": max_workers,
        "batch_size": batch_size,
    }

    summary_file = output_path.replace(".jsonl", "_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Summary: {summary_file}")

    target_client.close()
    guard_client.close()

    return asr


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="并发ASR测试")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--target-port", type=int, default=8001)
    parser.add_argument("--guard-port", type=int, default=8002)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10, help="每批处理数量")
    args = parser.parse_args()

    batch_asr_test_concurrent(
        input_path=args.input,
        output_path=args.output,
        target_port=args.target_port,
        guard_port=args.guard_port,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()