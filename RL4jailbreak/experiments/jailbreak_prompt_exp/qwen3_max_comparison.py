#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen3-max 对比实验脚本 - 实验1补充

根据 TODO.md "实验重做" 部分：
- 使用百炼 codingplan API 调用 qwen3-max
- 对 top3 策略进行重写实验
- 与本地 qwen3-4B 结果对比
- 分析旗舰LLM在jailbreak任务上的表现

目的：说明旗舰LLM在jailbreak任务上并无优势，分析其护栏问题或能力失配问题

API配置 (根据TODO.md):
- url: https://coding.dashscope.aliyuncs.com/v1
- api_key: sk-sp-eb50d67ca64a451b820cc4ab87ef8e6c
- model: qwen3-max-2026-01-23

使用方法:
    python qwen3_max_comparison.py \
        --test_set test.jsonl \
        --output_dir output/qwen3_max_comparison \
        --strategies hypothetical_scenario creative_writing role_playing \
        --api_key "sk-xxx" \
        --base_url "https://coding.dashscope.aliyuncs.com/v1" \
        --model "qwen3-max-2026-01-23" \
        --target_port 8001 \
        --guard_port 8002
"""

import os
import sys
import json
import argparse
import datetime
import time
import re
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.vllm_client import VLLMClient
from src.test import run_asr_test_serial
from experiments.jailbreak_prompt_exp.jailbreak_prompts import (
    JAILBREAK_PROMPTS,
    get_strategy_template,
)


class Qwen3MaxClient:
    """百炼 codingplan API 客户端"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        model: str = "qwen3-max-2026-01-23",
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
    
    def rewrite(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        使用 qwen3-max 重写 prompt
        
        根据 TODO.md 提供的示例代码：
        response = client.responses.create(
            model="qwen3-max-2026-01-23",
            input="你能做些什么？"
        )
        print(response.output_text)
        """
        try:
            # 使用 responses.create (百炼 codingplan API)
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )
            
            # 获取模型回复
            output = response.output_text
            
            # 处理 </think> 标签 (如果有)
            if "</think>" in output:
                output = output.split("</think>")[-1].strip()
            
            return output
            
        except Exception as e:
            print(f"[ERROR] qwen3-max API call failed: {e}")
            return ""
    
    def batch_rewrite(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_workers: int = 8,  # qwen3-max API并发数
        show_progress: bool = True,
    ) -> List[str]:
        """批量重写"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = [""] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(self.rewrite, p, temperature, max_tokens): i 
                          for i, p in enumerate(prompts)}
            
            iterator = as_completed(future_map)
            if show_progress:
                iterator = tqdm(iterator, total=len(prompts), desc="qwen3-max rewrite")
            
            for fut in iterator:
                i = future_map[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    print(f"[ERROR] rewrite {i}: {e}")
                    results[i] = ""
        
        return results


def load_test_set(path: str) -> List[Dict]:
    """加载测试集"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def run_asr_test(
    target_client: VLLMClient,
    guard_client: VLLMClient,
    originals: List[str],
    rewritten_buckets: List[List[str]],
    target_max_model_len: int = 8192,
    guard_max_model_len: int = 8192,
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
        return {"asr": 0.0, "total": 0}

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
            "gpu_id": "1",
            "timeout": 900,
            "gpu_memory_utilization": 0.4,
            "max_model_len": target_max_model_len,
        }
        guard_cfg = {
            "model_name": "guard",
            "model_path": "",
            "host": "127.0.0.1",
            "port": guard_client.port,
            "gpu_id": "2",
            "timeout": 900,
            "gpu_memory_utilization": 0.4,
            "max_model_len": guard_max_model_len,
        }

        metrics = run_asr_test_serial(
            prompt_path=temp_path,
            target_client_config=target_cfg,
            guard_client_config=guard_cfg,
            output_path=None,
            batch_size=64,
            max_workers=8,  # qwen3-max测试时降低并发数
            target_max_tokens=512,
            target_temperature=0.0,
            guard_max_tokens=256,
            guard_temperature=0.0,
            show_progress=True,
            sleep_s_between_stage=0.0,
            save_raw_results=save_raw_results,
        )

        if save_raw_results and raw_output_path and "results" in metrics:
            with open(raw_output_path, "w", encoding="utf-8") as f:
                json.dump(metrics["results"], f, ensure_ascii=False, indent=2)

        summary = {k: v for k, v in metrics.items() if k != "results"}
        return summary
    finally:
        os.unlink(temp_path)


def compare_results(
    local_results: Dict[str, float],
    qwen3max_results: Dict[str, float],
) -> Dict:
    """对比本地模型和 qwen3-max 的结果"""
    comparison = {
        "strategies": [],
        "analysis": {},
    }
    
    for strategy in local_results:
        local_asr = local_results[strategy]
        qwen3max_asr = qwen3max_results.get(strategy, 0.0)
        diff = local_asr - qwen3max_asr
        
        comparison["strategies"].append({
            "strategy": strategy,
            "local_asr": local_asr,
            "qwen3max_asr": qwen3max_asr,
            "diff": diff,
            "better": "local" if diff > 0 else "qwen3max" if diff < 0 else "equal",
        })
    
    # 分析
    avg_local = sum(local_results.values()) / len(local_results) if local_results else 0
    avg_qwen3max = sum(qwen3max_results.values()) / len(qwen3max_results) if qwen3max_results else 0
    
    comparison["analysis"] = {
        "avg_local_asr": avg_local,
        "avg_qwen3max_asr": avg_qwen3max,
        "avg_diff": avg_local - avg_qwen3max,
        "conclusion": "",
    }
    
    # 结论分析
    if avg_local > avg_qwen3max:
        comparison["analysis"]["conclusion"] = (
            "本地 qwen3-4B 平均ASR高于 qwen3-max，说明在jailbreak任务上 "
            "小规模专用模型可能更有优势。原因可能是：(1) qwen3-max有更强的护栏；"
            "(2) 旗舰模型能力失配，不适合这类特殊任务。"
        )
    else:
        comparison["analysis"]["conclusion"] = (
            "qwen3-max 平均ASR高于本地模型，说明旗舰LLM在jailbreak任务上仍有一定优势。"
        )
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="qwen3-max 对比实验")
    
    parser.add_argument("--test_set", type=str, required=True,
                        help="测试集路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--strategies", nargs="+", required=True,
                        help="要测试的策略列表 (top3)")
    parser.add_argument("--api_key", type=str, required=True,
                        help="百炼 API key")
    parser.add_argument("--base_url", type=str, 
                        default="https://coding.dashscope.aliyuncs.com/v1",
                        help="百炼 API base URL")
    parser.add_argument("--model", type=str,
                        default="qwen3-max-2026-01-23",
                        help="模型名称")
    parser.add_argument("--target_port", type=int, default=8001,
                        help="Target模型端口")
    parser.add_argument("--guard_port", type=int, default=8002,
                        help="Guard模型端口")
    parser.add_argument("--target_max_model_len", type=int, default=8192,
                        help="Target模型上下文长度")
    parser.add_argument("--guard_max_model_len", type=int, default=8192,
                        help="Guard模型上下文长度")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("qwen3-max 对比实验")
    print("="*80)
    print(f"测试集: {args.test_set}")
    print(f"策略: {args.strategies}")
    print(f"模型: {args.model}")
    print(f"API: {args.base_url}")
    print(f"输出目录: {args.output_dir}")
    print("="*80)
    
    # 加载测试集
    test_items = load_test_set(args.test_set)
    test_originals = [it.get("prompt", "") for it in test_items]
    print(f"加载测试集: {len(test_items)} 条数据")
    
    # 连接 Target 和 Guard (由 exp.sh 启动)
    print("\n连接 Target + Guard...")
    target_client = VLLMClient(
        model_name="target",
        model_path="/home/tiger/models/Qwen3-4B",
        host="127.0.0.1",
        port=args.target_port,
        launch_server=False,
        timeout=900,
    )
    guard_client = VLLMClient(
        model_name="guard",
        model_path="/home/tiger/models/Qwen3Guard-Gen-4B",
        host="127.0.0.1",
        port=args.guard_port,
        launch_server=False,
        timeout=900,
    )
    print("连接成功!")
    
    # 初始化 qwen3-max 客户端
    qwen3max_client = Qwen3MaxClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    
    # 加载本地模型结果 (从 experiment_summary.json)
    summary_path = os.path.dirname(args.output_dir) + "/experiment_summary.json"
    local_results = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
            local_results = summary.get("all_results", {})
        print(f"\n加载本地结果: {len(local_results)} 个策略")
    
    qwen3max_results = {}
    
    # 对每个策略进行测试
    for strategy_name in args.strategies:
        print("\n" + "="*60)
        print(f"策略: {strategy_name}")
        print("="*60)
        
        strategy_output_dir = os.path.join(args.output_dir, strategy_name)
        os.makedirs(strategy_output_dir, exist_ok=True)
        
        # 获取策略模板
        strategy_template = get_strategy_template(strategy_name)
        
        # 构建重写prompts
        rewrite_prompts = [strategy_template.format(original_prompt=o) for o in test_originals]
        
        print(f"使用 qwen3-max 重写...")
        rewritten = qwen3max_client.batch_rewrite(
            prompts=rewrite_prompts,
            temperature=0.7,
            max_tokens=2048,
            max_workers=8,  # qwen3-max API并发数
            show_progress=True,
        )
        
        # 统计重写成功率
        valid_count = sum(1 for r in rewritten if r.strip())
        print(f"重写成功: {valid_count}/{len(test_originals)} ({valid_count/len(test_originals)*100:.1f}%)")
        
        # ASR测试
        print(f"进行ASR测试...")
        rewritten_buckets = [[r] for r in rewritten]
        raw_output_path = os.path.join(strategy_output_dir, "raw_results.json")
        
        metrics = run_asr_test(
            target_client=target_client,
            guard_client=guard_client,
            originals=test_originals,
            rewritten_buckets=rewritten_buckets,
            target_max_model_len=args.target_max_model_len,
            guard_max_model_len=args.guard_max_model_len,
            save_raw_results=True,
            raw_output_path=raw_output_path,
        )
        
        asr = metrics.get("overall", {}).get("asr", 0.0)
        qwen3max_results[strategy_name] = asr
        
        print(f"ASR: {asr:.4f}")
        
        # 保存结果
        result = {
            "strategy": strategy_name,
            "model": args.model,
            "asr": asr,
            "valid_rewrite_count": valid_count,
            "total_count": len(test_originals),
            "metrics": metrics,
        }
        with open(os.path.join(strategy_output_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 对比分析
    print("\n" + "="*80)
    print("对比分析")
    print("="*80)
    
    # 获取本地模型的对应结果
    local_subset = {s: local_results.get(s, 0.0) for s in args.strategies}
    
    comparison = compare_results(local_subset, qwen3max_results)
    
    # 打印对比表格
    print(f"{'策略':<30} {'qwen3-4B ASR':<15} {'qwen3-max ASR':<15} {'差异':<15} {'更优':<10}")
    print("-"*80)
    for item in comparison["strategies"]:
        sign = "+" if item["diff"] > 0 else ""
        print(f"{item['strategy']:<30} {item['local_asr']:<15.4f} {item['qwen3max_asr']:<15.4f} "
              f"{sign}{item['diff']:<15.4f} {item['better']:<10}")
    print("="*80)
    
    print(f"\n平均 ASR:")
    print(f"  qwen3-4B: {comparison['analysis']['avg_local_asr']:.4f}")
    print(f"  qwen3-max: {comparison['analysis']['avg_qwen3max_asr']:.4f}")
    print(f"  差异: {comparison['analysis']['avg_diff']:+.4f}")
    
    print(f"\n结论:")
    print(comparison["analysis"]["conclusion"])
    
    # 保存对比结果
    comparison_path = os.path.join(args.output_dir, "comparison_summary.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"\n对比结果已保存: {comparison_path}")


if __name__ == "__main__":
    main()