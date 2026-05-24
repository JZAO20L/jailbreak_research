#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器端批量重写脚本

使用vLLM服务器（已通过sh脚本启动）调用baseline方法对test集进行重写
支持：DeepInception（无需模型）、Multilingual、Genetic（最多5轮）、PAIR（最多5轮）

GPU配置：单卡GPU0，同时加载Qwen3-4B（port 8001）和Qwen3Guard（port 8002）
各模型占用0.45显存，vLLM server通过sh脚本启动，此脚本只连接不启动

Usage:
    # 需要先通过sh脚本启动vLLM servers
    python baselines/rewrite_prompts_server.py --strategies deepinception multilingual pair genetic --limit 100
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import httpx

# Add project root and RL4jailbreak to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_DIR = os.path.join(BASE_DIR, "RL4jailbreak")
sys.path.insert(0, BASE_DIR)  # Add project root for baselines import
sys.path.insert(0, RL_DIR)     # Add RL4jailbreak for VLLMClient import

from src.vllm_client import VLLMClient
from baselines import get_attacker, list_strategies, STRATEGIES


# =============================================================================
# VLLM Client Wrapper (launch_server=False)
# =============================================================================
class VLLMClientWrapper:
    """
    VLLM客户端包装器（仅连接，不启动server）
    
    根据反馈memory：不使用model_name参数，只通过port连接
    """
    
    def __init__(
        self,
        port: int,
        host: str = "127.0.0.1",
        temperature: float = 0.7,
        timeout: float = 120.0,
        max_model_len: int = 4096,
    ):
        """
        Args:
            port: vLLM server端口
            host: 主机地址
            temperature: 生成温度
            timeout: 请求超时时间
            max_model_len: 最大模型长度（用于health check时推断）
        """
        self.port = port
        self.host = host
        self.base_url = f"http://{host}:{port}"
        self.base_url_v1 = f"{self.base_url}/v1"
        self.temperature = temperature
        self.timeout = timeout
        self.max_model_len = max_model_len
        
        # 使用httpx client
        self._http_client = httpx.Client(timeout=self.timeout)
        
        # 初始化OpenAI client（不传model_name）
        from openai import OpenAI
        self.openai_client = OpenAI(
            api_key="EMPTY",
            base_url=self.base_url_v1,
            http_client=self._http_client,
        )
        
        # 等待server就绪
        self._wait_for_server()
    
    def _wait_for_server(self, timeout_s: float = 60.0):
        """等待vLLM server就绪"""
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
    
    def llm_call(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        调用LLM生成响应
        
        注意：不使用model参数，直接通过port调用（根据反馈memory）
        
        Args:
            prompt: 用户输入prompt
            messages: messages列表（对话模式）
            system_prompt: 系统提示词
            max_tokens: 最大token数
            temperature: 生成温度
            stop: 停止词
            
        Returns:
            生成的响应文本
        """
        # 构建messages
        if messages is not None:
            final_messages = messages
        elif prompt is not None:
            final_messages = []
            if system_prompt:
                final_messages.append({"role": "system", "content": system_prompt})
            final_messages.append({"role": "user", "content": prompt})
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        
        # 调用OpenAI API（不传model参数）
        try:
            resp = self.openai_client.chat.completions.create(
                # 不使用model参数，根据反馈memory
                messages=final_messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            
            content = resp.choices[0].message.content
            return content if content else ""
            
        except Exception as e:
            print(f"API调用失败: {e}")
            return ""
    
    def close(self):
        """关闭客户端"""
        try:
            self._http_client.close()
        except Exception:
            pass


# =============================================================================
# Load and Save Functions
# =============================================================================
def load_prompts(input_path: str) -> List[Dict[str, Any]]:
    """从JSONL文件加载prompts"""
    prompts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                prompts.append(obj)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
    return prompts


def save_result(result: Dict, output_file: str):
    """保存单个结果到JSONL文件"""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# =============================================================================
# Strategy-Specific Rewrite Functions
# =============================================================================
def rewrite_deepinception(prompt_data: Dict, attacker, output_file: str) -> Dict:
    """
    DeepInception: 纯模板方法，无需模型
    """
    original_prompt = prompt_data.get("prompt", "")
    result_obj = attacker.attack(original_prompt, evaluate=False)
    
    result = {
        "id": prompt_data.get("id", ""),
        "original_prompt": original_prompt,
        "attack_prompt": result_obj.attack_prompt,
        "strategy": "deepinception",
        "iterations": result_obj.iterations,
        "time_cost": result_obj.time_cost,
    }
    
    save_result(result, output_file)
    return result


def rewrite_multilingual(prompt_data: Dict, attacker, output_file: str) -> Dict:
    """
    Multilingual: 使用translate_client翻译
    """
    original_prompt = prompt_data.get("prompt", "")
    result_obj = attacker.attack(original_prompt, evaluate=False)
    
    result = {
        "id": prompt_data.get("id", ""),
        "original_prompt": original_prompt,
        "attack_prompt": result_obj.attack_prompt,
        "strategy": "multilingual",
        "iterations": result_obj.iterations,
        "time_cost": result_obj.time_cost,
        "metadata": {"language": attacker.current_language},
    }
    
    save_result(result, output_file)
    return result


def rewrite_genetic(prompt_data: Dict, attacker, output_file: str) -> Dict:
    """
    Genetic: 遗传算法，最多5轮迭代
    需要target_client和guard_client评估fitness
    """
    original_prompt = prompt_data.get("prompt", "")
    
    # 执行攻击（会在内部进行遗传进化）
    result_obj = attacker.attack(original_prompt, evaluate=True)
    
    result = {
        "id": prompt_data.get("id", ""),
        "original_prompt": original_prompt,
        "attack_prompt": result_obj.attack_prompt,
        "strategy": "genetic",
        "iterations": result_obj.iterations,
        "time_cost": result_obj.time_cost,
        "target_response": result_obj.target_response,
        "guard_label": result_obj.guard_label,
        "is_success": result_obj.is_success,
        "metadata": result_obj.metadata,
    }
    
    save_result(result, output_file)
    return result


def rewrite_pair(prompt_data: Dict, attacker, output_file: str) -> Dict:
    """
    PAIR: 迭代优化，最多5轮迭代
    需要target_client和guard_client
    """
    original_prompt = prompt_data.get("prompt", "")
    
    # 执行攻击（会在内部进行迭代优化）
    result_obj = attacker.attack(original_prompt, evaluate=True)
    
    result = {
        "id": prompt_data.get("id", ""),
        "original_prompt": original_prompt,
        "attack_prompt": result_obj.attack_prompt,
        "strategy": "pair",
        "iterations": result_obj.iterations,
        "time_cost": result_obj.time_cost,
        "target_response": result_obj.target_response,
        "guard_label": result_obj.guard_label,
        "is_success": result_obj.is_success,
        "intermediate_results": result_obj.intermediate_results,
    }
    
    save_result(result, output_file)
    return result


# =============================================================================
# Batch Rewrite
# =============================================================================
def batch_rewrite(
    input_path: str,
    output_dir: str,
    strategies: List[str],
    limit: Optional[int] = None,
    rewrite_port: int = 8001,
    target_port: int = 8001,  # 使用同一个Qwen3-4B作为target
    guard_port: int = 8002,
    max_iterations: int = 5,
    verbose: bool = False,
):
    """
    批量重写prompts
    
    Args:
        input_path: 输入JSONL文件路径
        output_dir: 输出目录
        strategies: 策略列表
        limit: 限制处理的prompt数量
        rewrite_port: 重写模型端口（Qwen3-4B）
        target_port: Target模型端口（Qwen3-4B，与rewrite_port相同）
        guard_port: Guard模型端口（Qwen3Guard）
        max_iterations: 最大迭代次数（用于Genetic和PAIR）
        verbose: 是否显示详细输出
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载prompts
    prompts = load_prompts(input_path)
    if limit:
        prompts = prompts[:limit]
    
    print(f"\n{'='*70}")
    print(f"服务器端批量重写脚本")
    print(f"{'='*70}")
    print(f"输入文件: {input_path}")
    print(f"输出目录: {output_dir}")
    print(f"策略列表: {strategies}")
    print(f"Prompt数量: {len(prompts)}")
    print(f"重写/Target端口: {rewrite_port}")
    print(f"Guard端口: {guard_port}")
    print(f"最大迭代次数: {max_iterations}")
    print(f"{'='*70}\n")
    
    # 创建vLLM客户端（连接已启动的servers）
    rewrite_client = VLLMClientWrapper(port=rewrite_port)
    guard_client = VLLMClientWrapper(port=guard_port)
    
    # Target client使用同一个rewrite_client（Qwen3-4B）
    target_client = rewrite_client
    
    # 处理每个策略
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"处理策略: {strategy}")
        print(f"{'='*70}")
        
        # 创建输出文件
        output_file = os.path.join(output_dir, f"{strategy}.jsonl")
        
        # 根据策略类型创建attacker
        if strategy == "deepinception":
            # DeepInception不需要客户端
            attacker = get_attacker(strategy, verbose=verbose)
            rewrite_func = rewrite_deepinception
            
        elif strategy == "multilingual":
            # Multilingual需要translate_client
            attacker = get_attacker(
                strategy,
                translate_client=rewrite_client,
                verbose=verbose,
            )
            rewrite_func = rewrite_multilingual
            
        elif strategy == "genetic":
            # Genetic需要target_client和guard_client，最多5轮
            attacker = get_attacker(
                strategy,
                target_client=target_client,
                guard_client=guard_client,
                max_iterations=max_iterations,
                max_generations=max_iterations,  # 限制遗传代数
                population_size=5,  # 减小种群大小以加快速度
                verbose=verbose,
            )
            rewrite_func = rewrite_genetic
            
        elif strategy == "pair":
            # PAIR需要target_client和guard_client，最多5轮
            attacker = get_attacker(
                strategy,
                target_client=target_client,
                guard_client=guard_client,
                max_iterations=max_iterations,
                verbose=verbose,
            )
            rewrite_func = rewrite_pair
            
        else:
            print(f"不支持的策略: {strategy}")
            continue
        
        # 批量重写
        success_count = 0
        total_time = 0.0
        
        for prompt_data in tqdm(prompts, desc=f"[{strategy}]"):
            try:
                result = rewrite_func(prompt_data, attacker, output_file)
                success_count += 1
                total_time += result.get("time_cost", 0)
                
                # 延迟（避免API过载）
                time.sleep(0.3)
                
            except Exception as e:
                print(f"处理失败: {e}")
                continue
        
        # 统计
        avg_time = total_time / success_count if success_count > 0 else 0
        print(f"\n策略 {strategy} 完成:")
        print(f"  成功重写: {success_count}/{len(prompts)}")
        print(f"  平均耗时: {avg_time:.2f}s")
        print(f"  输出文件: {output_file}")
        
        # 保存summary
        summary = {
            "strategy": strategy,
            "total": len(prompts),
            "success": success_count,
            "avg_time": avg_time,
            "input": input_path,
            "output": output_file,
            "max_iterations": max_iterations,
            "rewrite_port": rewrite_port,
            "guard_port": guard_port,
        }
        
        summary_file = os.path.join(output_dir, f"{strategy}_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 关闭客户端
    rewrite_client.close()
    guard_client.close()
    
    print(f"\n{'='*70}")
    print(f"批量重写完成!")
    print(f"{'='*70}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="服务器端批量重写脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/dataset/processed/10k/test.jsonl",
        help="输入JSONL文件路径",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="baselines/output",
        help="输出目录",
    )
    
    parser.add_argument(
        "--strategies", "-s",
        type=str,
        nargs="+",
        default=["deepinception", "multilingual", "pair", "genetic"],
        help="策略列表",
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="限制处理的prompt数量",
    )
    
    parser.add_argument(
        "--rewrite-port",
        type=int,
        default=8001,
        help="重写模型端口（Qwen3-4B）",
    )
    
    parser.add_argument(
        "--target-port",
        type=int,
        default=8001,
        help="Target模型端口（与rewrite-port相同）",
    )
    
    parser.add_argument(
        "--guard-port",
        type=int,
        default=8002,
        help="Guard模型端口（Qwen3Guard）",
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="最大迭代次数（用于Genetic和PAIR）",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出",
    )
    
    args = parser.parse_args()
    
    # 运行批量重写
    batch_rewrite(
        input_path=args.input,
        output_dir=args.output,
        strategies=args.strategies,
        limit=args.limit,
        rewrite_port=args.rewrite_port,
        target_port=args.target_port,
        guard_port=args.guard_port,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()