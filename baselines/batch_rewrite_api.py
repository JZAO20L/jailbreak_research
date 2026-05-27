#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Rewrite Script using Qwen3-8B API (阿里云百炼API)

使用阿里云百炼API调用baseline方法对test集进行rewrite，合成新的攻击prompt数据集

Usage:
    # 方式1：通过命令行参数传递API Key（推荐）
    python baselines/batch_rewrite_api.py --api-key "sk-xxx" --limit 10

    # 方式2：使用默认硬编码API Key（临时方案）
    python baselines/batch_rewrite_api.py --limit 10

    # 方式3：从环境变量读取（需要在.env中设置DASHSCOPE_API_KEY）
    export DASHSCOPE_API_KEY="sk-xxx"
    python baselines/batch_rewrite_api.py --limit 10
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from openai import OpenAI

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from baselines import get_attacker, list_strategies, STRATEGIES


# =============================================================================
# 默认API Key（临时硬编码方案）
# =============================================================================
# 注意：请在此处填入您的阿里云百炼API Key
# 或者通过命令行参数 --api-key 传递
DEFAULT_API_KEY = "sk-xxx"  # 替换为您的实际API Key


# =============================================================================
# Qwen API Client (阿里云百炼API)
# =============================================================================
class QwenAPIClient:
    """
    使用阿里云百炼API调用Qwen3-8B模型
    
    用于baseline方法的rewrite操作（翻译、迭代优化等）
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen3-8b",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        enable_thinking: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 60.0,
    ):
        """
        Args:
            api_key: 阿里云百炼API Key
                优先级：
                1. 命令行参数 --api-key
                2. 环境变量 DASHSCOPE_API_KEY
                3. 硬编码默认值 DEFAULT_API_KEY
            model: 模型名称
            base_url: API base URL
            enable_thinking: 是否启用思考模式
            temperature: 生成温度
            max_tokens: 最大token数
            timeout: 请求超时时间
        """
        # API Key优先级：命令行参数 > 环境变量 > 默认硬编码
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or DEFAULT_API_KEY
        if not self.api_key or self.api_key == "sk-xxx":
            raise ValueError(
                "DASHSCOPE_API_KEY未设置！请使用以下方式之一提供API Key:\n"
                "  1. 命令行参数: --api-key 'sk-xxx'\n"
                "  2. 环境变量: export DASHSCOPE_API_KEY='sk-xxx'\n"
                "  3. 修改脚本中的DEFAULT_API_KEY变量\n"
                "详见：baselines/README_batch_rewrite.md"
            )
        
        self.model = model
        self.base_url = base_url
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
    
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        调用API生成响应
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示词
            temperature: 生成温度（覆盖默认值）
            max_tokens: 最大token数（覆盖默认值）
        
        Returns:
            生成的响应文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        extra_body = {}
        if self.enable_thinking:
            extra_body["enable_thinking"] = True
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                extra_body=extra_body,
                stream=False,
            )
            
            response = completion.choices[0].message.content
            return response.strip()
        
        except Exception as e:
            print(f"API调用失败: {e}")
            return ""
    
    def call_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        流式调用API（用于长时间响应）
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示词
            temperature: 生成温度
        
        Returns:
            完整响应文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        extra_body = {}
        if self.enable_thinking:
            extra_body["enable_thinking"] = True
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                extra_body=extra_body,
                stream=True,
            )
            
            response_parts = []
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    response_parts.append(chunk.choices[0].delta.content)
            
            return "".join(response_parts).strip()
        
        except Exception as e:
            print(f"API流式调用失败: {e}")
            return ""
    
    def close(self):
        """关闭客户端（API客户端无需显式关闭）"""
        pass


# =============================================================================
# Wrapper for Baseline Methods
# =============================================================================
class BaselineWrapper:
    """
    包装baseline方法，使其能够使用QwenAPIClient
    
    对于不需要客户端的方法（如deepinception），直接调用
    对于需要客户端的方法（如multilingual, pair），注入QwenAPIClient
    """
    
    def __init__(
        self,
        strategy: str,
        api_client: QwenAPIClient,
        **strategy_kwargs,
    ):
        """
        Args:
            strategy: baseline策略名称
            api_client: QwenAPIClient实例
            **strategy_kwargs: 策略特定参数
        """
        self.strategy = strategy
        self.api_client = api_client
        self.strategy_kwargs = strategy_kwargs
        
        # 根据策略类型注入客户端
        if strategy in ["multilingual", "multilingual_ensemble"]:
            # Multilingual需要translate_client
            self.attacker = get_attacker(
                strategy,
                translate_client=api_client,
                **strategy_kwargs,
            )
        elif strategy in ["pair", "genetic"]:
            # PAIR和Genetic需要target_client
            # 注意：这些方法需要迭代优化，可能需要guard_client评估
            # 这里简化处理，只提供target_client
            self.attacker = get_attacker(
                strategy,
                target_client=api_client,
                guard_client=None,  # 不评估guard
                **strategy_kwargs,
            )
        elif strategy in ["deepinception", "deepinception_multilayer", "persona"]:
            # DeepInception不需要客户端（纯模板方法）
            self.attacker = get_attacker(strategy, **strategy_kwargs)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
    
    def rewrite(self, original_prompt: str) -> str:
        """
        Rewrite一个prompt
        
        Args:
            original_prompt: 原始prompt
        
        Returns:
            Rewrite后的prompt
        """
        # 使用baseline方法的generate_attack_prompt或attack方法
        result = self.attacker.attack(original_prompt, evaluate=False)
        return result.attack_prompt


# =============================================================================
# Batch Processing
# =============================================================================
def load_prompts(input_path: str) -> List[Dict[str, Any]]:
    """Load prompts from JSONL file."""
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


def batch_rewrite(
    input_path: str,
    output_dir: str,
    strategies: List[str],
    limit: Optional[int] = None,
    api_key: Optional[str] = None,
    model: str = "qwen3-8b",
):
    """
    批量rewrite prompts
    
    Args:
        input_path: 输入JSONL文件路径
        output_dir: 输出目录
        strategies: 策略列表
        limit: 限制处理的prompt数量
        api_key: API Key
        model: 模型名称
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载prompts
    prompts = load_prompts(input_path)
    if limit:
        prompts = prompts[:limit]
    
    print(f"加载了 {len(prompts)} 个prompts")
    print(f"策略列表: {strategies}")
    print(f"输出目录: {output_dir}")
    
    # 创建API客户端
    api_client = QwenAPIClient(
        api_key=api_key,
        model=model,
        enable_thinking=False,  # rewrite不需要思考模式
        temperature=0.7,
        timeout=60.0,
    )
    
    # 处理每个策略
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"处理策略: {strategy}")
        print(f"{'='*60}")
        
        # 创建输出文件
        output_file = os.path.join(output_dir, f"{strategy}.jsonl")
        
        # 创建策略wrapper
        try:
            wrapper = BaselineWrapper(strategy, api_client)
        except ValueError as e:
            print(f"策略 {strategy} 不支持: {e}")
            continue
        
        # 批量rewrite
        results = []
        success_count = 0
        
        for item in tqdm(prompts, desc=f"[{strategy}]"):
            original_prompt = item.get("prompt", "")
            if not original_prompt:
                continue
            
            try:
                # Rewrite
                attack_prompt = wrapper.rewrite(original_prompt)
                
                # 保存结果
                result = {
                    "id": item.get("id", ""),
                    "original_prompt": original_prompt,
                    "attack_prompt": attack_prompt,
                    "strategy": strategy,
                    "source": item.get("source", "unknown"),
                    "original_label": item.get("label"),
                }
                results.append(result)
                success_count += 1
                
                # 写入文件
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                # 延迟（避免API rate limit）
                time.sleep(0.5)
            
            except Exception as e:
                print(f"处理失败: {e}")
                continue
        
        # 统计
        print(f"\n策略 {strategy} 完成:")
        print(f"  成功rewrite: {success_count}/{len(prompts)}")
        print(f"  输出文件: {output_file}")
        
        # 保存summary
        summary = {
            "strategy": strategy,
            "total": len(prompts),
            "success": success_count,
            "input": input_path,
            "output": output_file,
            "model": model,
        }
        
        summary_file = os.path.join(output_dir, f"{strategy}_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Batch Rewrite using Qwen3-8B API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/dataset/processed/10k/test.jsonl",
        help="Input JSONL file path",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="baselines/output",
        help="Output directory",
    )
    
    parser.add_argument(
        "--strategies", "-s",
        type=str,
        nargs="+",
        default=["deepinception", "multilingual", "pair", "genetic"],
        help="Strategies to use (default: all except no_rewrite and template_rewrite)",
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of prompts to process",
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key (default: from DASHSCOPE_API_KEY env var)",
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen3-8b",
        help="Model name",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Batch Rewrite Script")
    print(f"使用阿里云百炼API (Qwen3-8B)")
    print(f"{'='*60}")
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"策略列表: {args.strategies}")
    print(f"模型: {args.model}")
    if args.limit:
        print(f"限制数量: {args.limit}")
    print(f"{'='*60}\n")
    
    # 运行批量rewrite
    batch_rewrite(
        input_path=args.input,
        output_dir=args.output,
        strategies=args.strategies,
        limit=args.limit,
        api_key=args.api_key,
        model=args.model,
    )
    
    print(f"\n{'='*60}")
    print(f"批量rewrite完成!")
    print(f"{'='*60}")
    print(f"输出目录: {args.output}")
    print(f"每个策略一个JSONL文件: {args.strategies}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()