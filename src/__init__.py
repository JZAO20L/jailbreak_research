"""
jailbreak_research 通用模块

包含多个子项目共享的工具类：
- VLLMClient: vLLM 服务客户端
- prompts: 提示词模板
- config: 配置管理
"""

from .vllm_client import VLLMClient

__all__ = ["VLLMClient"]