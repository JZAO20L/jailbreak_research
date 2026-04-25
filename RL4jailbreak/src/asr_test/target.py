"""
Target Model Handler

Manages target model loading, inference, and offloading.
"""

import os
import gc
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class TargetResponse:
    """Response from target model."""
    prompt: str
    response: str
    prompt_id: str
    error: Optional[str] = None


class TargetModel:
    """
    Target model handler with automatic offloading.

    Usage:
        with TargetModel(model_path, port=8001) as target:
            responses = target.generate(prompts)
        # Model is automatically offloaded here
    """

    def __init__(
        self,
        model_path: str,
        port: int = 8001,
        gpu_id: str = "0",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 4096,
        dtype: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 600.0,
        log_file: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to target model
            port: vLLM server port
            gpu_id: GPU ID(s) to use
            tensor_parallel_size: Tensor parallel size
            gpu_memory_utilization: GPU memory utilization
            max_model_len: Maximum model context length
            dtype: Data type
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            timeout: Server startup timeout
            log_file: Log file path
        """
        self.model_path = model_path
        self.port = port
        self.gpu_id = gpu_id
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.log_file = log_file

        self.client = None
        self._is_loaded = False

    def load(self):
        """Load the model (start vLLM server)."""
        if self._is_loaded:
            return

        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.vllm_client import VLLMClient

        self.client = VLLMClient(
            model_name="target",
            model_path=self.model_path,
            port=self.port,
            gpu_id=self.gpu_id,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            timeout=self.timeout,
            log_file=self.log_file,
        )
        self._is_loaded = True

    def offload(self):
        """Offload the model (stop vLLM server)."""
        if not self._is_loaded:
            return

        if self.client is not None:
            self.client.close()
            self.client = None

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

        self._is_loaded = False

    def generate(self, prompt: str) -> str:
        """Generate response for a single prompt."""
        if not self._is_loaded:
            self.load()

        try:
            response = self.client.llm_call(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response
        except Exception as e:
            return f"Error: {e}"

    def generate_batch(
        self,
        prompts: List[str],
        prompt_ids: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> List[TargetResponse]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompts
            prompt_ids: Optional list of prompt IDs
            show_progress: Show progress bar

        Returns:
            List of TargetResponse objects
        """
        if not self._is_loaded:
            self.load()

        if prompt_ids is None:
            prompt_ids = [str(i) for i in range(len(prompts))]

        responses = []
        iterator = tqdm(prompts, desc="Target") if show_progress else prompts

        for i, prompt in enumerate(iterator):
            response = self.generate(prompt)
            responses.append(TargetResponse(
                prompt=prompt,
                response=response,
                prompt_id=prompt_ids[i],
            ))

        return responses

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.offload()
        return False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded