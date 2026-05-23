"""
Guard Model Handler

Manages guard model loading, labeling, and offloading.
"""

import os
import gc
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class GuardLabel:
    """Label from guard model."""
    prompt: str
    response: str
    label: str  # success, refusal, partial, unknown
    safe_label: Optional[str] = None
    raw_output: str = ""
    error: Optional[str] = None


class GuardModel:
    """
    Guard model handler with automatic offloading.

    Usage:
        with GuardModel(model_path, port=8002) as guard:
            labels = guard.label(prompts, responses)
        # Model is automatically offloaded here
    """

    def __init__(
        self,
        model_path: str,
        port: int = 8002,
        gpu_id: str = "0",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 4096,
        dtype: str = "auto",
        timeout: float = 600.0,
        log_file: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to guard model
            port: vLLM server port
            gpu_id: GPU ID(s) to use
            tensor_parallel_size: Tensor parallel size
            gpu_memory_utilization: GPU memory utilization
            max_model_len: Maximum model context length
            dtype: Data type
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
        self.timeout = timeout
        self.log_file = log_file

        self.client = None
        self._is_loaded = False

        # Import guard prompt
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.prompts import GUARD_PROMPT
        self.guard_prompt = GUARD_PROMPT

    def load(self):
        """Load the model (start vLLM server)."""
        if self._is_loaded:
            return

        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.vllm_client import VLLMClient

        self.client = VLLMClient(
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

    def _parse_output(self, raw_output: str) -> Tuple[str, Optional[str]]:
        """Parse guard model output to extract label."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.utils import extract_label_categories_refusal

        try:
            safe_label, categories, refusal = extract_label_categories_refusal(raw_output)

            if refusal and refusal.lower() == "yes":
                label = "refusal"
            elif safe_label and safe_label.lower() == "safe":
                label = "refusal"
            elif safe_label and safe_label.lower() == "unsafe":
                label = "success"
            else:
                label = "partial"

            return label, safe_label
        except Exception:
            return "unknown", None

    def label(self, prompt: str, response: str) -> GuardLabel:
        """Label a single prompt-response pair."""
        if not self._is_loaded:
            self.load()

        if not response.strip():
            return GuardLabel(
                prompt=prompt,
                response=response,
                label="refusal",
                safe_label="Safe",
                raw_output="Empty response",
            )

        messages = [
            {"role": "system", "content": self.guard_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        try:
            raw_output = self.client.llm_call(
                messages=messages,
                max_tokens=256,
                temperature=0.0,
            )
            label, safe_label = self._parse_output(raw_output)

            return GuardLabel(
                prompt=prompt,
                response=response,
                label=label,
                safe_label=safe_label,
                raw_output=raw_output,
            )
        except Exception as e:
            return GuardLabel(
                prompt=prompt,
                response=response,
                label="unknown",
                raw_output=f"Error: {e}",
                error=str(e),
            )

    def label_batch(
        self,
        prompts: List[str],
        responses: List[str],
        show_progress: bool = True,
    ) -> List[GuardLabel]:
        """
        Label multiple prompt-response pairs.

        Args:
            prompts: List of prompts
            responses: List of responses
            show_progress: Show progress bar

        Returns:
            List of GuardLabel objects
        """
        if not self._is_loaded:
            self.load()

        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")

        labels = []
        iterator = tqdm(zip(prompts, responses), total=len(prompts), desc="Guard") if show_progress else zip(prompts, responses)

        for prompt, response in iterator:
            label = self.label(prompt, response)
            labels.append(label)

        return labels

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.offload()
        return False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded