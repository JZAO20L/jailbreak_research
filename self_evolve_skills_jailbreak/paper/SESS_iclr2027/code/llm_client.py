#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal LLM client over a vLLM OpenAI-compatible HTTP server.

Only a running `vllm serve` endpoint is required -- the client auto-detects
the served model name and calls `chat.completions` (the OpenAI SDK).
"""

import re

import openai


class LLMClient:
    def __init__(self, port: int = 8000, temperature: float = 0.7,
                 timeout: float = 600.0, host: str = "127.0.0.1"):
        self.port = port
        self.temperature = temperature
        self.client = openai.OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="EMPTY",          # vLLM does not require a real key
            timeout=timeout,
            max_retries=2,
        )
        # auto-detect the served model name
        try:
            self.model = self.client.models.list().data[0].id
        except Exception:
            self.model = "default"

    def call(self, prompt=None, messages=None, system_prompt=None,
             max_tokens=1024, temperature=None):
        """Unified call: raw prompt (+optional system) or explicit messages."""
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
        )
        content = getattr(resp.choices[0].message, "content", None) or ""
        return strip_thinking(content)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 thinking blocks so only the visible answer remains."""
    if "</t_h>" in text:
        text = text.split("</t_h>")[-1]
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL)
    return text.strip()
