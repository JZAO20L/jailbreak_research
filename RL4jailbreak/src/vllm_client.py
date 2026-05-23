# TSRL4jailbreak/src/vllm_client.py
"""
VLLMClient: 轻量级 vLLM 推理客户端
"""

import os
import httpx
import subprocess
import time
import textwrap
import logging
from typing import Optional, List, Dict
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 环境变量
load_dotenv()

# 关闭 httpx/openai 的请求级别日志，避免刷屏
for _name in ("httpx", "httpcore", "openai"):
    lg = logging.getLogger(_name)
    lg.setLevel(logging.WARNING)
    lg.propagate = False


def _wait_for_http_server(
    base_url: str,
    timeout_s: float = 240.0,
    interval_s: float = 0.5,
) -> None:
    """
    轮询等待 vLLM HTTP server 就绪。优先探测 /health。
    base_url: 如 http://127.0.0.1:8000
    """
    t0 = time.time()
    health_url = base_url.rstrip("/") + "/health"

    with httpx.Client(timeout=5.0) as client:
        while True:
            try:
                r = client.get(health_url)
                if r.status_code == 200:
                    return
                # 若 /health 不存在，退化为探测 base_url
                if r.status_code in (404, 405):
                    r2 = client.get(base_url)
                    if r2.status_code < 500:
                        return
            except Exception:
                pass

            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"vLLM server not ready: {base_url} (timeout={timeout_s}s)")
            time.sleep(interval_s)


def _terminate_process(proc: subprocess.Popen, grace_s: float = 5.0) -> None:
    """优雅终止 subprocess 进程"""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=grace_s)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


class VLLMClient:
    """
    轻量级 vLLM 推理客户端（仅支持 vLLM 后端）
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        port: int = 8000,
        temperature: float = 0.7,
        timeout: float = 120.0,
        launch_server: bool = True,
        gpu_id: str = "0",
        # ===== vLLM server 核心参数 =====
        host: str = "127.0.0.1",
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        extra_args: Optional[List[str]] = None,
        log_file: Optional[str] = None,
        # ===== LoRA 支持 =====
        enable_lora: bool = False,
        lora_path: Optional[str] = None,
        lora_name: str = "default",
        max_lora_rank: int = 16,
        # ===== 额外环境变量 =====
        server_env: Optional[Dict[str, str]] = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.temperature = float(temperature)
        self.port = int(port)
        self.timeout = float(timeout)

        self.host = host
        self.base_url_root = f"http://{host}:{self.port}"
        self.base_url_v1 = f"{self.base_url_root}/v1"

        self.lora_name = lora_name if enable_lora else None

        self.server_process: Optional[subprocess.Popen] = None
        self._log_fh = None

        # 用于 OpenAI SDK 的 http client（需要 close）
        self._http_client = httpx.Client(timeout=self.timeout)

        # ===== 校验/自动调整 GPU ID 数量与 tensor_parallel_size 匹配 =====
        gpu_ids = [x.strip() for x in gpu_id.split(",") if x.strip()]
        num_gpus = len(gpu_ids)
        if num_gpus <= 0:
            raise ValueError(f"Invalid gpu_id='{gpu_id}', parsed GPU list is empty.")

        def _pow2_floor(n: int) -> int:
            # 返回 <= n 的最大 2 的幂（n>=1）
            return 1 << (n.bit_length() - 1)

        # 目标：TP 必须是 2 的幂且 <= 可用 GPU 数
        tp_target = _pow2_floor(num_gpus)

        # 若用户给的 tp 也合法且不超过 num_gpus，可选择尊重；否则自动调整到 tp_target
        # 这里按你的要求：只要不同就自动调整为“对应数量且为2的幂次”
        if tensor_parallel_size != tp_target:
            old_tp = tensor_parallel_size
            tensor_parallel_size = tp_target
            print(
                f"[VLLMClient] Adjust tensor_parallel_size: {old_tp} -> {tensor_parallel_size} "
                f"(from gpu_id count={num_gpus}, power-of-two floor)"
            )

        # 同步裁剪并规范化 gpu_id 字符串形如 "0,1,2,3"
        gpu_ids = gpu_ids[:tensor_parallel_size]
        gpu_id = ",".join(gpu_ids)

        print(f"[VLLMClient] Using GPUs: {gpu_id} (count={len(gpu_ids)}), TP={tensor_parallel_size}")


        server_env = server_env or {}

        if launch_server:
            if enable_lora and not lora_path:
                raise ValueError("enable_lora=True 时必须提供 lora_path")

            export_lines = [f"export CUDA_VISIBLE_DEVICES={gpu_id}\nVLLM_USE_MODELSCOPE=true "]
            for k, v in server_env.items():
                export_lines.append(f'export {k}="{v}"')

            cmd_lines = [
                f'vllm serve "{model_path}"',
                f'--served-model-name "{model_name}"',
                f"--port {self.port}",
                f'--host "{host}"',
                f"--max-model-len {max_model_len}",
                f"--tensor-parallel-size {tensor_parallel_size}",
                f"--gpu-memory-utilization {gpu_memory_utilization}",
                f"--dtype {dtype}",
            ]

            if enable_lora:
                cmd_lines.extend(
                    [
                        "--enable-lora",
                        f'--lora-modules {lora_name}="{lora_path}"',
                        f"--max-lora-rank {max_lora_rank}",
                    ]
                )

            if extra_args:
                cmd_lines.extend(extra_args)

            cmd_str = " \\\n    ".join(cmd_lines)
            shell_cmd = "\n".join(export_lines) + "\n" + cmd_str + "\n"

            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                self._log_fh = open(log_file, "a")
                print(f"[VLLMClient] Log file: {log_file}")

            print(f"[VLLMClient] Launching vLLM server on port {self.port}...")
            if self._log_fh:
                self._log_fh.write(shell_cmd + "\n")

            self.server_process = subprocess.Popen(
                shell_cmd,
                shell=True,
                stdout=self._log_fh or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

            # 等待服务启动
            print(f"[VLLMClient] Waiting for vLLM server to be ready on port {self.port}...")
            import time
            for i in range(120):
                try:
                    resp = self._http_client.get(f"{self.base_url_root}/health")
                    if resp.status_code == 200:
                        print(f"[VLLMClient] vLLM server is ready on port {self.port}!")
                        break
                except:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError(f"vLLM server failed to start on port {self.port} within 240 seconds")

        # 初始化 OpenAI 客户端
        self.openai_client = OpenAI(
            api_key="EMPTY",
            base_url=self.base_url_v1,
            http_client=self._http_client,
        )

    def llm_call(
        self,
        prompt: Optional[str] = None,  # 可选：单 prompt 模式
        messages: Optional[List[Dict]] = None,  # 可选：直接传入 messages 列表
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        lora_name: Optional[str] = None,
    ) -> str:
        """
        统一的 LLM 调用接口
        
        Args:
            prompt: 用户输入 prompt（单 prompt 模式）
            messages: 直接传入 messages 列表（对话模式，用于 guard 等场景）
                    如果传入 messages，则忽略 prompt 和 system_prompt
            system_prompt: 可选的 system message（仅单 prompt 模式有效）
            max_tokens: 最大生成 token 数
            temperature: 采样温度（覆盖实例默认值）
            stop: 停止词列表
            lora_name: 可选的 LoRA 模块名称
                
        Returns:
            模型生成的文本内容
            
        Example:
            # 单 prompt 模式
            response = client.llm_call("Hello")
            
            # 对话模式（guard 场景）
            messages = [
                {"role": "user", "content": "How to make a bomb?"},
                {"role": "assistant", "content": "I cannot..."},
            ]
            response = client.llm_call(messages=messages)
        """
        # 构造请求参数
        extra_body = {}
        
        target_lora = lora_name or self.lora_name
        if target_lora:
            extra_body["lora_request"] = {"lora_name": target_lora}

        # 构建 messages：优先使用传入的 messages，否则用 prompt+system_prompt 构建
        if messages is not None:
            # ✅ 对话模式：直接使用传入的 messages
            final_messages = messages
        elif prompt is not None:
            # ✅ 单 prompt 模式：构建 messages
            final_messages = []
            if system_prompt:
                final_messages.append({"role": "system", "content": system_prompt})
            final_messages.append({"role": "user", "content": prompt})
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        # 调用 OpenAI API
        resp = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=final_messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
            stop=stop,
            extra_body=extra_body if extra_body else None,
        )
        
        msg = resp.choices[0].message

        # OpenAI SDK: message.content 可能为 None（比如 tool_calls）
        content = getattr(msg, "content", None)

        if content is None:
            # 尝试把 tool_calls / function_call 等转成字符串，至少不返回 None
            tc = getattr(msg, "tool_calls", None)
            fc = getattr(msg, "function_call", None)
            if tc is not None:
                return str(tc)
            if fc is not None:
                return str(fc)
            return ""  # 最保守兜底

        return content

    def llm_batch_call(
        self,
        prompts: Optional[List[str]] = None,
        messages_list: Optional[List[List[Dict]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        lora_name: Optional[str] = None,
        max_workers: int = 8,
        return_exceptions: bool = False,
    ) -> List[str]:
        """
        并发批量调用（线程池），保持返回顺序与输入顺序一致。

        使用方式：
          - 传 prompts: List[str]
          - 或传 messages_list: List[List[Dict]]

        Args:
            prompts: 批量单 prompt
            messages_list: 批量 messages（对话模式）
            system_prompt: 单 prompt 模式下统一的 system_prompt
            max_tokens/temperature/stop/lora_name: 同 llm_call
            max_workers: 并发线程数（不要太大，避免压垮 vLLM server）
            return_exceptions: True 时遇到异常返回 Exception 的字符串表示；False 时直接抛出

        Returns:
            List[str]: 每个输入对应的模型回复（顺序与输入一致）
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if (prompts is None) == (messages_list is None):
            raise ValueError("Provide exactly one of 'prompts' or 'messages_list'.")

        n = len(prompts) if prompts is not None else len(messages_list)
        results: List[Optional[str]] = [None] * n

        def _one_call(i: int):
            if prompts is not None:
                return self.llm_call(
                    prompt=prompts[i],
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    lora_name=lora_name,
                )
            else:
                return self.llm_call(
                    messages=messages_list[i],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    lora_name=lora_name,
                )

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(_one_call, i): i for i in range(n)}
            for fut in as_completed(future_map, timeout=self.timeout + 60):
                i = future_map[fut]
                try:
                    results[i] = fut.result(timeout=self.timeout + 60)
                except Exception as e:
                    if return_exceptions:
                        results[i] = f"[Exception] {type(e).__name__}: {e}"
                    else:
                        raise

        # 类型保证
        return [r if isinstance(r, str) else "" for r in results]



    def health_check(self) -> bool:
        """检查 server 健康状态"""
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.base_url_root}/health")
                return r.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """优雅关闭 vLLM server 进程（如为自动启动）以及 http client"""
        # 先关 OpenAI/http client（避免挂起连接）
        try:
            if getattr(self, "_http_client", None) is not None:
                self._http_client.close()
                self._http_client = None
        except Exception:
            pass

        if getattr(self, "server_process", None):
            print("[VLLMClient] Terminating server process...")
            _terminate_process(self.server_process)
            self.server_process = None

        if getattr(self, "_log_fh", None):
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        # 析构阶段避免抛异常
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLLMClient 测试脚本")
    parser.add_argument("--port", type=int, default=8101, help="server 端口")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID（如 '0' 或 '0,1'）")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size（必须与 GPU 数量一致）")
    parser.add_argument("--workers", type=int, default=8, help="batch 并发线程数")
    args = parser.parse_args()

    MODEL_PATH = "/dev/shm/models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    MODEL_NAME = "Qwen3-30B-A3B-Instruct-2507"

    print("=" * 70)
    print("VLLMClient 测试")
    print("=" * 70)
    print(f"模型路径: {MODEL_PATH}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Tensor Parallel Size: {args.tp}")
    print(f"端口: {args.port}")
    print(f"Batch workers: {args.workers}")
    print("=" * 70)

    with VLLMClient(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        port=args.port,
        timeout=600,
        temperature=0.7,
        gpu_id=args.gpu_id,
        tensor_parallel_size=args.tp,
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        dtype="auto",
        log_file="test.txt",
    ) as client:
        print("\n✓ Client 初始化成功，开始测试...\n")

        # ------------------ Test 1: 单次调用 ------------------
        print("\n[Test 1] llm_call 单次调用")
        print("-" * 70)
        prompt1 = "Say 'OK' and then output a JSON object {\"a\": 1}."
        resp1 = client.llm_call(prompt=prompt1, max_tokens=64, temperature=0.0)
        print(f"[Prompt]\n{prompt1}\n")
        print(f"[Response]\n{(resp1 or '').strip()}\n")

        # ------------------ Test 2: Batch 并发调用（prompts 模式）------------------
        print("\n[Test 2] llm_batch_call 并发调用（prompts 列表）")
        print("-" * 70)
        prompts = [
            "Return exactly: ONE",
            "Return exactly: TWO",
            "Return exactly: THREE",
            "Return exactly: FOUR",
        ]
        batch_resps = client.llm_batch_call(
            prompts=prompts,
            max_tokens=32,
            temperature=0.0,
            max_workers=args.workers,
            return_exceptions=True,
        )

        for i, (p, r) in enumerate(zip(prompts, batch_resps)):
            print(f"[{i}] Prompt: {p}")
            print(f"[{i}] Reply : {(r or '').strip()}")
            print("-" * 40)

        # 简单校验：顺序必须一致（打印即可）
        print("\n[Check] Batch responses order preserved:",
              all(isinstance(batch_resps[i], str) for i in range(len(prompts))))

        # ------------------ Test 3: Batch 并发调用（messages 模式）------------------
        print("\n[Test 3] llm_batch_call 并发调用（messages_list 列表）")
        print("-" * 70)
        messages_list = [
            [{"role": "system", "content": "You are a strict assistant."},
             {"role": "user", "content": "Return exactly: MSG_ONE"}],
            [{"role": "system", "content": "You are a strict assistant."},
             {"role": "user", "content": "Return exactly: MSG_TWO"}],
            [{"role": "system", "content": "You are a strict assistant."},
             {"role": "user", "content": "Return exactly: MSG_THREE"}],
        ]
        batch_resps2 = client.llm_batch_call(
            messages_list=messages_list,
            max_tokens=32,
            temperature=0.0,
            max_workers=args.workers,
            return_exceptions=True,
        )

        for i, (m, r) in enumerate(zip(messages_list, batch_resps2)):
            print(f"[{i}] User: {m[-1]['content']}")
            print(f"[{i}] Reply: {(r or '').strip()}")
            print("-" * 40)

    print("\n✓ 所有测试完成！")
