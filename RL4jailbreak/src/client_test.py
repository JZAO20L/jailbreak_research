import sys
import os
import logging
import httpx

BASE_DIR = "/home/tiger/jailbreak_research/RL4jailbreak"
sys.path.insert(0, BASE_DIR)

from src.vllm_client import VLLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_remote_vllm")

HOST = "127.0.0.1"
PORT = 8101
BASE_URL = f"http://{HOST}:{PORT}"

# 你部署 vllm serve 时的 served-model-name，必须一致
# 例如：vllm serve ... --served-model-name "judge"
MODEL_NAME = "Judge"

def health_check():
    with httpx.Client(timeout=5.0) as c:
        r = c.get(f"{BASE_URL}/health")
        logger.info(f"GET /health -> {r.status_code}, body={r.text[:200]}")
        r.raise_for_status()

def main():
    health_check()

    # ✅ 不启动 server，只连已有端口
    client = VLLMClient(
        model_name=MODEL_NAME,
        model_path="judge",     # launch_server=False 时不会用到
        host=HOST,
        port=PORT,
        launch_server=False,     # 关键
        timeout=600,
        temperature=0.0,
    )

    try:
        # ---- 单次调用（prompt 模式）----
        prompt = "Return exactly: OK"
        resp = client.llm_call(prompt=prompt, max_tokens=32, temperature=0.0)
        logger.info(f"[single] prompt={prompt}\n[single] resp={resp}")

        # ---- batch 调用（prompts 模式）----
        prompts = ["Return exactly: ONE", "Return exactly: TWO", "Return exactly: THREE"]
        resps = client.llm_batch_call(
            prompts=prompts,
            max_tokens=32,
            temperature=0.0,
            max_workers=8,
            return_exceptions=True,
        )
        for i, (p, r) in enumerate(zip(prompts, resps)):
            logger.info(f"[batch {i}] {p} -> {r}")

        # ---- messages 模式（可选）----
        messages_list = [
            [{"role": "system", "content": "You are strict."}, {"role": "user", "content": "Return exactly: MSG_ONE"}],
            [{"role": "system", "content": "You are strict."}, {"role": "user", "content": "Return exactly: MSG_TWO"}],
        ]
        resps2 = client.llm_batch_call(
            messages_list=messages_list,
            max_tokens=32,
            temperature=0.0,
            max_workers=8,
            return_exceptions=True,
        )
        for i, r in enumerate(resps2):
            logger.info(f"[messages {i}] -> {r}")

    finally:
        client.close()

if __name__ == "__main__":
    main()
