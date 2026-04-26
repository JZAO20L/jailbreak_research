# TSRL4jailbreak/src/generate.py
"""
最简 Jailbreak prompt 改写生成

只提供一个函数 rewrite_prompts_k：
- 输入：client + 原始 prompts
- 输出：每个 prompt 改写出 k 个新 prompt（List[List[str]]）
- 使用 src.prompts.REWRITE_PROMPT.format(original_prompt=...) 构造输入
- 使用 src.utils.extract_tag_content 提取 <new_jailbreak_prompt> 标签内容
- 内部使用 client.llm_batch_call 并发，保持顺序
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import re
import sys
from tqdm.auto import tqdm 

# 移除硬编码路径，直接使用相对导入
# BASE_DIR = "/home/jiazixiao.jzx/TSRL4jailbreak"
# sys.path.insert(0, BASE_DIR)

from src.vllm_client import VLLMClient
from src.prompts import REWRITE_PROMPT
from src.utils import extract_tag_content


def _post_process(text: str, require_tag: bool = True, tag_name: str = "new_jailbreak_prompt") -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    t = text.strip()

    # 显式检查并移除 <think>...</think> 标签
    # 如果有<think>标签，只保留</think>之后的内容
    think_pattern = r"<think>.*?</think>"
    think_matches = re.findall(think_pattern, t, re.DOTALL)
    if think_matches:
        # 移除所有<think>标签及其内容
        t = re.sub(think_pattern, "", t, flags=re.DOTALL)
        t = t.strip()

    if require_tag:
        extracted = extract_tag_content(t, tag_name)
        return extracted.strip() if extracted else ""

    # require_tag=False：做最少清理
    low = t.lower()
    if low.startswith("rewritten prompt:"):
        t = t[len("rewritten prompt:"):].strip()

    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        t = t[1:-1].strip()

    return t


def rewrite_prompts_k(
    client: VLLMClient,
    prompts: Sequence[str],
    k: int = 1,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    stop: Optional[List[str]] = None,
    max_workers: int = 16,
    require_tag: bool = True,
    tag_name: str = "new_jailbreak_prompt",
    # tqdm / batching
    show_progress: bool = True,
    tqdm_desc: str = "rewrite",
    batch_size: int = 256,
) -> List[List[str]]:
    """
    对每个原始 prompt 改写生成 k 个新 prompt。
    保留 client.llm_batch_call，实现 tqdm：按 batch 更新进度。

    Returns:
        out[i] 是第 i 个输入 prompt 的 k 个改写结果（长度固定为 k，失败则为 ""）
    """
    prompts = list(prompts)
    if k <= 0:
        raise ValueError("k must be >= 1")
    if not prompts:
        return []

    stop = stop or []

    expanded_inputs: List[str] = []
    for p in prompts:
        full = REWRITE_PROMPT.format(original_prompt=p)
        expanded_inputs.extend([full] * k)

    total = len(expanded_inputs)
    if batch_size <= 0:
        batch_size = total

    outputs_all: List[str] = []
    pbar = tqdm(total=total, desc=tqdm_desc, disable=not show_progress)

    # 分批调用 llm_batch_call，这样可以更新进度条
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_prompts = expanded_inputs[start:end]

        batch_outs = client.llm_batch_call(
            prompts=batch_prompts,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            max_workers=max_workers,
            return_exceptions=True,
        )
        outputs_all.extend(batch_outs)
        pbar.update(len(batch_prompts))

    pbar.close()

    # 组装回 (n_prompts, k)
    results: List[List[str]] = []
    idx = 0
    for _ in range(len(prompts)):
        bucket: List[str] = []
        for _ in range(k):
            bucket.append(_post_process(outputs_all[idx], require_tag=require_tag, tag_name=tag_name))
            idx += 1
        results.append(bucket)

    return results



# =============================================================================
# After-train: Generate rewritten jailbreak prompts on eval set (base + LoRA via vLLM)
# =============================================================================
if __name__ == "__main__":
    
    eval_items = load_jsonl(EVAL_PATH)
    eval_originals = [it.get("prompt", "") for it in eval_items]

    gen_out_path = os.path.join(OUTPUT_DIR, "eval_generated_prompts.jsonl")

    LORA_DIR=""

    policy_vllm = VLLMClient(
        model_name="policy_lora",
        model_path="/home/jiazixiao.jzx/models/Qwen/Qwen3-4B",
        port=8100,
        gpu_id='0',
        launch_server=True,
        timeout=900,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        enable_lora=True,
        lora_path=LORA_DIR,
        lora_name="s1_lora",
    )

    try:
        # 一次性 batch 改写（k=1）
        rewritten_buckets = rewrite_prompts_k(
            client=policy_vllm,
            prompts=eval_originals,
            k=1,
            temperature=0.7,
            max_workers=32,
            require_tag=True,
            tag_name="new_jailbreak_prompt",
        )
    finally:
        policy_vllm.close()
        policy_vllm = None
        print("Policy vLLM closed after generation.")

    # 写出 jsonl（展开 k 条；这里 k=1）
    n_written = 0
    with open(gen_out_path, "w", encoding="utf-8") as f:
        for item, outs in zip(eval_items, rewritten_buckets):
            base_id = str(item.get("id", ""))
            original_prompt = item.get("prompt", "")
            for j, new_prompt in enumerate(outs):
                new_prompt = (new_prompt or "").strip()
                if not new_prompt:
                    continue
                out_obj = {
                    "prompt": new_prompt,
                    "original_prompt": original_prompt,
                    "source": item.get("source", "unknown"),
                    "id": f"{base_id}_{j}",
                    "original_label": item.get("original_label"),
                }
                f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                n_written += 1


    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Cleanup done.")