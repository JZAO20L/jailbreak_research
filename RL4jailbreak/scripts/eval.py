#!/usr/bin/env python3
"""
模型+prompt 评测脚本 (eval.py)

功能:
1. 对指定的评估数据集 (默认 val.jsonl) 进行 jailbreak prompt 重写 + ASR 评测
2. 支持多个 LoRA 权重文件依次评测，或 None (base model)
3. 支持多个 jailbreak prompt 模板依次评测
4. 固定: policy=GPU0, target+guard=GPU1

使用方式:
  # 命令行参数方式
  python scripts/eval.py --lora_paths /path/to/lora1 /path/to/lora2 --prompts prompt1 prompt2

  # 或直接修改脚本末尾的默认配置后运行
  python scripts/eval.py
"""

from __future__ import annotations

import os
import sys
import json
import re
import gc
import time
import argparse
import datetime
import logging
import itertools
import torch
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# =========================
# 路径配置
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.vllm_client import VLLMClient
from src.generate import rewrite_prompts_k
from src.test import run_asr_test_serial
from src.prompts import REWRITE_PROMPT, GUARD_PROMPT

# =========================
# 默认配置 (无命令行参数时生效)
# =========================
DEFAULT_CONFIG = {
    # 评估数据集 (默认 val.jsonl)
    "eval_path": os.path.join(BASE_DIR, "../data/dataset/processed/10k/val.jsonl"),

    # LoRA 路径列表 (空列表=base model)
    "lora_paths": [],

    # Jailbreak prompt 列表 (每个元素为完整的 rewrite prompt 模板字符串)
    # 默认使用 src.prompts.REWRITE_PROMPT
    "prompts": [REWRITE_PROMPT],

    # 模型路径
    "base_model_path": "/root/autodl-tmp/models/Qwen/Qwen3-4B",
    "target_model_path": "/root/autodl-tmp/models/Qwen/Qwen3-4B",
    "guard_model_path": "/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B",

    # 端口
    "policy_port": 8003,
    "target_port": 8001,
    "guard_port": 8002,
    "host": "127.0.0.1",

    # vLLM 配置
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096,
    "timeout": 900,

    # 生成配置
    "rewrite_temperature": 0.7,
    "rewrite_max_tokens": 2048,
    "rewrite_max_workers": 16,
    "rewrite_batch_size": 64,
    "k": 1,

    # ASR 测试配置
    "test_batch_size": 64,
    "test_max_workers": 16,
    "target_max_tokens": 512,
    "target_temperature": 0.0,
    "guard_max_tokens": 256,
    "guard_temperature": 0.0,
    "sleep_between_stage": 5.0,

    # 输出
    "output_root": os.path.join(BASE_DIR, "output/eval"),
}

# -------------------------
# helpers
# -------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def safe_name(s: str, max_len: int = 80) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:max_len]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    handler_file = logging.FileHandler(log_file, encoding="utf-8")
    handler_console = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler_file.setFormatter(fmt)
    handler_console.setFormatter(fmt)
    logger.addHandler(handler_file)
    logger.addHandler(handler_console)
    return logger


def run_single_eval(
    logger: logging.Logger,
    eval_items: List[Dict[str, Any]],
    eval_originals: List[str],
    *,
    base_model_path: str,
    lora_path: Optional[str],
    rewrite_prompt: str,
    target_model_path: str,
    guard_model_path: str,
    policy_port: int,
    target_port: int,
    guard_port: int,
    host: str,
    out_dir: str,
    # rewrite params
    k: int = 1,
    rewrite_temperature: float = 0.7,
    rewrite_max_tokens: int = 2048,
    rewrite_max_workers: int = 16,
    rewrite_batch_size: int = 64,
    # ASR test params
    test_batch_size: int = 64,
    test_max_workers: int = 16,
    target_max_tokens: int = 512,
    target_temperature: float = 0.0,
    guard_max_tokens: int = 256,
    guard_temperature: float = 0.0,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    timeout: int = 900,
    sleep_between_stage: float = 5.0,
) -> Dict[str, Any]:
    """
    执行单次评估 (自动启动 vLLM 服务):
    1. 启动 policy (GPU0) -> 重写 prompts
    2. 启动 target (GPU1) + guard (GPU1) -> ASR 测试
    """
    lora_label = "base" if lora_path is None else safe_name(os.path.basename(lora_path.rstrip("/")))

    # ---- Step 1: 启动 Policy (GPU0) ----
    logger.info(f"[1/4] Launch policy vLLM (GPU0:{policy_port}): lora={lora_label}")
    
    # 判断是否使用 LoRA
    use_lora = lora_path is not None
    
    policy_client = VLLMClient(
        model_name="policy",
        model_path=base_model_path,
        host=host,
        port=policy_port,
        launch_server=True,  # ✅ 自动启动服务
        timeout=timeout,
        gpu_id="0",  # GPU0
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_lora=use_lora,
        lora_path=lora_path if use_lora else None,
        lora_name="eval_lora" if use_lora else None,
    )

    # 临时替换全局 REWRITE_PROMPT
    original_rewrite_prompt = None
    if rewrite_prompt != REWRITE_PROMPT:
        import src.generate as gen_mod
        original_rewrite_prompt = getattr(gen_mod, "REWRITE_PROMPT", None)
        gen_mod.REWRITE_PROMPT = rewrite_prompt

    try:
        rewritten_buckets = rewrite_prompts_k(
            client=policy_client,
            prompts=eval_originals,
            k=k,
            temperature=rewrite_temperature,
            max_tokens=rewrite_max_tokens,
            stop=None,
            max_workers=rewrite_max_workers,
            batch_size=rewrite_batch_size,
            require_tag=False,
            show_progress=True,
            tqdm_desc=f"rewrite_{lora_label}",
        )
    finally:
        if original_rewrite_prompt is not None:
            import src.generate as gen_mod
            gen_mod.REWRITE_PROMPT = original_rewrite_prompt
        # 关闭 policy 服务
        policy_client.close()
        policy_client = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[2/4] Policy rewrite done (lora={lora_label})")

    # 收集重写后的 prompts
    rewritten_rows: List[Dict[str, Any]] = []
    n_written = 0
    for item, outs in zip(eval_items, rewritten_buckets):
        base_id = str(item.get("id", ""))
        original_prompt = item.get("prompt", "")
        for j, new_prompt in enumerate(outs):
            new_prompt = (new_prompt or "").strip()
            if not new_prompt:
                continue
            rewritten_rows.append({
                "prompt": new_prompt,
                "original_prompt": original_prompt,
                "source": item.get("source", "unknown"),
                "id": f"{base_id}_{j}",
                "original_label": item.get("original_label"),
            })
            n_written += 1

    rewritten_jsonl = os.path.join(out_dir, f"rewritten_{lora_label}.jsonl")
    write_jsonl(rewritten_jsonl, rewritten_rows)
    logger.info(f"[2/4] Rewritten: {n_written} prompts -> {rewritten_jsonl}")

    # ---- Step 2: 连接已启动的 Target + Guard (GPU1) 进行 ASR 测试 ----
    logger.info(f"[3/4] Connect to ASR test services (GPU1:{target_port}/{guard_port}): lora={lora_label}")
    target_cfg = {
        "model_name": "target",
        "model_path": target_model_path,
        "host": host,
        "port": target_port,
        "gpu_id": "0",  # GPU1 (CUDA_VISIBLE_DEVICES=1)
        "timeout": timeout,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
    }
    guard_cfg = {
        "model_name": "guard",
        "model_path": guard_model_path,
        "host": host,
        "port": guard_port,
        "gpu_id": "1",  # GPU1 (CUDA_VISIBLE_DEVICES=1)
        "timeout": timeout,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
    }

    test_report_path = os.path.join(out_dir, f"asr_report_{lora_label}.json")
    metrics = run_asr_test_serial(
        prompt_path=rewritten_jsonl,
        target_client_config=target_cfg,
        guard_client_config=guard_cfg,
        output_path=test_report_path,
        batch_size=test_batch_size,
        max_workers=test_max_workers,
        target_max_tokens=target_max_tokens,
        target_temperature=target_temperature,
        target_stop=None,
        guard_max_tokens=guard_max_tokens,
        guard_temperature=guard_temperature,
        show_progress=True,
        sleep_s_between_stage=sleep_between_stage,
        save_raw_results=False,
    )

    return {
        "lora": lora_label,
        "rewritten_count": n_written,
        "metrics": metrics,
        "rewritten_jsonl": rewritten_jsonl,
        "test_report_path": test_report_path,
    }


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="模型+prompt 评测脚本")

    # 输入配置
    parser.add_argument("--eval_path", type=str, default=None,
                        help="评估数据集路径 (默认: data/dataset/processed/10k/val.jsonl)")
    parser.add_argument("--lora_paths", type=str, nargs="*", default=None,
                        help="LoRA 权重路径列表 (空=base model)")
    parser.add_argument("--prompt_ids", type=str, nargs="*", default=None,
                        help="用于测试的 jailbreak prompt ID 列表 (默认: 使用 REWRITE_PROMPT)")
    parser.add_argument("--strategy_name", type=str, default=None,
                        help="jailbreak prompt 策略名称 (从 jailbreak_prompts.py 加载)")

    # 模型路径
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--target_model_path", type=str, default=None)
    parser.add_argument("--guard_model_path", type=str, default=None)

    # 端口
    parser.add_argument("--policy_port", type=int, default=None)
    parser.add_argument("--target_port", type=int, default=None)
    parser.add_argument("--guard_port", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)

    # vLLM 配置
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)

    # 生成配置
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--rewrite_temperature", type=float, default=None)
    parser.add_argument("--rewrite_max_tokens", type=int, default=None)
    parser.add_argument("--rewrite_max_workers", type=int, default=None)
    parser.add_argument("--rewrite_batch_size", type=int, default=None)

    # ASR 测试配置
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--test_max_workers", type=int, default=None)
    parser.add_argument("--target_max_tokens", type=int, default=None)
    parser.add_argument("--target_temperature", type=float, default=None)
    parser.add_argument("--guard_max_tokens", type=int, default=None)
    parser.add_argument("--guard_temperature", type=float, default=None)
    parser.add_argument("--sleep_between_stage", type=float, default=None)

    # 输出
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args(argv)

    # =========================
    # 合并默认配置 + 命令行覆盖
    # =========================
    cfg = dict(DEFAULT_CONFIG)
    for key, val in vars(args).items():
        if val is not None:
            cfg[key] = val

    # 如果指定了 strategy_name, 从 jailbreak_prompts.py 加载对应的 prompt
    if cfg.get("strategy_name"):
        try:
            import sys as _sys
            _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _base_dir not in _sys.path:
                _sys.path.insert(0, _base_dir)
            from experiments.jailbreak_prompt_exp.jailbreak_prompts import (
                get_strategy_template,
                JAILBREAK_PROMPTS,
            )
            strategy_name = cfg["strategy_name"]
            if strategy_name not in JAILBREAK_PROMPTS:
                raise ValueError(f"未知策略: {strategy_name}. 可选: {list(JAILBREAK_PROMPTS.keys())}")
            cfg["prompts"] = [get_strategy_template(strategy_name)]
            print(f"[eval.py] 已加载策略: {strategy_name}")
        except ImportError as e:
            raise ImportError(f"无法加载 jailbreak_prompts: {e}")

    # 设置 GPU (policy=GPU0, target+guard=GPU1 分开运行)
    # 注意: 由于 policy 和 target/guard 不在同一时间运行, 可以共用 visible devices
    # 但为了简化, 我们设置 CUDA_VISIBLE_DEVICES=0,1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # 输出目录
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.get("run_name") or f"eval_{ts}"
    out_dir = os.path.join(cfg["output_root"], run_name)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))

    logger = setup_logger(os.path.join(out_dir, "eval.log"))

    logger.info("=" * 90)
    logger.info("模型+prompt 评测脚本")
    logger.info(f"eval_path: {cfg['eval_path']}")
    logger.info(f"lora_paths: {cfg['lora_paths']}")
    logger.info(f"num_prompts: {len(cfg['prompts'])}")
    logger.info(f"output_root: {out_dir}")
    logger.info("=" * 90)

    # 保存配置快照
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)

    # 加载评估数据
    eval_items = load_jsonl(cfg["eval_path"])
    eval_originals = [it.get("prompt", "") for it in eval_items]
    logger.info(f"Loaded {len(eval_items)} prompts from {cfg['eval_path']}")

    # =========================
    # 执行评估矩阵
    # =========================
    # 如果 lora_paths 为空, 使用 [None] 表示 base model
    lora_paths_to_test = cfg["lora_paths"] if cfg["lora_paths"] else [None]
    prompts_to_test = cfg["prompts"]

    all_results: List[Dict[str, Any]] = []
    total_combos = len(lora_paths_to_test) * len(prompts_to_test)
    logger.info(f"Total combinations: {total_combos} ({len(lora_paths_to_test)} LoRA x {len(prompts_to_test)} prompts)")

    combo_idx = 0
    for lora_path, prompt_template in itertools.product(lora_paths_to_test, prompts_to_test):
        combo_idx += 1
        lora_label = "base" if lora_path is None else safe_name(os.path.basename(lora_path.rstrip("/")))
        prompt_label = safe_name(prompt_template[:50]) if isinstance(prompt_template, str) else str(combo_idx)

        combo_dir = os.path.join(out_dir, f"combo_{combo_idx:03d}_{lora_label}_{prompt_label}")
        ensure_dir(combo_dir)
        ensure_dir(os.path.join(combo_dir, "logs"))

        logger.info(f"--- [{combo_idx}/{total_combos}] LoRA={lora_label}, Prompt={prompt_label} ---")

        try:
            result = run_single_eval(
                logger=logger,
                eval_items=eval_items,
                eval_originals=eval_originals,
                base_model_path=cfg["base_model_path"],
                lora_path=lora_path,
                rewrite_prompt=prompt_template,
                target_model_path=cfg["target_model_path"],
                guard_model_path=cfg["guard_model_path"],
                policy_port=cfg["policy_port"],
                target_port=cfg["target_port"],
                guard_port=cfg["guard_port"],
                host=cfg["host"],
                out_dir=combo_dir,
                k=cfg["k"],
                rewrite_temperature=cfg["rewrite_temperature"],
                rewrite_max_tokens=cfg["rewrite_max_tokens"],
                rewrite_max_workers=cfg["rewrite_max_workers"],
                rewrite_batch_size=cfg["rewrite_batch_size"],
                test_batch_size=cfg["test_batch_size"],
                test_max_workers=cfg["test_max_workers"],
                target_max_tokens=cfg["target_max_tokens"],
                target_temperature=cfg["target_temperature"],
                guard_max_tokens=cfg["guard_max_tokens"],
                guard_temperature=cfg["guard_temperature"],
                gpu_memory_utilization=cfg["gpu_memory_utilization"],
                max_model_len=cfg["max_model_len"],
                timeout=cfg["timeout"],
                sleep_between_stage=cfg["sleep_between_stage"],
            )
            result["prompt_label"] = prompt_label
            all_results.append(result)

            # 打印本次结果摘要
            overall = result["metrics"].get("overall", {})
            logger.info(f"  -> ASR: {overall.get('asr', 0):.4f}, Refusal: {overall.get('refusal_rate', 0):.4f}")

        except Exception as e:
            logger.error(f"  -> FAILED: {e}", exc_info=True)
            all_results.append({
                "lora": lora_label,
                "prompt_label": prompt_label,
                "error": str(e),
            })

    # =========================
    # 汇总结果
    # =========================
    summary_path = os.path.join(out_dir, "summary.json")
    summary_table = []
    for r in all_results:
        if "error" in r:
            summary_table.append({
                "lora": r.get("lora"),
                "prompt": r.get("prompt_label"),
                "error": r["error"],
            })
        else:
            overall = r["metrics"].get("overall", {})
            summary_table.append({
                "lora": r.get("lora"),
                "prompt": r.get("prompt_label"),
                "asr": overall.get("asr", 0),
                "refusal_rate": overall.get("refusal_rate", 0),
                "partial_rate": overall.get("partial_rate", 0),
                "success_rate": overall.get("success_rate", 0),
                "total": overall.get("total", 0),
                "valid": overall.get("valid", 0),
            })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"combinations": summary_table}, f, ensure_ascii=False, indent=2)

    logger.info("=" * 90)
    logger.info("评估完成! 汇总结果:")
    for row in summary_table:
        logger.info(f"  {row}")
    logger.info(f"汇总结果已保存: {summary_path}")
    logger.info(f"所有输出目录: {out_dir}")
    logger.info("=" * 90)


if __name__ == "__main__":
    # 检测是否有命令行参数
    if len(sys.argv) > 1:
        main()
    else:
        # ========== 默认配置直接运行 ==========
        # 修改此处可快速执行
        # =========================

        # 示例: 评估 base model + 1个 LoRA, 使用默认 prompt
        EVAL_ARGS = []  # 空列表=使用默认配置

        # 如果需要自定义, 例如:
        # EVAL_ARGS = [
        #     "--eval_path", "data/dataset/processed/10k/test.jsonl",
        #     "--lora_paths", "/path/to/lora1", "/path/to/lora2",
        #     "--output_root", "output/my_eval",
        # ]

        if EVAL_ARGS:
            main(EVAL_ARGS)
        else:
            print("=" * 80)
            print("使用默认配置运行评估...")
            print("=" * 80)
            main()
