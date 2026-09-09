"""V100 (SM 7.0) SDPA math-fallback patch for long-sequence GRPO training.

问题 (09-08 二次 OOM 根因):
  torch SDPA 在 V100 上 flash (需 SM80+) 与 mem-efficient (cutlass, 需 SM75+)
  后端均不可用, 一律回退 math 后端 —— 完整物化 [B, H, M, M] attention 矩阵。
  10 轮 GRPO 轨迹序列最长 ~18.8k token, 32 heads 单条轨迹就要 ~21GB,
  加上 policy(8GB) + ref(8GB) 常驻, V100-32GB 必 OOM。

修复:
  将 transformers.integrations.sdpa_attention.sdpa_attention_forward 重定向到
  xformers.ops.memory_efficient_attention (cutlass fmha, 原生支持 SM70,
  显存 O(M)), 语义等价 (fp16 / causal / additive mask / GQA repeat / scale)。

仅在 SM < 8.0 的 CUDA 设备上生效; 其他环境保持 transformers 原行为。
由 plugin.py (swift external_plugins) 在 trainer 进程内加载。
评估侧 conv_eval 不加载 plugin, 不受影响; vLLM rollout 服务独立进程,
自带 V100 后端选择, 也不受影响。
"""

import torch

_PATCHED = False


def _bool_to_additive(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """bool mask (True=keep) -> additive float mask (0 / finfo.min)."""
    out = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return out.masked_fill(~mask, torch.finfo(dtype).min)


def _patched_sdpa_forward(module, query, key, value, attention_mask,
                          dropout=0.0, scaling=None, is_causal=None, **kwargs):
    import xformers.ops as xops

    # xformers 无 enable_gqa, 一律显式 repeat kv (与 transformers repeat_kv 语义一致)
    if hasattr(module, "num_key_value_groups"):
        n_rep = module.num_key_value_groups
        if n_rep > 1:
            batch, kv_heads, slen, head_dim = key.shape
            key = key[:, :, None, :, :].expand(batch, kv_heads, n_rep, slen, head_dim)
            key = key.reshape(batch, kv_heads * n_rep, slen, head_dim)
            value = value[:, :, None, :, :].expand(batch, kv_heads, n_rep, slen, head_dim)
            value = value.reshape(batch, kv_heads * n_rep, slen, head_dim)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    elif attention_mask is not None and attention_mask.ndim == 2:
        # [B, S] padding mask -> [B, 1, M, S] additive
        attention_mask = attention_mask[:, None, None, :].to(query.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(query.dtype).min

    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # SDPA 布局 [B, H, M, D] -> xformers [B, M, H, D]
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # V100 (SM7.0) 的 cutlass fmha 不支持 bf16 ("bf16 only on A100+"):
    # trainer 的 no_grad logps pass 按 config dtype (bf16) 加载权重、无 autocast,
    # q/k/v 会是 bf16 —— 透明转 fp16 计算, 输出转回 bf16 (fp16 尾数更多, 精度无损)
    out_dtype = q.dtype
    if out_dtype == torch.bfloat16:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

    bias = None
    if attention_mask is not None:
        bias = _bool_to_additive(attention_mask, q.dtype) \
            if attention_mask.dtype == torch.bool else attention_mask.to(q.dtype)
        # xformers 0.0.35 要求 bias 与 head 数精确同形; expand 是 stride-0 视图,
        # 不物化 [B,H,M,N] (否则 18.8k seq 又是 ~21GB)。cutlass 内核支持 stride-0。
        if bias.ndim == 4 and bias.shape[1] != q.shape[2] and bias.shape[1] == 1:
            bias = bias.expand(q.shape[0], q.shape[2], bias.shape[2], bias.shape[3])
    elif is_causal:
        bias = xops.LowerTriangularMask()

    attn_output = xops.memory_efficient_attention(
        q, k, v, attn_bias=bias, p=dropout, scale=scaling)

    if out_dtype != attn_output.dtype:
        attn_output = attn_output.to(out_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def maybe_patch():
    """SM < 8.0 且 xformers 可用时替换 transformers SDPA; 幂等。"""
    global _PATCHED
    if _PATCHED:
        return False
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(0)
    if major >= 8:
        return False
    try:
        import xformers.ops  # noqa: F401
        import transformers.integrations.sdpa_attention as sdpa_mod
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        # 1) 模块属性 (直接 import 引用方)
        sdpa_mod.sdpa_attention_forward = _patched_sdpa_forward
        # 2) attention 注册表 (modeling_*.py 运行时经
        #    ALL_ATTENTION_FUNCTIONS[config._attn_implementation] 分发 —— 09-08
        #    教训: 只改 1) 时 qwen3 走的仍是原函数, patch 未生效)
        ALL_ATTENTION_FUNCTIONS['sdpa'] = _patched_sdpa_forward
        assert ALL_ATTENTION_FUNCTIONS['sdpa'] is _patched_sdpa_forward
        _PATCHED = True
        print("[v100_attn_patch] transformers SDPA -> xformers mem-efficient "
              "(SM 7.0, long-seq math-fallback OOM fix; registry+module)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[v100_attn_patch] NOT applied: {e}")
        return False


maybe_patch()
