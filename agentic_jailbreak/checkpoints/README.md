# Chapter 3 LoRA Checkpoints（入库版，仅 LoRA，base 不入库）

> 只提交 adapter 权重（`adapter_model.safetensors` + 配置）；base 模型（Qwen3-4B）与
> merged 权重（7.6G 级）不入库，需用 `model/` 下载或从 `output/` 恢复。
> 完整训练产出（含 optimizer/rng，可续训）在 `agentic_jailbreak/output/`（不入库）。

## 目录

| 目录 | 来源 | 初始权重 | 训练 | ASR（test C 300） |
|------|------|----------|------|-------------------|
| `M2_rft_sft_conv10turn_e2/` | `output/rft_sft_conv10turn_e2/v0-*/checkpoint-134` | Qwen3-4B (base) | RFT SFT 2 epochs（530 条全轨迹样本） | **61.0%**（A800 历史 62.3% @300 / 59.5% @1000） |
| `M3_grpo_10turn/` | `output/multi_turn_10_agent_rft/v0-*/checkpoint-300` | M2 merged | vanilla GRPO@10，300 步（0.3 epoch） | **56.67%** |

A 轴排序：**M2 61.0% > M3 56.7% > M0 51.7% > M1 50.7%**（本机 A100 口径）——vanilla GRPO 在 base 与 RFT 初始上均负收益，RFT (M2) 是唯一有效后训练臂。

## Merge 用法（⚠️ M3 的 base 是 M2 merged，不是 base！）

```bash
V=/home/tiger/jailbreak_research/.venv/bin

# M2: base = Qwen3-4B
MERGED_DIR=output/rft_sft_conv10turn_e2_merged \
SFT_DIR=agentic_jailbreak/checkpoints/M2_rft_sft_conv10turn_e2 \
  bash scripts/merge_rft_lora.sh   # 注意: merge_rft_lora.sh 按 checkpoint-* 目录搜索, 此处直接给路径亦可换用 swift export

# M3: base = M2 merged
CUDA_VISIBLE_DEVICES=3 "$V/swift" export \
    --model output/rft_sft_conv10turn_e2_merged \
    --adapters agentic_jailbreak/checkpoints/M3_grpo_10turn/adapter_model.safetensors \
    --merge_lora true --output_dir output/m3_10turn_merged --max_length 4096
```

（历史完成的 merged 权重仍在 `output/`：`rft_sft_conv10turn_e2_merged/`、`m3_10turn_merged/`）

## 后续臂（计划）

- M4 DAPO（`output/multi_turn_10_agent_rft_dapo/`）：M2 merged 初始 + dapo loss + dynamic_sample，训练完成并入本目录
- M5/M6 长臂续训：见 `docs/TODO.md` 09-14 计划