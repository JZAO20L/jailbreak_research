# M5b 长臂 GRPO LoRA ckpts（累计 2.0 epoch 计划的中间产物）

> 防服务器释放丢失产物。仅 adapter（33MB/档）+ 配置；base/merged 不入库。

## 血统（2026-09-15/16）

- 初始 base：`output/m3_10turn_merged`（= M2 merged + M3 LoRA merge，策略 = M3 终态）
- **v4pdb2-ckpt50** — v4 run（PDB=2 快跑版，run dir `output/multi_turn_10_agent_rft_long/v4-20260915-195128`）checkpoint-50
  = 累计 300(M3) + 100 = 400 条；因显存红线（74G/80G）于 ~step 86 止损，此 ckpt 经 merge 成为 v5 的初始（`output/m5v5_init_ckpt50`）
- **v5-ckpt50 / v5-ckpt100 / v5-ckpt150** — v5 run（PDB=1 安全版，run dir `v5-20260916-001837`，G=8 / GBS=8 / GA=8 / liger，1600 步计划）
  = 累计 300 + 100 + {50,100,150} 条

## 重建方法

```bash
V=/opt/tiger/JudgeForge/jailbreak_research/.venv/bin
# 例：从 v5-ckpt150 续跑（先 merge 到 v5 的 base 上）
CUDA_VISIBLE_DEVICES=3 $V/swift export \
    --model <v5 base：output/m5v5_init_ckpt50> \
    --adapters checkpoints/M5b_grpo_10turn/v5-ckpt150 \
    --merge_lora true --output_dir output/m5b_from150_merged --max_length 4096
# 再用 /tmp/m5_train_cmd_v5.sh 模板：--model 换 merged，--max_steps 按剩余条数调整（PDB=1 → 步数=条数）
```

⚠️ base 对应关系见各 `adapter_config.json` 的 `base_model_name_or_path`：
- v4pdb2-ckpt50 → `output/m3_10turn_merged`
- v5-ckpt* → `output/m5v5_init_ckpt50`（= m3_10turn_merged + v4pdb2-ckpt50 的 merge）

## 评估计划

- 选点累计 0.6 / 1.0 / 1.5 / 2.0 epoch（RUN_TAG=m5_*）
- 结论须标注：G=8 与 M1-M4（G=16）口径不同（零组率实测 ~0.65 vs M3 0.54，以最终统计为准）