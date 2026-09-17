# 训练曲线数据（论文绘图用）

每行 = 一个 logging step（每 5 步）的 JSON（swift GRPO 指标）。关键字段：

- `global_step/max_steps`、`epoch`：步数/进度
- `reward`、`rewards/gym_reward/mean|std`、`frac_reward_zero_std`：奖励与零组率（**核心曲线**）
- `kl`、`loss`、`grad_norm`、`learning_rate`：优化侧（⚠️ kl 含尖峰，归因见 LOG 09-17）
- `num_turns`、`completions/mean_length|max_length`：轨迹行为
- `step_time`、`memory(GiB)`：效率/显存

| 文件 | 臂 | 步数 | 说明 |
|------|----|------|------|
| `m3_grpo.jsonl` | M3（M2 merged + vanilla GRPO） | 300 | A 轴对照（56.67%） |
| `m4_dapo.jsonl` | M4（M2 merged + DAPO） | 300 | 唯一正收益 RL（64.33%） |
| `m5_v5_final.jsonl` | M5 v5（M3 终态续训至 1.0ep，vanilla） | ~602 | **主曲线**（48.0%，含 kl 尖峰） |
| `m5_v0_dead160.jsonl` | M5 初版（G16；被 SIGHUP 误杀） | 160 | 血统碎片 |
| `m5_v3p1_pdb2.jsonl` | M5 v3.1（G8/PDB2；显存止损） | ~25 | 血统碎片 |
| `m5_v4_pdb2.jsonl` | M5 v4（G8/PDB2；显存红线止损） | ~10 | 血统碎片 |

绘制示例：

```python
import json, pandas as pd
rows = [json.loads(l) for l in open("m5_v5_final.jsonl")]
df = pd.DataFrame(rows)
df["step"] = df["global_step/max_steps"].str.split("/").str[0].astype(int)
df.plot(x="step", y=["reward", "frac_reward_zero_std"])
```

注意：
- M3/M4/M5 曲线来自不同 run；M5 的累计 epoch = (300 + 100 + step) / 1000（口径见 `checkpoints/M5b_grpo_10turn/README.md`）
- `epoch` 字段是 swift 内部归一（≠累计口径），绘图请用 `step` 自行换算