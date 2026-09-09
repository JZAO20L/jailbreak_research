# 03 · 毕业论文三章内容介绍

> 仓库 `JZAO20L/jailbreak_research`，三章递进：**单轮 RL 重写（Ch1）→ 多轮规则化 skill 进化（Ch2）→ RL 训练的多轮 agentic 攻击（Ch3）**。

| 章 | 主题 | 代码目录 | 训练框架 | 论文状态 |
|---|---|---|---|---|
| 第一章 | AHR-GRPO：自适应混合奖励的单轮 jailbreak RL | `RL4jailbreak/` | TRL 直接调用 | 实验完成，EMNLP 冲刺 |
| 第二章 | SESS：自进化 Skills 库的多轮迭代攻击 | `self_evolve_skills_jailbreak/` | 无训练（vLLM 推理） | 实验完成，AAAI 2027 重投准备 |
| 第三章 | Agentic-RL：对话式 harness + 多轮 GRPO | `agentic_jailbreak/` | **ms-swift 4.3.2** | 实验执行中 ★ 当前重心 |

---

## 第一章 · AHR-GRPO（自适应混合奖励）

**方法**：GRPO + LoRA 训练 jailbreak prompt 重写模型。混合奖励 = ASR 结果奖励（稀疏）+ Judge 过程奖励（稠密）。
核心创新：按组内方差比经 sigmoid + EMA 自适应调整权重 λ（`λ_raw = σ(α·log ratio + δ)`，EMA 平滑，clip [0.1, 0.9]）。

**关键结果**（口径：`docs/experiment_results_midterm.md`，55 组 / ~45 GPU 时）：

| 实验 | 结果 |
|---|---|
| 实验1 攻击策略筛选（1000 条） | top-3：hypothetical_scenario 30.8% / creative_writing 28.3% / role_playing 25.0% |
| 实验2 Judge 维度 12 组合 | 最佳 hypothetical+idea_preservation **32.3%（+1.5）**；通用维度优于专用维度 |
| 实验3 自适应混合奖励消融 | 自适应 β=0 **33.2%**（+2.4 vs baseline）；仅 ASR 25.0（-5.8，稀疏信号） |
| 定位 | 低成本重写式：500 步 GRPO vs PAIR 1000+ API 调用；PAIR 91.8%（上界）/ genetic 47.9% |

**核心结论**：①自适应权重 > 固定权重（+0.9pp）；②混合 Reward > 单一 Reward；③通用 Judge 维度 > 专用；④低成本重写式定位。
②③④ 直接构成第三章 C 轴（reward 组合 + AHR λ 迁移）的动机链。

**口径注意**：本章数字统一中期报告版；`docs/AHR-GRPO.md` 为原始记录（评估模板不同，数字不可混用）。
Baselines 的 target=Qwen3-4B（非 SafeRL），与第三章 plain-4B target 口径可比。

---

## 第二章 · SESS（自进化 Skills 库）

**方法**：自进化 Skills 库多轮攻击。从成功轨迹抽象可复用 skill（模板 + 成功率统计 + 关键词 + harm_type），
按质量分 × 关键词匹配动态检索注入；三阶段 Cold Start → Evolution → Test；统计维护（剪枝/合并）。

**关键结果**（中期口径，217 组 / ~133 GPU 时）：

| 实验 | 最佳结果 |
|---|---|
| Layer 1 方法组合 grid | single_call+trajectory+statistical **79.1%** |
| Layer 2 数据策略消融 | small(300)+early(30%CS) **80.6%** |
| Layer 3 DAN 模板验证 | DAN+full_evolve **98.8%** |
| Layer 4 DAN 数据策略 | **99.7%** |
| Evolution 必要性消融 | 平均仅 **+0.9pp**（可省 80% 训练时间） |
| 跨模型迁移 | 同族 0.6B 95.1% / 4B 86.7% / 14B 86.5%；**跨族 gpt-oss-20b 仅 32.5%（-66.5pp）** |

**核心结论**：Skill System（CS+检索+维护）是主要贡献；Evolution 高水位下边际；强先验（DAN）可直达最优；
**跨族迁移是真正挑战 → 第三章动机之一**。

**论文**：`paper/`（目标 AAAI 2027）。⚠️ **2026-09-08 因超 9 页被 desk reject**（格式问题，未评审），
需压缩重投（目标会议/页数待定）。Round 1 内审 Weak Reject (4.5/10)，v1→v2 已重构叙事为 "skill system"。

---

## 第三章 · Agentic-RL（对话式 harness + 多轮 GRPO）★ 当前重心

**方法**：**PAIR 骨架 + Skills 库（LLM 自选）** 的 agentic 攻击 agent，与第二章一脉相承。
对话式 harness（system → user 初始观察+候选技能 → assistant 动作 → user guard 反馈，累积式多轮）；
每轮动作二选一 —— **自由改写攻击 prompt**（PAIR 式）或 **`Selection: Skill [i]` + 适配**。
主配置 = **`skill_decide@10skills`**（精选库 `seed_skills_top10.json`，54 库按实测 ASR 筛 top-10）。
训练 = **ms-swift 4.3.2** `swift rlhf --rlhf_type grpo`（包 TRL 0.29 GRPOTrainer，经 gym 插件对接多轮 env）；
奖励现阶段 ASR-only（Guard 判定）。

**已完成**：

| 项 | 结果 |
|---|---|
| Harness 设计空间消融 | 对话式 +7pp 定稿 |
| Skill 注入消融 | 强制注入 6-9%、LLM 自决 19% vs no_skill 23.2% → 精选 top-10 消除候选不对齐 |
| 口径定稿（09-02） | **skill_decide@10skills**，后续核心实验统一 |
| GRPO exp01/02/03（500 步） | 1 轮 10.0% / 3 轮 25.8% / **5 轮 33.9%**（beam-2 44.4%）；RL 增益随轮数放大 |
| 搜索类基线 | Crescendo 1.8% / TAP 7.2%（对照 AutoDAN 19%） |
| RFT 冷启动 v2 | 负结果 6.8%（协议失配，已改 conv 原生数据） |

**实验矩阵（10 轮协议，统一超参）**：

| 臂 | 初始权重 | 训练 | 状态（09-09） |
|---|---|---|---|
| M0 agent@10 | base | 无 | 待正式评估 |
| M1 只GRPO@10 | base | GRPO 300 步 | 🔄 **发散修复中 → 恢复臂 = plain-4B target + 5 轮 + bf16** |
| M2 RFT@10 | base→SFT(conv v3) | SFT 2 epochs | ✅ 完成（10.67%，+1.34pp vs base 9.33%） |
| M3 RFT+GRPO@10 | M2 merged | 同 M1 | ⬜ 待 M1 恢复后推进 |

**三轴规划**（漏斗式，每轴在最优配置上做下一轴）：
- **A 轴 后训练配方**（主轴）：RFT 与 RL 的各自贡献与叠加 = M0-M3 主矩阵（+DAPO 臂视稀疏度）
- **B 轴 上下文维护**：全量累积 / 分层工作记忆 / 滑动窗口（推理侧消融，C1/C2/C3）
- **C 轴 Reward 组合**：R1 ASR-only / R1+P 过程 / R1+E 效率 / **R1+λ AHR 自适应迁移**

**09-09 关键决策（记录在案）**：M1 训练在 fp16 下数值发散（首个非零奖励批 grad 270→NaN），
根因 = V100 无 bf16 计算 → 切 fp16 + liger 的数值副作用。修复 = **bf16 原生训练**（exp03 已验证 500 步稳定）。
同时采纳用户建议：**target 换 plain Qwen3-4B**（提升奖励稠密度）+ **轮次降到 5**（缩短序列、贴近 exp03 验证区间）。

---

## 三章关联

```
Ch1 单轮 RL 重写（低成本，AHR 混合奖励）
   └─> Ch2 多轮规则化 skill 进化（SESS，高 ASR 但跨族迁移差、依赖人工规则）
        └─> Ch3 RL 训练的多轮 agentic 攻击（PAIR 骨架 + LLM 自选 skill + GRPO）
             └─> 直接回应 Ch2 局限：skill 注入去人工化、跨族迁移、reward 稀疏（C 轴）
```

**数字口径警告**：Ch1 中期口径 33.2%（hypothetical 模板）与 Ch3 exp03 33.9%（对话式 5 轮）**不是同一评估设置，不可直接比较**；跨章对比一律需同 target / 同评估模板重跑。
