# 毕业论文三章进度报告

> 更新日期：2026-09-02
> 项目：`/home/tiger/jailbreak_research`（GitHub: JZAO20L/jailbreak_research, main 已同步至 b328c92）
> 环境：4×V100-SXM2-32GB | Guard=Qwen3Guard-Gen-4B / Target=Qwen3-4B-SafeRL / Policy=Qwen3-4B（全项目统一）

---

## 总览

| 章 | 主题 | 形态 | 完成度 | 当前阶段 |
|---|---|---|---|---|
| 第一章 | AHR-GRPO：自适应混合奖励的单轮 jailbreak RL | 方法 + 实验 | 实验完成（中期口径 55 组） | EMNLP 冲刺投稿 |
| 第二章 | SESS：基于自进化 Skills 库的多轮迭代攻击 | 系统 + 大规模实验 | 实验完成（中期口径 217 组） | 论文撰写（AAAI 2027 投稿准备） |
| 第三章 | Agentic Jailbreak：对话式 harness + 多轮 GRPO 训练 | Harness + AgenticRL | harness ~95%，训练 ~70%，奖励 ~30% | 10 轮实验矩阵执行中（本周） |

三章递进逻辑：**单轮 RL 重写（Ch1）→ 多轮规则化 skill 进化（Ch2）→ RL 训练的多轮 agentic 攻击（Ch3）**；
Ch3 直接回应 Ch2 的两个局限：skill 注入依赖人工规则、跨族迁移差（gpt-oss-20b 仅 32.5%）。

---

## 第一章：AHR-GRPO（`RL4jailbreak/`）

### 方法
GRPO + LoRA 训练 jailbreak prompt 重写模型。混合奖励 = ASR 结果奖励（稀疏）+ Judge 过程奖励（稠密）；
核心创新：按组内方差比经 sigmoid + EMA 自适应调整权重 λ（`λ_raw = σ(α·log ratio + δ)`，EMA 平滑，clip [0.1, 0.9]）。

### 进度（数据口径：中期报告版 `docs/experiment_results_midterm.md`，55 组实验 / ~45 GPU 时）
> ⚠️ 口径注记：本章数字统一采用中期报告版口径；`docs/AHR-GRPO.md` 为原始实验工作记录（部分表格使用不同 prompt 评估模板，数字不同，两份文档数字不可混用）。Baselines 均为 target = Qwen3-4B（非 SafeRL）口径，与第三章不可直接比。

| 项 | 状态 | 结果（中期口径） |
|---|---|---|
| 实验1：24 种攻击策略筛选（1000 条） | ✅ | Top-3 入选（ASR>25%）：hypothetical_scenario **30.8%** / creative_writing 28.3% / role_playing 25.0%；其余 16 策略 7.7~18.8% |
| 实验2：Judge 维度 12 组合 | ✅ | **全部正向改进**；最佳 hypothetical+idea_preservation **32.3%（+1.5）**；idea_preservation 平均 +1.2 最稳 → **通用维度优于专用维度** |
| 实验3：自适应混合奖励消融 | ✅ | 自适应 β=0 **33.2%（vs 实验2最佳 +0.9，vs baseline +2.4）**、β=0.8 32.9%、β=0.9 32.6%；Reward 类型消融：仅 ASR 25.0（**-5.8，信号稀疏**）/ 仅 Judge 28.5 / 固定 1:1 32.3 / **自适应 33.2 最佳** |
| 方法定位（表1.4） | ✅ | 低成本重写式攻击：500 步 GRPO vs PAIR 1000+ API 调用；PAIR 91.8%（多轮上界）/ genetic 47.9% / deepinception 37.3% / multilingual 2.1% |
| 论文 | 🔄 冲刺中 | 完整大纲 + 各章草稿（EMNLP 模板）；9 天冲刺 checklist（双盲检查、页数、OpenReview 提交） |

核心结论：①自适应权重超越固定权重（+0.9pp）；②混合 Reward 显著优于单一 Reward（仅 ASR 掉 5.8pp——稀疏性直接证据）；③通用 Judge 维度（idea_preservation）优于专用维度；④低成本重写式定位（绝对值低于 PAIR，成本差两个数量级）。②③④ 直接构成第三章 C 轴（reward 组合 + AHR λ 迁移）的动机链。

### 待办
- EMNLP 冲刺 checklist 执行（双盲检查、页数控制、OpenReview 提交）
- 论文占位补齐：OOD 泛化、Case Study、附录完整表
- λ 演化曲线可视化（SwanLab CSV 已导出）

---

## 第二章：SESS（`self_evolve_skills_jailbreak/`）

### 方法
自进化 Skills 库多轮攻击：从成功轨迹抽象可复用 skill（模板 + 成功率统计 + 关键词 + harm_type），
按质量分×关键词匹配动态检索注入；三阶段流水线 Cold Start → Evolution → Test；统计维护（剪枝/合并）。

### 进度（数据口径：中期报告版，217 组实验 / ~133 GPU 时）
| 实验 | 规模 | 最佳结果 |
|---|---|---|
| Layer 1 方法组合 grid | 16 组 | single_call+trajectory+statistical **79.1%**（28 skills）；single_call 优于 every_iteration +12.7pp |
| Layer 2 数据策略消融 | 36 组 | small(300)+early(30%CS) **80.6%**；数据量影响 <2%（300 条足够） |
| Layer 3 DAN 模板验证 | 17 组 | DAN+full_evolve **98.8%**（6 个 DAN 模板直接最优，无需 Cold Start/Evolution） |
| Layer 4 DAN 数据策略 | 12 组 | **99.7%**（medium+evo / small+full_evolve 并列） |
| Evolution 必要性消融 | 16 组 | 平均贡献仅 **+0.9pp**（最佳组合 +0.5），可省 80% 训练时间 |
| 跨模型迁移 | 140 组 | 同族（Qwen 系）：0.6B 95.1% / 4B 86.7% / 14B 86.5%（平均 +4.7 vs baseline）；跨族 gpt-oss-20b：autodan_skills_54 最高 **32.5%**（同族 ~99%） |

### 核心结论
- Skill System（CS + 检索 + 维护）是主要贡献来源；Evolution 高水位下边际收益（+0.9pp）
- 强先验（DAN 模板）可直接达到最优，6 模板免进化
- 同族迁移良好；**跨族迁移是真正挑战（-66.5pp）** → 第三章动机之一

### 论文进度（`paper/`，目标 AAAI 2027）
- Round 1 模拟审稿：Weak Reject (4.5/10)；v1→v2 叙事重构为 "skill system" 已完成
- `outline_v5.md` 待 LaTeX 编译、图表嵌入、Algorithm 伪代码
- 待补实验：多种子统计、LLaMA/Mistral 跨族扩展、TAP/GAP/Metis/ASTRA 基线

---

## 第三章：Agentic Jailbreak（`agentic_jailbreak/`）★ 当前重心

### 方法
**PAIR 骨架 + Skills 库（LLM 自选）** 的 agentic 攻击 agent，与第二章 SESS 一脉相承：
对话式 harness（system → user(初始观察+候选技能) → assistant(动作) → user(guard 反馈) 累积式多轮交互）；
每轮动作空间二选一 —— **自由改写攻击 prompt**（PAIR 式迭代）或 **`Selection: Skill [i]` + 适配**（调用 skill）。
Skills 库统一使用 **10-skill 精选库**（54 库按实测单调用 ASR 筛选的 top-10，`seed_skills_top10.json`）；
ms-swift 4.3.2 async GYM 插件对接 GRPO 训练；奖励现阶段 ASR-only。

### 已完成（8 月，已入库 git）
| 项 | 结果 |
|---|---|
| Harness 设计空间消融（无状态 6 变体 → 对话式） | 对话式 +7pp，主形态定稿 = 对话式多轮 |
| Skill 注入消融（A1，54 库全局 top-5） | 强制注入 6-9%、LLM 自决 19% vs no_skill 23.2% → 根因是全局 top-5 候选与 prompt 不对齐 |
| Skill 精选（实测 ASR sweep） | 54 库逐一实测 → **top-10 精选库**（`seed_skills_top10.json`），消除候选不对齐 |
| **口径定稿（09-02）** | **主配置 = skill_decide@10skills**：LLM 自选 + PAIR 式自由改写，后续核心实验统一此协议 |
| 正式 GRPO 训练 exp01/02/03（500 步 ×3） | 单轮 10.0% / 3 轮 25.8% / 5 轮 33.9%（beam-2: 17.1/35.2/44.4%）；RL 增益随轮数放大 |
| 搜索类基线 | Crescendo 1.8% / TAP 7.2%（对照 AutoDAN 19%） |
| RFT 冷启动 v2（beam 协议来源） | **负结果** 6.8%：协议失配（主因）+ 前缀展开过加权早期动作 |
| RFT+GRPO 复训 v10（5 轮） | 跑完 200 步但 reward 饥饿：66% 步组内奖励全零，学习信号近零 |

### 关键教训（已固化到文档/记忆）
1. swift `default` loss 对多轮 messages 所有 assistant 轮算 loss → RFT 应使用全轨迹样本（每动作监督一次）
2. TRL 0.29 默认 `loss_type=dapo` → vanilla GRPO 对照必须显式 `--loss_type grpo`
3. beam-2 本质是每轮 best-of-2 采样放大 → 主口径改单轨迹，beam 降级附录
4. 8 月"GRPO 崩溃"实为端口冲突误诊，V100 链路可用

### 进行中：10 轮实验矩阵（M0-M3，DAPO 臂暂缓）
**全部臂统一协议：skill_decide@10skills**（PAIR 骨架 + LLM 自选 skill + 自由改写），max_turns=10。

| 臂 | 初始权重 | 训练 | 状态 |
|---|---|---|---|
| M0 agent@10 | base | 无 | ⬜ test C 正式评估待跑 |
| M1 只GRPO@10 | base | vanilla GRPO 300 步，num_gen=16 | ⬜ 训练脚本就绪（plugin 默认已切 skill_decide@10skills@10轮） |
| M2 RFT@10 | base→SFT(conv v3 数据) | SFT 2 epochs | 🔄 采集运行中（skill_decide 协议，09-02 重采） |
| M3 RFT+GRPO@10 | M2 merged | 同 M1 | ⬜ 阻塞于 M2 |

统一口径：max_turns=10 / num_generations=16 / lr 1e-5 / β 0.05 / temp 0.9 / 单轨迹评估。
RFT v3 数据链路（conv 协议原生采集 → 全轨迹样本，assistant 侧监督原始输出含 Selection 标记）代码已完成。

### 第三章三轴实验规划（09-02 定稿）

前两章实验结果已够用；第三章按"agent 实现 + agentic 训练"三轴推进，漏斗式叠加（每轴只在当前最优配置上做下一轴）：

| 轴 | 内容 | 臂 | 状态 |
|---|---|---|---|
| **A 轴 后训练配方**（主轴） | RFT 与 RL 的各自贡献与叠加 | M0-M3 主矩阵 + M4/M5(DAPO) 视稀疏度 | 🔄 进行中 |
| **B 轴 上下文维护**（前置，harness 定稿用） | 全量累积 / 分层工作记忆 / 滑动窗口 | C1/C2/C3 推理侧消融（纯 eval，base + 最优配方权重各一遍） | ⬜ 排期 9/6-9/8 |
| **C 轴 Reward 组合** | 稀疏缓解 + 第一章方法迁移 | R1 ASR-only / R1+P 过程 / R1+E 效率 / **R1+λ AHR 自适应迁移** | ⬜ 排期 9/8-9/14，3 新臂全跑 |

执行顺序：A 主矩阵（9/2-9/6）→ B 推理侧消融定稿 harness（9/6-9/8）→ C reward 三臂（9/8-9/14，每次 GRPO ~19h）→ Phase 4。
C 轴代码工作项：rewards.py R2/R3 接入 plugin（现为 ASR-only）；R1+λ 需 env 返回奖励分量 + 训练侧组级自适应混合（唯一非平凡改造）。

### 暂缓
- M4/M5 DAPO 臂（dynamic_sample + epsilon_high，swift 支持已确认）：视 M1/M3 的 reward 稀疏程度决定
- Phase 4：多 benchmark（advbench/harmbench/jailbreakBench）、vs SESS/PAIR/AHR-GRPO 对比、REPORT 单元 8
- Phase 5：RSI 递归自我改进（`docs/RSI_DESIGN.md`，三轴闭环，未启动）

---

## 后续排期

### 本周（9/2 - 9/8）
| 日期 | 事项 |
|---|---|
| 9/2 | 起服务；RFT conv 采集重跑（skill_decide@10skills，~3-5h）；方法口径定稿（PAIR+10skill）+ 三轴规划定稿 |
| 9/3 | M2: SFT（1-2h）→ merge → 评估；M1 启动训练（~19h） |
| 9/4 | M1 评估；M3 启动训练（~19h） |
| 9/5 | M3 评估；M0 test C 正式评估；A 轴主矩阵表格成型 |
| 9/6-9/8 | **B 轴上下文消融**（C1/C2/C3 推理侧，base+最优配方权重）；视结果定稿 harness；决定 DAPO 臂 |

### 下一步（9 月中下旬）
| 日期 | 事项 |
|---|---|
| 9/8-9/14 | **C 轴 reward 三臂**（R1+P / R1+E / R1+λ），先完成 plugin 奖励接入与 AHR λ 迁移代码 |
| 9/14-9/20 | Phase 4 对比实验（vs SESS / PAIR / AHR-GRPO，多 benchmark）；最优配置补 DAPO 臂（如需） |
| 9/20+ | 第三章 REPORT 单元 8 定稿 + 毕业论文章节撰写；第一章 EMNLP 提交（9 天冲刺 checklist）；第二章论文 LaTeX 编译 + AAAI 2027 投稿 |

### 风险与备忘
1. M0 同口径基线必须在 test C 上单独评估（不能拿 split A 采集数字）
2. num_generations=16 的 generation_batch_size=32 batch 语义需在 M1 脚本中现场验证
3. 磁盘：清理后 47G 可用；merged 权重已删（可从 LoRA 再生），Phase 4 评估 exp03 policy 前需先重新 merge
4. 第三章 vs TAP/Crescendo 的口径公平性（搜索式 vs 单轨迹）需在论文中预埋说明

---

## 附录：关键文件索引
| 内容 | 路径 |
|---|---|
| 第三章状态权威来源 | `agentic_jailbreak/docs/TODO.md` / `LOG.md` / `REPORT.md` |
| 第三章实验结果汇总 | `agentic_jailbreak/exp/README.md` |
| RFT v3 链路 | `agentic_jailbreak/scripts/{run_rft_collect_conv.sh, build_rft_data_conv.py, rft_sft_conv.sh}` |
| 10-skill 精选库（主配置） | `agentic_jailbreak/exp/skill_asr_sweep/seed_skills_top10.json`（构建器 `scripts/build_seed_skills.py`） |
| **两章实验数据口径权威来源（中期版）** | `docs/experiment_results_midterm.md` |
| 第一章原始实验记录 | `docs/AHR-GRPO.md`（评估模板口径不同，勿与中期数字混用） |
| 第二章实验总汇 | `self_evolve_skills_jailbreak/exp/COMPLETE_EXPERIMENT_SUMMARY.md` |
| 第二章论文工程 | `self_evolve_skills_jailbreak/paper/` |
| 第一章实验规划 | `RL4jailbreak/TODO.md` / `docs/EXP3_GUIDE.md` |
