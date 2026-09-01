# SESS 论文修改计划

**目标会议**: AAAI 2027
**制定日期**: 2026-07-20
**最后更新**: 2026-07-20

---

## 总体定位调整

| 当前定位 | 目标定位 |
|----------|----------|
| "Self-Evolving Skills System" — 强调自进化生成更好的技能 | "跨实例经验积累机制" — 强调成功轨迹的知识不会丢弃，可被后续攻击复用 |
| Self-evolution 作为核心卖点 | Self-evolution 降级为可选模块（+0.9pp 不足以支撑主论述） |
| 标题平淡 | 突出"经验积累"、"跨实例复用" |

---

## Phase 1：纯文字修改（1-2天）

### 1.1 标题重写
**参考候选：**
- *From Isolated Attacks to Shared Memory: Reusable Skills for Automated Red Teaming*
- *Attack Experience Accumulation: A Reusable-Skill Mechanism for Automated Jailbreaking*
- *Beyond One-Shot Jailbreaking: Cross-Instance Experience Accumulation via Reusable Skills*

### 1.2 摘要重写
**结构：**
1. 首句直接点出 jailbreak 问题，不用 LLM 开头
2. 什么经验被丢弃：角色扮演、情景构造、指令覆盖等结构性策略
3. 为什么 PAIR/TAP/AutoDAN 没解决：优化当前实例/种群，不形成跨实例可检索记忆
4. 本文 RQs：
   - RQ1: Can successful attack trajectories become reusable cross-instance memory?
   - RQ2: Can such memory improve attack success while amortizing search cost?
5. SESS 如何回答：从成功轨迹抽取技能 → 按新任务检索 → 依据后续成功/失败维护技能库
6. 证据：主实验（4模型×5数据集）、跨模型迁移、消融、端到端查询成本
7. 关键结果：技能系统 +22.7pp，evolution +0.9pp（可选优化）

### 1.3 Introduction 重构（RQ 驱动）
**写法：**
- 提出合理问题（为什么每次攻击都要从头搜索？成功的攻击经验能否复用？）
- 依次回答 RQ1、RQ2
- 保留答案拼接上下文 → Method → Experiments → Conclusion

### 1.4 删除独立 Analysis and Discussion 章节
- 改为每个实验小节的结果表后用 2-3 句话解释发现
- 腾出版面给问题描述图和补充实验

### 1.5 Transferability 小节明确引用
- 必须明确对应具体表格（Table X）和图（Figure Y）

---

## Phase 2：图表修改（1天）

### 2.1 新增 Fig. 1（问题描述图）
**内容：** 展示"经验为什么被浪费"
- 任务 A → 多轮搜索 → 成功轨迹 → 任务结束，经验丢弃
- 任务 B → 再次多轮搜索 → 再次发现类似结构
- 只出现抽象策略类型（role-playing / scenario framing / instruction override）
- 不放具体攻击提示词

**工具：** drawio 绘制

### 2.2 Table 6 细化
**当前问题：** 只比较 "PAIR baseline / skill system / evolution / DAN prior"，太粗
**改为：**
| 配置 | ASR | Δ |
|------|-----|---|
| PAIR baseline | 55.5% | — |
| + 固定人工模板 | XX.X% | +X.X |
| + 原始成功轨迹复用 | XX.X% | +X.X |
| + 随机 skill | XX.X% | +X.X |
| + 静态 skill library（只建库不维护） | XX.X% | +X.X |
| + skill retrieval | XX.X% | +X.X |
| + retrieval + maintenance（完整 SESS） | 79.1% | +22.7 |
| + retrieval + maintenance + evolution | 80.0% | +0.9 |

### 2.3 主表重构（source/target 明确切分）
**当前问题：** 没有把"历史经验是否真被复用"作为主问题来组织
**改为三个层次：**
1. 同一模型、不同未见请求（source: Qwen3-4B/WildTeaming → target: Qwen3-4B/其他请求）
2. 同一模型、跨数据集（source: WildTeaming → target: AdvBench/HarmBench/JailbreakBench）
3. 跨模型（同族 vs 跨族）

**需明确说明：**
- source 与 target 如何去重
- 是否存在近重复
- 训练/建库与测试是否来自不同攻击类别

---

## Phase 3：补充实验（2-4天）

### 3.1 🟢 低成本（用已有数据即可，0.5天）

#### 3.1.1 新增指标提取
- 每个任务的平均在线模型调用次数
- 成功所需平均轮数
- 多随机种子的均值与标准差
- **来源：** 已有实验日志，无需重跑

#### 3.1.2 Judge 一致性检验
- 从已有结果中随机抽样 100-200 条
- 用第二个独立 judge（如 Llama-Guard 或人工）重判
- 报告一致性（Cohen's Kappa 或简单一致率）

### 3.2 🟡 中成本（需少量额外运行，2-3天）

#### 3.2.1 细粒度消融实验
**实验组（相同攻击器 PAIR、相同预算、相同测试集）：**
| 组别 | 配置 | 样本量 | 预计时间 |
|------|------|--------|----------|
| A | PAIR baseline | 200 | 已有 |
| B | PAIR + 固定人工模板（6个DAN） | 200 | ~2h |
| C | PAIR + 原始成功轨迹（直接复用历史轨迹） | 200 | ~2h |
| D | PAIR + 随机 skill（验证不是随便 skill 都能提升） | 200 | ~2h |
| E | PAIR + 静态 skill library（只建库不维护） | 200 | ~2h |
| F | PAIR + skill retrieval | 200 | ~2h |
| G | PAIR + retrieval + maintenance（完整 SESS） | 200 | 已有 |
| H | PAIR + retrieval + maintenance + evolution | 200 | 已有 |

**关键对比：**
- B vs C：固定模板 vs 原始轨迹复用（验证 skill abstraction 的必要性）
- D vs E：随机 skill vs 真实 skill（验证 skill 质量的重要性）
- E vs G：静态库 vs 维护库（验证 maintenance 的必要性）
- G vs H：maintenance vs +evolution（验证 evolution 的增益）

#### 3.2.2 Evolution 纵向实验
**设计：**
1. 用第一批攻击任务（200样本）建初始库
2. 后续按批次（每批 200 样本）输入新任务
3. 每批后测试固定的、未见测试集（200 样本）
4. 比较三组：
   - Static library：只建库，不维护不进化
   - Maintenance only：定期维护但不进化
   - Maintenance + evolution：完整 SESS
5. 同时报告：技能库大小、重复率、低质量技能比例、ASR

**预计时间：** 3 批 × 200 样本 = 600 样本，~4-6h

---

## 文件清单

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `SESS/SESS.tex` | 标题、摘要、Introduction、删除Analysis章节、Table 6、新增Fig 1 引用 | Phase 1-2 |
| `SESS/Figures/` | 新增 `problem_illustration.pdf`（问题描述图） | Phase 2 |
| `SESS/Tables/` | Table 6 细化、主表重构 | Phase 2 |
| `exp/` | 补充实验结果写入 | Phase 3 |

---

## 时间规划

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| Phase 1 | 纯文字修改（标题、摘要、Intro、删除Analysis） | 1-2 天 |
| Phase 2 | 图表修改（Fig 1、Table 6、主表重构） | 1 天 |
| Phase 3 | 补充实验（细粒度消融、Evolution纵向、指标提取、Judge检验） | 2-4 天 |
| **总计** | | **4-7 天** |

---

## 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| 原始轨迹复用效果已经很好 | 反而证明 skill abstraction 必要性不足 | 正面结果，如实报告，强调 abstraction 的价值在于泛化而非复制 |
| Evolution 纵向实验增益仍很小 | 无法支撑 self-evolving 论述 | 将 evolution 降级为可选模块，从标题/摘要中移除 |
| Judge 一致性低 | 评估有效性受质疑 | 报告一致性数据，讨论评估局限，建议社区标准化 |
