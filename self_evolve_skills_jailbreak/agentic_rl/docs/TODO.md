# Agentic-RL 待验证 TODO

## 实验设计更新（2026-08-04）

### 简化方向

**算法对比**：只对比 GRPO vs GSPO（TRL 有实现，DAPO 没有）
**重点**：Reward 建模方法的对比（核心贡献）

### 实验矩阵

#### 主实验：Reward 建模对比

| 方法 | Reward 设计 | 说明 |
|------|-------------|------|
| **GRPO-ASR** | R_asr only | 单一 outcome reward |
| **GRPO-Fixed** | 0.5 * R_asr + 0.5 * R_judge | 固定权重混合 |
| **GRPO-EMA** | EMA 平滑权重（无 variance ratio） | 简单时间平滑 |
| **GRPO-AHR (Ours)** | variance ratio + sigmoid + EMA | 自适应权重 |

#### 算法对比（次要）

| 方法 | 说明 |
|------|------|
| GRPO | 标准 GRPO（TRL 默认） |
| GSPO | 序列级 GRPO（TRL experimental） |

#### 消融实验

| 维度 | 变量 | 证明什么 |
|------|------|----------|
| Skill conditioning | w/ vs w/o | skill 库的价值 |
| 冷启动方式 | RFT vs SFT vs None | RFT 的必要性 |
| Reward 组合 | ASR only → Fixed → EMA → AHR | 自适应权重的价值 |
| 算法 | GRPO vs GSPO | 算法选择的影响 |

#### 迁移实验

| 训练目标 | 测试目标 | 对比 SESS |
|----------|----------|-----------|
| Qwen3-4B | gpt-oss-20b | 32.5% → ? |

### 论文故事

**核心贡献**：
1. Skill-conditioned agentic 攻击（vs 无 skill 的纯 RL）
2. RFT 冷启动（vs 直接 RL 或 SFT）
3. **自适应混合奖励**（variance ratio 机制，vs 单一/固定/简单平滑）

**次要贡献**：
- GRPO vs GSPO 的对比（证明算法选择不是关键）

---

## 关键决策：训练目标模型选择

**问题**：第三章的训练目标应该用哪个模型？

### 选项 A：Qwen3-4B（同族）

**优势**：
- SESS 在同族上达到 99.7%，有明确的 baseline
- 本地部署成本低，训练速度快
- 可以直接对比 SESS vs Agentic-RL（同条件下）

**劣势**：
- 99.7% 已经接近天花板，提升空间有限
- 无法证明跨族迁移能力

### 选项 B：gpt-oss-20b（跨族）

**优势**：
- SESS 跨族只有 32.5%，提升空间大
- 直接证明 Agentic-RL 的泛化能力
- 论文价值更高（解决跨族迁移问题）

**劣势**：
- 需要验证 gpt-oss 是否可用（API？本地？）
- 训练成本可能更高
- 需要确认训练数据是否足够

### 需要验证的问题

1. **gpt-oss 可用性**
   - [ ] 检查是否有 API 访问权限
   - [ ] 检查是否可以本地部署
   - [ ] 评估推理成本（每次攻击的 token 消耗）

2. **训练可行性**
   - [ ] 确认 gpt-oss 的输入格式限制
   - [ ] 评估训练数据量是否足够（SESS 跨族成功率低，可能数据不足）
   - [ ] 测试 RFT 在 gpt-oss 上的初步效果

3. **实验设计**
   - [ ] 如果选 gpt-oss，是否需要同时保留 Qwen 实验作为对比？
   - [ ] 是否需要做"Qwen 训练 → gpt-oss 测试"的迁移实验？

### 建议

**短期**：先用 Qwen3-4B 验证完整 pipeline（RFT + AHR-GRPO）
**中期**：如果 pipeline 验证成功，迁移到 gpt-oss-20b
**长期**：论文中同时报告同族和跨族结果

---

## 实现优先级

1. **Phase 1: RFT 数据构建**
   - [x] 实现 `trajectory_generator.py`：单轮 RFT 模式
   - [x] 添加多维度评估（格式正确性 + ASR + 适配质量）
   - [ ] 生成 RFT 训练数据（需要 vLLM servers）

2. **Phase 2: RFT 训练**
   - [x] 修改 `sft_train.py` 为 RFT 模式
   - [ ] 验证 RFT checkpoint 的 ASR

3. **Phase 3: AHR-GRPO**
   - [x] 实现 `rewards.py`：R_asr + R_judge + 自适应权重
   - [x] 实现 `grpo_train.py`：从 RFT checkpoint 出发
   - [ ] 对比不同 reward 组合（ASR only / Fixed / EMA / AHR）

4. **Phase 4: 消融实验**
   - [ ] Skill conditioning: w/ vs w/o
   - [ ] 冷启动: RFT vs SFT vs None
   - [ ] 算法: GRPO vs GSPO

### 实验矩阵

- [ ] 主实验：4 种 reward 组合 × 2 种算法 = 8 runs
- [ ] 消融：3 个维度 × 2-3 variants = ~10 runs
- [ ] 迁移：Qwen → gpt-oss

### 论文写作

- [ ] 方法章节：强调"单轮 step-level reward"的简化
- [ ] 实验章节：突出 reward 建模的对比
- [ ] 相关工作中对比 SEMA、AdvGRPO 等
