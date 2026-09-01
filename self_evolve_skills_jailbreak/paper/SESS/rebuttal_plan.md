# SESS论文Review响应计划

**评审分数**: 6/10 (Marginally above acceptance threshold)
**目标**: 提升至7-8分（Weak Accept）

---

## 一、致命问题（必须立即修复）

### 1. 缺失关键相关工作 ⚠️ 最严重

**问题**:
- MemoAttack (Zhang et al. 2026) - "skill-structured memory"
- SRTJ (Li et al. 2026) - "self-evolving rule-driven"
- Persona Attack (Park et al. 2026) - "incremental memory injection"

**影响**: 直接威胁novelty claim

**修复方案**:
1. **验证引用真实性**（防止再次出现幻觉引用）
2. 在Related Work中添加新子章节："Memory-Driven Jailbreak Methods"
3. 创建对比表：SESS vs MemoAttack vs SRTJ vs Persona Attack
4. 强调SESS的差异化：
   - 多因子检索 vs 单一相似度
   - 结构化模板 vs 规则/记忆
   - 冷启动机制 vs 端到端训练

**工作量**: 中等（需要阅读3篇论文，撰写对比段落）

---

### 2. 检索机制过于简单

**问题**: 
- 仅用lexical matching（14个pattern）
- 无法处理同义词、改写

**修复方案**:
- **短期**: 在Limitations中诚实承认，并说明设计理由
- **中期**: 添加语义检索对比实验（SBERT vs lexical）
- **长期**: 实现hybrid检索

**Rebuttal策略**:
- 强调效率优势（lexical检索快10-100倍）
- 承认精度劣势，但展示实际效果已足够好

**工作量**: 低（短期），高（中期）

---

### 3. 强先验依赖

**问题**: 
- DAN模板贡献了大部分性能（99.7% vs 79.1%）
- 空库启动效果差

**修复方案**:
- **重新定位**: 从"autonomous skill discovery"改为"prior-amplification framework"
- **诚实承认**: 在Abstract中明确说明依赖强先验
- **补充分析**: 分析PAIR初始化为何效果差

**Rebuttal策略**:
- 强调实用价值："红队测试通常有先验知识"
- DAN模板是开源的，可复用

**工作量**: 低（主要是重新表述）

---

## 二、重要问题（建议修复）

### 4. 演化机制贡献微弱

**问题**: 只有+0.9pp提升

**修复方案**:
- 重新定位为"optional refinement"而非"core mechanism"
- 补充实验：在更长演化周期下的效果

**工作量**: 中等

---

### 5. 缺乏统计显著性

**问题**: 没有置信区间、标准差

**修复方案**:
- 补充实验：3个随机种子，计算mean±std
- 添加statistical significance tests

**工作量**: 高（需要重新跑实验）

---

### 6. 跨模型迁移弱

**问题**: GPT-OSS-20B只有32.5%

**修复方案**:
- 补充分析：为什么迁移效果差？
- 讨论tokenization、alignment差异

**工作量**: 中等

---

### 7. 模型覆盖有限

**问题**: 缺少GPT-4o, Claude等

**修复方案**:
- 在Limitations中诚实承认
- 补充Llama-3-70B实验（如果计算资源允许）

**工作量**: 高

---

## 三、次要问题（可快速修复）

### 8. 图表可视化

**修复**:
- Figure 2: 添加error bars
- Figure 3: 添加数值标签
- 统一y-axis scaling

**工作量**: 低

### 9. 写作澄清

- Page 1: 分离"8.8 calls"和"32.5% ASR"
- 添加代码公开声明

**工作量**: 低

---

## 四、Rebuttal策略

### 核心辩护点

1. **Novelty**: 
   - SESS有独特的cold-start机制
   - 多因子检索vs单一相似度
   - 详细的成本分析（首次量化API调用降低）

2. **Practical Value**:
   - 成本降低78-90%对红队测试有实际价值
   - DAN模板已开源，可复用

3. **Honest Limitation Acknowledgment**:
   - 诚实承认依赖先验
   - 承认检索机制的局限性
   - 承认跨模型迁移的困难

---

## 五、修改优先级

| 优先级 | 问题 | 工作量 | 影响 |
|--------|------|--------|------|
| **P0** | 缺失相关工作 | 中 | 致命 |
| **P1** | 统计显著性 | 高 | 重要 |
| **P1** | 重新定位（先验依赖） | 低 | 重要 |
| **P2** | 检索机制局限 | 低-中 | 中等 |
| **P2** | 演化机制定位 | 低 | 中等 |
| **P3** | 图表改进 | 低 | 次要 |

---

## 六、时间规划

### Rebuttl前（紧急）
- [ ] 验证MemoAttack/SRTJ/Persona Attack引用真实性
- [ ] 添加相关工作对比段落
- [ ] 重新定位论文contribution（诚实承认先验依赖）

### Camera Ready前
- [ ] 补充统计显著性（3 seeds）
- [ ] 改进图表
- [ ] 添加代码公开链接

---

## 七、需要验证的引用

**必须先验证真实性**：
1. MemoAttack (Zhang et al. 2026)
2. SRTJ (Li et al. 2026)  
3. Persona Attack (Park et al. 2026)
4. Many-shot jailbreaking (Anil et al. 2024)
5. Guiding not Forcing (Yang et al. 2025)

**验证方法**：
- 检查arXiv是否存在
- 检查论文标题是否匹配
- 阅读摘要确认内容

---

## 八、预期结果

**如果成功修复P0-P1问题**:
- 分数可从6分提升至7-8分
- 有较大概率接受（70-80%）

**如果仅修复P0**:
- 分数维持在6-7分
- 接受概率50-60%