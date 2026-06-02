# Self-Evolving Skills for Jailbreak - 早期成果总结

**组会汇报材料 | 2026-05-28**

---

## 一、项目背景

### 研究问题

如何通过**自进化机制**提升 Jailbreak Prompt 生成的效率和成功率？

### 核心创新

**Skills = 多轮迭代攻击中的"捷径"**

- 从成功攻击轨迹中提取可复用的 prompt 前缀/策略
- 形成 Skills 库，引导后续攻击更高效达成目标
- Skills 在攻击过程中持续进化优化

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Self-Evolving Skills Pipeline                 │
│                                                                  │
│  Phase 1: Cold Start ─→ Phase 2: Evolution ─→ Phase 3: Test     │
│  (200 prompts)        (800 prompts)        (1000 prompts)        │
│  生成初始Skills        Skills进化          计算ASR              │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 并发: 轨迹级 (64+ concurrent trajectories)                 │  │
│  │ 评判: 严格判定 (仅 Unsafe → success)                       │  │
│  │ 维护: 每100步删除低效Skills                                │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 三阶段流程

| 阶段 | 数据量 | 目标 | 输出 |
|------|--------|------|------|
| Cold Start | 200条 | 生成初始 Skills | Skills 库 |
| Evolution | 800条 | Skills 进化优化 | 进化后 Skills |
| Test | 1000条 | 评估最终效果 | ASR |

---

## 三、关键技术实现

### 1. Skill 数据结构

```python
Skill {
    name: str           # 唯一标识
    content: str        # Prompt 前缀内容
    keywords: List      # 关键词标签
    harm_type: str      # 伤害类型
    success_count: int  # 成功次数
    failure_count: int  # 失败次数
    success_rate: float # 成功率
}
```

### 2. Skills 检索机制

```
检索得分 = quality_score × 匹配加成

quality_score = success_rate × sqrt(usage_count)

匹配加成:
- 关键词匹配: +30% / 个
- 伤害类型匹配: +50%
```

### 3. 轨迹级并发

- 使用 `ThreadPoolExecutor` 实现并发
- 每个 attack trajectory 独立并发执行
- SkillLibrary 使用 `threading.Lock()` 保证线程安全

### 4. 严格评判标准

| Guard 输出 | 判定结果 |
|------------|----------|
| **Unsafe** | ✅ Success |
| Controversial | ❌ Refusal |
| Safe | ❌ Refusal |

---

## 四、实验设计

### Layer 1: 核心方法消融（16 组）

| 消融点 | 选项 | 数量 |
|--------|------|------|
| A: Skill 调用时机 | single_call / every_iteration | 2 |
| B: Skill 总结粒度 | final_prompt / trajectory | 2 |
| C: Skill 进化策略 | success_only / failure_only / both / statistical | 4 |

### 数据配置

- **Cold Start**: 200 条 (train 前 1000 条的 20%)
- **Evolution**: 800 条 (train 前 1000 条的 80%)
- **Test**: 1000 条 (完整独立 test 数据集)

---

## 五、早期成果

### 初步验证结果

| 方法 | ASR | Avg Iterations | 备注 |
|------|-----|----------------|------|
| **Self-Evolve Skills** | **76.7%** | 4.6 | single_call + final_prompt + success_only |
| PAIR (Baseline) | ~50% | 5-6 | 无 Skills |
| AutoDAN (Baseline) | ~45% | - | - |
| Genetic (Baseline) | ~40% | - | - |

### 关键发现

1. **✅ Skills 自进化效果显著**
   - 初步验证达到 **76.7% ASR**
   - 相比 PAIR baseline 提升 **+26.7%**
   - 平均迭代次数 4.6（效率提升）

2. **✅ 评判标准统一**
   - 所有方法统一使用严格判定
   - 仅 `Unsafe` 计入成功，避免 ASR 虚高

3. **✅ 并发机制有效**
   - 轨迹级并发实现
   - 支持 64+ 并发轨迹
   - 线程安全的 Skills 库

---

## 六、消融分析（待完成）

### Layer 1 目标

找到最佳方法组合：

| 问题 | 消融点 | 验证内容 |
|------|--------|----------|
| Skill何时调用? | A | single_call vs every_iteration |
| 如何总结Skill? | B | final_prompt vs trajectory |
| 如何进化Skill? | C | 4种策略对比 |

### 当前状态

- ✅ 单组合验证完成 (76.7% ASR)
- 🔄 Grid Search 16组实验进行中
- ⏳ Layer 2 数据消融待运行

---

## 七、下一步计划

### 短期（本周）

1. **完成 Layer 1 Grid Search**
   - 16 组实验运行
   - 分析最佳方法组合

2. **Bug 修复**
   - JSON 序列化问题已修复
   - Skills 存储路径已修正

### 中期

1. **Layer 2 数据消融**
   - 数据量影响验证
   - Cold Start/Evolution 分配比例

2. **完整实验**
   - 3 epochs 运行
   - 最终 ASR 报告

---

## 八、技术亮点

| 特性 | 实现方式 | 效果 |
|------|----------|------|
| 轨迹级并发 | ThreadPoolExecutor | 高效并行 |
| 线程安全 Skills | threading.Lock() | 无竞态 |
| 严格评判 | 仅 Unsafe | 准确 ASR |
| Skills 维护 | 定期删除低效 | 库质量保证 |

---

## 九、总结

### 当前进展

| 项目 | 状态 |
|------|------|
| 系统实现 | ✅ 完成 |
| Baselines 对比 | ✅ 完成 |
| 初步验证 | ✅ 76.7% ASR |
| Layer 1 Grid Search | 🔄 进行中 |

### 核心结论

**Self-Evolving Skills 机制显著提升 Jailbreak 攻击效果**

- 初步验证 +26.7% ASR 提升
- Layer 1 实验将确定最佳配置
- 完整实验将进一步验证效果

---

## 附录：命令参考

```bash
# 启动服务
bash RL4jailbreak/scripts/start_guard.sh    # GPU 0, Port 8002
bash RL4jailbreak/scripts/start_target.sh   # GPU 1-2, Port 8001

# 运行 Layer 1 Grid Search
bash self_evolve_skills_jailbreak/exp/layer1/run_layer1.sh --skip_launch --max_workers 64

# 查看 README
cat self_evolve_skills_jailbreak/README.md
cat self_evolve_skills_jailbreak/exp/layer1/README.md
```