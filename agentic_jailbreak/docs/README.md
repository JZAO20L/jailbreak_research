# Agentic Jailbreak 文档

## 文档导航

| 文档 | 内容 |
|------|------|
| [../README.md](../README.md) | 项目概述、环境准备、快速开始 |
| [./TODO.md](./TODO.md) | **唯一待办清单**(P0 阻塞项 + RL 训练 + 评估) |
| [./LOG.md](./LOG.md) | **唯一事件时间线**(决策、卡点、实验起止) |
| [./REPORT.md](./REPORT.md) | **唯一研究结论**(尝试-结果-结论单元,结论可修订) |
| [./DOC_ORGANIZATION.md](./DOC_ORGANIZATION.md) | **文档组织方案说明**(五文档人类/Agent 视角) |
| [../src/README.md](../src/README.md) | 源代码说明 |
| [../scripts/README.md](../scripts/README.md) | 实验脚本说明 |
| [../data/README.md](../data/README.md) | 数据文件说明 |
| [../configs/README.md](../configs/README.md) | 配置文件说明 |
| [../output/README.md](../output/README.md) | 实验输出说明 |

## 参考文档

### 项目内

- [SESS 第二章](../../self_evolve_skills_jailbreak/) - 自进化技能系统
- [AHR-GRPO 第一章](../../RL4jailbreak/) - 自适应混合奖励 GRPO
- [基线方法](../../baselines/) - PAIR、AutoDAN 等基线实现

### 外部

- [PAIR 论文](https://arxiv.org/abs/2310.08414) - Prompt Automatic Iterative Refinement
- [ms-swift 多轮 GRPO](../../docs/swift/grpo_multi_turn.md) - 多轮训练文档
- [ms-swift GYM 环境](../../docs/swift/gym_env.md) - GYM 环境接口

## 设计文档

### Agent Loop 设计

```
while turn < max_turns:
    # 1. 分析（Analysis）
    analysis = agent.analyze(prompt, history, memory)

    # 2. 动作选择（Action Selection）
    action = agent.select_action(analysis, skills, history)

    # 3. 执行攻击
    attack_prompt = action.adapted_content + "\n\n" + prompt
    target_response = gpt_oss(attack_prompt)
    guard_label = guard(attack_prompt, target_response)

    # 4. 更新记忆
    memory.update({turn, action, response, success})

    # 5. 检查是否成功
    if guard_label == "Unsafe":
        reward = 1.0
        break

    turn += 1
```

### 与 SESS 的关系

- **Skills 来源**：复用 SESS 提取的有效 skills
- **Skill 选择**：Agent 学习（vs SESS 的启发式规则）
- **Skill 适配**：Agent 学习（vs SESS 的无适配）
- **失败后策略**：Agent 根据反馈调整（vs SESS 的固定 rewrite）

### 与 PAIR 的区别

- **Rewrite 逻辑**：Agent 分析 + 决策（vs PAIR 的固定 judge + rewrite）
- **Skill 使用**：SESS skills（vs PAIR 的无 skill）
- **记忆机制**：可选的长期记忆（vs PAIR 的无记忆）
- **训练方式**：多轮 RL（vs PAIR 的无训练）

## 常见问题

### Q: 为什么不用 AHR-GRPO？

A: AHR-GRPO 是单轮 RL，而本章是多轮 Agent。Agent 的奖励就是 ASR，不需要 Judge reward。

### Q: 为什么 Target 用 GPT-OSS？

A: 跨族迁移，论文价值更高。SESS 在 Qwen 上 ASR 已达 99.7%，提升空间有限。
(注:当前 2-GPU 环境先用 Qwen3-4B-SafeRL 作 Target 验证 pipeline，跨族迁移后置)

### Q: 为什么只用 ASR reward？

A: Agent loop 中每轮都有 ASR 反馈，不需要额外的 Judge。简化设计，聚焦核心贡献。

### Q: 记忆机制是必须的吗？

A: 不是。记忆是可选的增强功能，用于积累攻击经验。可以先不用记忆，验证基础 Agent 效果。