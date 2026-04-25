# S1 训练超参数调整方案

基于 Prompt 策略优化后的 S1 重新训练计划

## 一、Prompt 策略测试（第一步）

### 测试命令
```bash
cd /home/jiazixiao.jzx/.nanobot/TSRL4jailbreak
source /opt/conda/bin/activate rlgen

# 测试 5 个策略（包含 no_rewrite baseline）
python scripts/test_rewrite_strategies.py \
  --strategies baseline direct_aggressive academic_frame fiction_creative few_shot no_rewrite \
  --eval_data data/dataset/processed/s1_1k/eval.jsonl \
  --output_root output/rewrite_strategy_test \
  --sample_size 50  # 先用 50 条快速测试
```

### 预期输出
```
output/rewrite_strategy_test/
├── strategy_baseline/
│   ├── 01_rewritten_prompts.jsonl
│   ├── 02_asr_report.json
│   └── summary.json
├── strategy_direct_aggressive/
├── ...
└── summary.json  # 对比报告
```

### 决策标准
- **ASR > 20%**: 优秀，保留该策略
- **ASR 15-20%**: 可接受，考虑使用
- **ASR < 15%**: 放弃

---

## 二、S1 训练超参数调整（第二步）

基于最优 Prompt 策略，调整以下超参数：

### 2.1 Learning Rate 实验
| 实验名 | lr | 预期效果 |
|--------|-----|----------|
| s1_lr1e-4 | 1e-4 | 更快收敛，可能过拟合 |
| s1_lr5e-5 | 5e-5 | baseline，稳定 |
| s1_lr3e-5 | 3e-5 | 更慢但更稳定 |

### 2.2 LoRA Target 模块实验
| 实验名 | lora_target_type | 微调模块 | 预期效果 |
|--------|------------------|----------|----------|
| s1_lora_a | a | q,k,v,o_proj | baseline，参数少 |
| s1_lora_all | all | 所有线性层 | 表达能力更强 |
| s1_lora_mlp | mlp | 仅 FFN 层 | 测试 FFN 重要性 |

### 2.3 Beta (KL 系数) 实验
| 实验名 | beta | 预期效果 |
|--------|------|----------|
| s1_beta0.02 | 0.02 | 更少 KL 约束，更激进 |
| s1_beta0.05 | 0.05 | baseline |
| s1_beta0.1 | 0.1 | 更强 KL 约束，更保守 |

### 2.4 Reward 权重实验（基于 5221 优化）
| 实验名 | w_idea | w_div | w_jb | w_format | 说明 |
|--------|--------|-------|------|----------|------|
| s1_5221 | 0.5 | 0.2 | 0.2 | 0.1 | baseline (最优) |
| s1_6211 | 0.6 | 0.2 | 0.1 | 0.1 | 更强 idea 保持 |
| s1_4321 | 0.4 | 0.3 | 0.2 | 0.1 | 平衡 idea+div |
| s1_7111 | 0.7 | 0.1 | 0.1 | 0.1 | 极端 idea 保持 |

---

## 三、推荐实验顺序

### Phase 1: Prompt 策略筛选 (1-2 天)
1. 测试 6 个策略 (50 条样本快速筛选)
2. 选出 ASR top-2 策略
3. 用完整 eval (200 条) 验证

### Phase 2: 单变量超参数实验 (3-5 天)
基于最优 Prompt 策略 + baseline 超参数：
1. **LR 实验**: 3 个配置 → 选最优
2. **LoRA 实验**: 3 个配置 → 选最优
3. **Beta 实验**: 3 个配置 → 选最优
4. **Reward 实验**: 4 个配置 → 选最优

### Phase 3: 最优组合验证 (1-2 天)
- 使用 Phase 2 选出的最优配置
- 完整数据集训练 (2000 条)
- 完整评估 (1000 条 test)

---

## 四、快速测试脚本

### 4.1 单策略快速测试 (50 条)
```bash
python scripts/test_rewrite_strategies.py \
  --strategies direct_aggressive no_rewrite \
  --sample_size 50
```

### 4.2 单变量实验脚本模板
```bash
# LR 实验
bash scripts/run_s1_lr_ablation.sh

# LoRA 实验
bash scripts/run_s1_lora_ablation.sh

# Beta 实验
bash scripts/run_s1_beta_ablation.sh

# Reward 实验
bash scripts/run_s1_reward_ablation_v2.sh
```

---

## 五、成功标准

| 阶段 | 指标 | 目标 |
|------|------|------|
| Prompt 优化 | Base ASR | > 20% (vs. 当前 5%) |
| S1 训练后 | ASR | > 30% (vs. 当前 6-7%) |
| S2 训练后 | ASR | > 50% (最终目标) |

---

## 六、风险与应对

| 风险 | 应对方案 |
|------|----------|
| 所有策略 ASR 都低 | 检查 Target/Guard 配置，确认 ASR 测试流程正确 |
| 超参数实验结果接近 | 增加样本量或使用统计检验 |
| 训练 OOM | 降低 batch_size 或 gpu_memory_utilization |
| Judge 服务不稳定 | 使用更小的模型或增加重试机制 |
