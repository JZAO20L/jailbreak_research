# 实验1: Jailbreak Prompt 策略测试

## 实验目标

测试约24种不同的jailbreak prompt重写策略，在test集上评估它们的ASR (Attack Success Rate)，筛选出表现优秀的策略。

## 实验内容

1. **基线测试**: 测试原始prompt的ASR（不进行任何重写）
2. **策略测试**: 依次测试所有24种重写策略的ASR
3. **结果筛选**: 根据设定的条件筛选优秀策略

## 文件说明

本实验目录包含以下核心文件：

### 1. `jailbreak_prompts.py` - Prompt策略定义

**功能**: 定义所有要测试的jailbreak prompt重写策略模板

**使用方法**:

```python
# 查看所有策略
python experiments/jailbreak_prompt_exp/jailbreak_prompts.py

# 在代码中使用
from experiments.jailbreak_prompt_exp.jailbreak_prompts import (
    JAILBREAK_PROMPTS,          # 所有策略字典
    get_all_strategy_names,     # 获取所有策略名称列表
    get_strategy_template,      # 获取单个策略模板
    get_strategy_info,          # 获取策略详细信息
)

# 获取特定策略
template = get_strategy_template("urgent_situation")
info = get_strategy_info("academic_research")
```

**策略格式**: 每个策略包含 `name`（名称）、`description`（描述）、`template`（prompt模板）。

---

### 2. `jailbreak_prompt_exp.py` - Python实验脚本

**功能**: 执行完整的实验流程，包括基线测试、策略测试、结果筛选

**使用方法**:

```bash
# 基本用法
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py

# 指定要测试的策略
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py \
    --strategies urgent_situation academic_research creative_writing

# 使用Top-K筛选（只输出前5个最佳策略）
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py --topk 5

# 使用Gap筛选（差距>5%时舍弃后续策略）
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py --gap_threshold 0.05

# 指定测试集
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py \
    --test_set data/dataset/processed/10k/test.jsonl

# 指定输出目录
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py \
    --output_dir experiments/jailbreak_prompt_exp/my_output

# 调整评估间隔时间
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py \
    --sleep_between_evals 30

# 只验证配置，不实际执行（调试用）
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py --dry_run
```

**主要参数**:
- `--strategies`: 策略名称列表（空格分隔）
- `--test_set`: 测试集路径
- `--output_dir`: 输出目录
- `--topk`: 只保留前K个最佳策略
- `--gap_threshold`: Gap阈值（如0.05表示5%）
- `--sleep_between_evals`: 两次评估之间的等待时间（秒）
- `--dry_run`: 只打印配置，不执行

---

### 3. `exp.sh` - Shell实验脚本

**功能**: 使用bash执行实验，更适合在服务器上长时间运行

**使用方法**:

```bash
# 基本用法 - 运行所有策略
bash experiments/jailbreak_prompt_exp/exp.sh

# 运行指定策略
bash experiments/jailbreak_prompt_exp/exp.sh urgent_situation academic_research

# 使用Top-K筛选
bash experiments/jailbreak_prompt_exp/exp.sh --topk 5

# 使用Gap筛选
bash experiments/jailbreak_prompt_exp/exp.sh --gap_threshold 0.05

# 指定测试集（环境变量方式）
TEST_SET=data/dataset/processed/10k/test.jsonl \
    bash experiments/jailbreak_prompt_exp/exp.sh

# 指定输出目录
OUTPUT_DIR=experiments/jailbreak_prompt_exp/my_output \
    bash experiments/jailbreak_prompt_exp/exp.sh

# 调整评估间隔
bash experiments/jailbreak_prompt_exp/exp.sh --sleep 30

# 查看帮助
bash experiments/jailbreak_prompt_exp/exp.sh --help
```

**支持的环境变量**:
- `TEST_SET`: 测试集路径
- `OUTPUT_DIR`: 输出目录
- `BASE_MODEL`: 基础模型路径
- `TARGET_MODEL`: 目标模型路径
- `GUARD_MODEL`:  Guard模型路径

**支持的主要参数**:
- `--topk N`: 只输出前N个最佳策略
- `--gap_threshold FLOAT`: Gap阈值
- `--test_set PATH`: 测试集路径
- `--output_dir PATH`: 输出目录
- `--sleep SECONDS`: 评估间隔时间
- `--help`: 显示帮助信息

---

### 4. `README.md` - 实验文档

**功能**: 实验说明文档，包含目标、方法、使用指南和结果解释

---

## 使用方法

### 方式1: 使用 shell 脚本 (推荐)

```bash
# 运行全部24种策略 + 基线测试
bash experiments/jailbreak_prompt_exp/exp.sh

# 只运行指定的几个策略
bash experiments/jailbreak_prompt_exp/exp.sh urgent_situation academic_research

# 使用 Top-K 筛选 (只输出前5个最佳策略)
bash experiments/jailbreak_prompt_exp/exp.sh --topk 5

# 使用 Gap 筛选 (当策略间差距>5%时舍弃后续策略)
bash experiments/jailbreak_prompt_exp/exp.sh --gap_threshold 0.05

# 指定测试集
TEST_SET=data/dataset/processed/10k/test.jsonl bash experiments/jailbreak_prompt_exp/exp.sh

# 查看帮助
bash experiments/jailbreak_prompt_exp/exp.sh --help
```

### 方式2: 使用 Python 脚本

```bash
# 运行全部策略
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py

# 指定策略
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py --strategies urgent_situation academic_research

# Top-K 筛选
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py --topk 5

# Gap 筛选
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py --gap_threshold 0.05

# 指定测试集
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py --test_set data/dataset/processed/10k/test.jsonl

# 只验证配置，不实际执行
python experiments/jailbreak_prompt_exp/jailbreak_prompt_exp.py --dry_run
```

## 输出结果

实验完成后，结果保存在 `experiments/jailbreak_prompt_exp/output/` 目录下：

```
output/
├── baseline_original/          # 原始prompt基线测试结果
│   └── summary.json
├── urgent_situation/           # 每个策略的单独结果
│   └── summary.json
├── academic_research/
│   └── summary.json
├── ...
└── experiment_summary.json     # 汇总结果
```

### 汇总结果格式

`experiment_summary.json` 包含：

```json
{
  "experiment": "jailbreak_prompt_exp_1",
  "timestamp": "2026-04-25T...",
  "test_set": "data/dataset/processed/10k/test.jsonl",
  "baseline_asr": 0.1234,
  "all_results": {
    "urgent_situation": 0.3456,
    "academic_research": 0.2345,
    ...
  },
  "filtered_results": {
    "urgent_situation": 0.3456,
    ...
  },
  "filter_criteria": {
    "topk": 5,
    "gap_threshold": null
  }
}
```

## 筛选逻辑

### Top-K 筛选

只保留ASR最高的前K个策略。

```bash
bash exp.sh --topk 5
```

### Gap Threshold 筛选

按ASR降序排列，当相邻策略的ASR差距超过阈值时，舍弃后续所有策略。

```bash
# 当差距>5%时舍弃
bash exp.sh --gap_threshold 0.05
```

例如：
- Strategy A: 0.40
- Strategy B: 0.38  (差距 0.02 < 0.05，保留)
- Strategy C: 0.32  (差距 0.06 > 0.05，舍弃C及后续)

## 进度显示

实验运行时会有进度显示：
- Shell脚本: 显示 `[当前/总数]` 和进度条
- Python脚本: 使用 `tqdm` 显示实时进度条

## 策略列表

实验包含24种有区分度的重写策略：

1. urgent_situation - 紧急情境
2. academic_research - 学术研究
3. creative_writing - 创意写作
4. cybersecurity_defense - 网络安全防御
5. historical_analysis - 历史分析
6. legal_compliance - 法律合规
7. psychological_counseling - 心理咨询
8. medical_research - 医学研究
9. journalistic_investigation - 新闻调查
10. policy_analysis - 政策分析
11. role_playing - 角色扮演
12. hypothetical_scenario - 假设场景
13. technical_documentation - 技术文档
14. reverse_engineering - 逆向工程
15. debate_preparation - 辩论准备
16. ethics_discussion - 伦理讨论
17. risk_assessment - 风险评估
18. training_materials - 培训材料
19. data_protection - 数据保护
20. social_engineering_defense - 社会工程防御
21. comparative_analysis - 对比分析
22. case_study - 案例研究
23. system_testing - 系统测试
24. red_teaming - 红队演练
