# Data Generation for Thesis Chapter 2

## 目标

使用 Skills 系统生成合成数据：
- **输入**：`data/dataset/processed/10k/train.jsonl`（8000条）
- **输出**：种子 prompt → 最终攻击成功 prompt 数据集
- **方法**：pair_skills_28 和 autodan_skills_54

---

## 数据来源

| 文件 | 数量 | 格式 |
|------|------|------|
| **train.jsonl** | **8000** | JSONL（id + prompt） |

---

## 目录结构

```
exp/data_generate/
├── README.md           # 本文档
├── generate_data.py    # 数据生成脚本
├── run_generate.sh     # 运行脚本（自动启动服务）
├── results/            # 输出目录
│   ├── pair_skills_28/
│   │   ├── success_cases.json   # 成功案例
│   │   └── summary.json         # 统计信息
│   └── autodan_skills_54/
│   │   ├── success_cases.json
│   │   └── summary.json
│   └── ALL_SUMMARY.json         # 全部汇总
└── logs/               # vLLM 服务日志
    ├── guard.log
    ├── attacker.log
    └── target.log
```

---

## 使用方法

### 1. 完整运行（自动启动服务）

```bash
# 生成 pair_skills_28 数据
bash run_generate.sh --method pair_skills_28

# 生成 autodan_skills_54 数据
bash run_generate.sh --method autodan_skills_54

# 生成两种方法的数据
bash run_generate.sh --method both

# 指定并发数
bash run_generate.sh --method both --max_workers 64
```

### 2. 使用已有服务

如果 vLLM 服务已启动（端口 8001, 8002, 8003）：

```bash
# 直接运行生成脚本
python generate_data.py --method pair_skills_28
python generate_data.py --method autodan_skills_54
python generate_data.py --method both
```

---

## 输出格式

### success_cases.json

```json
[
  {
    "id": "22041",
    "original_prompt": "As a creative writing assistant...",
    "attack_prompt": "You are an expert in creative writing...",
    "iterations": 3,
    "skill_used": "skill_extracted_6",
    "skill_content": "You are an expert in [field]...",
    "trajectory": [...],
    "target_response": "...",
    "guard_label": "unsafe"
  },
  ...
]
```

### summary.json

```json
{
  "method": "pair_skills_28",
  "total_prompts": 8000,
  "success_count": 6400,
  "failed_count": 1600,
  "success_rate": 0.80,
  "avg_iterations": 3.5,
  "skills_count": 28,
  "data_source": "data/dataset/processed/10k/train.jsonl"
}
```

---

## Skills 配置

| 方法 | Skills 来源 | 数量 | 调用模式 |
|------|-------------|------|----------|
| **pair_skills_28** | Layer 1 演化 | 28 | single_call |
| **autodan_skills_54** | Layer 4 演化 | 54 | every_iteration |

---

## GPU 分配

| GPU | 角色 | 模型 | 端口 |
|-----|------|------|------|
| GPU 0 | Guard | Qwen3Guard-Gen-4B | 8002 |
| GPU 1 | Attacker | Qwen3-4B | 8003 |
| GPU 2,3 | Target | Qwen3-4B (TP=2) | 8001 |

---

## 预期结果

基于之前的实验结果：

| 方法 | 预期成功率 | 预期平均迭代 |
|------|-----------|-------------|
| pair_skills_28 | ~80% | 3-4 次 |
| autodan_skills_54 | ~99% | 1-2 次 |

---

## 时间预估（8000条数据）

| 方法 | 数据量 | 预期时间 |
|------|--------|----------|
| pair_skills_28 | 8000 | ~8 小时 |
| autodan_skills_54 | 8000 | ~4 小时 |

---

*Created: 2026-06-10*