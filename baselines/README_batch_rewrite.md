# Batch Rewrite Script - 使用阿里云百炼API

## 功能

使用阿里云百炼API（Qwen3-8B）调用baseline方法对test集进行rewrite，合成新的攻击prompt数据集。

## 前置要求

1. **API Key**: 阿里云百炼API Key
   - 从`.env`文件中获取`DASHSCOPE_API_KEY`
   - 或在命令行中指定`--api-key`

2. **依赖**: OpenAI Python SDK
   ```bash
   pip install openai
   ```

## 使用方法

### 1. 基本用法

```bash
# 使用默认参数
python baselines/batch_rewrite_api.py

# 默认参数：
# - 输入: data/dataset/processed/10k/test.jsonl
# - 输出: baselines/output
# - 策略: deepinception, multilingual, pair, genetic
# - 模型: qwen3-8b
```

### 2. 自定义参数

```bash
# 指定输入文件和输出目录
python baselines/batch_rewrite_api.py \
    --input data/dataset/processed/10k/test.jsonl \
    --output baselines/output

# 只使用特定策略
python baselines/batch_rewrite_api.py \
    --strategies deepinception multilingual

# 限制处理数量（用于测试）
python baselines/batch_rewrite_api.py \
    --limit 100

# 指定API Key（如果环境变量未设置）
python baselines/batch_rewrite_api.py \
    --api-key "sk-xxx"

# 使用不同的模型
python baselines/batch_rewrite_api.py \
    --model qwen-plus
```

### 3. 完整命令示例

```bash
# 处理完整test集，使用所有baseline方法
python baselines/batch_rewrite_api.py \
    --input data/dataset/processed/10k/test.jsonl \
    --output baselines/output \
    --strategies deepinception multilingual pair genetic \
    --model qwen3-8b

# 快速测试（只处理100条）
python baselines/batch_rewrite_api.py \
    --limit 100 \
    --strategies deepinception multilingual
```

## 支持的策略

根据要求，排除了`no_rewrite`和`template_rewrite`，支持以下策略：

| 策略 | 描述 | 是否需要API调用 | 输出文件 |
|------|------|----------------|----------|
| **deepinception** | 使用角色扮演和虚构场景绕过安全过滤器 | ❌ 纯模板方法 | `baselines/output/deepinception.jsonl` |
| **multilingual** | 翻译到低资源语言绕过安全过滤器 | ✅ 需要API翻译 | `baselines/output/multilingual.jsonl` |
| **pair** | PAIR迭代优化方法 | ✅ 需要API迭代 | `baselines/output/pair.jsonl` |
| **genetic** | 基于遗传算法的优化方法 | ✅ 需要API迭代 | `baselines/output/genetic.jsonl` |

## 输出格式

每个策略生成一个JSONL文件，包含以下字段：

```json
{
  "id": "原始prompt ID",
  "original_prompt": "原始攻击prompt",
  "attack_prompt": "rewrite后的攻击prompt",
  "strategy": "使用的策略名称",
  "source": "数据来源",
  "original_label": "原始标签"
}
```

同时生成summary文件：

```json
{
  "strategy": "deepinception",
  "total": 1000,
  "success": 1000,
  "input": "data/dataset/processed/10k/test.jsonl",
  "output": "baselines/output/deepinception.jsonl",
  "model": "qwen3-8b"
}
```

## 输出目录结构

```
baselines/output/
├── deepinception.jsonl          # DeepInception方法结果
├── deepinception_summary.json   # DeepInception统计
├── multilingual.jsonl           # Multilingual方法结果
├── multilingual_summary.json    # Multilingual统计
├── pair.jsonl                   # PAIR方法结果
├── pair_summary.json            # PAIR统计
├── genetic.jsonl                # Genetic方法结果
└── genetic_summary.json         # Genetic统计
```

## API调用说明

### 阿里云百炼API

- **Base URL**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Model**: `qwen3-8b`
- **API Key**: 从环境变量`DASHSCOPE_API_KEY`获取

### 调用限制

- 每个API调用后延迟0.5秒（避免rate limit）
- 超时时间60秒
- Temperature: 0.7
- Max tokens: 2048

## 时间预估

根据策略不同，处理1000条prompts的时间：

| 策略 | 预估时间 | 原因 |
|------|----------|------|
| **deepinception** | ~10分钟 | 纯模板方法，无需API调用 |
| **multilingual** | ~30分钟 | 每条需要1-2次API翻译 |
| **pair** | ~2-4小时 | 每条需要迭代优化（最多20次） |
| **genetic** | ~2-4小时 | 每条需要进化优化（多代） |

**建议**：先使用`--limit 100`测试，确认正常后再处理全量数据。

## 注意事项

1. **API Key**: 确保`.env`文件中有`DASHSCOPE_API_KEY`
2. **Rate Limit**: 内置延迟避免触发API限制
3. **超时处理**: API超时会跳过当前prompt继续处理
4. **进度保存**: 实时写入文件，中断后可检查部分结果

## 示例：快速测试

```bash
# 测试API是否正常
python baselines/batch_rewrite_api.py --limit 10 --strategies deepinception

# 测试multilingual方法
python baselines/batch_rewrite_api.py --limit 10 --strategies multilingual

# 测试所有方法
python baselines/batch_rewrite_api.py --limit 50
```

## 下一步

生成的rewrite数据可用于：
1. 作为训练数据扩充训练集
2. 评估不同baseline方法的攻击效果
3. 与RL训练的模型进行对比分析