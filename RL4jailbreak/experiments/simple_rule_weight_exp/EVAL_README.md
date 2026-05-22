# Simple Rule Weight Experiments - Eval 修复说明

## 问题描述

之前的 eval 结果中出现 `[Exception] APIConnectionError: Connection error.` 错误，原因是：
- `eval.py` 使用 `launch_server=False` 连接已有的 vLLM 服务
- 但实际上 vLLM 服务并未启动
- 错误被 `return_exceptions=True` 捕获并当作文本写入 JSONL 文件

## 解决方案

新增 `eval.sh` 脚本，统一管理 vLLM server 生命周期：

### 文件结构
```
experiments/simple_rule_weight_exp/
├── exp.sh          # 训练脚本（3个实验依次运行）
├── eval.sh         # 评估脚本（批量评估所有 final_lora）
├── simple_rule_weight_grpo.py  # 训练主脚本
└── output/
    ├── exp1/
    │   ├── final_lora/          # 训练保存的 LoRA 权重
    │   ├── logs/                # 训练和 vLLM 日志
    │   └── eval_results/        # eval.sh 生成的评估结果
    ├── exp2/
    │   ├── final_lora/
    │   └── eval_results/
    ├── exp3/
    │   ├── final_lora/
    │   └── eval_results/
    └── eval_summary_all.json   # 汇总报告
```

## 使用方法

### 方式 1：训练 + 评估一键完成
```bash
bash experiments/simple_rule_weight_exp/exp.sh
```
这会：
1. 依次运行 3 个训练实验
2. 每个实验完成后保存 final_lora
3. 所有训练完成后，自动调用 `eval.sh` 批量评估所有 checkpoint

### 方式 2：单独运行评估
```bash
# 评估所有已有的 final_lora
bash experiments/simple_rule_weight_exp/eval.sh

# 评估指定的实验
bash experiments/simple_rule_weight_exp/eval.sh exp1 exp2
```

### 方式 3：单独训练（不评估）
```bash
# 运行所有训练
bash experiments/simple_rule_weight_exp/exp.sh

# 或只运行某个实验
bash experiments/simple_rule_weight_exp/exp.sh 1  # 只运行 exp1
```

## eval.sh 工作流程

对于每个实验：

```
1. 检查 final_lora 是否存在
   ↓
2. 清理之前失败的 eval 结果（包含 APIConnectionError 的文件）
   ↓
3. 检查 eval 是否已成功完成（如果有则跳过）
   ↓
4. 启动 vLLM servers:
   - Policy (GPU0, port 8003, 带 LoRA)
   - Target (GPU1, port 8001, base model)
   - Guard  (GPU1, port 8002, base model)
   ↓
5. 等待所有 server 就绪（最多 120s）
   ↓
6. 运行 eval.py（连接已有服务）
   ↓
7. 停止所有 vLLM servers
   ↓
8. 继续下一个实验
```

## 失败结果清理

`eval.sh` 会自动检测并清理：
- 包含 `APIConnectionError` 的 JSONL 文件
- 包含 `Connection error` 的 ASR report 文件
- 不完整的 combo 目录（缺少 summary.json 或空文件）

如果 eval 已成功完成，会跳过该实验。

## 评估结果

评估完成后生成：

### 1. 详细结果（每个实验）
```
output/exp1/eval_results/eval_exp1/
├── combo_001_final_lora_hypothetical_scenario/
│   ├── rewritten_final_lora.jsonl    # 重写后的 prompts
│   ├── asr_report_final_lora.json    # ASR 统计
│   └── summary.json                  # 本次评估摘要
└── eval.log                          # 评估日志
```

### 2. 汇总报告
```json
{
  "exp1": {"status": "success", "asr": "0.2700", ...},
  "exp2": {"status": "success", "asr": "0.2500", ...},
  "exp3": {"status": "success", "asr": "0.2600", ...}
}
```

文本格式：
```
Experiment           | ASR        | Refusal    | Partial    | Success    | Total     
------------------------------------------------------------------------------------------
exp1                 | 0.2700     | 0.5000     | 0.1000     | 0.4000     | 1000      
exp2                 | 0.2500     | 0.5200     | 0.1100     | 0.3700     | 1000      
exp3                 | 0.2600     | 0.5100     | 0.0900     | 0.4000     | 1000      
```

## 处理已有的错误 eval 结果

如果你已经有包含 `APIConnectionError` 的旧 eval 结果：

### 方法 1：让 eval.sh 自动清理
```bash
bash experiments/simple_rule_weight_exp/eval.sh
```
脚本会自动检测并删除失败的文件，然后重新评估。

### 方法 2：手动清理后重新评估
```bash
# 删除所有旧的 eval 结果
rm -rf experiments/simple_rule_weight_exp/output/*/eval_results

# 重新评估
bash experiments/simple_rule_weight_exp/eval.sh
```

### 方法 3：只清理包含错误的文件
```bash
# 查找并删除包含 APIConnectionError 的文件
find experiments/simple_rule_weight_exp/output -name "*.jsonl" -exec grep -l "APIConnectionError" {} \; -delete

# 重新评估
bash experiments/simple_rule_weight_exp/eval.sh
```

## vLLM Server 管理

`eval.sh` 包含完整的 server 管理功能：

- `check_vllm_server port` - 检查 server 是否就绪
- `launch_vllm_server model port gpu_id log_file` - 启动 server
- `wait_for_server port name` - 等待 server 就绪
- `kill_vllm_server port name` - 停止 server

所有 server 日志保存在：
```
output/expX/logs/vllm_policy_eval.log
output/expX/logs/vllm_target_eval.log
output/expX/logs/vllm_guard_eval.log
```

## 注意事项

1. **GPU 资源**: eval 时需要 2 个 GPU（GPU0 给 policy，GPU1 给 target+guard）
2. **端口占用**: 确保端口 8001、8002、8003 未被占用
3. **模型路径**: 确保模型路径与训练时一致
4. **失败重试**: 如果某个实验 eval 失败，脚本会继续评估下一个，最后生成汇总报告
