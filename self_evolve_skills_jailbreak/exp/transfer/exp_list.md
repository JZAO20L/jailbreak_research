# Target Models (4个)

需要下载到 `/home/tiger/models/<家族>/<模型名>` 目录。

## 模型下载命令

### 方式1: HuggingFace CLI
```bash
# 创建目录结构
mkdir -p /home/tiger/models/Qwen
mkdir -p /home/tiger/models/google
mkdir -p /home/tiger/models/openai-mirror

# Qwen3-0.6B
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /home/tiger/models/Qwen/Qwen3-0.6B

# Qwen3-14B-FP8
huggingface-cli download Qwen/Qwen3-14B-FP8 --local-dir /home/tiger/models/Qwen/Qwen3-14B-FP8

# gemma-4-12B-it
huggingface-cli download google/gemma-4-12B-it --local-dir /home/tiger/models/google/gemma-4-12B-it

# gpt-oss-20b
huggingface-cli download openai-mirror/gpt-oss-20b --local-dir /home/tiger/models/openai-mirror/gpt-oss-20b
```

### 方式2: ModelScope (国内推荐)
```bash
# 设置环境变量
export VLLM_USE_MODELSCOPE=true

# 创建目录结构
mkdir -p /home/tiger/models/Qwen
mkdir -p /home/tiger/models/google
mkdir -p /home/tiger/models/openai-mirror

# Qwen3-0.6B
modelscope download --model Qwen/Qwen3-0.6B --local_dir /home/tiger/models/Qwen/Qwen3-0.6B

# Qwen3-14B-FP8
modelscope download --model Qwen/Qwen3-14B-FP8 --local_dir /home/tiger/models/Qwen/Qwen3-14B-FP8

# gemma-4-12B-it
modelscope download --model AI-ModelScope/gemma-4-12B-it --local_dir /home/tiger/models/google/gemma-4-12B-it

# gpt-oss-20b
modelscope download --model LLM-Research/gpt-oss-20b --local_dir /home/tiger/models/openai-mirror/gpt-oss-20b
```

### 方式3: Python脚本
```python
from huggingface_hub import snapshot_download

models = [
    ("Qwen/Qwen3-0.6B", "/home/tiger/models/Qwen/Qwen3-0.6B"),
    ("Qwen/Qwen3-14B-FP8", "/home/tiger/models/Qwen/Qwen3-14B-FP8"),
    ("google/gemma-4-12B-it", "/home/tiger/models/google/gemma-4-12B-it"),
    ("openai-mirror/gpt-oss-20b", "/home/tiger/models/openai-mirror/gpt-oss-20b"),
]

for repo_id, local_dir in models:
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"Downloaded: {repo_id} -> {local_dir}")
```

## 模型列表
| HuggingFace ID | 本地路径 | 规模 |
|----------------|----------|------|
| Qwen/Qwen3-0.6B | /home/tiger/models/Qwen/Qwen3-0.6B | 0.6B |
| Qwen/Qwen3-14B-FP8 | /home/tiger/models/Qwen/Qwen3-14B-FP8 | 14B |
| google/gemma-4-12B-it | /home/tiger/models/google/gemma-4-12B-it | 12B |
| openai-mirror/gpt-oss-20b | /home/tiger/models/openai-mirror/gpt-oss-20b | 20B |

---

# Benchmark Datasets (4个)

已在 `jailbreak_research/data/benchmark` 目录下转换为 jsonl 格式：

| 文件 | 来源 | 数量 |
|------|------|------|
| advbench.jsonl | AdvBench | 520 |
| harmbench_contextual.jsonl | HarmBench Contextual | 100 |
| harmbench_standard.jsonl | HarmBench Standard | 200 |
| jailbreakBench.jsonl | JailbreakBench | 100 |