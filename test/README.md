# SafeRL 模型测试

本目录包含 Qwen3-4B-SafeRL 模型的部署和测试脚本。

## 文件说明

- `start_safarl.sh` - 启动 vLLM 服务
- `test_safarl.sh` - 测试模型推理（支持 thinking/no_think 模式）

## 环境要求

- 虚拟环境：`/home/tiger/jailbreak_research/.venv`
- GPU：至少 1 张 GPU（推荐 A100 80GB）
- 模型路径：`/home/tiger/models/Qwen/Qwen3-4B-SafeRL`

## 使用方法

### 1. 启动服务

```bash
bash test/start_safarl.sh
```

服务将在 `0.0.0.0:8000` 启动，使用 GPU 0。

启动参数：
- `--max-model-len 8192` - 最大序列长度
- `--gpu-memory-utilization 0.9` - GPU 显存利用率
- `--trust-remote-code` - 信任远程代码

### 2. 测试模型

#### 默认模式（启用思考）

```bash
bash test/test_safarl.sh
```

#### No-think 模式（关闭思考）

```bash
bash test/test_safarl.sh --no_think
```

### 3. API 调用示例

```bash
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/tiger/models/Qwen/Qwen3-4B-SafeRL",
    "messages": [
      {"role": "user", "content": "/no_think 你好"}
    ],
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

## 服务状态检查

```bash
curl http://0.0.0.0:8000/health
```

## 日志查看

服务日志输出到：`/tmp/vllm_safereal.log`

```bash
tail -f /tmp/vllm_safereal.log
```

## 停止服务

```bash
pkill -f "vllm.*Qwen3-4B-SafeRL"
```

## 注意事项

1. 启动服务需要约 2-3 分钟（模型加载 + torch.compile）
2. 首次启动会进行 torch.compile 编译，后续启动会使用缓存
3. 服务使用虚拟环境中的 vllm 版本（0.18.0）
4. 环境变量 `FLASHINFER_DISABLE_VERSION_CHECK=1` 用于绕过版本检查

## 故障排查

### 服务启动失败

检查日志：
```bash
cat /tmp/vllm_safereal.log
```

常见错误：
- GPU 显存不足：调整 `--gpu-memory-utilization` 参数
- 端口被占用：修改 `--port` 参数或停止其他服务
- 模型路径错误：确认模型已下载到正确位置

### 推理超时

- 检查 GPU 利用率：`nvidia-smi`
- 检查服务日志：`tail -f /tmp/vllm_safereal.log`
- 重启服务：先停止，再重新启动

## 性能参考

- 模型加载时间：约 18 秒
- torch.compile 时间：约 35 秒（首次）
- GPU 显存占用：约 8.3 GB
- 推理速度：取决于序列长度和 batch size
