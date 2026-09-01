#!/bin/bash
# 测试 Qwen3-4B-SafeRL 模型
# 用法: bash test_safarl.sh [--no_think]
#
# 前置条件:
#   - 使用虚拟环境: /home/tiger/jailbreak_research/.venv
#   - 启动服务: bash test/start_safarl.sh

set -e

MODEL_PATH="/home/tiger/models/Qwen/Qwen3-4B-SafeRL"
PORT=8000
HOST="0.0.0.0"
VENV_PATH="/home/tiger/jailbreak_research/.venv"

# 解析参数
NO_THINK=""
if [[ "$1" == "--no_think" ]]; then
    NO_THINK="/no_think "
    echo "启用 /no_think 模式（关闭思考）"
else
    echo "默认模式（启用思考）"
fi

echo "=========================================="
echo "测试 Qwen3-4B-SafeRL 模型"
echo "模型路径: $MODEL_PATH"
echo "端口: $PORT"
echo "虚拟环境: $VENV_PATH"
echo "=========================================="

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型目录不存在: $MODEL_PATH"
    exit 1
fi

# 检查服务是否运行
echo "检查服务状态..."
if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
    echo "✓ 服务已运行"
else
    echo "✗ 服务未运行，请先启动服务:"
    echo "  bash test/start_safarl.sh"
    exit 1
fi

# 测试推理
echo ""
echo "=========================================="
echo "测试推理"
echo "=========================================="

TEST_PROMPT="${NO_THINK}请解释什么是强化学习？"

echo "测试 Prompt: $TEST_PROMPT"
echo ""

RESPONSE=$(curl -s "http://$HOST:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_PATH\",
        \"messages\": [
            {\"role\": \"user\", \"content\": \"$TEST_PROMPT\"}
        ],
        \"max_tokens\": 500,
        \"temperature\": 0.7
    }")

echo "响应:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

# 提取生成的文本
GENERATED_TEXT=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'choices' in data and len(data['choices']) > 0:
        print(data['choices'][0]['message']['content'])
    else:
        print('无法解析响应')
except Exception as e:
    print(f'解析错误: {e}')
" 2>/dev/null)

echo ""
echo "=========================================="
echo "生成的文本:"
echo "=========================================="
echo "$GENERATED_TEXT"

echo ""
echo "=========================================="
echo "✓ 测试完成"
echo "=========================================="

