#!/bin/bash
# =============================================================================
# 一次性环境搭建 (临时 GPU 环境 / 无历史 venv 时)
# =============================================================================
# 依赖清单唯一来源 = 仓库根 requirements.txt (版本严格复刻历史环境)。
#
# Usage:
#   bash docs/cookbook/bootstrap_env.sh              # 建 venv + 装依赖 + 冻结 lock
#   DOWNLOAD_MODELS=1 bash docs/cookbook/bootstrap_env.sh   # 追加下载模型
#   HF_ENDPOINT=https://hf-mirror.com DOWNLOAD_MODELS=1 bash docs/cookbook/bootstrap_env.sh
# =============================================================================
set -e

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY_VERSION="${PY_VERSION:-3.12}"
VENV="${VENV:-$REPO/.venv}"

# -----------------------------------------------------------------------------
# 1) 兼容脚本内硬编码的 PROJECT_ROOT=/home/tiger/jailbreak_research
#    (common.sh 里 PROJECT_ROOT/PATH 不可用环境变量覆盖, 用符号链接零改动适配)
# -----------------------------------------------------------------------------
HIST_LINK="${HIST_LINK:-/home/tiger/jailbreak_research}"
if [ "$REPO" != "$HIST_LINK" ]; then
    if [ -L "$HIST_LINK" ]; then
        echo "[1/5] 符号链接已存在: $HIST_LINK -> $(readlink "$HIST_LINK")"
    elif [ -e "$HIST_LINK" ]; then
        echo "⚠️  $HIST_LINK 已存在且不是符号链接, 跳过 (确认内容后再处理)"
    else
        mkdir -p "$(dirname "$HIST_LINK")"
        ln -s "$REPO" "$HIST_LINK"
        echo "[1/5] 已建立符号链接 $HIST_LINK -> $REPO"
    fi
else
    echo "[1/5] 仓库就在约定路径, 无需链接"
fi

# -----------------------------------------------------------------------------
# 2) uv (临时环境首选: 自带解释器管理 + 无需 ensurepip)
# -----------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "[2/5] 安装 uv"
    pip install --user -q uv
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
export PATH="$(dirname "$UV"):$PATH"
echo "[2/5] uv $($UV --version | awk '{print $2}')"

# -----------------------------------------------------------------------------
# 3) venv + 依赖
# -----------------------------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
    echo "[3/5] 创建 venv (python $PY_VERSION) -> $VENV"
    "$UV" venv --python "$PY_VERSION" "$VENV"
else
    echo "[3/5] venv 已存在, 复用: $VENV"
fi

echo "[3/5] 安装 requirements.txt"
if ! "$UV" pip install -p "$VENV/bin/python" -r requirements.txt; then
    echo "⚠️  整体解析失败, 回退: 核心依赖正常装 + xformers/liger 用 --no-deps"
    grep -vE '^(xformers|liger-kernel)==' requirements.txt > /tmp/req_core.txt
    "$UV" pip install -p "$VENV/bin/python" -r /tmp/req_core.txt
    "$UV" pip install -p "$VENV/bin/python" --no-deps xformers==0.0.35 liger-kernel==0.8.2
fi

# -----------------------------------------------------------------------------
# 4) 冻结 lock (临时环境易重建, 留一份精确版本存证)
# -----------------------------------------------------------------------------
"$UV" pip freeze -p "$VENV/bin/python" > requirements.lock.txt
echo "[4/5] 已写 requirements.lock.txt ($(wc -l < requirements.lock.txt) 行)"
echo "      关键版本: $(grep -E '^(torch|vllm|ms-swift|trl|transformers|xformers|liger-kernel)==' requirements.lock.txt | tr '\n' ' ')"

# -----------------------------------------------------------------------------
# 5) 模型 (可选)
# -----------------------------------------------------------------------------
if [ "${DOWNLOAD_MODELS:-0}" = "1" ]; then
    echo "[5/5] 下载模型 -> $REPO/model/ (HF_ENDPOINT=${HF_ENDPOINT:-默认})"
    HF="${VENV}/bin/hf"
    [ -x "$HF" ] || HF="${VENV}/bin/huggingface-cli"
    bash docs/cookbook/download_models.sh
else
    echo "[5/5] 跳过模型下载 (DOWNLOAD_MODELS=1 开启)"
fi

echo "============================================================================="
echo "完成。运行时:"
echo "  source $VENV/bin/activate"
echo "  模型在 $REPO/model/ 时 common.sh 会自动探测, 无需导出环境变量"
echo "============================================================================="
