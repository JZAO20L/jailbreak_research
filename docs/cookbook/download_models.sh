#!/bin/bash
# =============================================================================
# 下载第三章实验所需模型 (默认放 jailbreak_research/model/)
#   只需两个模型: Qwen3-4B (policy+target) + Qwen3Guard-Gen-4B (guard)
#   国内镜像: HF_ENDPOINT=https://hf-mirror.com bash docs/cookbook/download_models.sh
# Usage: bash docs/cookbook/download_models.sh
# =============================================================================
set -e
cd "$(dirname "$0")/../.."   # jailbreak_research/
ROOT="$PWD"
mkdir -p "$ROOT/model"
source "$ROOT/.venv/bin/activate"

# huggingface_hub 1.x 起 huggingface-cli 改名为 hf, 两个都试
if command -v hf >/dev/null 2>&1; then HF_BIN=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HF_BIN=huggingface-cli
else echo "ERROR: 找不到 hf / huggingface-cli, 先 pip install -U huggingface_hub"; exit 1; fi

download() {
    local repo="$1" dir="$2"
    if [ -f "$ROOT/$dir/config.json" ]; then
        echo "[skip] $dir 已存在"
        return
    fi
    echo "== 下载 $repo -> $dir =="
    "$HF_BIN" download "$repo" --local-dir "$ROOT/$dir"
}

download Qwen/Qwen3-4B          model/Qwen3-4B
download Qwen/Qwen3Guard-Gen-4B model/Qwen3Guard-Gen-4B

echo "== 校验 =="
for d in model/Qwen3-4B model/Qwen3Guard-Gen-4B; do
    [ -f "$ROOT/$d/config.json" ] && echo "OK: $d" || { echo "MISSING: $d"; exit 1; }
done
echo "完成。运行时导出环境变量:"
echo "  export BASE_MODEL=$ROOT/model/Qwen3-4B"
echo "  export GUARD_MODEL=$ROOT/model/Qwen3Guard-Gen-4B"
echo "  export TARGET_MODEL=$ROOT/model/Qwen3-4B"
