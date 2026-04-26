#!/bin/bash
# Jailbreak Prompt 实验脚本 - 实验1
# 对所有策略在test集上进行ASR测试
#
# 用法:
#   bash experiments/jailbreak_prompt_exp/exp.sh                          # 运行所有策略
#   bash experiments/jailbreak_prompt_exp/exp.sh --topk 5                 # 只输出top5策略
#   bash experiments/jailbreak_prompt_exp/exp.sh --gap_threshold 0.05     # gap>5%时舍弃
#   bash experiments/jailbreak_prompt_exp/exp.sh s1 s2                    # 运行指定策略
#   TEST_SET=/path/to/test.jsonl bash experiments/jailbreak_prompt_exp/exp.sh

set -e

# =========================
# 配置区 - 可根据需要修改
# =========================

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 测试集路径 (使用test集而非val集)
TEST_SET="${TEST_SET:-$BASE_DIR/../data/dataset/processed/10k/test.jsonl}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"

# 模型路径
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
TARGET_MODEL="${TARGET_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B}"

# vLLM端口
POLICY_PORT=8003
TARGET_PORT=8001
GUARD_PORT=8002

# 生成配置
K=1
REWRITE_TEMP=0.7
REWRITE_MAX_TOKENS=2048
MAX_MODEL_LEN=4096

# 两次评估之间的等待时间 (秒)
SLEEP_BETWEEN_EVALS=10

# =========================
# 参数解析
# =========================
TOPK=""
GAP_THRESHOLD=""
SELECTED_STRATEGIES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --topk)
            TOPK="$2"
            shift 2
            ;;
        --gap_threshold)
            GAP_THRESHOLD="$2"
            shift 2
            ;;
        --test_set)
            TEST_SET="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --sleep)
            SLEEP_BETWEEN_EVALS="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项] [策略1 策略2 ...]"
            echo ""
            echo "选项:"
            echo "  --topk N                只输出表现最好的N个策略"
            echo "  --gap_threshold FLOAT   当策略间ASR差距大于此值时，舍弃后续策略 (如 0.05 表示5%)"
            echo "  --test_set PATH         测试集路径"
            echo "  --output_dir PATH       输出目录"
            echo "  --sleep SECONDS         两次评估之间的等待时间"
            echo "  --help                  显示帮助"
            exit 0
            ;;
        *)
            SELECTED_STRATEGIES+=("$1")
            shift
            ;;
    esac
done

# =========================
# 策略列表 (24种 + 原始基线)
# =========================
# 如果传入了命令行参数，使用传入的策略；否则使用默认全部策略
if [ ${#SELECTED_STRATEGIES[@]} -gt 0 ]; then
    STRATEGIES=("${SELECTED_STRATEGIES[@]}")
else
    STRATEGIES=(
        "urgent_situation"
        "academic_research"
        "creative_writing"
        "cybersecurity_defense"
        "historical_analysis"
        "legal_compliance"
        "psychological_counseling"
        "medical_research"
        "journalistic_investigation"
        "policy_analysis"
        "role_playing"
        "hypothetical_scenario"
        "technical_documentation"
        "reverse_engineering"
        "debate_preparation"
        "ethics_discussion"
        "risk_assessment"
        "training_materials"
        "data_protection"
        "social_engineering_defense"
        "comparative_analysis"
        "case_study"
        "system_testing"
        "red_teaming"
    )
fi

# =========================
# 辅助函数
# =========================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# 进度条显示函数
print_progress() {
    local current=$1
    local total=$2
    local pct=$((current * 100 / total))
    local filled=$((pct / 2))
    local empty=$((50 - filled))

    printf -v bar '%*s' "$filled" ''
    bar=${bar// /#}
    printf -v spaces '%*s' "$empty" ''

    printf "\r  [%s%s] %d%% (%d/%d)" "$bar" "$spaces" "$pct" "$current" "$total"
}

# =============================================================================
# GPU状态检查和模型加载函数
# =============================================================================

# 检查GPU1状态
check_gpu1_status() {
    local gpu1_memory
    gpu1_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d '[:space:]')

    if [ -z "$gpu1_memory" ]; then
        echo "error"
        return 1
    fi

    if [ "$gpu1_memory" -lt 2000 ]; then
        echo "idle"
    elif [ "$gpu1_memory" -gt 15000 ]; then
        echo "loaded"
    else
        echo "busy"
    fi
}

# 检查指定端口是否有vLLM服务在运行
check_port_active() {
    local port=$1
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q "200"; then
        return 0
    else
        return 1
    fi
}

# 释放GPU1显存
release_gpu1_memory() {
    log "检查GPU1状态..."

    local status
    status=$(check_gpu1_status)
    log "GPU1当前状态: $status"

    case $status in
        "idle")
            log "GPU1空闲,无需释放"
            ;;
        "loaded")
            log "GPU1已加载模型,检查服务是否活跃..."

            local target_active=false
            local guard_active=false

            if check_port_active "$TARGET_PORT"; then
                target_active=true
                log "Target服务已在端口 $TARGET_PORT 运行"
            fi

            if check_port_active "$GUARD_PORT"; then
                guard_active=true
                log "Guard服务已在端口 $GUARD_PORT 运行"
            fi

            if $target_active && $guard_active; then
                log "所有服务已就绪,无需重新启动"
                return 0
            else
                log "部分服务未就绪,需要重新启动"
                return 1
            fi
            ;;
        "busy")
            log "GPU1非空闲但可能没加载目标模型"
            log "建议手动检查或重启服务"
            return 1
            ;;
        "error")
            log "无法获取GPU状态"
            return 1
            ;;
    esac

    return 0
}

# 启动Target模型服务 (GPU1)
start_target_service() {
    log "启动Target模型服务 (端口: $TARGET_PORT)..."

    if check_port_active "$TARGET_PORT"; then
        log "Target服务已在运行,跳过启动"
        return 0
    fi

    if [ -f "$BASE_DIR/scripts/start_target.sh" ]; then
        log "执行 start_target.sh..."
        bash "$BASE_DIR/scripts/start_target.sh" --port "$TARGET_PORT" --gpu 1 &

        log "等待Target服务启动..."
        for i in $(seq 1 60); do
            if check_port_active "$TARGET_PORT"; then
                log "Target服务启动成功!"
                return 0
            fi
            sleep 2
        done

        log "Target服务启动超时!"
        return 1
    else
        log "错误: 找不到 start_target.sh 脚本"
        log "请手动启动Target模型服务到端口 $TARGET_PORT"
        return 1
    fi
}

# 启动Guard模型服务 (GPU1)
start_guard_service() {
    log "启动Guard模型服务 (端口: $GUARD_PORT)..."

    if check_port_active "$GUARD_PORT"; then
        log "Guard服务已在运行,跳过启动"
        return 0
    fi

    if [ -f "$BASE_DIR/scripts/start_guard.sh" ]; then
        log "执行 start_guard.sh..."
        bash "$BASE_DIR/scripts/start_guard.sh" --port "$GUARD_PORT" --gpu 1 &

        log "等待Guard服务启动..."
        for i in $(seq 1 60); do
            if check_port_active "$GUARD_PORT"; then
                log "Guard服务启动成功!"
                return 0
            fi
            sleep 2
        done

        log "Guard服务启动超时!"
        return 1
    else
        log "错误: 找不到 start_guard.sh 脚本"
        log "请手动启动Guard模型服务到端口 $GUARD_PORT"
        return 1
    fi
}

# 启动Policy模型服务 (GPU0)
start_policy_service() {
    log "启动Policy模型服务 (端口: $POLICY_PORT)..."

    if check_port_active "$POLICY_PORT"; then
        log "Policy服务已在运行,跳过启动"
        return 0
    fi

    if [ -f "$BASE_DIR/scripts/start_policy.sh" ]; then
        log "执行 start_policy.sh..."
        bash "$BASE_DIR/scripts/start_policy.sh" --port "$POLICY_PORT" --gpu 0 &

        log "等待Policy服务启动..."
        for i in $(seq 1 60); do
            if check_port_active "$POLICY_PORT"; then
                log "Policy服务启动成功!"
                return 0
            fi
            sleep 2
        done

        log "Policy服务启动超时!"
        return 1
    else
        log "错误: 找不到 start_policy.sh 脚本"
        log "请手动启动Policy模型服务到端口 $POLICY_PORT"
        return 1
    fi
}

# 确保所有需要的服务都在运行
ensure_services_running() {
    log "============================================================"
    log "确保模型服务就绪..."
    log "============================================================"

    release_gpu1_memory
    start_target_service
    start_guard_service
    start_policy_service

    log "所有服务检查完成!"
}

# 从eval结果中提取ASR
extract_asr() {
    local output_dir=$1
    local summary_file="$output_dir/strategy_*/summary.json"
    
    # 查找summary.json文件
    local found_summary=""
    for f in "$output_dir"/*/summary.json; do
        if [ -f "$f" ]; then
            found_summary="$f"
            break
        fi
    done
    
    if [ -z "$found_summary" ]; then
        echo "0"
        return
    fi
    
    # 提取ASR (假设JSON中有 overall.asr 字段)
    python3 -c "
import json, sys
try:
    with open('$found_summary') as f:
        data = json.load(f)
    asr = data.get('combinations', [{}])[0].get('asr', 0)
    print(f'{asr:.4f}')
except:
    print('0')
" 2>/dev/null || echo "0"
}

# =========================
# 实验主流程
# =========================
TOTAL_EVALS=$((${#STRATEGIES[@]} + 1))  # +1 为原始prompt基线

log "============================================================"
log "Jailbreak Prompt 实验 - 实验1"
log "============================================================"
log "测试集: $TEST_SET"
log "策略数量: ${#STRATEGIES[@]}"
log "总评估次数: $TOTAL_EVALS (含原始基线)"
log "输出目录: $OUTPUT_DIR"
if [ -n "$TOPK" ]; then
    log "Top-K筛选: 只保留前 $TOPK 个策略"
fi
if [ -n "$GAP_THRESHOLD" ]; then
    log "Gap筛选: 差距阈值 = $GAP_THRESHOLD"
fi
log "============================================================"

# 创建实验输出目录
mkdir -p "$OUTPUT_DIR"

# ----------------------------------------------------------------
# Step 0: 确保模型服务就绪
# ----------------------------------------------------------------
ensure_services_running

# 记录实验开始时间
START_TIME=$(date +%s)

# 存储结果
declare -A ASR_RESULTS
CURRENT_EVAL=0

# ----------------------------------------------------------------
# Step 1: 测试原始prompt的ASR (基线)
# ----------------------------------------------------------------
CURRENT_EVAL=$((CURRENT_EVAL + 1))
print_progress $CURRENT_EVAL $TOTAL_EVALS
log ""
log "[$CURRENT_EVAL/$TOTAL_EVALS] 评估基线: 原始prompt (无重写)"
log "============================================================"

BASELINE_OUTPUT="$OUTPUT_DIR/baseline_original"
mkdir -p "$BASELINE_OUTPUT"

python "$BASE_DIR/scripts/eval.py" \
    --eval_path "$TEST_SET" \
    --base_model_path "$BASE_MODEL" \
    --target_model_path "$TARGET_MODEL" \
    --guard_model_path "$GUARD_MODEL" \
    --policy_port "$POLICY_PORT" \
    --target_port "$TARGET_PORT" \
    --guard_port "$GUARD_PORT" \
    --k "$K" \
    --rewrite_temperature "$REWRITE_TEMP" \
    --rewrite_max_tokens "$REWRITE_MAX_TOKENS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --output_root "$BASELINE_OUTPUT" \
    --run_name "baseline_original" \
    --prompt_ids "original"

BASELINE_ASR=$(extract_asr "$BASELINE_OUTPUT")
ASR_RESULTS["baseline_original"]="$BASELINE_ASR"
log "基线ASR: $BASELINE_ASR"

if [ "$CURRENT_EVAL" -lt "$TOTAL_EVALS" ]; then
    log "等待 $SLEEP_BETWEEN_EVALS 秒..."
    sleep "$SLEEP_BETWEEN_EVALS"
fi

# ----------------------------------------------------------------
# Step 2: 依次评估各策略
# ----------------------------------------------------------------
for i in "${!STRATEGIES[@]}"; do
    STRATEGY="${STRATEGIES[$i]}"
    CURRENT_EVAL=$((CURRENT_EVAL + 1))
    
    print_progress $CURRENT_EVAL $TOTAL_EVALS
    log ""
    log "[$CURRENT_EVAL/$TOTAL_EVALS] 评估策略: $STRATEGY"
    log "============================================================"
    
    # 创建策略输出目录
    STRATEGY_OUTPUT="$OUTPUT_DIR/$STRATEGY"
    mkdir -p "$STRATEGY_OUTPUT"
    
    # 调用eval脚本进行评估
    log "启动评估: $STRATEGY"
    log "输出目录: $STRATEGY_OUTPUT"
    
    python "$BASE_DIR/scripts/eval.py" \
        --eval_path "$TEST_SET" \
        --base_model_path "$BASE_MODEL" \
        --target_model_path "$TARGET_MODEL" \
        --guard_model_path "$GUARD_MODEL" \
        --policy_port "$POLICY_PORT" \
        --target_port "$TARGET_PORT" \
        --guard_port "$GUARD_PORT" \
        --k "$K" \
        --rewrite_temperature "$REWRITE_TEMP" \
        --rewrite_max_tokens "$REWRITE_MAX_TOKENS" \
        --max_model_len "$MAX_MODEL_LEN" \
        --output_root "$STRATEGY_OUTPUT" \
        --run_name "strategy_${STRATEGY}" \
        --strategy_name "$STRATEGY"
    
    # 提取ASR
    STRATEGY_ASR=$(extract_asr "$STRATEGY_OUTPUT")
    ASR_RESULTS["$STRATEGY"]="$STRATEGY_ASR"
    log "策略 $STRATEGY ASR: $STRATEGY_ASR"
    
    # 等待一段时间让vLLM服务完全关闭
    if [ "$CURRENT_EVAL" -lt "$TOTAL_EVALS" ]; then
        log "等待 $SLEEP_BETWEEN_EVALS 秒，确保服务完全关闭..."
        sleep "$SLEEP_BETWEEN_EVALS"
    fi
done

# 记录实验结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

# =================================================================
# Step 3: 结果筛选与汇总
# =================================================================
log ""
log "============================================================"
log "实验评估完成! 开始结果筛选..."
log "============================================================"

# 生成结果汇总
python3 << 'PYTHON_SCRIPT'
import json
import os
import sys

# 从环境变量读取参数
output_dir = os.environ.get("OUTPUT_DIR", "")
baseline_asr = float(os.environ.get("BASELINE_ASR", "0"))
topk = os.environ.get("TOPK", "")
gap_threshold = os.environ.get("GAP_THRESHOLD", "")

# 构建结果字典
results = {}
results["baseline_original"] = baseline_asr

# 读取各策略的ASR (从环境变量中解析)
# 环境变量格式: ASR_STRATEGY_NAME=0.1234
for key, val in os.environ.items():
    if key.startswith("ASR_") and key != "ASR_RESULTS":
        strategy_name = key[4:].lower()
        try:
            results[strategy_name] = float(val)
        except:
            pass

# 按ASR降序排序
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("\n" + "="*80)
print("实验1 结果汇总 (按ASR降序)")
print("="*80)
print(f"{'排名':<6} {'策略':<30} {'ASR':<10} {'vs基线':<10}")
print("-"*80)

baseline = results.get("baseline_original", 0)
for rank, (name, asr) in enumerate(sorted_results, 1):
    vs_baseline = asr - baseline
    sign = "+" if vs_baseline >= 0 else ""
    print(f"{rank:<6} {name:<30} {asr:<10.4f} {sign}{vs_baseline:.4f}")

print("="*80)

# 应用筛选逻辑
if topk:
    topk = int(topk)
    filtered = sorted_results[:topk]
    print(f"\nTop-{topk} 筛选结果:")
    for rank, (name, asr) in enumerate(filtered, 1):
        print(f"  {rank}. {name}: {asr:.4f}")
elif gap_threshold:
    gap_threshold = float(gap_threshold)
    filtered = []
    prev_asr = None
    for name, asr in sorted_results:
        if prev_asr is None or (prev_asr - asr) <= gap_threshold:
            filtered.append((name, asr))
            prev_asr = asr
        else:
            print(f"\nGap筛选: {name} 与上一策略差距为 {prev_asr - asr:.4f} > {gap_threshold}，舍弃及后续策略")
            break
    
    print(f"\nGap筛选结果 (阈值={gap_threshold}):")
    for rank, (name, asr) in enumerate(filtered, 1):
        print(f"  {rank}. {name}: {asr:.4f}")
else:
    filtered = sorted_results

# 保存筛选结果
summary = {
    "baseline": baseline,
    "all_results": {name: asr for name, asr in sorted_results},
    "filtered_results": {name: asr for name, asr in filtered},
    "filter_criteria": {
        "topk": topk if topk else None,
        "gap_threshold": float(gap_threshold) if gap_threshold else None,
    }
}

summary_path = os.path.join(output_dir, "experiment_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n汇总结果已保存: {summary_path}")
PYTHON_SCRIPT

log ""
log "============================================================"
log "实验完成!"
log "============================================================"
log "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
log "结果保存在: $OUTPUT_DIR"
log "============================================================"

log "\n提示: 可使用以下命令查看结果:"
log "  cat $OUTPUT_DIR/experiment_summary.json | python -m json.tool"
log "  ls -la $OUTPUT_DIR/*/"
