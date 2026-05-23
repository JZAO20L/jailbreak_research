#!/bin/bash
# =============================================================================
# Batch Evaluation Script for Simple Rule Weight Experiments
#
# Purpose: Run ASR evaluation on ALL existing final_lora checkpoints
# Usage:
#   bash experiments/simple_rule_weight_exp/eval.sh              # Eval all experiments
#   bash experiments/simple_rule_weight_exp/eval.sh exp1 exp2    # Eval specific ones
#
# Features:
# 1. Auto-detects existing final_lora directories
# 2. Manages vLLM server lifecycle (start → eval → stop)
# 3. Cleans up failed eval results before re-running
# 4. Generates summary report after all evals complete
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT_DIR="$BASE_DIR/experiments/simple_rule_weight_exp"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Model paths
POLICY_MODEL="/root/autodl-tmp/models/Qwen/Qwen3-4B"
TARGET_MODEL="/root/autodl-tmp/models/Qwen/Qwen3-4B"
GUARD_MODEL="/root/autodl-tmp/models/Qwen/Qwen3Guard-Gen-4B"

# Eval data
EVAL_DATA="$BASE_DIR/../data/dataset/processed/10k/test.jsonl"

# vLLM ports
POLICY_PORT=8003
TARGET_PORT=8001
GUARD_PORT=8002

# Attack prompt strategy
ATTACK_PROMPT="hypothetical_scenario"

# =============================================================================
# Helper Functions
# =============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if vLLM server is ready
check_vllm_server() {
    local port=$1
    local max_wait=${2:-60}
    
    for i in $(seq 1 $max_wait); do
        if curl -s "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# Launch vLLM server in background
launch_vllm_server() {
    local model_path=$1
    local port=$2
    local gpu_id=$3
    local log_file=$4
    local enable_lora=${5:-false}
    local lora_dir=${6:-""}
    local gpu_memory_utilization=${7:-0.9}
    local max_model_len=${8:-4096}
    
    log "Launching vLLM server: port=$port, GPU=$gpu_id, LoRA=$enable_lora"
    log "  Model: $model_path"
    [ -n "$lora_dir" ] && log "  LoRA dir: $lora_dir"
    
    mkdir -p "$(dirname "$log_file")"
    
    local cmd="CUDA_VISIBLE_DEVICES=$gpu_id VLLM_USE_MODELSCOPE=true vllm serve $model_path"
    cmd="$cmd --served-model-name $(basename $model_path)"
    cmd="$cmd --port $port --host 127.0.0.1"
    cmd="$cmd --max-model-len $max_model_len --tensor-parallel-size 1"
    cmd="$cmd --gpu-memory-utilization $gpu_memory_utilization --dtype auto"
    
    if [ "$enable_lora" = "true" ] && [ -n "$lora_dir" ]; then
        cmd="$cmd --enable-lora --lora-modules eval_lora=$lora_dir --max-lora-rank 16"
    fi
    
    log "Command: $cmd"
    
    eval "$cmd" > "$log_file" 2>&1 &
    local pid=$!
    log "Server started with PID $pid (log: $log_file)"
    echo $pid
}

# Wait for server to be ready
wait_for_server() {
    local port=$1
    local server_name=$2
    local max_wait=${3:-120}
    
    log "Waiting for $server_name server on port $port..."
    
    if check_vllm_server "$port" "$max_wait"; then
        log "✓ $server_name server ready on port $port"
        return 0
    else
        log "✗ $server_name server failed to start on port $port"
        return 1
    fi
}

# Stop server on port
kill_vllm_server() {
    local port=$1
    local server_name=$2
    
    log "Stopping $server_name server (port $port)..."
    
    local pid=$(lsof -ti :$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        kill $pid 2>/dev/null || true
        sleep 2
        kill -9 $pid 2>/dev/null || true
        log "✓ $server_name server (PID $pid) stopped"
    else
        log "No server found on port $port"
    fi
}

# Clean up failed eval results (containing APIConnectionError)
cleanup_failed_eval() {
    local eval_dir=$1
    
    if [ ! -d "$eval_dir" ]; then
        return 0
    fi
    
    log "Checking for failed eval results in $eval_dir..."
    
    local failed_count=0
    
    # Find and remove jsonl files containing APIConnectionError
    while IFS= read -r -d '' file; do
        if grep -q "APIConnectionError" "$file" 2>/dev/null; then
            log "  Removing failed file: $file"
            rm -f "$file"
            ((failed_count++))
        fi
    done < <(find "$eval_dir" -name "*.jsonl" -print0 2>/dev/null)
    
    # Also check asr report files for errors
    while IFS= read -r -d '' file; do
        if grep -q "APIConnectionError\|Connection error" "$file" 2>/dev/null; then
            log "  Removing failed file: $file"
            rm -f "$file"
            ((failed_count++))
        fi
    done < <(find "$eval_dir" -name "asr_report_*.json" -print0 2>/dev/null)
    
    if [ $failed_count -gt 0 ]; then
        log "Cleaned up $failed_count failed eval files"
    else
        log "No failed eval results found"
    fi
}

# Check if eval completed successfully
check_eval_success() {
    local combo_dir=$1
    
    # Check if summary.json exists and has valid content
    local summary_file="$combo_dir/summary.json"
    if [ ! -f "$summary_file" ]; then
        return 1
    fi
    
    # Check if there are any failed entries
    if grep -q "error\|APIConnectionError" "$summary_file" 2>/dev/null; then
        return 1
    fi
    
    # Check if rewritten jsonl files exist and are non-empty
    local jsonl_count=$(find "$combo_dir" -name "rewritten_*.jsonl" -size +0c 2>/dev/null | wc -l)
    if [ "$jsonl_count" -eq 0 ]; then
        return 1
    fi
    
    # Check if asr report files exist and are non-empty
    local asr_count=$(find "$combo_dir" -name "asr_report_*.json" -size +0c 2>/dev/null | wc -l)
    if [ "$asr_count" -eq 0 ]; then
        return 1
    fi
    
    return 0
}

# Run eval for a single LoRA checkpoint
run_single_eval() {
    local lora_dir=$1
    local exp_id=$2
    
    log "=========================================="
    log "Starting eval for: $exp_id"
    log "LoRA dir: $lora_dir"
    log "=========================================="
    
    # Validate LoRA directory exists
    if [ ! -d "$lora_dir" ]; then
        log "ERROR: LoRA directory not found: $lora_dir"
        return 1
    fi
    
    # Setup eval output directory
    local eval_output_dir="$OUTPUT_DIR/$exp_id/eval_results"
    mkdir -p "$eval_output_dir"
    
    # Clean up any previous failed eval results
    cleanup_failed_eval "$eval_output_dir"
    
    # Check if eval already completed successfully
    # Find combo directories
    local combo_dirs=$(find "$eval_output_dir" -maxdepth 1 -type d -name "combo_*" 2>/dev/null)
    local all_success=true
    
    for combo_dir in $combo_dirs; do
        if ! check_eval_success "$combo_dir"; then
            all_success=false
            log "Previous eval incomplete/failed for: $combo_dir"
        fi
    done
    
    if [ "$all_success" = true ] && [ -n "$combo_dirs" ]; then
        log "✓ Eval already completed successfully for $exp_id, skipping"
        return 0
    fi
    
    # =================================================================
    # Start vLLM servers
    # =================================================================
    log "Starting vLLM servers for eval..."
    
    # Policy server (GPU0, port 8003, with LoRA)
    kill_vllm_server "$POLICY_PORT" "policy"  # Ensure port is free
    launch_vllm_server \
        "$POLICY_MODEL" \
        "$POLICY_PORT" \
        "0" \
        "$OUTPUT_DIR/$exp_id/logs/vllm_policy_eval.log" \
        "true" \
        "$lora_dir" \
        "0.9" \
        "4096"
    wait_for_server "$POLICY_PORT" "policy" 120 || {
        log "ERROR: Policy server failed to start"
        kill_vllm_server "$POLICY_PORT" "policy"
        return 1
    }
    
    # Target server (GPU1, port 8001, base model)
    kill_vllm_server "$TARGET_PORT" "target"
    launch_vllm_server \
        "$TARGET_MODEL" \
        "$TARGET_PORT" \
        "1" \
        "$OUTPUT_DIR/$exp_id/logs/vllm_target_eval.log" \
        "false" \
        "" \
        "0.45" \
        "4096"
    wait_for_server "$TARGET_PORT" "target" 120 || {
        log "ERROR: Target server failed to start"
        kill_vllm_server "$POLICY_PORT" "policy"
        kill_vllm_server "$TARGET_PORT" "target"
        return 1
    }
    
    # Guard server (GPU1, port 8002, base model)
    kill_vllm_server "$GUARD_PORT" "guard"
    launch_vllm_server \
        "$GUARD_MODEL" \
        "$GUARD_PORT" \
        "1" \
        "$OUTPUT_DIR/$exp_id/logs/vllm_guard_eval.log" \
        "false" \
        "" \
        "0.45" \
        "4096"
    wait_for_server "$GUARD_PORT" "guard" 120 || {
        log "ERROR: Guard server failed to start"
        kill_vllm_server "$POLICY_PORT" "policy"
        kill_vllm_server "$TARGET_PORT" "target"
        kill_vllm_server "$GUARD_PORT" "guard"
        return 1
    }
    
    log "✓ All vLLM servers ready for eval"
    
    # =================================================================
    # Run eval
    # =================================================================
    local eval_script="$BASE_DIR/scripts/eval.py"
    local run_name="eval_${exp_id}"
    
    log "Running eval script..."
    log "Eval command:"
    log "  python $eval_script \\"
    log "    --eval_path $EVAL_DATA \\"
    log "    --lora_paths $lora_dir \\"
    log "    --strategy_name $ATTACK_PROMPT \\"
    log "    --output_root $eval_output_dir \\"
    log "    --run_name $run_name"
    
    local eval_success=false
    
    python "$eval_script" \
        --eval_path "$EVAL_DATA" \
        --lora_paths "$lora_dir" \
        --strategy_name "$ATTACK_PROMPT" \
        --base_model_path "$POLICY_MODEL" \
        --target_model_path "$TARGET_MODEL" \
        --guard_model_path "$GUARD_MODEL" \
        --policy_port "$POLICY_PORT" \
        --target_port "$TARGET_PORT" \
        --guard_port "$GUARD_PORT" \
        --output_root "$eval_output_dir" \
        --run_name "$run_name" && eval_success=true
    
    # =================================================================
    # Stop vLLM servers
    # =================================================================
    log "Stopping vLLM servers..."
    kill_vllm_server "$POLICY_PORT" "policy"
    kill_vllm_server "$TARGET_PORT" "target"
    kill_vllm_server "$GUARD_PORT" "guard"
    
    if [ "$eval_success" = true ]; then
        log "✓ Eval completed successfully for $exp_id"
        return 0
    else
        log "✗ Eval failed for $exp_id"
        return 1
    fi
}

# Generate summary report from all eval results
generate_summary() {
    log "=========================================="
    log "Generating eval summary report..."
    log "=========================================="
    
    local summary_file="$OUTPUT_DIR/eval_summary_all.json"
    local summary_txt="$OUTPUT_DIR/eval_summary_all.txt"
    
    echo "{" > "$summary_file"
    echo "# Eval Summary - All Experiments" > "$summary_txt"
    echo "# Generated: $(date '+%Y-%m-%d %H:%M:%S')" >> "$summary_txt"
    echo "" >> "$summary_txt"
    printf "%-20s | %-10s | %-10s | %-10s | %-10s | %-10s\n" "Experiment" "ASR" "Refusal" "Partial" "Success" "Total" >> "$summary_txt"
    echo "$(printf '%0.s-' {1..90})" >> "$summary_txt"
    
    local first=true
    
    for exp_dir in "$OUTPUT_DIR"/exp*; do
        local exp_id=$(basename "$exp_dir")
        local eval_dir="$exp_dir/eval_results"
        
        [ ! -d "$eval_dir" ] && continue
        
        # Find the eval run directory (eval_*)
        local eval_run_dir=$(find "$eval_dir" -maxdepth 1 -type d -name "eval_*" | head -1)
        [ -z "$eval_run_dir" ] && continue
        
        # Find summary.json
        local combo_summary=$(find "$eval_run_dir" -name "summary.json" | head -1)
        if [ -f "$combo_summary" ]; then
            # Extract ASR and other metrics
            local asr=$(python3 -c "
import json
with open('$combo_summary') as f:
    data = json.load(f)
combos = data.get('combinations', [])
if combos:
    c = combos[0]
    if 'error' in c:
        print('ERROR: ' + c['error'])
    else:
        print(f\"{c.get('asr', 0):.4f}|{c.get('refusal_rate', 0):.4f}|{c.get('partial_rate', 0):.4f}|{c.get('success_rate', 0):.4f}|{c.get('total', 0)}\")
else:
    print('NO_DATA')
" 2>/dev/null || echo "PARSE_ERROR")
            
            if [[ "$asr" == ERROR:* ]]; then
                asr_val="FAILED"
                status="failed"
            elif [[ "$asr" == "NO_DATA" || "$asr" == "PARSE_ERROR" ]]; then
                asr_val="N/A"
                status="incomplete"
            else
                IFS='|' read -r asr_val refusal partial success total <<< "$asr"
                status="success"
            fi
            
            if [ "$first" = true ]; then
                first=false
            else
                echo "," >> "$summary_file"
            fi
            
            echo "  \"$exp_id\": {\"status\": \"$status\", \"asr\": \"$asr_val\", \"summary\": \"$combo_summary\"}" >> "$summary_file"
            printf "%-20s | %-10s | %-10s | %-10s | %-10s | %-10s\n" "$exp_id" "$asr_val" "${refusal:-N/A}" "${partial:-N/A}" "${success:-N/A}" "${total:-N/A}" >> "$summary_txt"
        else
            echo "  \"$exp_id\": {\"status\": \"no_summary\"}" >> "$summary_file"
            printf "%-20s | %-10s | %-10s | %-10s | %-10s | %-10s\n" "$exp_id" "N/A" "N/A" "N/A" "N/A" "N/A" >> "$summary_txt"
        fi
    done
    
    echo "}" >> "$summary_file"
    
    log "Summary saved to:"
    log "  JSON: $summary_file"
    log "  Text: $summary_txt"
    log ""
    cat "$summary_txt"
}

# =============================================================================
# Main Execution
# =============================================================================
log "=========================================="
log "Batch Eval for Simple Rule Weight Experiments"
log "=========================================="
log "Output directory: $OUTPUT_DIR"
log "Eval data: $EVAL_DATA"
log "Attack prompt: $ATTACK_PROMPT"
log "=========================================="

# Determine which experiments to eval
EXPS_TO_EVAL=()

if [ $# -gt 0 ]; then
    # Use provided experiment IDs
    for exp_id in "$@"; do
        exp_dir="$OUTPUT_DIR/$exp_id"
        if [ -d "$exp_dir" ]; then
            EXPS_TO_EVAL+=("$exp_id")
        else
            log "WARNING: Experiment directory not found: $exp_id (skipping)"
        fi
    done
else
    # Auto-detect all experiment directories with final_lora
    for exp_dir in "$OUTPUT_DIR"/exp*; do
        if [ -d "$exp_dir" ]; then
            exp_id=$(basename "$exp_dir")
            lora_dir="$exp_dir/final_lora"
            if [ -d "$lora_dir" ]; then
                EXPS_TO_EVAL+=("$exp_id")
            else
                log "WARNING: No final_lora found in $exp_id (skipping)"
            fi
        fi
    done
fi

if [ ${#EXPS_TO_EVAL[@]} -eq 0 ]; then
    log "ERROR: No experiments found to evaluate!"
    exit 1
fi

log "Experiments to eval: ${EXPS_TO_EVAL[*]}"
log "Total: ${#EXPS_TO_EVAL[@]}"
log "=========================================="

# Run evals
SUCCESS_COUNT=0
FAIL_COUNT=0

for exp_id in "${EXPS_TO_EVAL[@]}"; do
    lora_dir="$OUTPUT_DIR/$exp_id/final_lora"
    
    if run_single_eval "$lora_dir" "$exp_id"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
        log "WARNING: Eval failed for $exp_id, continuing..."
    fi
    
    # Optional: pause between experiments
    log "Waiting 5 seconds before next experiment..."
    sleep 5
done

# =============================================================================
# Generate Summary
# =============================================================================
log "=========================================="
log "Batch Eval Complete!"
log "=========================================="
log "Success: $SUCCESS_COUNT"
log "Failed:  $FAIL_COUNT"
log "=========================================="

generate_summary

log ""
log "=========================================="
log "All done!"
log "=========================================="
