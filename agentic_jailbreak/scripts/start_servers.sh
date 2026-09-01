#!/bin/bash
# =============================================================================
# 启动 vLLM Servers
# =============================================================================
#
# 启动 Guard、Target、Rollout 三个 server
#
# Usage:
#   bash scripts/start_servers.sh
#
# =============================================================================

set -e

source "$(dirname "$0")/common.sh"

log_section "启动 vLLM Servers"

# 清理 GPU
cleanup_gpus

# 启动 Guard server (GPU 0)
start_guard_server

# 启动 Target server (GPU 1)
start_target_server

# 启动 Rollout server (GPU 2-3)
start_rollout_server

log_section "所有 Servers 已启动"
log_info "Guard:  port $GUARD_PORT (GPU $GUARD_GPU)"
log_info "Target: port $TARGET_PORT (GPU $TARGET_GPU)"
log_info "Rollout: port $ROLLOUT_PORT (GPU $ROLLOUT_GPUS)"
log_info ""
log_info "下一步: bash scripts/exp01_single_turn.sh"
