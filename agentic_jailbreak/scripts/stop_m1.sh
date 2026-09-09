#!/bin/bash
# 停止发散的 M1 trainer (09-09 凌晨, step ~80 NaN 权重无挽救价值)
pkill -f 'swift.cli.rlhf'
sleep 3
pkill -9 -f 'swift.cli.rlhf' 2>/dev/null
if pgrep -f 'swift.cli.rlhf' >/dev/null; then
    echo STILL-ALIVE
else
    echo trainer-stopped
fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n 4p
