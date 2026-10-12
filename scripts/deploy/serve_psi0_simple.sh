#!/bin/bash
# Serve a trained psi0 run over HTTP (POST /act, GET /health).
#
# Usage: scripts/deploy/serve_psi0_simple.sh RUN_DIR CKPT_STEP
#   RUN_DIR    training run dir (holds run_config.json, argv.txt, checkpoints/ckpt_<step>)
#   CKPT_STEP  checkpoint step, e.g. 40000
# Env overrides: PORT (22085), HOST (0.0.0.0), ACTION_EXEC_HORIZON (30, the
# LEGS chunk), RTC (1 = training-time RTC on), CUDA_VISIBLE_DEVICES (0).
# Run from the Psi0 repo root.

set -e

if [ "$#" -ne 2 ]; then
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

RUN_DIR=$1
CKPT_STEP=$2
PORT=${PORT:-22085}
HOST=${HOST:-0.0.0.0}
ACTION_EXEC_HORIZON=${ACTION_EXEC_HORIZON:-30}
RTC=${RTC:-1}

source .venv-psi/bin/activate

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "Serving $RUN_DIR (step $CKPT_STEP) on GPU $CUDA_VISIBLE_DEVICES, port $PORT"

RTC_FLAG=()
if [ "$RTC" = "1" ]; then
    RTC_FLAG=(--rtc)
fi

serve_psi0 \
    --host "$HOST" \
    --port "$PORT" \
    --policy=psi0 \
    --run-dir="$RUN_DIR" \
    --ckpt-step="$CKPT_STEP" \
    --action-exec-horizon="$ACTION_EXEC_HORIZON" \
    "${RTC_FLAG[@]}"
