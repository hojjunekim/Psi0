#!/bin/bash
# Serve a fine-tuned pi0.5 checkpoint over websocket (openpi protocol).
#
# Usage: baselines/pi05/serve_pi05.sh <config> <checkpoint_dir> [port]
#   <config>          training config name, e.g. legs_turn_uni_pi05
#   <checkpoint_dir>  e.g. .runs/openpi-05/<config>/<exp>/40000
#   [port]            default 9000 (or set PORT)
# Env overrides: CUDA_VISIBLE_DEVICES (0), VENV (.venv-openpi).
# Run from the Psi0 repo root.

if [ "$#" -lt 2 ]; then
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

CONFIG=$1
CKPT_DIR=$2
PORT=${3:-${PORT:-9000}}
VENV=${VENV:-.venv-openpi}

source "$VENV/bin/activate"
export UV_ENV_FILE=$VENV
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Serving pi0.5 $CONFIG from $CKPT_DIR on GPU $CUDA_VISIBLE_DEVICES, port $PORT"

python src/openpi/deploy/serve_policy.py \
    --port="$PORT" \
    policy:checkpoint \
    --policy.config="$CONFIG" \
    --policy.dir="$CKPT_DIR"
