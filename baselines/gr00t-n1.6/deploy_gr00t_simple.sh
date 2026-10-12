#!/usr/bin/env bash
# Serve a fine-tuned GR00T N1.6 checkpoint over HTTP (POST /act, GET /health).
#
# Usage: baselines/gr00t-n1.6/deploy_gr00t_simple.sh <checkpoint_dir> [port]
#   <checkpoint_dir>  e.g. .runs/gr00t/<run>/checkpoint-40000 (or set MODEL_PATH)
#   [port]            default 5556 (or set PORT)
# Env overrides: EMBODIMENT_TAG (G1_EEF_UNI, the LEGS 18-D EEF schema),
# ACTION_EXEC_HORIZON (16), DEVICE (cuda:0), HOST (0.0.0.0).
# Run from the Psi0 repo root; an optional .env there is sourced.
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
PORT="${2:-${PORT:-5556}}"
if [[ -z "$MODEL_PATH" ]]; then
  sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi
EMBODIMENT_TAG="${EMBODIMENT_TAG:-G1_EEF_UNI}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-16}"
DEVICE="${DEVICE:-cuda:0}"
HOST="${HOST:-0.0.0.0}"

export PYTHONPATH="$PWD/src:$PWD/src/gr00t${PYTHONPATH:+:$PYTHONPATH}"
source ./src/gr00t/.venv/bin/activate
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE=0

python -m gr00t.deploy.gr00t_serve_simple \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --model-path "$MODEL_PATH" \
  --device "$DEVICE" \
  --host "$HOST" \
  --port "$PORT" \
  --action-exec-horizon "$ACTION_EXEC_HORIZON" \
  --use-sim-policy-wrapper \
  --strict
