#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
: "${PYTHON_BIN:?Set PYTHON_BIN}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/assets/models/Qwen3-8B-Base}

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_exact_kl_hvp.py" \
    --model-path "${MODEL_PATH}" --output "${OUTPUT_ROOT}" \
    --gpus "${N_GPUS:-4}" --steps "${HVP_POWER_STEPS:-20}" \
    --tolerance "${HVP_POWER_TOLERANCE:-0.001}" --seed "${SEED:-20261005}" "$@"
