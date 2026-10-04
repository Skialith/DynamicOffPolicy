#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROBE_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PROBE_PYTHON=/home/ymy/cyt/dynamicoffpolicy/verl-staleness/.venv/bin/python
PROBE_MODEL=/home/ymy/cyt/dynamicoffpolicy/verl-staleness/assets/models/Qwen3-0.6B-Base
REFERENCE_ROOT=${PROBE_ROOT}/raw/qwen06_20261004_194558_cnh3Ei
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | awk 'NF {found=1} END {exit !found}'; then
    echo "GPU 0 already has compute processes; not launching." >&2
    exit 1
fi
test -f "${REFERENCE_ROOT}/reference.pt"
mkdir -p "${PROBE_ROOT}/raw"
RUN_ROOT=$(mktemp -d "${PROBE_ROOT}/raw/single_reference_$(date +%Y%m%d_%H%M%S)_XXXXXX")
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
{
    date -Is
    hostname
    nvidia-smi -i 0 --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
    sha256sum "${SCRIPT_DIR}/compare_kl_hvp.py" "${BASH_SOURCE[0]}"
} | tee "${RUN_ROOT}/launch.txt"

# Each seed gets a new directory. Historical two-GPU outputs remain read-only.
for PROBE_SEED in 20261005 20261007; do
    "$PROBE_PYTHON" "${SCRIPT_DIR}/compare_kl_hvp.py" exact --placement single \
        --model qwen06 --model-path "$PROBE_MODEL" --reference "$REFERENCE_ROOT" \
        --power-seed "$PROBE_SEED" --steps 80 --tolerance 1e-5 \
        --output "${RUN_ROOT}/seed_${PROBE_SEED}" 2>&1 | tee "${RUN_ROOT}/seed_${PROBE_SEED}.log"
done
date -Is | tee "${RUN_ROOT}/completed.txt"
