#!/usr/bin/env bash
set -euo pipefail

# Run on 165 only. No installs, model downloads, optimizer, or background jobs.
PHASE=${1:-tiny}
case "$PHASE" in
    tiny|qwen06) ;;
    *) echo "Usage: $0 [tiny|qwen06]" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROBE_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PROBE_PYTHON=/home/ymy/cyt/dynamicoffpolicy/verl-staleness/.venv/bin/python
PROBE_MODEL=/home/ymy/cyt/dynamicoffpolicy/verl-staleness/assets/models/Qwen3-0.6B-Base
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Recheck immediately before launch, rather than relying on an earlier inventory.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | awk 'NF {found=1} END {exit !found}'; then
    echo "Compute processes already exist on this two-GPU server; not launching." >&2
    exit 1
fi
mkdir -p "${PROBE_ROOT}/raw"
RUN_ROOT=$(mktemp -d "${PROBE_ROOT}/raw/${PHASE}_$(date +%Y%m%d_%H%M%S)_XXXXXX")
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
{
    date -Is
    hostname
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
    sha256sum "${SCRIPT_DIR}/compare_kl_hvp.py" "${BASH_SOURCE[0]}"
} | tee "${RUN_ROOT}/launch.txt"

"$PROBE_PYTHON" "${SCRIPT_DIR}/compare_kl_hvp.py" exact \
    --model "$PHASE" --model-path "$PROBE_MODEL" --output "$RUN_ROOT" \
    --steps "${PROBE_STEPS:-80}" --tolerance 1e-3 2>&1 | tee "${RUN_ROOT}/exact.log"
"$PROBE_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
    "${SCRIPT_DIR}/compare_kl_hvp.py" fd --model "$PHASE" \
    --model-path "$PROBE_MODEL" --output "$RUN_ROOT" \
    --steps "${PROBE_STEPS:-80}" --tolerance 1e-3 2>&1 | tee "${RUN_ROOT}/fd.log"
date -Is | tee "${RUN_ROOT}/completed.txt"
