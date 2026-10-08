#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
: "${KL_PAIR_DIR:?Set the shared, unique pair output directory}"
: "${REUSE_N:?Set REUSE_N to 4 or 8}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export FULL_KL_EXPERIMENT=1
export TARGET_OPTIMIZER_STEP=96
export EXPERIMENT_NAME=full_kl_n${REUSE_N}_u0096
export OUTPUT_DIR=${OUTPUT_ROOT}/${EXPERIMENT_NAME}
bash "${SCRIPT_DIR}/run_staleness.sh" anchor "$@"
"${PYTHON_BIN}" "${REPO_ROOT}/tests/checks/verify_full_kl_run.py" "${OUTPUT_DIR}" --n "${REUSE_N}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/finish_full_kl_pair.py" \
    --run "${OUTPUT_DIR}" --pair-dir "${KL_PAIR_DIR}" --n "${REUSE_N}"
