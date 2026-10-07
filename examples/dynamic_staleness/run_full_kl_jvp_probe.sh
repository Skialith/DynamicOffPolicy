#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${PYTHON_BIN:?Set PYTHON_BIN}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

export FULL_KL_EXPERIMENT=1 FULL_KL_JVP=1
export REUSE_N=4 TARGET_OPTIMIZER_STEP=4
export ENABLE_BENCHMARK_EVAL=0 SAVE_FREQ=-1
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-fisher_jvp_probe_n4_u0004}

bash "${SCRIPT_DIR}/run_staleness.sh" anchor "$@"
"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_full_kl_jvp.py" \
    "${OUTPUT_ROOT}/${EXPERIMENT_NAME}" --updates 4
