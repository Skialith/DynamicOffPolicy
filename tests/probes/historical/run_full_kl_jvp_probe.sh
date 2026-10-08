#!/usr/bin/env bash
set -euo pipefail

echo "Online legacy-FSDP functional parameter JVP is paused; historical commands follow in comments." >&2
exit 2

# Historical implementation, paused on 2026-10-07.
# set -euo pipefail

# SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
# : "${PYTHON_BIN:?Set PYTHON_BIN}"
# : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

# export FULL_KL_EXPERIMENT=1 FULL_KL_JVP=1
# export REUSE_N=4 TARGET_OPTIMIZER_STEP=4
# export ENABLE_BENCHMARK_EVAL=0 SAVE_FREQ=-1
# export EXPERIMENT_NAME=${EXPERIMENT_NAME:-fisher_jvp_probe_n4_u0004}

# bash "${REPO_ROOT}/examples/dynamic_staleness/run_staleness.sh" anchor "$@"
# "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_full_kl_jvp.py" \
#     "${OUTPUT_ROOT}/${EXPERIMENT_NAME}" --updates 4
