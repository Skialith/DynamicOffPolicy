#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/dynamic_staleness}
ANCHOR_TARGET_OPTIMIZER_STEP=${ANCHOR_TARGET_OPTIMIZER_STEP:-100}
BRANCH_TARGET_OPTIMIZER_STEP=${BRANCH_TARGET_OPTIMIZER_STEP:-196}

if ((ANCHOR_TARGET_OPTIMIZER_STEP % 4 != 0)); then
    echo "ANCHOR_TARGET_OPTIMIZER_STEP must be divisible by 4" >&2
    exit 2
fi
if (((BRANCH_TARGET_OPTIMIZER_STEP - ANCHOR_TARGET_OPTIMIZER_STEP) % 8 != 0)); then
    echo "N=8 branch updates must be divisible by 8" >&2
    exit 2
fi

anchor_experiment="anchor_n4_u$(printf '%04d' "${ANCHOR_TARGET_OPTIMIZER_STEP}")"
anchor_global_step=$((ANCHOR_TARGET_OPTIMIZER_STEP / 4))
anchor_checkpoint="${OUTPUT_ROOT}/${anchor_experiment}/global_step_${anchor_global_step}"

echo "chain stage 1/2: N=4, optimizer update 0 -> ${ANCHOR_TARGET_OPTIMIZER_STEP}"
env \
    -u ANCHOR_CKPT \
    -u EXPERIMENT_NAME \
    -u LOGGER_EXPERIMENT_NAME \
    -u OUTPUT_DIR \
    -u VALIDATION_DATA_DIR \
    REUSE_N=4 \
    TARGET_OPTIMIZER_STEP="${ANCHOR_TARGET_OPTIMIZER_STEP}" \
    SAVE_FREQ=1000000 \
    "${SCRIPT_DIR}/run_staleness.sh" anchor "$@"

state_file="${anchor_checkpoint}/staleness_state.json"
if [[ ! -f "${state_file}" ]]; then
    echo "anchor finished without the expected checkpoint state: ${state_file}" >&2
    exit 1
fi
"${PYTHON_BIN}" -c \
    'import json,sys; state=json.load(open(sys.argv[1])); expected=int(sys.argv[2]); assert state["optimizer_step"] == expected, state' \
    "${state_file}" "${ANCHOR_TARGET_OPTIMIZER_STEP}"

echo "chain stage 2/2: N=8, optimizer update ${ANCHOR_TARGET_OPTIMIZER_STEP} -> ${BRANCH_TARGET_OPTIMIZER_STEP}"
env \
    -u EXPERIMENT_NAME \
    -u LOGGER_EXPERIMENT_NAME \
    -u OUTPUT_DIR \
    -u VALIDATION_DATA_DIR \
    ANCHOR_CKPT="${anchor_checkpoint}" \
    REUSE_N=8 \
    TARGET_OPTIMIZER_STEP="${BRANCH_TARGET_OPTIMIZER_STEP}" \
    SAVE_FREQ=-1 \
    "${SCRIPT_DIR}/run_staleness.sh" branch "$@"

echo "chain completed: anchor_checkpoint=${anchor_checkpoint}"
echo "chain outputs=${OUTPUT_ROOT}"
