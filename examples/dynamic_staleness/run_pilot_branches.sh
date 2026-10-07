#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
: "${ANCHOR_CKPT:?Set ANCHOR_CKPT to the timestamped shared anchor checkpoint}"
TARGET_OPTIMIZER_STEP=${TARGET_OPTIMIZER_STEP:-196}

for reuse_n in 4 8 16; do
    env \
        ANCHOR_CKPT="${ANCHOR_CKPT}" \
        REUSE_N="${reuse_n}" \
        TARGET_OPTIMIZER_STEP="${TARGET_OPTIMIZER_STEP}" \
        "${SCRIPT_DIR}/run_staleness.sh" branch "$@"
done
