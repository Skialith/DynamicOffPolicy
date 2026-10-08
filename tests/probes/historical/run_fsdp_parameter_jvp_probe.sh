#!/usr/bin/env bash
set -euo pipefail

echo "Legacy-FSDP functional parameter JVP is paused; historical commands follow in comments." >&2
exit 2

# Historical implementation, paused on 2026-10-07.
# set -euo pipefail

# SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
# PYTHON_BIN=${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}
# N_GPUS=${N_GPUS:-4}

# exec "${PYTHON_BIN}" -m torch.distributed.run \
#     --standalone \
#     --nproc_per_node="${N_GPUS}" \
#     "${SCRIPT_DIR}/verify_fsdp_parameter_jvp.py"
