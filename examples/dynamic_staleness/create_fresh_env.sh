#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
ENV_PREFIX=${ENV_PREFIX:-${REPO_ROOT}/.venv}
CONDA_BIN=${CONDA_BIN:-conda}

"${CONDA_BIN}" create --prefix "${ENV_PREFIX}" python=3.12 pip -y
"${ENV_PREFIX}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_PREFIX}/bin/python" -m pip install \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
"${ENV_PREFIX}/bin/python" -m pip install vllm==0.11.0
"${ENV_PREFIX}/bin/python" -m pip install --no-build-isolation flash-attn==2.8.3
"${ENV_PREFIX}/bin/python" -m pip install -e "${REPO_ROOT}[vllm,math,test]" matplotlib

PYTHONPATH="${REPO_ROOT}" "${ENV_PREFIX}/bin/python" "${SCRIPT_DIR}/check_env.py" \
    --repo-root "${REPO_ROOT}"
