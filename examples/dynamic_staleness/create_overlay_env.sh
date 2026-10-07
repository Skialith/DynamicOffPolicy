#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
BASE_PYTHON=${BASE_PYTHON:-$(command -v python)}
ENV_DIR=${ENV_DIR:-${REPO_ROOT}/.venv}

"${BASE_PYTHON}" -m venv --system-site-packages "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install \
    'numpy<2.0' \
    'codetiming>=1.4' \
    pybind11 \
    pylatexenc \
    matplotlib

PYTHONPATH="${REPO_ROOT}" "${ENV_DIR}/bin/python" "${SCRIPT_DIR}/check_env.py" \
    --repo-root "${REPO_ROOT}"
