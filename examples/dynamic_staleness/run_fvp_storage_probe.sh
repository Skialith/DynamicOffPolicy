#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "${PYTHON_BIN}" - <<'PY_CPU_CHECK'
import sys, torch, unittest
torch.set_num_threads(4)
suite = unittest.TestSuite()
for pattern in ("test_training_fisher.py", "test_exact_kl_hvp.py"):
    suite.addTests(unittest.defaultTestLoader.discover("tests/workers/actor", pattern=pattern))
result = unittest.TextTestRunner(verbosity=1).run(suite)
sys.exit(not result.wasSuccessful())
PY_CPU_CHECK
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_fvp_storage_probe.py" \
    --model-path "${MODEL_PATH}" --output "${OUTPUT_ROOT}" --gpus "${N_GPUS}" \
    --length 4095 --repeats 3 --seed 20261008 "$@"
