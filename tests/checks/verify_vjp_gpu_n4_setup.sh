#!/usr/bin/env bash
# CPU numerical and resolved-config gate, on the allocated compute node.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
vjp_setup_started=$(date +%s)
vjp_cpu_log=$(mktemp "${TMPDIR:?}/vjp-gpu-cpu-check.XXXXXX.log")
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "${PYTHON_BIN}" - > "${vjp_cpu_log}" <<'PY_CPU_CHECK'
import sys
import torch
import unittest
torch.set_num_threads(4)
suite = unittest.TestSuite()
for pattern in ("test_training_fisher.py", "test_exact_kl_hvp.py"):
    suite.addTests(unittest.defaultTestLoader.discover("tests/workers/actor", pattern=pattern))
result = unittest.TextTestRunner(verbosity=1).run(suite)
sys.exit(not result.wasSuccessful())
PY_CPU_CHECK
echo "vjp_gpu_preflight_cpu_log=${vjp_cpu_log}"
vjp_cfg=$(mktemp "${TMPDIR}/vjp-gpu-config.XXXXXX.yaml")
CUDA_VISIBLE_DEVICES= DRY_RUN=1 bash "${REPO_ROOT}/examples/dynamic_staleness/run_staleness.sh" anchor > "${vjp_cfg}"
"${PYTHON_BIN}" - "${vjp_cfg}" <<'PY_CONFIG_CHECK'
from pathlib import Path
import sys
import yaml
text = Path(sys.argv[1]).read_text()
config = yaml.safe_load(text[text.index("actor_rollout_ref:\n"):])
actor = config["actor_rollout_ref"]["actor"]
checks = {
    "train_batch_size": config["data"]["train_batch_size"],
    "ppo_mini_batch_size": actor["ppo_mini_batch_size"],
    "rollout_n": config["actor_rollout_ref"]["rollout"]["n"],
    "vjp_cpu_offload": actor["full_kl_hvp_vjp_cpu_offload"],
    "stress_prefix_length": actor["full_kl_hvp_stress_prefix_length"],
    "compute_quadratic": actor["full_kl_hvp_compute_quadratic"],
    "actor_kl_measurement": actor["full_kl_actor_measurement"],
    "total_training_steps": config["trainer"]["total_training_steps"],
    "test_freq": config["trainer"]["test_freq"],
    "save_freq": config["trainer"]["save_freq"],
}
expected = dict(train_batch_size=1024, ppo_mini_batch_size=256, rollout_n=8,
                vjp_cpu_offload=False, stress_prefix_length=4095, compute_quadratic=False,
                actor_kl_measurement=False, total_training_steps=1, test_freq=-1, save_freq=-1)
assert checks == expected, (checks, expected)
print("vjp_gpu_preflight_config_passed:", checks, flush=True)
PY_CONFIG_CHECK
echo "vjp_gpu_preflight_seconds=$(( $(date +%s) - vjp_setup_started ))"
