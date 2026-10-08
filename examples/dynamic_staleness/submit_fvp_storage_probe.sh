#!/usr/bin/env bash
# One four-GPU engineering job; two longest rows, baseline and optimized FVP.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"
module load miniforge3/25.11.0-1 cuda/12.8
ASSET_REPO=/data/run01/scyb980/cyt/src/verl-staleness
unset OUTPUT_DIR OUTPUT_INSTANCE_ROOT RUN_INSTANCE_TAG RUN_TIMESTAMP KL_PAIR_DIR
export PARTITION=gpu_a800 N_GPUS=4 GPU_PROFILE=a800 ROLLOUT_TP_SIZE=2
export PYTHON_BIN=${ASSET_REPO}/.venv/bin/python
export MODEL_PATH=${ASSET_REPO}/assets/models/Qwen3-8B-Base
export PERSISTENT_CACHE_ROOT=${ASSET_REPO}/.cache/paracloud
export JOB_NAME=fisher-inputs-accum-g4 TIME_LIMIT=01:00:00
export RUNTIME_PROFILE=sis_offload SEED=20261008 FISHER_VJP_N4_PREFLIGHT=0
export RUN_CONFIG_TAG=g4_seed20261008_inputs_gpu_direct_accum
export OUTPUT_ROOT=${REPO_ROOT}/outputs/fisher_gpu_inputs_accum/raw/probe
bash examples/dynamic_staleness/submit_slurm.sh fisher_fvp_probe "$@"
