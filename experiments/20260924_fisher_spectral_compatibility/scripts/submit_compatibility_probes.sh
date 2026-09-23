#!/usr/bin/env bash
set -euo pipefail

cd /data/run01/scyb980/cyt/src/verl-staleness

export PARTITION=gpu_a800 N_GPUS=4 GPU_PROFILE=a800
export PYTHON_BIN="$PWD/.venv/bin/python"

exact_output=$(
    FSDP_USE_ORIG_PARAMS=1 TIME_LIMIT=00:10:00 JOB_NAME=fsdp-jvp-orig-params \
    bash examples/dynamic_staleness/submit_slurm.sh jvp_compat_probe
)
echo "${exact_output}"

fvp_output=$(
    FVP_EPSILON=0.01 TIME_LIMIT=00:10:00 JOB_NAME=fsdp-fisher-grad-diff \
    bash examples/dynamic_staleness/submit_slurm.sh fvp_compat_probe
)
echo "${fvp_output}"
