#!/usr/bin/env bash
# Run from the independent Paracloud deployment; one submission per invocation.
set -euo pipefail
SETTING=${1:?Use probe, n4 or n8}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"
module load miniforge3/25.11.0-1 cuda/12.8
unset KL_PAIR_DIR OUTPUT_DIR OUTPUT_INSTANCE_ROOT RUN_INSTANCE_TAG RUN_TIMESTAMP
unset RUN_CONFIG_TAG LOGGER_EXPERIMENT_NAME

ASSET_REPO=/data/run01/scyb980/cyt/src/verl-staleness
export PARTITION=gpu_a800 N_GPUS=4 GPU_PROFILE=a800
export PYTHON_BIN=${ASSET_REPO}/.venv/bin/python
export MODEL_PATH=${ASSET_REPO}/assets/models/Qwen3-8B-Base
export TRAIN_FILE=${ASSET_REPO}/assets/datasets/SIS-Math-GRPO/data/pilot-seed1-u0292.parquet
export MATH500_FILE=${ASSET_REPO}/assets/datasets/MATH-500/data/math500-sis.parquet
export RUNTIME_PROFILE=sis_offload ROLLOUT_TP_SIZE=2
export FULL_KL_EXPERIMENT=1 FULL_KL_HVP=1 FULL_KL_JVP=0
export DRY_RUN=0 SMOKE=0 SEED=1 TRAIN_SHUFFLE=false
export RESPONSES_PER_PROMPT=8 LEARNING_RATE=1e-6 LR_WARMUP_STEPS=0
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=3072 KL_LOSS_COEF=0.001
export KL_NUM_PROMPTS=64 KL_POSITIONS_PER_RESPONSE=8 KL_MEASUREMENT_SEED=20260903
export HVP_POWER_STEPS=200 HVP_POWER_TOLERANCE=0.001 HVP_MEASUREMENT_TIMEOUT=21600
export SAVE_FREQ=-1
export PERSISTENT_CACHE_ROOT=${ASSET_REPO}/.cache/paracloud

case "${SETTING}" in
    probe)
        # Real rollout, all eight ages and full q; reduced training batch only.
        export REUSE_N=8 TARGET_OPTIMIZER_STEP=8 MINI_PROMPT_BATCH=8
        export ENABLE_BENCHMARK_EVAL=0 VAL_BEFORE_TRAIN=false
        export JOB_NAME=fisher-kqb-integration-g4 TIME_LIMIT=24:00:00
        export EXPERIMENT_NAME=fisher_kqb_probe_n8_u0008
        export OUTPUT_ROOT=${REPO_ROOT}/experiments/20261005_fisher_kqb_n4_n8/n8/raw/integration_probe
        export RUN_CONFIG_TAG=g4_tp2_seed1_hvp_integration_probe
        ;;
    n4|n8)
        : "${SLURM_DEPENDENCY:?Formal runs require afterok of the passed integration probe}"
        if [[ ! "${SLURM_DEPENDENCY}" =~ ^afterok:[0-9]+$ ]]; then
            echo "Use SLURM_DEPENDENCY=afterok:<integration-job-id>" >&2
            exit 2
        fi
        export REUSE_N=${SETTING#n} TARGET_OPTIMIZER_STEP=96 MINI_PROMPT_BATCH=256
        export ENABLE_BENCHMARK_EVAL=1 VAL_BEFORE_TRAIN=true TEST_FREQ=1
        export JOB_NAME=fisher-kqb-${SETTING}-g4 TIME_LIMIT=7-00:00:00
        export EXPERIMENT_NAME=fisher_kqb_${SETTING}_u0096
        export OUTPUT_ROOT=${REPO_ROOT}/experiments/20261005_fisher_kqb_n4_n8/${SETTING}/raw/formal
        export RUN_CONFIG_TAG=g4_tp2_seed1_hvp_formal_${SETTING}
        ;;
    *) echo "Use probe, n4 or n8" >&2; exit 2 ;;
esac
bash examples/dynamic_staleness/submit_slurm.sh anchor
