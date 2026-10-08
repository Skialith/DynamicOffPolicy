#!/usr/bin/env bash
# Single normal-batch N=4 rollout; current layer VJP activations stay on GPU.
set -euo pipefail
REUSE_N=4
TARGET_OPTIMIZER_STEP=4
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"
module load miniforge3/25.11.0-1 cuda/12.8
unset KL_PAIR_DIR OUTPUT_DIR OUTPUT_INSTANCE_ROOT RUN_INSTANCE_TAG RUN_TIMESTAMP
unset RUN_CONFIG_TAG LOGGER_EXPERIMENT_NAME

ASSET_REPO=/data/run01/scyb980/cyt/src/verl-staleness
export PARTITION=gpu_a800 N_GPUS=4 GPU_PROFILE=a800
export PYTHON_BIN=${ASSET_REPO}/.venv/bin/python
export MODEL_PATH=${ASSET_REPO}/assets/models/Qwen3-8B-Base
export TRAIN_FILE=${ASSET_REPO}/assets/datasets/SIS-Math-GRPO/data/pilot-seed1-u0292.parquet
export VAL_FILE=${TRAIN_FILE} TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=16
export RUNTIME_PROFILE=sis_offload ROLLOUT_TP_SIZE=2
export FULL_KL_EXPERIMENT=1 FULL_KL_HVP=1 FULL_KL_JVP=0
export FULL_KL_ACTOR_MEASUREMENT=0 HVP_COMPUTE_QUADRATIC=0 HVP_VJP_CPU_OFFLOAD=0
export HVP_STRESS_PREFIX_LENGTH=4095 FISHER_VJP_N4_PREFLIGHT=1
export DRY_RUN=${DRY_RUN:-0} SMOKE=0 SEED=1 TRAIN_SHUFFLE=false
export REUSE_N TARGET_OPTIMIZER_STEP MINI_PROMPT_BATCH=256 RESPONSES_PER_PROMPT=8
export LEARNING_RATE=1e-6 LR_WARMUP_STEPS=0 KL_LOSS_COEF=0.001
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=3072 DATA_TRUNCATION=right
export ACTOR_MAX_TOKENS_PER_GPU=16384 INFER_MAX_TOKENS_PER_GPU=16384
export GPU_MEMORY_UTILIZATION=0.7 ROLLOUT_MAX_BATCHED_TOKENS=8192 ROLLOUT_MAX_NUM_SEQS=512
export ROLLOUT_MAX_MODEL_LEN=4096 REWARD_NUM_WORKERS=8
export ENABLE_BENCHMARK_EVAL=0 VAL_BEFORE_TRAIN=false TEST_FREQ=-1
export KL_NUM_PROMPTS=64 KL_POSITIONS_PER_RESPONSE=8 KL_MEASUREMENT_SEED=20260903
export HVP_POWER_STEPS=0 HVP_POWER_TOLERANCE=${HVP_POWER_TOLERANCE:-0.001}
export SAVE_FREQ=-1 PERSISTENT_CACHE_ROOT=${ASSET_REPO}/.cache/paracloud
export JOB_NAME=fisher-vjp-gpu-n4-g4 TIME_LIMIT=0
export EXPERIMENT_NAME=fisher_vjp_gpu_n4_u0004
export OUTPUT_ROOT=${REPO_ROOT}/outputs/fisher_vjp_gpu_n4/raw/probe
export RUN_CONFIG_TAG=g4_tp2_seed1_vjp_gpu_n4_one_rollout

# Submit only this GPU-VJP storage probe; existing deployments are untouched.
# An optional SLURM_DEPENDENCY is forwarded unchanged.
if [[ "${DRY_RUN}" == 1 ]]; then
    export TMPDIR=${TMPDIR:-/tmp/ds-vjp-gpu-n4-config}
    exec bash examples/dynamic_staleness/run_staleness.sh anchor "$@"
fi
bash examples/dynamic_staleness/submit_slurm.sh anchor "$@"
