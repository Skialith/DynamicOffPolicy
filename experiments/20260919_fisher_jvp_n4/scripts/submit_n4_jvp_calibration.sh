#!/usr/bin/env bash
set -euo pipefail

cd /data/run01/scyb980/cyt/src/verl-staleness

: "${JVP_COMPAT_JOB:?Set JVP_COMPAT_JOB to the successful prerequisite job id}"
unset KL_PAIR_DIR OUTPUT_DIR OUTPUT_INSTANCE_ROOT RUN_INSTANCE_TAG RUN_TIMESTAMP
unset RUN_CONFIG_TAG LOGGER_EXPERIMENT_NAME

export PARTITION=gpu_a800 N_GPUS=4 GPU_PROFILE=a800
export RUNTIME_PROFILE=sis_offload ROLLOUT_TP_SIZE=2
export PYTHON_BIN="$PWD/.venv/bin/python"
export FULL_KL_EXPERIMENT=1 FULL_KL_JVP=1 DRY_RUN=0 SMOKE=0 SEED=1
export MINI_PROMPT_BATCH=256 RESPONSES_PER_PROMPT=8
export LEARNING_RATE=1e-6 LR_WARMUP_STEPS=0
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=3072 KL_LOSS_COEF=0.001
export MODEL_PATH="$PWD/assets/models/Qwen3-8B-Base"
export TRAIN_FILE="$PWD/assets/datasets/SIS-Math-GRPO/data/pilot-seed1-u0292.parquet"
export MATH500_FILE="$PWD/assets/datasets/MATH-500/data/math500-sis.parquet"
export TRAIN_SHUFFLE=false
export KL_NUM_PROMPTS=32 KL_POSITIONS_PER_RESPONSE=8 KL_MEASUREMENT_SEED=20260903
export KL_SELF_CHECK=true ENABLE_BENCHMARK_EVAL=0 SAVE_FREQ=-1

run_tag=$(date +%Y%m%d-%H%M%S)
export OUTPUT_ROOT="$PWD/outputs/fisher_jvp_n4/$run_tag"

export SLURM_DEPENDENCY="afterok:${JVP_COMPAT_JOB}"
probe_output=$(
    KL_NUM_PROMPTS=8 TIME_LIMIT=16:00:00 JOB_NAME=fisher-jvp-probe-n4 \
    EXPERIMENT_NAME=fisher_jvp_probe_n4_u0004 \
    bash examples/dynamic_staleness/submit_slurm.sh jvp_probe
)
echo "${probe_output}"
probe_job=$(awk '/submitted job/{print $3}' <<<"${probe_output}")
if [[ ! "${probe_job}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse probe job id" >&2
    exit 2
fi

export SLURM_DEPENDENCY="afterok:${probe_job}"
KL_NUM_PROMPTS=32 REUSE_N=4 TARGET_OPTIMIZER_STEP=40 \
TIME_LIMIT=48:00:00 JOB_NAME=fisher-jvp-n4-g4 \
EXPERIMENT_NAME=fisher_jvp_n4_u0040 \
bash examples/dynamic_staleness/submit_slurm.sh anchor
