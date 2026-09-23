#!/usr/bin/env bash
# Run on the Paracloud login node. This script submits two GPU jobs.
# Same mini-batch, no warmup; 24 rollout cycles, evaluated after every cycle.
set -euo pipefail

cd /data/run01/scyb980/cyt/src/verl-staleness

# The old pair wrapper forces 96 updates. Let each standalone run use its target.
# Let slurm_job.sh create fresh output instances with timestamps and job IDs.
unset KL_PAIR_DIR OUTPUT_DIR OUTPUT_INSTANCE_ROOT RUN_INSTANCE_TAG RUN_TIMESTAMP
unset RUN_CONFIG_TAG LOGGER_EXPERIMENT_NAME SLURM_DEPENDENCY

export PARTITION=gpu_a800 N_GPUS=4 GPU_PROFILE=a800
export RUNTIME_PROFILE=sis_offload ROLLOUT_TP_SIZE=2
export PYTHON_BIN="$PWD/.venv/bin/python"
export FULL_KL_EXPERIMENT=1 DRY_RUN=0 SMOKE=0 SEED=1
export MINI_PROMPT_BATCH=256 RESPONSES_PER_PROMPT=8
export LEARNING_RATE=1e-6 LR_WARMUP_STEPS=0
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=3072 KL_LOSS_COEF=0.001
export MODEL_PATH="$PWD/assets/models/Qwen3-8B-Base"
export TRAIN_FILE="$PWD/assets/datasets/SIS-Math-GRPO/data/pilot-seed1-u0292.parquet"
export MATH500_FILE="$PWD/assets/datasets/MATH-500/data/math500-sis.parquet"
export TRAIN_SHUFFLE=false
export KL_NUM_PROMPTS=64 KL_POSITIONS_PER_RESPONSE=8 KL_MEASUREMENT_SEED=20260903
export ENABLE_BENCHMARK_EVAL=1 VAL_BEFORE_TRAIN=true SAVE_FREQ=1000000

rerun_tag=$(date +%Y%m%d-%H%M%S)
export OUTPUT_ROOT="$PWD/outputs/full_kl_rerun/$rerun_tag"

REUSE_N=4 TARGET_OPTIMIZER_STEP=96 TEST_FREQ=1 \
TIME_LIMIT=24:00:00 JOB_NAME=fullkl2-rerun-n4-g4 \
EXPERIMENT_NAME=full_kl_n4_u0096 \
bash examples/dynamic_staleness/submit_slurm.sh anchor

REUSE_N=8 TARGET_OPTIMIZER_STEP=192 TEST_FREQ=1 \
TIME_LIMIT=48:00:00 JOB_NAME=fullkl2-rerun-n8-g4 \
EXPERIMENT_NAME=full_kl_n8_u0192 \
bash examples/dynamic_staleness/submit_slurm.sh anchor
