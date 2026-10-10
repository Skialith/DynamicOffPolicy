#!/usr/bin/env bash
# Forward-only cross-rollout observations; no automatic N controller.
set -euo pipefail
SETTING=${1:-n4}
shift "$(( $# > 0 ? 1 : 0 ))"
case "${SETTING}" in
    n4) REUSE_N=4 ;;
    n8) REUSE_N=8 ;;
    *) echo "Use n4 or n8" >&2; exit 2 ;;
esac
if [[ "${FULL_KL_HVP:-0}" != 0 || "${FULL_KL_JVP:-0}" != 0 || "${FISHER_VJP_N4_PREFLIGHT:-0}" != 0 ]]; then
    echo "The delayed-KL route forbids inherited JVP/HVP/Fisher preflight settings" >&2
    exit 2
fi
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
export FULL_KL_EXPERIMENT=1 FULL_KL_ACTOR_MEASUREMENT=1
export DELAYED_KL=1 DELAYED_KL_HORIZON=${DELAYED_KL_HORIZON:-$((2 * REUSE_N))}
export DELAYED_KL_ANCHOR_EVERY_ROLLOUTS=${DELAYED_KL_ANCHOR_EVERY_ROLLOUTS:-1}
export DRY_RUN=${DRY_RUN:-0} SMOKE=0 SEED=1 TRAIN_SHUFFLE=false
export REUSE_N TARGET_OPTIMIZER_STEP=${TARGET_OPTIMIZER_STEP:-$((2 * REUSE_N))}
export MINI_PROMPT_BATCH=256 RESPONSES_PER_PROMPT=8
export LEARNING_RATE=1e-6 LR_WARMUP_STEPS=0 KL_LOSS_COEF=0.001
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=3072 DATA_TRUNCATION=right
export ACTOR_MAX_TOKENS_PER_GPU=16384 INFER_MAX_TOKENS_PER_GPU=16384
export GPU_MEMORY_UTILIZATION=0.7 ROLLOUT_MAX_BATCHED_TOKENS=8192 ROLLOUT_MAX_NUM_SEQS=512
export ROLLOUT_MAX_MODEL_LEN=4096 REWARD_NUM_WORKERS=8
export ENABLE_BENCHMARK_EVAL=0 VAL_BEFORE_TRAIN=false TEST_FREQ=-1 SAVE_FREQ=-1
export KL_NUM_PROMPTS=64 KL_POSITIONS_PER_RESPONSE=8 KL_MEASUREMENT_SEED=20260903
export PERSISTENT_CACHE_ROOT=${ASSET_REPO}/.cache/paracloud
export JOB_NAME=delayed-kl-${SETTING}-g4 TIME_LIMIT=${TIME_LIMIT:-0}
export EXPERIMENT_NAME=delayed_kl_${SETTING}_u$(printf '%04d' "${TARGET_OPTIMIZER_STEP}")
export OUTPUT_ROOT=${REPO_ROOT}/outputs/delayed_anchor_kl/${SETTING}/raw/formal
export RUN_CONFIG_TAG=g4_tp2_seed1_forward_only_${SETTING}
if [[ "${DRY_RUN}" == 1 ]]; then
    exec bash examples/dynamic_staleness/run_staleness.sh anchor "$@"
fi
exec bash examples/dynamic_staleness/submit_slurm.sh anchor "$@"
