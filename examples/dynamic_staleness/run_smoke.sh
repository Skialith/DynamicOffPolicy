#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
SMOKE_ROOT=${SMOKE_ROOT:-${REPO_ROOT}/outputs/smoke_staleness}
SMOKE_MODEL=${SMOKE_MODEL:-${REPO_ROOT}/assets/models/Qwen3-0.6B-Base}

env \
    SMOKE=1 \
    MODEL_PATH="${SMOKE_MODEL}" \
    FSDP_PARAM_OFFLOAD=false \
    FSDP_OPTIMIZER_OFFLOAD=false \
    FSDP_OFFLOAD_POLICY=false \
    REF_PARAM_OFFLOAD=false \
    OUTPUT_ROOT="${SMOKE_ROOT}" \
    EXPERIMENT_NAME=smoke_anchor_n4_u0004 \
    MINI_PROMPT_BATCH=1 \
    RESPONSES_PER_PROMPT=2 \
    REUSE_N=4 \
    TARGET_OPTIMIZER_STEP=4 \
    SAVE_FREQ=1000000 \
    TRAIN_MAX_SAMPLES=32 \
    VAL_MAX_SAMPLES=8 \
    MAX_PROMPT_LENGTH=512 \
    MAX_RESPONSE_LENGTH=64 \
    ACTOR_MAX_TOKENS_PER_GPU=1024 \
    INFER_MAX_TOKENS_PER_GPU=1024 \
    ROLLOUT_MAX_BATCHED_TOKENS=576 \
    ROLLOUT_MAX_NUM_SEQS=8 \
    "${SCRIPT_DIR}/run_staleness.sh" anchor actor_rollout_ref.actor.use_kl_loss=false "$@"

ANCHOR_CKPT=${SMOKE_ROOT}/smoke_anchor_n4_u0004/global_step_1
env \
    SMOKE=1 \
    MODEL_PATH="${SMOKE_MODEL}" \
    FSDP_PARAM_OFFLOAD=false \
    FSDP_OPTIMIZER_OFFLOAD=false \
    FSDP_OFFLOAD_POLICY=false \
    REF_PARAM_OFFLOAD=false \
    OUTPUT_ROOT="${SMOKE_ROOT}" \
    EXPERIMENT_NAME=smoke_branch_n8_u0004_0012 \
    MINI_PROMPT_BATCH=1 \
    RESPONSES_PER_PROMPT=2 \
    REUSE_N=8 \
    TARGET_OPTIMIZER_STEP=12 \
    SAVE_FREQ=1000000 \
    ANCHOR_CKPT="${ANCHOR_CKPT}" \
    TRAIN_MAX_SAMPLES=32 \
    VAL_MAX_SAMPLES=8 \
    MAX_PROMPT_LENGTH=512 \
    MAX_RESPONSE_LENGTH=64 \
    ACTOR_MAX_TOKENS_PER_GPU=1024 \
    INFER_MAX_TOKENS_PER_GPU=1024 \
    ROLLOUT_MAX_BATCHED_TOKENS=576 \
    ROLLOUT_MAX_NUM_SEQS=16 \
    "${SCRIPT_DIR}/run_staleness.sh" branch actor_rollout_ref.actor.use_kl_loss=false "$@"
