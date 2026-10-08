#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# Exercise one complete N=4 outer step with the formal model and sequence limits.
# The tiny prompt batch keeps this a memory/runtime probe; it is not a scientific
# branch and its metrics must not be mixed with the 256-prompt experiment.
env \
    MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/assets/models/Qwen3-8B-Base}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/resource_probe}" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_base_fsdp2_n4_u0004}" \
    MINI_PROMPT_BATCH=1 \
    RESPONSES_PER_PROMPT=8 \
    REUSE_N=4 \
    TARGET_OPTIMIZER_STEP=4 \
    ENABLE_BENCHMARK_EVAL=0 \
    TRAIN_MAX_SAMPLES=16 \
    VAL_MAX_SAMPLES=8 \
    MAX_PROMPT_LENGTH=1024 \
    MAX_RESPONSE_LENGTH=3072 \
    FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}" \
    FSDP_PARAM_OFFLOAD="${FSDP_PARAM_OFFLOAD:-false}" \
    FSDP_OPTIMIZER_OFFLOAD="${FSDP_OPTIMIZER_OFFLOAD:-false}" \
    FSDP_OFFLOAD_POLICY="${FSDP_OFFLOAD_POLICY:-true}" \
    REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-true}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}" \
    ROLLOUT_MAX_BATCHED_TOKENS=4096 \
    ROLLOUT_MAX_NUM_SEQS=32 \
    "${REPO_ROOT}/examples/dynamic_staleness/run_staleness.sh" anchor "$@"
