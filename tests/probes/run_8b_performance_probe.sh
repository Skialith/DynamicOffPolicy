#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# One full scientific-size N=4 outer step, isolated from formal outputs. Unlike
# run_8b_resource_probe.sh, this keeps MINI_PROMPT_BATCH=256 so it can validate
# first-Adam memory and measure realistic throughput for a runtime profile.
env \
    MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/assets/models/Qwen3-8B-Base}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/performance_probe}" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_${RUNTIME_PROFILE:-memory_safe}_g${N_GPUS:-1}_n4_u0004}" \
    MINI_PROMPT_BATCH=256 \
    RESPONSES_PER_PROMPT=8 \
    REUSE_N=4 \
    TARGET_OPTIMIZER_STEP=4 \
    ENABLE_BENCHMARK_EVAL=0 \
    TRAIN_MAX_SAMPLES=-1 \
    VAL_MAX_SAMPLES=16 \
    MAX_PROMPT_LENGTH=1024 \
    MAX_RESPONSE_LENGTH=3072 \
    SAVE_FREQ=-1 \
    "${REPO_ROOT}/examples/dynamic_staleness/run_staleness.sh" anchor "$@"
