#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# One real optimizer.step() is enough to materialize the complete sharded Adam
# state. This probe intentionally uses a tiny prompt batch and N=1; it validates
# optimizer-state memory only, not formal-batch activation memory or throughput.
env \
    MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/assets/models/Qwen3-8B-Base}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/adam_probe}" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_${RUNTIME_PROFILE:-sis_offload}_g${N_GPUS:-1}_adam}" \
    SMOKE=1 \
    MINI_PROMPT_BATCH=1 \
    RESPONSES_PER_PROMPT=8 \
    REUSE_N=1 \
    TARGET_OPTIMIZER_STEP=1 \
    ENABLE_BENCHMARK_EVAL=0 \
    TRAIN_MAX_SAMPLES=1 \
    VAL_MAX_SAMPLES=1 \
    MAX_PROMPT_LENGTH=1024 \
    MAX_RESPONSE_LENGTH=3072 \
    SAVE_FREQ=-1 \
    "${REPO_ROOT}/examples/dynamic_staleness/run_staleness.sh" anchor "$@"
