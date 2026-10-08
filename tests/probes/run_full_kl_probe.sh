#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "${SCRIPT_DIR}/full_kl_fsdp_probe.py"
export FULL_KL_EXPERIMENT=1 RUNTIME_PROFILE=sis_offload KL_SELF_CHECK=true
export MINI_PROMPT_BATCH=4 RESPONSES_PER_PROMPT=8 KL_NUM_PROMPTS=64
export ENABLE_BENCHMARK_EVAL=1 VAL_MAX_SAMPLES=4 FILTER_PROMPT_WORKERS=2
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=256
export MODEL_PATH=${REPO_ROOT}/assets/models/Qwen3-0.6B-Base
export TARGET_OPTIMIZER_STEP=16
for probe_n in 4 8; do
    export REUSE_N=${probe_n} TEST_FREQ=$((8 / probe_n))
    export OUTPUT_DIR=${OUTPUT_ROOT}/small_n${probe_n}
    EXPERIMENT_NAME=kl_probe_small_n${probe_n} bash "${REPO_ROOT}/examples/dynamic_staleness/run_staleness.sh" anchor
    "${PYTHON_BIN}" "${REPO_ROOT}/tests/checks/verify_full_kl_run.py" "${OUTPUT_DIR}" \
        --updates 16 --n "${probe_n}" --eval-steps 0,8,16 --eval-count 4
done
"${PYTHON_BIN}" "${REPO_ROOT}/examples/dynamic_staleness/endpoint_full_kl.py" \
    --run-n4 "${OUTPUT_ROOT}/small_n4" --run-n8 "${OUTPUT_ROOT}/small_n8" \
    --prompts-per-branch 8 --output "${OUTPUT_ROOT}/endpoint_kl.json"

# Exercise the actual 8B model and full context length without a large checkpoint.
export MODEL_PATH=${REPO_ROOT}/assets/models/Qwen3-8B-Base
export MAX_RESPONSE_LENGTH=3072 TARGET_OPTIMIZER_STEP=4 REUSE_N=4
export ENABLE_BENCHMARK_EVAL=0 VAL_BEFORE_TRAIN=false TEST_FREQ=-1 SAVE_FREQ=-1
export OUTPUT_DIR=${OUTPUT_ROOT}/eight_b_n4
EXPERIMENT_NAME=kl_probe_8b_n4 bash "${REPO_ROOT}/examples/dynamic_staleness/run_staleness.sh" anchor
"${PYTHON_BIN}" "${REPO_ROOT}/tests/checks/verify_full_kl_run.py" "${OUTPUT_DIR}" \
    --updates 4 --n 4 --eval-steps '' --no-endpoint
echo "FULL_KL_PROBE_PASSED"
