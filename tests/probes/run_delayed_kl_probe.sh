#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
if [[ "${FULL_KL_HVP:-0}" != 0 || "${FULL_KL_JVP:-0}" != 0 || "${FISHER_VJP_N4_PREFLIGHT:-0}" != 0 ]]; then
    echo "Delayed KL probe forbids JVP/HVP/Fisher preflight" >&2
    exit 2
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 "${SCRIPT_DIR}/delayed_kl_fsdp_probe.py"
# Explicit integration probe: reduced batch/length, never a formal cost or capability result.
export FULL_KL_EXPERIMENT=1 FULL_KL_ACTOR_MEASUREMENT=1
export DELAYED_KL=1 DELAYED_KL_HORIZON=8 DELAYED_KL_ANCHOR_EVERY_ROLLOUTS=1
export RUNTIME_PROFILE=sis_offload REUSE_N=4 TARGET_OPTIMIZER_STEP=8
export MINI_PROMPT_BATCH=4 RESPONSES_PER_PROMPT=8 KL_NUM_PROMPTS=8 KL_POSITIONS_PER_RESPONSE=8
export MODEL_PATH=${REPO_ROOT}/assets/models/Qwen3-8B-Base
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=256 ROLLOUT_MAX_MODEL_LEN=1280
export FILTER_PROMPT_WORKERS=2 REWARD_NUM_WORKERS=2 LR_WARMUP_STEPS=0
export ENABLE_BENCHMARK_EVAL=0 VAL_BEFORE_TRAIN=false TEST_FREQ=-1 SAVE_FREQ=-1
export OUTPUT_DIR=${OUTPUT_ROOT}/eight_b_n4 EXPERIMENT_NAME=delayed_kl_8b_integration_probe
unset KL_PAIR_DIR
bash "${REPO_ROOT}/examples/dynamic_staleness/run_staleness.sh" anchor "$@"
"${PYTHON_BIN}" "${REPO_ROOT}/tests/checks/verify_delayed_kl_run.py" "${OUTPUT_DIR}" --updates 8 --reuse-n 4 --horizon 8
