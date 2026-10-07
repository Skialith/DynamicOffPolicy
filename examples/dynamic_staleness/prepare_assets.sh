#!/usr/bin/env bash
set -euo pipefail

WHAT=${1:-dataset}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
HF_BIN=${HF_BIN:-}
if [[ -z "${HF_BIN}" ]]; then
    if [[ -x "${REPO_ROOT}/.venv/bin/hf" ]]; then
        HF_BIN=${REPO_ROOT}/.venv/bin/hf
    else
        HF_BIN=$(command -v hf || true)
    fi
fi
PYTHON_BIN=${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}
DATASET_DIR=${DATASET_DIR:-${REPO_ROOT}/assets/datasets/DAPO-Math-17k}
MATH500_DIR=${MATH500_DIR:-${REPO_ROOT}/assets/datasets/MATH-500}
SIS_MATH_DIR=${SIS_MATH_DIR:-${REPO_ROOT}/assets/datasets/SIS-Math-GRPO}
SCHEDULE_SEED=${SCHEDULE_SEED:-1}
SCHEDULE_OPTIMIZER_STEPS=${SCHEDULE_OPTIMIZER_STEPS:-292}
SCHEDULE_PROMPTS_PER_UPDATE=${SCHEDULE_PROMPTS_PER_UPDATE:-256}
MODEL_DIR=${MODEL_DIR:-${REPO_ROOT}/assets/models/Qwen3-8B-Base}

case "${WHAT}" in
    dataset|all)
        : "${HF_BIN:?Could not find the hf executable; set HF_BIN explicitly}"
        "${HF_BIN}" download BytedTsinghua-SIA/DAPO-Math-17k \
            data/dapo-math-17k.parquet \
            --repo-type dataset \
            --local-dir "${DATASET_DIR}"
        ;;
esac

case "${WHAT}" in
    math500|all)
        : "${HF_BIN:?Could not find the hf executable; set HF_BIN explicitly}"
        "${HF_BIN}" download HuggingFaceH4/MATH-500 \
            test.jsonl \
            --repo-type dataset \
            --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
            --local-dir "${MATH500_DIR}/raw"
        ;;
esac

case "${WHAT}" in
    sis|all)
        "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_sis_math.py" \
            --input "${DATASET_DIR}/data/dapo-math-17k.parquet" \
            --output "${SIS_MATH_DIR}/data/train.parquet"
        "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_pilot_schedule.py" \
            --input "${SIS_MATH_DIR}/data/train.parquet" \
            --output "${SIS_MATH_DIR}/data/pilot-seed${SCHEDULE_SEED}-u$(printf '%04d' "${SCHEDULE_OPTIMIZER_STEPS}").parquet" \
            --optimizer-steps "${SCHEDULE_OPTIMIZER_STEPS}" \
            --prompts-per-update "${SCHEDULE_PROMPTS_PER_UPDATE}" \
            --seed "${SCHEDULE_SEED}"
        ;;
esac

case "${WHAT}" in
    math500|sis|all)
        "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_math500.py" \
            --input "${MATH500_DIR}/raw/test.jsonl" \
            --output "${MATH500_DIR}/data/math500-sis.parquet"
        ;;
esac

case "${WHAT}" in
    model|all)
        : "${HF_BIN:?Could not find the hf executable; set HF_BIN explicitly}"
        "${HF_BIN}" download Qwen/Qwen3-8B-Base --local-dir "${MODEL_DIR}"
        ;;
esac

if [[ "${WHAT}" != dataset && "${WHAT}" != math500 && "${WHAT}" != sis && "${WHAT}" != model && "${WHAT}" != all ]]; then
    echo "Usage: $0 {dataset|math500|sis|model|all}" >&2
    exit 2
fi
