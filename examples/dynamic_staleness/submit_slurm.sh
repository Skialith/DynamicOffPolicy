#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:-probe}
if [[ $# -gt 0 ]]; then
    shift
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

case "${TARGET}" in
    smoke|probe|adam_probe|performance_probe|kl_probe|geometry_probe|jvp_compat_probe|anchor|branch|n4_then_n8) ;;
    *)
        echo "Usage: $0 {smoke|probe|adam_probe|performance_probe|kl_probe|geometry_probe|jvp_compat_probe|anchor|branch|n4_then_n8} [extra Hydra overrides...]" >&2
        exit 2
        ;;
esac

: "${PARTITION:?Set PARTITION to the Slurm queue shown by sinfo}"
N_GPUS=${N_GPUS:-8}
GPU_PROFILE=${GPU_PROFILE:-auto}
PYTHON_BIN=${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/slurm}
JOB_NAME=${JOB_NAME:-staleness-${TARGET}}
SLURM_DEPENDENCY=${SLURM_DEPENDENCY:-}
SLURM_EXCLUDE_NODES=${SLURM_EXCLUDE_NODES:-}

if [[ ! "${N_GPUS}" =~ ^[1-8]$ ]]; then
    echo "N_GPUS must be an integer from 1 to 8, got: ${N_GPUS}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available; run this script on the cluster login node" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
export N_GPUS GPU_PROFILE PYTHON_BIN REPO_ROOT

submit_args=(
    --parsable
    --nodes=1
    --ntasks=1
    --partition="${PARTITION}"
    --gpus="${N_GPUS}"
    --job-name="${JOB_NAME}"
    --chdir="${REPO_ROOT}"
    --output="${LOG_DIR}/%x-%j.out"
    --error="${LOG_DIR}/%x-%j.out"
    --export=ALL
)
if [[ -n "${TIME_LIMIT:-}" ]]; then
    submit_args+=(--time="${TIME_LIMIT}")
fi
if [[ -n "${SLURM_DEPENDENCY}" ]]; then
    submit_args+=(--dependency="${SLURM_DEPENDENCY}")
fi
if [[ -n "${SLURM_EXCLUDE_NODES}" ]]; then
    submit_args+=(--exclude="${SLURM_EXCLUDE_NODES}")
fi

job_id=$(sbatch "${submit_args[@]}" "${SCRIPT_DIR}/slurm_job.sh" "${TARGET}" "$@")
echo "submitted job ${job_id} (${TARGET}, ${N_GPUS} GPU, partition=${PARTITION})"
if [[ -n "${SLURM_DEPENDENCY}" ]]; then
    echo "dependency: ${SLURM_DEPENDENCY}"
fi
if [[ -n "${SLURM_EXCLUDE_NODES}" ]]; then
    echo "excluded nodes: ${SLURM_EXCLUDE_NODES}"
fi
echo "log: ${LOG_DIR}/${JOB_NAME}-${job_id}.out"
