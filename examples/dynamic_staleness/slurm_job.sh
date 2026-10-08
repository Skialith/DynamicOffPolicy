#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:-probe}
if [[ $# -gt 0 ]]; then
    shift
fi

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
SCRIPT_DIR=${REPO_ROOT}/examples/dynamic_staleness
N_GPUS=${N_GPUS:-${SLURM_GPUS_ON_NODE:-2}}
PYTHON_BIN=${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}
GPU_PROFILE=${GPU_PROFILE:-auto}
SEED=${SEED:-1}
RUNTIME_PROFILE=${RUNTIME_PROFILE:-memory_safe}

case "${TARGET}" in
    smoke|probe|adam_probe|performance_probe|kl_probe|jvp_probe|jvp_compat_probe|fvp_compat_probe|exact_hvp_probe|fisher_fvp_probe|anchor|branch|n4_then_n8) ;;
    *)
        echo "Usage: $0 {smoke|probe|adam_probe|performance_probe|kl_probe|jvp_probe|jvp_compat_probe|fvp_compat_probe|exact_hvp_probe|fisher_fvp_probe|anchor|branch|n4_then_n8} [extra Hydra overrides...]" >&2
        exit 2
        ;;
esac

if [[ ! "${N_GPUS}" =~ ^[1-8]$ ]]; then
    echo "N_GPUS must be an integer from 1 to 8, got: ${N_GPUS}" >&2
    exit 2
fi

if [[ "${GPU_PROFILE}" == auto ]]; then
    gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
    case "${gpu_names,,}" in
        *a800*) GPU_PROFILE=a800 ;;
        *5090*) GPU_PROFILE=rtx5090 ;;
        *) GPU_PROFILE=generic ;;
    esac
fi

case "${GPU_PROFILE}" in
    a800)
        # Qwen3-8B fits on one A800, but TP=2 leaves more room for KV cache.
        # Odd GPU counts cannot be divided into TP=2 rollout groups.
        DEFAULT_ROLLOUT_TP_SIZE=$((N_GPUS % 2 == 0 ? 2 : 1))
        ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-${DEFAULT_ROLLOUT_TP_SIZE}}
        GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
        UPDATE_WEIGHTS_BUCKET_MEGABYTES=${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-3072}
        ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-16384}
        INFER_MAX_TOKENS_PER_GPU=${INFER_MAX_TOKENS_PER_GPU:-16384}
        ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS:-8192}
        ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-512}
        ;;
    rtx5090)
        # TP=1 avoids making rollout inference depend on fast GPU-to-GPU links.
        ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
        GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.55}
        UPDATE_WEIGHTS_BUCKET_MEGABYTES=${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-512}
        ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-8192}
        INFER_MAX_TOKENS_PER_GPU=${INFER_MAX_TOKENS_PER_GPU:-8192}
        ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS:-4096}
        ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}
        ;;
    generic)
        DEFAULT_ROLLOUT_TP_SIZE=$((N_GPUS % 2 == 0 ? 2 : 1))
        ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-${DEFAULT_ROLLOUT_TP_SIZE}}
        GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
        UPDATE_WEIGHTS_BUCKET_MEGABYTES=${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-1024}
        ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-8192}
        INFER_MAX_TOKENS_PER_GPU=${INFER_MAX_TOKENS_PER_GPU:-8192}
        ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS:-4096}
        ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}
        ;;
    *)
        echo "GPU_PROFILE must be auto, a800, rtx5090, or generic; got: ${GPU_PROFILE}" >&2
        exit 2
        ;;
esac

if [[ ! "${ROLLOUT_TP_SIZE}" =~ ^[1-8]$ ]] || ((N_GPUS % ROLLOUT_TP_SIZE != 0)); then
    echo "ROLLOUT_TP_SIZE must be a positive divisor of N_GPUS (${N_GPUS}), got: ${ROLLOUT_TP_SIZE}" >&2
    exit 2
fi

# Keep scientific outputs from different hardware layouts, seeds, and repeated
# submissions separate. The Slurm job id prevents collisions when two jobs start
# in the same second. Reuse RUN_INSTANCE_TAG explicitly only when resuming the
# same run instance.
RUN_CONFIG_TAG=${RUN_CONFIG_TAG:-g${N_GPUS}_tp${ROLLOUT_TP_SIZE}_seed${SEED}}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}
RUN_INSTANCE_TAG=${RUN_INSTANCE_TAG:-${RUN_TIMESTAMP}_job${SLURM_JOB_ID:-local}_${RUN_CONFIG_TAG}}
case "${TARGET}" in
    smoke)
        SMOKE_BASE_ROOT=${SMOKE_ROOT:-${REPO_ROOT}/outputs/smoke_staleness}
        SMOKE_ROOT=${OUTPUT_INSTANCE_ROOT:-${SMOKE_BASE_ROOT}/${RUN_INSTANCE_TAG}}
        EFFECTIVE_OUTPUT_ROOT=${SMOKE_ROOT}
        export SMOKE_ROOT
        ;;
    probe|kl_probe|jvp_probe|jvp_compat_probe|fvp_compat_probe|exact_hvp_probe|fisher_fvp_probe)
        OUTPUT_BASE_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/resource_probe}
        OUTPUT_ROOT=${OUTPUT_INSTANCE_ROOT:-${OUTPUT_BASE_ROOT}/${RUN_INSTANCE_TAG}}
        EFFECTIVE_OUTPUT_ROOT=${OUTPUT_ROOT}
        export OUTPUT_ROOT
        ;;
    performance_probe)
        OUTPUT_BASE_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/performance_probe}
        OUTPUT_ROOT=${OUTPUT_INSTANCE_ROOT:-${OUTPUT_BASE_ROOT}/${RUN_INSTANCE_TAG}}
        EFFECTIVE_OUTPUT_ROOT=${OUTPUT_ROOT}
        export OUTPUT_ROOT
        ;;
    adam_probe)
        OUTPUT_BASE_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/adam_probe}
        OUTPUT_ROOT=${OUTPUT_INSTANCE_ROOT:-${OUTPUT_BASE_ROOT}/${RUN_INSTANCE_TAG}}
        EFFECTIVE_OUTPUT_ROOT=${OUTPUT_ROOT}
        export OUTPUT_ROOT
        ;;
    anchor|branch|n4_then_n8)
        OUTPUT_BASE_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/dynamic_staleness}
        OUTPUT_ROOT=${OUTPUT_INSTANCE_ROOT:-${OUTPUT_BASE_ROOT}/${RUN_INSTANCE_TAG}}
        EFFECTIVE_OUTPUT_ROOT=${OUTPUT_ROOT}
        export OUTPUT_ROOT
        ;;
esac

# Persist only the reusable Parquet -> Arrow conversion on shared storage.
# Compilation, vLLM, and Ray caches stay on the compute node's local filesystem.
PERSISTENT_CACHE_ROOT=${PERSISTENT_CACHE_ROOT:-${REPO_ROOT}/.cache/paracloud}
LOCAL_SCRATCH_ROOT=${SLURM_TMPDIR:-/tmp}
JOB_LOCAL_ROOT=${JOB_LOCAL_ROOT:-${LOCAL_SCRATCH_ROOT}/ds-${SLURM_JOB_ID:-$$}}

export HF_HOME=${PERSISTENT_CACHE_ROOT}/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_XET_CACHE=${HF_HOME}/xet
export HF_DATASETS_CACHE=${PERSISTENT_CACHE_ROOT}/hf-datasets

export TMPDIR=${JOB_LOCAL_ROOT}/tmp
export RAY_TMPDIR=${JOB_LOCAL_ROOT}/ray
export XDG_CACHE_HOME=${JOB_LOCAL_ROOT}/xdg-cache
export XDG_CONFIG_HOME=${JOB_LOCAL_ROOT}/xdg-config
export VLLM_CACHE_ROOT=${JOB_LOCAL_ROOT}/vllm
export VLLM_CONFIG_ROOT=${JOB_LOCAL_ROOT}/vllm-config
export TRITON_CACHE_DIR=${JOB_LOCAL_ROOT}/triton
export TORCHINDUCTOR_CACHE_DIR=${JOB_LOCAL_ROOT}/torchinductor
export TORCH_EXTENSIONS_DIR=${JOB_LOCAL_ROOT}/torch-extensions
export CUDA_CACHE_PATH=${JOB_LOCAL_ROOT}/cuda
export NUMBA_CACHE_DIR=${JOB_LOCAL_ROOT}/numba
export MPLCONFIGDIR=${JOB_LOCAL_ROOT}/matplotlib

mkdir -p \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${HF_XET_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${TMPDIR}" \
    "${RAY_TMPDIR}" \
    "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" \
    "${VLLM_CACHE_ROOT}" \
    "${VLLM_CONFIG_ROOT}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${CUDA_CACHE_PATH}" \
    "${NUMBA_CACHE_DIR}" \
    "${MPLCONFIGDIR}"

export N_GPUS PYTHON_BIN GPU_PROFILE ROLLOUT_TP_SIZE SEED RUN_CONFIG_TAG RUNTIME_PROFILE
export RUN_TIMESTAMP RUN_INSTANCE_TAG
export GPU_MEMORY_UTILIZATION UPDATE_WEIGHTS_BUCKET_MEGABYTES
export ACTOR_MAX_TOKENS_PER_GPU INFER_MAX_TOKENS_PER_GPU
export ROLLOUT_MAX_BATCHED_TOKENS ROLLOUT_MAX_NUM_SEQS
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1

echo "job=${SLURM_JOB_ID:-none} host=$(hostname) target=${TARGET} gpu_profile=${GPU_PROFILE} runtime_profile=${RUNTIME_PROFILE} gpus=${N_GPUS} rollout_tp=${ROLLOUT_TP_SIZE}"
echo "python=${PYTHON_BIN} repo=${REPO_ROOT}"
echo "run_instance=${RUN_INSTANCE_TAG}"
echo "output_root=${EFFECTIVE_OUTPUT_ROOT}"
echo "persistent_cache=${PERSISTENT_CACHE_ROOT} job_local_cache=${JOB_LOCAL_ROOT}"
echo "execution_limits=actor_tokens_per_gpu:${ACTOR_MAX_TOKENS_PER_GPU},infer_tokens_per_gpu:${INFER_MAX_TOKENS_PER_GPU},rollout_batched_tokens:${ROLLOUT_MAX_BATCHED_TOKENS},rollout_max_seqs:${ROLLOUT_MAX_NUM_SEQS}"
nvidia-smi -L

if [[ "${FISHER_VJP_N4_PREFLIGHT:-0}" == 1 ]]; then
    bash "${SCRIPT_DIR}/verify_vjp_gpu_n4_setup.sh"
fi

case "${TARGET}" in
    fisher_fvp_probe)
        exec bash "${SCRIPT_DIR}/run_fvp_storage_probe.sh" "$@"
        ;;
    kl_probe)
        exec bash "${SCRIPT_DIR}/run_full_kl_probe.sh" "$@"
        ;;
    jvp_probe)
        exec bash "${SCRIPT_DIR}/run_full_kl_jvp_probe.sh" "$@"
        ;;
    jvp_compat_probe)
        exec bash "${SCRIPT_DIR}/run_fsdp_parameter_jvp_probe.sh" "$@"
        ;;
    fvp_compat_probe)
        exec bash "${SCRIPT_DIR}/run_fsdp_fisher_vector_product_probe.sh" "$@"
        ;;
    exact_hvp_probe)
        exec bash "${SCRIPT_DIR}/run_exact_kl_hvp_probe.sh" "$@"
        ;;
    smoke)
        exec "${SCRIPT_DIR}/run_smoke.sh" "$@"
        ;;
    probe)
        # The cloud probe checks the 8B execution path without writing a very
        # large optimizer checkpoint. Set SAVE_FREQ to a positive value only
        # after confirming that the shared-storage quota is large enough.
        export SAVE_FREQ=${SAVE_FREQ:--1}
        exec "${SCRIPT_DIR}/run_8b_resource_probe.sh" "$@"
        ;;
    adam_probe)
        # One tiny-batch optimizer step validates sharded Adam-state creation.
        # It deliberately skips eval, checkpointing, and throughput claims.
        export SAVE_FREQ=-1
        exec "${SCRIPT_DIR}/run_8b_adam_probe.sh" "$@"
        ;;
    performance_probe)
        # A full 256-prompt/update outer step. This validates first-Adam memory
        # and throughput for an unverified runtime profile without saving a
        # checkpoint or mixing probe metrics into the formal run directory.
        export SAVE_FREQ=-1
        exec "${SCRIPT_DIR}/run_8b_performance_probe.sh" "$@"
        ;;
    anchor)
        if [[ "${FULL_KL_EXPERIMENT:-0}" == 1 && -n "${KL_PAIR_DIR:-}" ]]; then
            exec bash "${SCRIPT_DIR}/run_full_kl_experiment.sh" "$@"
        fi
        exec "${SCRIPT_DIR}/run_staleness.sh" anchor "$@"
        ;;
    branch)
        exec "${SCRIPT_DIR}/run_staleness.sh" branch "$@"
        ;;
    n4_then_n8)
        exec "${SCRIPT_DIR}/run_n4_then_n8.sh" "$@"
        ;;
    *)
        echo "Usage: $0 {smoke|probe|adam_probe|performance_probe|kl_probe|jvp_probe|jvp_compat_probe|fvp_compat_probe|exact_hvp_probe|fisher_fvp_probe|anchor|branch|n4_then_n8} [extra Hydra overrides...]" >&2
        exit 2
        ;;
esac
