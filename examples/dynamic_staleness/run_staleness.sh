#!/usr/bin/env bash
set -euo pipefail

PHASE=${1:-anchor}
if [[ $# -gt 0 ]]; then
    shift
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    DEFAULT_PYTHON_BIN=${REPO_ROOT}/.venv/bin/python
else
    DEFAULT_PYTHON_BIN=$(command -v python || true)
fi
PYTHON_BIN=${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}
N_GPUS=${N_GPUS:-2}
if [[ ! "${N_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_GPUS must be a positive integer, got: ${N_GPUS}" >&2
    exit 2
fi
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

MINI_PROMPT_BATCH=${MINI_PROMPT_BATCH:-256}
RESPONSES_PER_PROMPT=${RESPONSES_PER_PROMPT:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-10}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
DATA_TRUNCATION=${DATA_TRUNCATION:-right}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
UPDATE_WEIGHTS_BUCKET_MEGABYTES=${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-3072}
SEED=${SEED:-1}
MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/assets/models/Qwen3-8B-Base}
TRAIN_FILE=${TRAIN_FILE:-${REPO_ROOT}/assets/datasets/SIS-Math-GRPO/data/pilot-seed${SEED}-u0292.parquet}
MATH500_FILE=${MATH500_FILE:-${REPO_ROOT}/assets/datasets/MATH-500/data/math500-sis.parquet}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
TRAIN_SHUFFLE=${TRAIN_SHUFFLE:-false}
FILTER_PROMPT_WORKERS=${FILTER_PROMPT_WORKERS:-16}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/dynamic_staleness}
PROJECT_NAME=${PROJECT_NAME:-dynamic-staleness-pilot}
RUNTIME_PROFILE=${RUNTIME_PROFILE:-memory_safe}

# Runtime profiles only change how the same scientific batch is executed. Keep
# one profile fixed across an anchor and all of its branches. `memory_safe` is
# the verified Paracloud default. `sis_offload` mirrors the SIS Qwen3-8B recipe's
# legacy FSDP stage-level offload and must pass a full-batch performance probe
# on the selected GPU count before a formal run. `gpu_resident` is an explicit
# no-offload experiment and has the highest first-Adam OOM risk.
case "${RUNTIME_PROFILE}" in
    memory_safe)
        FSDP_STRATEGY=${FSDP_STRATEGY:-fsdp2}
        FSDP_PARAM_OFFLOAD=${FSDP_PARAM_OFFLOAD:-${FSDP_OFFLOAD:-false}}
        FSDP_OPTIMIZER_OFFLOAD=${FSDP_OPTIMIZER_OFFLOAD:-${FSDP_OFFLOAD:-false}}
        FSDP_OFFLOAD_POLICY=${FSDP_OFFLOAD_POLICY:-true}
        REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-${FSDP_OFFLOAD:-true}}
        ;;
    sis_offload)
        FSDP_STRATEGY=${FSDP_STRATEGY:-fsdp}
        FSDP_PARAM_OFFLOAD=${FSDP_PARAM_OFFLOAD:-${FSDP_OFFLOAD:-true}}
        FSDP_OPTIMIZER_OFFLOAD=${FSDP_OPTIMIZER_OFFLOAD:-${FSDP_OFFLOAD:-true}}
        FSDP_OFFLOAD_POLICY=${FSDP_OFFLOAD_POLICY:-false}
        REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-${FSDP_OFFLOAD:-true}}
        ;;
    gpu_resident)
        FSDP_STRATEGY=${FSDP_STRATEGY:-fsdp2}
        FSDP_PARAM_OFFLOAD=${FSDP_PARAM_OFFLOAD:-false}
        FSDP_OPTIMIZER_OFFLOAD=${FSDP_OPTIMIZER_OFFLOAD:-false}
        FSDP_OFFLOAD_POLICY=${FSDP_OFFLOAD_POLICY:-false}
        REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-true}
        ;;
    *)
        echo "RUNTIME_PROFILE must be memory_safe, sis_offload, or gpu_resident; got: ${RUNTIME_PROFILE}" >&2
        exit 2
        ;;
esac

# Match the public SIS Qwen3-8B GRPO execution ceilings. These are packing and
# scheduler limits, not scientific batch sizes: increasing them reduces the
# number of actor/ref/log-prob micro-batches and lets vLLM schedule more work at
# once without changing prompts/update, responses/prompt, or optimizer steps.
ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-16384}
INFER_MAX_TOKENS_PER_GPU=${INFER_MAX_TOKENS_PER_GPU:-16384}
ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-512}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-$((N_GPUS < 2 ? N_GPUS : 2))}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-8}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
SMOKE=${SMOKE:-0}
DRY_RUN=${DRY_RUN:-0}
FULL_KL_EXPERIMENT=${FULL_KL_EXPERIMENT:-0}
FULL_KL_GEOMETRY=${FULL_KL_GEOMETRY:-0}
FULL_KL_JVP=${FULL_KL_JVP:-0}
FULL_KL_HVP=${FULL_KL_HVP:-0}
FULL_KL_ACTOR_MEASUREMENT=${FULL_KL_ACTOR_MEASUREMENT:-1}
HVP_COMPUTE_QUADRATIC=${HVP_COMPUTE_QUADRATIC:-1}
if [[ ! "${FULL_KL_ACTOR_MEASUREMENT}" =~ ^[01]$ || ! "${HVP_COMPUTE_QUADRATIC}" =~ ^[01]$ ]]; then
    echo "FULL_KL_ACTOR_MEASUREMENT and HVP_COMPUTE_QUADRATIC must be 0 or 1" >&2
    exit 2
fi
if [[ "${FULL_KL_HVP}" != 0 && "${FULL_KL_HVP}" != 1 ]]; then
    echo "FULL_KL_HVP must be 0 or 1" >&2
    exit 2
fi
if [[ "${FULL_KL_HVP}" == 1 ]] && [[ "${FULL_KL_EXPERIMENT}" != 1 || "${FULL_KL_JVP}" != 0 || "${N_GPUS}" != 4 || "${RUNTIME_PROFILE}" != sis_offload ]]; then
    echo "Exact HVP requires full-KL, no JVP, four GPUs and sis_offload" >&2
    exit 2
fi
if [[ "${FULL_KL_GEOMETRY}" != 0 && "${FULL_KL_GEOMETRY}" != 1 ]]; then
    echo "FULL_KL_GEOMETRY must be 0 or 1, got: ${FULL_KL_GEOMETRY}" >&2
    exit 2
fi
if [[ "${FULL_KL_GEOMETRY}" == 1 && "${FULL_KL_EXPERIMENT}" != 1 ]]; then
    echo "FULL_KL_GEOMETRY=1 requires FULL_KL_EXPERIMENT=1" >&2
    exit 2
fi
if [[ "${FULL_KL_JVP}" != 0 && "${FULL_KL_JVP}" != 1 ]]; then
    echo "FULL_KL_JVP must be 0 or 1, got: ${FULL_KL_JVP}" >&2
    exit 2
fi
if [[ "${FULL_KL_JVP}" == 1 && "${FULL_KL_GEOMETRY}" != 1 ]]; then
    echo "FULL_KL_JVP=1 requires FULL_KL_GEOMETRY=1" >&2
    exit 2
fi
if [[ "${FULL_KL_EXPERIMENT}" == 1 ]]; then
    if [[ "${PHASE}" != anchor ]]; then
        echo "The full-KL experiment starts fresh and does not resume optimizer state" >&2
        exit 2
    fi
    TARGET_OPTIMIZER_STEP=${TARGET_OPTIMIZER_STEP:-96}
    KL_NUM_PROMPTS=${KL_NUM_PROMPTS:-64}
    KL_POSITIONS_PER_RESPONSE=${KL_POSITIONS_PER_RESPONSE:-8}
    KL_MEASUREMENT_SEED=${KL_MEASUREMENT_SEED:-20260903}
    KL_SELF_CHECK=${KL_SELF_CHECK:-false}
    EVAL_TEMPERATURE=1.0
    EVAL_TOP_P=0.95
    EVAL_DO_SAMPLE=true
    EVAL_N=1
    SAVE_FREQ=${SAVE_FREQ:-1000000}
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-full_kl_n${REUSE_N:-4}_u$(printf '%04d' "${TARGET_OPTIMIZER_STEP}")}
    PROJECT_NAME=dynamic-staleness-full-kl
fi

if [[ ! "${ROLLOUT_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || (( N_GPUS % ROLLOUT_TP_SIZE != 0 )); then
    echo "ROLLOUT_TP_SIZE must be a positive divisor of N_GPUS (${N_GPUS}), got: ${ROLLOUT_TP_SIZE}" >&2
    exit 2
fi
if [[ ! "${REWARD_NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REWARD_NUM_WORKERS must be a positive integer, got: ${REWARD_NUM_WORKERS}" >&2
    exit 2
fi

case "${PHASE}" in
    anchor)
        REUSE_N=${REUSE_N:-4}
        TARGET_OPTIMIZER_STEP=${TARGET_OPTIMIZER_STEP:-100}
        START_OPTIMIZER_STEP=0
        START_ROLLOUT_STEP=0
        RESUME_MODE=disable
        RESUME_FROM=null
        ;;
    branch)
        : "${ANCHOR_CKPT:?Set ANCHOR_CKPT to the shared global_step checkpoint}"
        REUSE_N=${REUSE_N:?Set REUSE_N to 1, 4, 8, 16, or 32}
        TARGET_OPTIMIZER_STEP=${TARGET_OPTIMIZER_STEP:-196}
        STATE_FILE=${ANCHOR_CKPT}/staleness_state.json
        if [[ ! -f "${STATE_FILE}" ]]; then
            echo "Missing checkpoint staleness state: ${STATE_FILE}" >&2
            exit 2
        fi
        read -r START_OPTIMIZER_STEP START_ROLLOUT_STEP STATE_MINI_BATCH STATE_RESPONSES < <(
            "${PYTHON_BIN}" -c \
                'import json,sys; s=json.load(open(sys.argv[1])); print(s["optimizer_step"],s["rollout_step"],s["mini_prompt_batch_size"],s["responses_per_prompt"])' \
                "${STATE_FILE}"
        )
        if [[ "${STATE_MINI_BATCH}" -ne "${MINI_PROMPT_BATCH}" || "${STATE_RESPONSES}" -ne "${RESPONSES_PER_PROMPT}" ]]; then
            echo "Branch must keep mini-prompt batch and responses per prompt unchanged" >&2
            exit 2
        fi
        RESUME_MODE=resume_path
        RESUME_FROM=${ANCHOR_CKPT}
        ;;
    *)
        echo "Usage: $0 {anchor|branch} [extra Hydra overrides...]" >&2
        exit 2
        ;;
esac

if [[ "${FULL_KL_EXPERIMENT}" == 1 ]]; then
    TEST_FREQ=${TEST_FREQ:-$((48 / REUSE_N))}
fi

# Formal runs evaluate MATH500 after every complete rollout/update cycle. This
# keeps the controlled N consecutive optimizer steps uninterrupted. Smoke and
# resource probes opt out because they validate execution, not model quality.
if [[ -z "${ENABLE_BENCHMARK_EVAL+x}" ]]; then
    if [[ "${SMOKE}" == 1 ]]; then
        ENABLE_BENCHMARK_EVAL=0
    else
        ENABLE_BENCHMARK_EVAL=1
    fi
fi
if [[ "${ENABLE_BENCHMARK_EVAL}" != 0 && "${ENABLE_BENCHMARK_EVAL}" != 1 ]]; then
    echo "ENABLE_BENCHMARK_EVAL must be 0 or 1, got: ${ENABLE_BENCHMARK_EVAL}" >&2
    exit 2
fi

if [[ "${ENABLE_BENCHMARK_EVAL}" == 1 ]]; then
    VAL_FILE=${VAL_FILE:-${MATH500_FILE}}
    VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
    TEST_FREQ=${TEST_FREQ:-1}
    if [[ "${PHASE}" == anchor ]]; then
        VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
    else
        # update 100 was already evaluated by the shared anchor
        VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
    fi
    EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.0}
    EVAL_TOP_P=${EVAL_TOP_P:-1.0}
    EVAL_DO_SAMPLE=${EVAL_DO_SAMPLE:-false}
    EVAL_N=${EVAL_N:-1}
else
    VAL_FILE=${VAL_FILE:-${TRAIN_FILE}}
    VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-16}
    TEST_FREQ=${TEST_FREQ:--1}
    VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
    EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.0}
    EVAL_TOP_P=${EVAL_TOP_P:-1.0}
    EVAL_DO_SAMPLE=${EVAL_DO_SAMPLE:-false}
    EVAL_N=${EVAL_N:-1}
fi

# A positive save_freq still forces a save on the last outer step. By default
# only the shared anchor is resumable; branches keep metrics/generations only.
if [[ -z "${SAVE_FREQ+x}" ]]; then
    if [[ "${PHASE}" == anchor ]]; then
        SAVE_FREQ=1000000
    else
        SAVE_FREQ=-1
    fi
fi

if [[ "${SMOKE}" != 1 ]]; then
    case "${REUSE_N}" in
        1|4|8|16|32) ;;
        *)
            echo "REUSE_N must be 1, 4, 8, 16, or 32" >&2
            exit 2
            ;;
    esac
fi
if (( TARGET_OPTIMIZER_STEP <= START_OPTIMIZER_STEP )); then
    echo "TARGET_OPTIMIZER_STEP must exceed the checkpoint optimizer step" >&2
    exit 2
fi
REMAINING_UPDATES=$((TARGET_OPTIMIZER_STEP - START_OPTIMIZER_STEP))
if (( REMAINING_UPDATES % REUSE_N != 0 )); then
    echo "Remaining optimizer updates (${REMAINING_UPDATES}) must be divisible by N=${REUSE_N}" >&2
    exit 2
fi

ADDED_ROLLOUT_STEPS=$((REMAINING_UPDATES / REUSE_N))
TOTAL_ROLLOUT_STEPS=$((START_ROLLOUT_STEP + ADDED_ROLLOUT_STEPS))
TRAIN_PROMPT_BATCH=$((MINI_PROMPT_BATCH * REUSE_N))
RESPONSES_PER_UPDATE=$((MINI_PROMPT_BATCH * RESPONSES_PER_PROMPT))
if [[ "${PHASE}" == anchor ]]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-anchor_n${REUSE_N}_u$(printf '%04d' "${TARGET_OPTIMIZER_STEP}")}
else
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-branch_n${REUSE_N}_u$(printf '%04d' "${START_OPTIMIZER_STEP}")_$(printf '%04d' "${TARGET_OPTIMIZER_STEP}")}
fi
# OUTPUT_ROOT already carries the timestamped run instance when launched by
# slurm_job.sh. TensorBoard stores runs outside OUTPUT_ROOT, so prefix its name
# separately to avoid collisions between repeated submissions.
if [[ -n "${RUN_INSTANCE_TAG:-}" ]]; then
    LOGGER_EXPERIMENT_NAME=${LOGGER_EXPERIMENT_NAME:-${RUN_INSTANCE_TAG}_${EXPERIMENT_NAME}}
else
    LOGGER_EXPERIMENT_NAME=${LOGGER_EXPERIMENT_NAME:-${EXPERIMENT_NAME}}
fi
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}
if [[ "${ENABLE_BENCHMARK_EVAL}" == 1 ]]; then
    VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${OUTPUT_DIR}/eval_generations}
else
    VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-null}
fi

echo "phase=${PHASE} N=${REUSE_N} gpus=${N_GPUS} rollout_tp=${ROLLOUT_TP_SIZE} prompts/rollout=${TRAIN_PROMPT_BATCH} responses/update=${RESPONSES_PER_UPDATE}"
echo "optimizer updates ${START_OPTIMIZER_STEP} -> ${TARGET_OPTIMIZER_STEP}; rollout steps ${START_ROLLOUT_STEP} -> ${TOTAL_ROLLOUT_STEPS}"
echo "output=${OUTPUT_DIR}"
echo "logger_experiment=${LOGGER_EXPERIMENT_NAME}"
echo "benchmark_eval=${ENABLE_BENCHMARK_EVAL} val_file=${VAL_FILE} test_freq_outer_steps=${TEST_FREQ}"
echo "reward_manager=remote reward_workers=${REWARD_NUM_WORKERS}"
echo "save_freq_outer_steps=${SAVE_FREQ} validation_data_dir=${VALIDATION_DATA_DIR}"
echo "runtime_profile=${RUNTIME_PROFILE} fsdp=${FSDP_STRATEGY} offload_policy=${FSDP_OFFLOAD_POLICY} param_offload=${FSDP_PARAM_OFFLOAD} optimizer_offload=${FSDP_OPTIMIZER_OFFLOAD} ref_offload=${REF_PARAM_OFFLOAD}"
echo "execution_limits=actor_tokens_per_gpu:${ACTOR_MAX_TOKENS_PER_GPU},infer_tokens_per_gpu:${INFER_MAX_TOKENS_PER_GPU},rollout_batched_tokens:${ROLLOUT_MAX_BATCHED_TOKENS},rollout_max_seqs:${ROLLOUT_MAX_NUM_SEQS}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ "${DRY_RUN}" != 1 ]]; then
    if [[ "${ENABLE_BENCHMARK_EVAL}" == 1 && ! -f "${VAL_FILE}" ]]; then
        echo "Missing benchmark validation parquet: ${VAL_FILE}" >&2
        exit 2
    fi
    PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" "${SCRIPT_DIR}/check_env.py" \
        --repo-root "${REPO_ROOT}" \
        --model "${MODEL_PATH}" \
        --train-file "${TRAIN_FILE}" \
        --require-assets \
        --require-gpus "${N_GPUS}"
fi

overrides=(
    "algorithm.adv_estimator=grpo"
    "algorithm.use_kl_in_reward=false"
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${VAL_FILE}"
    "data.prompt_key=prompt"
    "data.train_batch_size=${TRAIN_PROMPT_BATCH}"
    "data.train_max_samples=${TRAIN_MAX_SAMPLES}"
    "data.val_max_samples=${VAL_MAX_SAMPLES}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=true"
    "data.filter_overlong_prompts_workers=${FILTER_PROMPT_WORKERS}"
    "data.truncation=${DATA_TRUNCATION}"
    "data.shuffle=${TRAIN_SHUFFLE}"
    "data.validation_shuffle=false"
    "data.seed=${SEED}"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.use_remove_padding=true"
    "actor_rollout_ref.model.enable_gradient_checkpointing=true"
    "actor_rollout_ref.actor.strategy=${FSDP_STRATEGY}"
    "actor_rollout_ref.actor.fsdp_config.strategy=${FSDP_STRATEGY}"
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=${N_GPUS}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${FSDP_PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${FSDP_OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.offload_policy=${FSDP_OFFLOAD_POLICY}"
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_PROMPT_BATCH}"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.shuffle=false"
    "actor_rollout_ref.actor.use_dynamic_bsz=true"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKENS_PER_GPU}"
    "actor_rollout_ref.actor.use_kl_loss=true"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.actor.clip_ratio_low=0.2"
    "actor_rollout_ref.actor.clip_ratio_high=0.2"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.loss_agg_mode=token-mean"
    "actor_rollout_ref.actor.log_staleness_metrics=true"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.calculate_log_probs=true"
    "actor_rollout_ref.rollout.n=${RESPONSES_PER_PROMPT}"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.top_p=1.0"
    "actor_rollout_ref.rollout.top_k=-1"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MEGABYTES}"
    "actor_rollout_ref.rollout.enable_chunked_prefill=true"
    "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}"
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKENS_PER_GPU}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${EVAL_TEMPERATURE}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${EVAL_TOP_P}"
    "actor_rollout_ref.rollout.val_kwargs.top_k=-1"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=${EVAL_DO_SAMPLE}"
    "actor_rollout_ref.rollout.val_kwargs.n=${EVAL_N}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKENS_PER_GPU}"
    # Math-Verify uses signal.alarm() and cannot run in DAPO's thread pool.
    # The remote manager evaluates rewards in dedicated Ray actor processes.
    "reward.reward_manager.name=remote"
    "reward.num_workers=${REWARD_NUM_WORKERS}"
    "trainer.balance_batch=false"
    "trainer.use_legacy_worker_impl=enable"
    "trainer.logger=[console,tensorboard]"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${LOGGER_EXPERIMENT_NAME}"
    "trainer.nnodes=1"
    "trainer.n_gpus_per_node=${N_GPUS}"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.validation_data_dir=${VALIDATION_DATA_DIR}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.total_epochs=1000"
    "trainer.total_training_steps=${TOTAL_ROLLOUT_STEPS}"
    "trainer.default_local_dir=${OUTPUT_DIR}"
    "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}"
    "trainer.resume_mode=${RESUME_MODE}"
    "trainer.resume_from_path=${RESUME_FROM}"
)

if [[ "${FULL_KL_EXPERIMENT}" == 1 ]]; then
    FULL_KL_GEOMETRY_BOOL=false
    FULL_KL_ACTOR_BOOL=false
    if [[ "${FULL_KL_ACTOR_MEASUREMENT}" == 1 ]]; then
        FULL_KL_ACTOR_BOOL=true
    fi
    if [[ "${FULL_KL_GEOMETRY}" == 1 ]]; then
        FULL_KL_GEOMETRY_BOOL=true
    fi
    FULL_KL_JVP_BOOL=false
    if [[ "${FULL_KL_JVP}" == 1 ]]; then
        FULL_KL_JVP_BOOL=true
    fi
    overrides+=(
        "actor_rollout_ref.actor.full_kl_measurement=true"
        "actor_rollout_ref.actor.full_kl_actor_measurement=${FULL_KL_ACTOR_BOOL}"
        "actor_rollout_ref.actor.full_kl_geometry_measurement=${FULL_KL_GEOMETRY_BOOL}"
        "actor_rollout_ref.actor.full_kl_jvp_measurement=${FULL_KL_JVP_BOOL}"
        "actor_rollout_ref.actor.full_kl_num_prompts=${KL_NUM_PROMPTS}"
        "actor_rollout_ref.actor.full_kl_positions_per_response=${KL_POSITIONS_PER_RESPONSE}"
        "actor_rollout_ref.actor.full_kl_seed=${KL_MEASUREMENT_SEED}"
        "actor_rollout_ref.actor.full_kl_self_check=${KL_SELF_CHECK}"
        "actor_rollout_ref.actor.optim.step_unit=optimizer_update"
        "actor_rollout_ref.actor.optim.zero_indexed_step=false"
        "actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]"
    )
fi

if [[ "${FULL_KL_HVP}" == 1 ]]; then
    HVP_QUADRATIC_BOOL=false
    if [[ "${HVP_COMPUTE_QUADRATIC}" == 1 ]]; then
        HVP_QUADRATIC_BOOL=true
    fi
    overrides+=(
        "actor_rollout_ref.actor.full_kl_hvp_measurement=true"
        "actor_rollout_ref.actor.full_kl_hvp_compute_quadratic=${HVP_QUADRATIC_BOOL}"
        "actor_rollout_ref.actor.full_kl_hvp_steps=${HVP_POWER_STEPS:-200}"
        "actor_rollout_ref.actor.full_kl_hvp_tolerance=${HVP_POWER_TOLERANCE:-0.001}"
        "actor_rollout_ref.actor.full_kl_hvp_visible_devices='${CUDA_VISIBLE_DEVICES}'"
        "actor_rollout_ref.actor.full_kl_hvp_scratch=${TMPDIR:?HVP requires Slurm job-local TMPDIR}/fisher-hvp"
        "actor_rollout_ref.actor.full_kl_hvp_timeout=${HVP_MEASUREMENT_TIMEOUT:-21600}"
        "++actor_rollout_ref.nccl_timeout=${HVP_NCCL_TIMEOUT:-24000}"
    )
fi

if [[ "${DRY_RUN}" == 1 ]]; then
    exec "${PYTHON_BIN}" -m verl.trainer.main_ppo --cfg job "${overrides[@]}" "$@"
fi
if [[ "${FULL_KL_HVP}" == 1 ]]; then
    "${PYTHON_BIN}" -m verl.trainer.main_ppo "${overrides[@]}" "$@"
    validation_args=(--run-dir "${OUTPUT_DIR}" --updates "${REMAINING_UPDATES}" --reuse-n "${REUSE_N}"
                     --prompts "${KL_NUM_PROMPTS}" --tolerance "${HVP_POWER_TOLERANCE:-0.001}"
                     --scratch "${TMPDIR}/fisher-hvp")
    if [[ "${HVP_COMPUTE_QUADRATIC}" == 0 ]]; then
        validation_args+=(--skip-quadratic)
    fi
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_training_fisher.py" "${validation_args[@]}"
fi
exec "${PYTHON_BIN}" -m verl.trainer.main_ppo "${overrides[@]}" "$@"
