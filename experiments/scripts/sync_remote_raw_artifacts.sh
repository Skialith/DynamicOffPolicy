#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ANALYSIS_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

REMOTE_HOST=${REMOTE_HOST:-scyb980@NMCC-N46H1@ssh.paracloud.com}
REMOTE_PORT=${REMOTE_PORT:-2222}
REMOTE_ROOT=${REMOTE_ROOT:-/data/run01/scyb980/cyt/src/verl-staleness}
SSH_COMMAND="ssh -p ${REMOTE_PORT} -o BatchMode=yes"

sync_run_stats() {
    local remote_path=$1
    local local_path=$2
    mkdir -p "${local_path}"
    rsync -av --prune-empty-dirs -e "${SSH_COMMAND}" \
        --exclude='kl_contexts/***' \
        --exclude='global_step_*/actor/***' \
        --include='*/' \
        --include='*.json' \
        --include='*.jsonl' \
        --include='*.yaml' \
        --include='*.yml' \
        --include='*.csv' \
        --include='*.txt' \
        --exclude='*' \
        "${REMOTE_HOST}:${REMOTE_ROOT}/${remote_path}/" \
        "${local_path}/"
}

sync_file() {
    local remote_path=$1
    local local_dir=$2
    mkdir -p "${local_dir}"
    rsync -av -e "${SSH_COMMAND}" \
        "${REMOTE_HOST}:${REMOTE_ROOT}/${remote_path}" \
        "${local_dir}/"
}

pilot_root="${ANALYSIS_ROOT}/20260902_staleness_pilot/raw"
pilot_run="outputs/dynamic_staleness/20260830-174008_job150042_g4_tp2_seed1"
sync_run_stats "${pilot_run}/anchor_n4_u0100" "${pilot_root}/anchor"
sync_run_stats "${pilot_run}/branch_n4_u0100_0196" "${pilot_root}/n4"
sync_run_stats "${pilot_run}/branch_n8_u0100_0196" "${pilot_root}/n8"
sync_run_stats "${pilot_run}/branch_n16_u0100_0196" "${pilot_root}/n16"
sync_file "tensorboard_log/dynamic-staleness-pilot/20260830-174008_job150042_g4_tp2_seed1_anchor_n4_u0100/events.out.tfevents.1788083054.d1n41a11g02.882852.0" "${pilot_root}/tensorboard/anchor"
sync_file "tensorboard_log/dynamic-staleness-pilot/20260831-142057_job151476_g4_tp2_seed1_branch_n4_u0100_0196/events.out.tfevents.1788157503.d1n41a22g03.3617015.0" "${pilot_root}/tensorboard/n4"
sync_file "tensorboard_log/dynamic-staleness-pilot/20260830-174008_job150042_g4_tp2_seed1_branch_n8_u0100_0196/events.out.tfevents.1788156929.d1n41a11g02.2093624.0" "${pilot_root}/tensorboard/n8"
sync_file "tensorboard_log/dynamic-staleness-pilot/20260901-172142_job151477_g4_tp2_seed1_branch_n16_u0100_0196/events.out.tfevents.1788254777.d1n41a11g02.3951187.0" "${pilot_root}/tensorboard/n16"
sync_file "logs/slurm/staleness-g4-sis-n4-then-n8-150042.out" "${pilot_root}/slurm"
sync_file "logs/slurm/staleness-g4-n4-u100-196-151476.out" "${pilot_root}/slurm"
sync_file "logs/slurm/staleness-g4-n16-u100-196-151477.out" "${pilot_root}/slurm"

warmup_root="${ANALYSIS_ROOT}/20260903_full_kl_warmup/raw"
sync_run_stats "outputs/dynamic_staleness/20260903-200830_job154027_g4_tp2_seed1/full_kl_n4_u0096" "${warmup_root}/n4"
sync_run_stats "outputs/dynamic_staleness/20260904-153215_job154028_g4_tp2_seed1/full_kl_n8_u0096" "${warmup_root}/n8"
sync_run_stats "outputs/dynamic_staleness/20260909-144336_job157388_g4_tp2_seed1/full_kl_n16_u0096" "${warmup_root}/n16"
sync_file "outputs/full_kl_pairs/20260903-124556_seed1_g4_tp2/endpoint_kl.json" "${warmup_root}/endpoint_compare"
sync_file "outputs/full_kl_pairs/20260903-124556_seed1_g4_tp2/ready_n4.json" "${warmup_root}/endpoint_compare"
sync_file "outputs/full_kl_pairs/20260903-124556_seed1_g4_tp2/ready_n8.json" "${warmup_root}/endpoint_compare"
sync_file "tensorboard_log/dynamic-staleness-full-kl/20260903-200830_job154027_g4_tp2_seed1_full_kl_n4_u0096/events.out.tfevents.1788437585.d1n41a22g03.410146.0" "${warmup_root}/tensorboard/n4"
sync_file "tensorboard_log/dynamic-staleness-full-kl/20260903-200830_job154027_g4_tp2_seed1_full_kl_n4_u0096/events.out.tfevents.1788688071.ln01.2510032.0" "${warmup_root}/tensorboard/n4"
sync_file "tensorboard_log/dynamic-staleness-full-kl/20260904-153215_job154028_g4_tp2_seed1_full_kl_n8_u0096/events.out.tfevents.1788507406.d1n41a22g03.1553769.0" "${warmup_root}/tensorboard/n8"
sync_file "tensorboard_log/dynamic-staleness-full-kl/20260904-153215_job154028_g4_tp2_seed1_full_kl_n8_u0096/events.out.tfevents.1788688073.ln01.2511915.0" "${warmup_root}/tensorboard/n8"
sync_file "tensorboard_log/dynamic-staleness-full-kl/20260909-144336_job157388_g4_tp2_seed1_full_kl_n16_u0096/events.out.tfevents.1788936486.d1n41a21g03.3467109.0" "${warmup_root}/tensorboard/n16"
sync_file "logs/slurm/fullkl-n4-g4-154027.out" "${warmup_root}/slurm"
sync_file "logs/slurm/fullkl-n8-g4-154028.out" "${warmup_root}/slurm"
sync_file "logs/slurm/fullkl-n16-g4-156023.out" "${warmup_root}/slurm"
sync_file "logs/slurm/fullkl-n16-g4-156172.out" "${warmup_root}/slurm"
sync_file "logs/slurm/fullkl-n16-g4-157388.out" "${warmup_root}/slurm"

no_warmup_root="${ANALYSIS_ROOT}/20260909_full_kl_no_warmup_n4_n8/raw"
no_warmup_run="outputs/full_kl_rerun/20260909-225922"
sync_run_stats "${no_warmup_run}/20260910-230512_job158523_g4_tp2_seed1/full_kl_n4_u0096" "${no_warmup_root}/n4"
sync_run_stats "${no_warmup_run}/20260910-230512_job158524_g4_tp2_seed1/full_kl_n8_u0192" "${no_warmup_root}/n8"
sync_file "tensorboard_log/dynamic-staleness-full-kl/20260910-230512_job158523_g4_tp2_seed1_full_kl_n4_u0096/events.out.tfevents.1789053095.d1n41a18g02.1620117.0" "${no_warmup_root}/tensorboard/n4"
sync_file "tensorboard_log/dynamic-staleness-full-kl/20260910-230512_job158524_g4_tp2_seed1_full_kl_n8_u0192/events.out.tfevents.1789053095.d1n41a18g02.1613113.0" "${no_warmup_root}/tensorboard/n8"
sync_file "logs/slurm/fullkl2-rerun-n4-g4-158523.out" "${no_warmup_root}/slurm"
sync_file "logs/slurm/fullkl2-rerun-n8-g4-158524.out" "${no_warmup_root}/slurm"

fisher_n4_root="${ANALYSIS_ROOT}/20260919_fisher_alignment_n4/raw"
fisher_n4_run="outputs/fisher_alignment_n4/20260919-182654"
sync_run_stats "${fisher_n4_run}/20260920-013935_job166277_g4_tp2_seed1/fisher_alignment_probe_n4_u0004" "${fisher_n4_root}/probe_job166277"
sync_run_stats "${fisher_n4_run}/20260920-022839_job166278_g4_tp2_seed1/fisher_alignment_n4_u0096" "${fisher_n4_root}/formal_job166278"
sync_file "tensorboard_log/dynamic-staleness-full-kl/20260920-013935_job166277_g4_tp2_seed1_fisher_alignment_probe_n4_u0004/events.out.tfevents.1789839810.d1n41a28g03.2665413.0" "${fisher_n4_root}/probe_job166277/tensorboard"
sync_file "tensorboard_log/dynamic-staleness-full-kl/20260920-022839_job166278_g4_tp2_seed1_fisher_alignment_n4_u0096/events.out.tfevents.1789842725.d1n41a28g03.2729048.0" "${fisher_n4_root}/formal_job166278/tensorboard"
sync_file "logs/slurm/fisher-align-probe-n4-166277.out" "${fisher_n4_root}/probe_job166277"
sync_file "logs/slurm/fisher-align-n4-g4-166278.out" "${fisher_n4_root}/formal_job166278"

large_n_root="${ANALYSIS_ROOT}/20260920_fisher_alignment_large_n/raw"
large_n_run="outputs/fisher_alignment_large_n/20260920-233439"
sync_run_stats "${large_n_run}/20260921-054628_job167220_g4_tp2_seed1/fisher_alignment_n16_u0096" "${large_n_root}/n16_job167220"
sync_run_stats "${large_n_run}/20260921-065849_job167221_g4_tp2_seed1/fisher_alignment_n32_u0096" "${large_n_root}/n32_job167221"
sync_file "tensorboard_log/dynamic-staleness-full-kl/20260921-054628_job167220_g4_tp2_seed1_fisher_alignment_n16_u0096/events.out.tfevents.1789941049.d1n41a28g03.4158120.0" "${large_n_root}/n16_job167220/tensorboard"
sync_file "tensorboard_log/dynamic-staleness-full-kl/20260921-065849_job167221_g4_tp2_seed1_fisher_alignment_n32_u0096/events.out.tfevents.1789945396.d1n41a12g03.878085.0" "${large_n_root}/n32_job167221/tensorboard"
sync_file "logs/slurm/fisher-align-n16-g4-167220.out" "${large_n_root}/n16_job167220"
sync_file "logs/slurm/fisher-align-n32-g4-167221.out" "${large_n_root}/n32_job167221"

n1_root="${ANALYSIS_ROOT}/20260921_fisher_alignment_n1_control/raw/job167543"
sync_run_stats "outputs/fisher_alignment_n1_control/20260921-122405/20260921-231510_job167543_g4_tp2_seed1/fisher_alignment_n1_u0096" "${n1_root}"
sync_file "tensorboard_log/dynamic-staleness-full-kl/20260921-231510_job167543_g4_tp2_seed1_fisher_alignment_n1_u0096/events.out.tfevents.1790003972.d1n41a28g03.977998.0" "${n1_root}/tensorboard"
sync_file "logs/slurm/fisher-align-n1-g4-167543.out" "${n1_root}"

echo "Remote raw statistics synchronized under ${ANALYSIS_ROOT}."
