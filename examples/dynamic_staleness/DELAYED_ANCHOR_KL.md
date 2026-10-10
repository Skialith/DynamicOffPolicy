# Forward-only delayed-anchor KL

This route retains sampled real-rollout contexts and BF16 actor full-vocabulary
log probabilities in CPU RAM. It measures the current actor against the same
anchor at later optimizer updates, including updates after rollout refresh.
There is no Fisher, JVP/VJP, HVP, parameter snapshot, displacement norm, Q/B, or
automatic change of N in this route. Training backward/optimizer updates continue.

Use the `codex/delayed-anchor-kl` worktree. JVP/HVP/Fisher implementations,
training hooks, configuration fields, and historical probes/recipes are removed
from this branch. The original worktree and Git history retain the old route.
Assets and `.venv` are
symlinks to the shared `verl-staleness` deployment.

```bash
bash examples/dynamic_staleness/submit_delayed_kl.sh n4
# Configuration only; does not submit or start training:
DRY_RUN=1 bash examples/dynamic_staleness/submit_delayed_kl.sh n4
```

The initial n4 recipe is two rollouts/eight true updates, with 256 prompt groups
and eight responses per optimizer update, `ppo_epochs=1`, 4xA800 legacy FSDP,
TP=2, sis_offload, LR=1e-6/no warmup, 1024/3072 length ceilings, 64 selected
prompt groups/one response/up to eight positions, no evaluation/checkpoint.
`n8` keeps the same per-update batch and runs two rollouts/16 updates by default.
These are observation recipes, not calibrated safe-N rules.

`DELAYED_KL_HORIZON` sets the oldest measured anchor age (default 2N in this
recipe), and `DELAYED_KL_ANCHOR_EVERY_ROLLOUTS` sets capture frequency (default 1).
An anchor is released after its horizon endpoint is measured. Empty DP ranks
retain one dummy context to join the same FSDP collective forwards. Existing
current-anchor KL forwards are reused; only older anchors need extra forwards.
Stale derivative environment flags are rejected before launch; removed Hydra
fields are no longer supported.

`kl_updates.jsonl` keeps the current-rollout measurements. Its
`delayed_kl_measurement_seconds` measures the extra per-update delayed-KL call,
including scoring, CPU KL aggregation and in-call synchronization/metadata
collectives. It excludes the subsequent timing-metric reduction, driver JSONL
write and the once-per-anchor capture reported below.

`delayed_kl_updates.jsonl` records each retained-anchor/current-update pair:

- `anchor_update`, `optimizer_step`, `anchor_age`, `behavior_anchor_update`,
  `policy_age`, `rollout_step`, `anchor_rollout_step`, `reuse_n`, and `training_path`.
- Direct `cumulative_kl`, full vocabulary, fixed causal contexts/weights,
  `bf16_actor` model precision and FP64 probability renormalization/reduction.
- `anchor_forward_seconds` (existing anchor scoring) and `anchor_cache_seconds`
  (extra selected-context CPU copy) only at anchor age 1; cache bytes are summed
  over DP ranks, times take the slowest rank. Cache cost is CPU RAM retention,
  not serialization of old distributions to disk.
- `extra_forward_and_kl_seconds` is zero for reused current-anchor measurements;
  older-anchor rows time their extra forward, aggregation and KL reduction.
  This component is already included in `delayed_kl_measurement_seconds`.

The existing trainer still saves selected contexts under `kl_contexts/`; that
selection/serialization occurs before actor update and is outside these timers.
Peak GPU memory and complete training overhead must be measured in the intended
deployment. Do not add overlapping timer fields as independent costs or mix
BF16 actor KL with historical FP32 Fisher-model KL.

For N=4, anchor age 8 is observed along a path that refreshed behavior data after
four updates. It is not the unexecuted N=8 path. All ages 1..8 are measured in the
initial probe; an endpoint alone does not establish the intervening peak.

CPU regressions and the Slurm-only integration probe:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD" .venv/bin/python tests/trainer/ppo/test_delayed_anchor_kl.py
# Set partition/runtime on the cluster before submitting:
PARTITION=gpu_a800 N_GPUS=4 ROLLOUT_TP_SIZE=2 RUNTIME_PROFILE=sis_offload \
  bash examples/dynamic_staleness/submit_slurm.sh delayed_kl_probe
```

The GPU probe first checks four-GPU FSDP with empty/uneven DP ranks, disables
derivative APIs, and compares training parameters/RNG with an unmeasured control.
It then runs an explicitly reduced-batch Qwen3-8B integration (4 prompt groups
per update, eight responses, response limit 256, eight sampled groups). This
checks cross-rollout hooks/artifacts; it does not establish normal-batch cost,
thresholds, capability, or the validity of upgrading N.
