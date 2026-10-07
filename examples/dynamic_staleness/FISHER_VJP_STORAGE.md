# Layerwise Fisher VJP activation storage

The default HVP_VJP_CPU_OFFLOAD=1 preserves the existing CPU saved-tensor hooks.
Set HVP_VJP_CPU_OFFLOAD=0 to keep only the current layer's ordinary VJP graph on
its GPU. Detached layer inputs remain on CPU; the whole-model second-order graph
is never retained. The switch also applies to the anchor-gradient check and the
four-GPU numerical self-check. The legacy whole-model reference is unchanged.

The independent measurement CLI accepts --vjp-gpu-activations. Each age report
records vjp_activations_cpu_offload and the saved-tensor storage description.

Submit one N=4 rollout with the normal 256-prompt update batch:

    bash examples/dynamic_staleness/submit_vjp_gpu_n4.sh

Inspect the real Hydra configuration on the login node without GPU execution:

    CUDA_VISIBLE_DEVICES= DRY_RUN=1 bash examples/dynamic_staleness/submit_vjp_gpu_n4.sh

Timing is recorded in power.history[*].seconds, each age report's seconds,
training-side hvp_measurement_seconds, and Slurm Start/End/Elapsed. The parent
measurement timer includes the child; do not add these overlapping times.

The one-rollout recipe enables HVP_STRESS_PREFIX_LENGTH=4095. At the initial
anchor it also runs a synthetic resource check with two 4095-token prefixes and
8 output positions each, using the full parameter direction. This covers the
per-row Fv plus the accumulation buffer. The synthetic inputs do not enter the
research q, spectral iteration or KL measurements. Its scalar report is
hvp_diagnostics/start_0000/long_prefix_stress.json; its time is included in the
age-0 measurement total, and the memory values are process peaks up to this check.

The recipe enables FISHER_VJP_N4_PREFLIGHT=1. On the allocated compute node,
CPU numerical regressions and the resolved Hydra configuration must pass before
rollout/training starts. Setup time is logged separately. This avoids relying on
a login-node Python import when its Ceph metadata requests are stalled.
