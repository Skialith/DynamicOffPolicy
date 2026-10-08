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

Current-prefix detached inputs can also remain on their layer GPUs with
HVP_LAYER_INPUTS_CPU_OFFLOAD=0 (CLI --layer-inputs-gpu). Each input is popped and
released after its VJP; no inputs from other prefixes or iterations are cached.
Production FVPs now add row gradients directly into one total buffer. The
metadata records layer_inputs_cpu_offload and fvp_accumulation=direct_into_total.

Run the standalone resource probe with:

    bash tests/probes/submit_fvp_storage_probe.sh

It compares the 195937 CPU-input/per-row-result baseline with GPU inputs/direct
accumulation on the same allocated GPUs. Each mode performs three FVPs over two
4095-token prefixes and eight positions, replacing the direction between calls.
CPU exact-HVP regressions and the four-GPU numerical check precede the 8B probe.
Per-call time includes FVP and norm calculation; peaks are reset per call. Only
scalar reports persist. No rollout, optimizer update, research lambda or KL is
measured. A later training integration uses HVP_LAYER_INPUTS_CPU_OFFLOAD=0.
