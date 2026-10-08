"""Synthetic longest-prefix FVP resource check; no rollout or optimizer."""
import argparse
import gc
import json
import math
import subprocess
import time

import torch
from transformers import Qwen3ForCausalLM

from examples.dynamic_staleness.measure_training_fisher import make_layerwise_product, selected_logits, verify_gpu_offload
from examples.dynamic_staleness.layerwise_fisher import layerwise_fisher_product
from examples.dynamic_staleness.verify_exact_kl_hvp import memory, normalize_in_place, place_layers, synchronize, vector_dot, write_json


def baseline_product(model, rows, anchors):
    """195937 storage/aggregation: CPU inputs, GPU VJP graph, per-row Fv."""
    def product(direction):
        total = None
        for row, anchor in zip(rows, anchors):
            part = layerwise_fisher_product(model, row, anchor, direction)
            if total is None:
                total = part
            else:
                for accumulator, value in zip(total, part):
                    accumulator.add_(value)
                del part
        return total
    return product


def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.device_count() != args.gpus:
        raise RuntimeError("Probe must see all allocated GPUs")
    args.output.mkdir(parents=True, exist_ok=False)
    self_check = verify_gpu_offload(args.gpus, False, False)
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float32, attn_implementation="eager",
        local_files_only=True,
    ).eval()
    model.gradient_checkpointing_disable()
    place_layers(model, args.gpus)
    parameters = list(model.parameters())
    versions = [p._version for p in parameters]
    row = {"input_ids": torch.arange(1, args.length + 1).remainder(model.config.vocab_size).unsqueeze(0),
           "position_ids": torch.arange(args.length).unsqueeze(0),
           "indices": torch.arange(args.length - 8, args.length),
           "weights": torch.full((8,), 1 / 16, dtype=torch.float64)}
    with torch.no_grad():
        anchor = selected_logits(model, row).double().log_softmax(-1).cpu()
    rows, anchors = [row, row], [anchor, anchor]
    report = {"purpose": "synthetic_resource_probe_not_research_q",
              "model_path": str(args.model_path),
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "parameter_count": sum(p.numel() for p in parameters),
              "parameter_dtype": "float32", "fisher_logits_dtype": "float64",
              "gpus": args.gpus, "prefix_length": args.length, "contexts": 2,
              "positions_per_context": 8, "repeats": args.repeats, "seed": args.seed,
              "rollout_cycles": 0, "optimizer_updates": 0, "gpu_self_check": self_check,
              "memory_scope": "measurement_process_per_call_pytorch_allocator",
              "runs": {}}
    for mode in ("baseline", "optimized"):
        print(json.dumps({"fvp_storage_mode_start": mode}), flush=True)
        generators = [torch.Generator(device=f"cuda:{i}").manual_seed(args.seed + i) for i in range(args.gpus)]
        direction = [torch.randn(p.shape, dtype=p.dtype, device=p.device,
                                 generator=generators[p.device.index]) for p in parameters]
        normalize_in_place(direction, vector_dot(direction, direction).sqrt())
        product = (baseline_product(model, rows, anchors) if mode == "baseline"
                   else make_layerwise_product(model, rows, anchors, False, False))
        calls = []
        for repeat in range(1, args.repeats + 1):
            synchronize(args.gpus)
            for device in range(args.gpus):
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            result = product(direction)
            norm = vector_dot(result, result).sqrt()
            if not math.isfinite(norm.item()) or norm.item() <= 0:
                raise RuntimeError("Non-finite or zero FVP")
            synchronize(args.gpus)
            item = {"mode": mode, "repeat": repeat, "seconds": time.perf_counter() - started,
                    "fvp_norm": norm.item(), "memory": memory(args.gpus)}
            # Like spectral iteration, replace rather than retain the previous direction.
            direction[:] = result
            del result
            normalize_in_place(direction, norm)
            del norm
            synchronize(args.gpus)
            item["post_call_allocated"] = [torch.cuda.memory_allocated(i) for i in range(args.gpus)]
            calls.append(item)
            write_json(args.output / f"{mode}_{repeat:02d}.json", item)
            print(json.dumps({"fvp_storage_call": item}), flush=True)
        report["runs"][mode] = calls
        del direction, product, generators
        gc.collect()
        for device in range(args.gpus):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    growth = [max(item["post_call_allocated"][i] for item in report["runs"]["optimized"])
              - min(item["post_call_allocated"][i] for item in report["runs"]["optimized"])
              for i in range(args.gpus)]
    report["post_call_allocation_range_bytes"] = growth
    report["memory_stable"] = all(value <= 1024**2 for value in growth)
    report["parameters_unchanged"] = versions == [p._version for p in parameters]
    report["passed"] = report["memory_stable"] and report["parameters_unchanged"]
    write_json(args.output / "fvp_storage_probe.json", report)
    print(json.dumps({"fvp_storage_probe_completed": report}), flush=True)
    if not report["passed"]:
        raise RuntimeError("Probe detected retained allocation growth or parameter mutation")


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--length", type=int, default=4095)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if args.length < 8 or args.repeats < 2:
        parser.error("Need at least 8 tokens and 2 repeated FVPs")
    run(args)
