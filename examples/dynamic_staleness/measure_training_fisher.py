"""Exact K/Q/B on one fixed rollout anchor; invoked while FSDP is CPU-offloaded.

Only scalar diagnostics are persistent. Anchor/current state_dicts and log-probability
caches live in a private, job-local directory and are removed by the training worker.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from examples.dynamic_staleness.verify_exact_kl_hvp import (
    exact_product, memory, normalize_in_place, place_layers, residual_norm, vector_dot, write_json,
)


def context_rows(payload):
    rows = []
    for index in range(len(payload["input_ids"])):
        valid = payload["attention_mask"][index].bool()
        selected_mask = payload["kl_positions"][index] >= 0
        selected = payload["kl_positions"][index][selected_mask]
        prompt_width = payload["input_ids"].shape[-1] - payload["responses"].shape[-1]
        absolute = prompt_width + selected - 1
        indices = valid.long().cumsum(0)[absolute] - 1
        if not len(indices) or (indices < 0).any() or not valid[absolute].all():
            raise ValueError("Selected position has no valid causal prefix")
        end = int(indices.max()) + 1
        rows.append({
            "input_ids": payload["input_ids"][index][valid][:end].unsqueeze(0),
            "position_ids": payload["position_ids"][index][valid][:end].unsqueeze(0),
            "indices": indices,
            "weights": payload["kl_weights"][index][selected_mask].double(),
        })
    total_weight = sum(row["weights"].sum().item() for row in rows)
    if abs(total_weight - 1.0) > 1e-10:
        raise ValueError(f"Global context weights do not sum to one: {total_weight}")
    return rows


def selected_logits(model, row):
    device = next(model.parameters()).device
    last_device = model.model.norm.weight.device
    return model(
        input_ids=row["input_ids"].to(device),
        position_ids=row["position_ids"].to(device),
        attention_mask=None, use_cache=False, logits_to_keep=row["indices"].to(last_device),
    ).logits[0]


def weighted_kl(logits, anchor, weights):
    anchor = anchor.to(logits.device)
    per_position = (anchor.exp() * (anchor - logits.double().log_softmax(-1))).sum(-1)
    return (per_position * weights.to(logits.device)).sum()


def offload_saved_activations(parameters, direction=()):
    """Offload graph storage, without copying weights/v that already stay on GPU."""
    persistent = {(value.device, value.untyped_storage().data_ptr())
                  for value in (*parameters, *direction)}

    def pack(tensor):
        device = tensor.device
        if device.type != "cuda" or (device, tensor.untyped_storage().data_ptr()) in persistent:
            return device, tensor.detach()
        saved = torch.empty_like(tensor, device="cpu")
        saved.copy_(tensor)
        return device, saved

    def unpack(packed):
        device, saved = packed
        return saved if saved.device == device else saved.to(device)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def make_product(model, rows, anchors):
    parameters = list(model.parameters())
    calls = 0

    def product(direction):
        nonlocal calls
        calls += 1
        total = None
        started = time.perf_counter()
        for index, (row, anchor) in enumerate(zip(rows, anchors), 1):
            # Preserve dtype/values/AD, but store saved graph activations on CPU.
            with offload_saved_activations(parameters, direction):
                loss = weighted_kl(selected_logits(model, row), anchor, row["weights"])
                part = exact_product(loss, parameters, direction)
            if total is None:
                total = part
            else:
                for accumulator, value in zip(total, part):
                    accumulator.add_(value)
                del part
            del loss
            if index == 1 or index % 16 == 0 or index == len(rows):
                print(json.dumps({"hvp_product_progress": {
                    "call": calls, "completed_rows": index, "total_rows": len(rows),
                    "seconds": time.perf_counter() - started,
                }}), flush=True)
        return total

    return product


def power_iteration(product, direction, steps, tolerance):
    normalize_in_place(direction, vector_dot(direction, direction).sqrt())
    history = []
    for iteration in range(1, steps + 1):
        started = time.perf_counter()
        result = product(direction)
        eigenvalue = vector_dot(direction, result)
        norm = vector_dot(result, result).sqrt()
        if not torch.isfinite(norm) or norm.item() <= 0 or eigenvalue.item() <= 0:
            raise RuntimeError("Invalid Fisher product or Rayleigh quotient")
        residual = (residual_norm(direction, result, eigenvalue) / norm).item()
        if not math.isfinite(residual):
            raise RuntimeError("Non-finite eigenpair residual")
        entry = {"iteration": iteration, "rayleigh": eigenvalue.item(),
                 "relative_residual": residual, "seconds": time.perf_counter() - started}
        history.append(entry)
        print(json.dumps({"hvp_power": entry}), flush=True)
        if residual <= tolerance:
            return {"lambda_estimate": eigenvalue.item(), "residual": residual,
                    "iterations": iteration, "history": history, "converged": True}
        # Reuse the caller's list so the initial 8B random vector is not retained.
        direction[:] = result
        normalize_in_place(direction, norm)
        del result
    return {"lambda_estimate": history[-1]["rayleigh"], "residual": history[-1]["relative_residual"],
            "iterations": steps, "history": history, "converged": False}


def displacement_direction(named_parameters, anchor, current):
    direction = []
    for name, parameter in named_parameters:
        if anchor[name].dtype != torch.float32 or current[name].dtype != torch.float32:
            raise ValueError("HVP requires stored FP32 actor parameters")
        if anchor[name].shape != parameter.shape or current[name].shape != parameter.shape:
            raise ValueError(f"Snapshot parameter mismatch: {name}")
        direction.append((current[name] - anchor[name]).to(parameter.device))
    return direction


def verify_gpu_offload(gpus):
    """Numerical startup check of the new hooks on the actual cross-GPU path."""
    torch.manual_seed(17)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=11, hidden_size=8, intermediate_size=12, num_hidden_layers=gpus,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        max_position_embeddings=16, tie_word_embeddings=False, attn_implementation="eager",
    )).eval()
    place_layers(model, gpus)
    row = {"input_ids": torch.tensor([[1, 2, 3, 4]]),
           "position_ids": torch.tensor([[0, 1, 2, 3]]), "indices": torch.tensor([1, 3]),
           "weights": torch.tensor([0.5, 0.5], dtype=torch.float64)}
    anchor = selected_logits(model, row).detach().double().log_softmax(-1)
    parameters = list(model.parameters())
    direction = [torch.randn_like(parameter) for parameter in parameters]
    resident = exact_product(weighted_kl(selected_logits(model, row), anchor, row["weights"]), parameters, direction)
    offloaded = make_product(model, [row], [anchor])(direction)
    for expected, actual in zip(resident, offloaded):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    report = {"passed": True, "gpus": gpus, "parameter_count": sum(p.numel() for p in parameters),
              "atol": 1e-6, "rtol": 1e-5}
    print(json.dumps({"hvp_gpu_offload_self_check": report}), flush=True)
    return report


def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.device_count() != args.gpus:
        raise RuntimeError("Measurement process must see all allocated GPUs")
    started = time.perf_counter()
    offload_check = verify_gpu_offload(args.gpus) if args.age == 0 and args.anchor_update == 0 else None
    payload = torch.load(args.context, map_location="cpu", weights_only=True)
    if payload["anchor_update"] != args.anchor_update:
        raise ValueError("Snapshot and rollout anchor do not match")
    rows = context_rows(payload)
    anchor_state = torch.load(args.cycle / "anchor.pt", map_location="cpu", mmap=True, weights_only=True)
    config = Qwen3Config.from_json_file(args.cycle / "config.json")
    model = Qwen3ForCausalLM.from_pretrained(
        None, config=config, state_dict=anchor_state, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    model.gradient_checkpointing_disable()
    place_layers(model, args.gpus)
    named_parameters = list(model.named_parameters())
    parameters = [parameter for _, parameter in named_parameters]
    if not all(parameter.requires_grad and parameter.dtype == torch.float32 for parameter in parameters):
        raise ValueError("Every model parameter must participate in FP32")
    cache = args.cycle / "anchor_logp.pt"
    if args.age == 0:
        with torch.no_grad():
            anchors = [selected_logits(model, row).double().log_softmax(-1).cpu() for row in rows]
        with cache.open("xb") as stream:
            torch.save(anchors, stream)
        maximum_gradient_norm = 0.0
        for row, anchor in zip(rows, anchors):
            with offload_saved_activations(parameters):
                loss = weighted_kl(selected_logits(model, row), anchor, row["weights"])
                gradient = torch.autograd.grad(loss, parameters)
            maximum_gradient_norm = max(maximum_gradient_norm, vector_dot(gradient, gradient).sqrt().item())
            del gradient, loss
        if maximum_gradient_norm > 1e-4:
            raise RuntimeError("Anchor first derivative is not near zero")
        generators = [torch.Generator(device=f"cuda:{i}").manual_seed(args.seed + args.anchor_update + i)
                      for i in range(args.gpus)]
        direction = [torch.randn(p.shape, dtype=p.dtype, device=p.device,
                                 generator=generators[p.device.index]) for p in parameters]
        power = power_iteration(make_product(model, rows, anchors), direction, args.steps, args.tolerance)
        del direction
        power["anchor_gradient_max_row_norm"] = maximum_gradient_norm
        write_json(args.cycle / "power.json", power)
        if not power["converged"]:
            write_json(args.output, {"anchor_update": args.anchor_update, "age_after": 0, "power": power})
            raise RuntimeError("Iteration limit reached without residual convergence")
        values = {"hvp_cumulative_kl": 0.0, "hvp_fisher_quadratic": 0.0,
                  "hvp_spectral_bound": 0.0, "hvp_displacement_norm": 0.0}
    else:
        anchors = torch.load(cache, map_location="cpu", weights_only=True)
        with (args.cycle / "power.json").open() as stream:
            power = json.load(stream)
        if not power["converged"]:
            raise RuntimeError("Cannot use an unconverged spectral estimate")
        current = torch.load(args.cycle / "current.pt", map_location="cpu", mmap=True, weights_only=True)
        direction = displacement_direction(named_parameters, anchor_state, current)
        norm = vector_dot(direction, direction).sqrt()
        if not torch.isfinite(norm):
            raise RuntimeError("Invalid cumulative displacement")
        if norm.item() == 0:
            quadratic = 0.0
        else:
            normalize_in_place(direction, norm)
            result = make_product(model, rows, anchors)(direction)
            quadratic = 0.5 * norm.item() ** 2 * vector_dot(direction, result).item()
            del result
        del direction
        # Evaluate K using the same FP32 measurement model/contexts, not the BF16 actor scores.
        model.load_state_dict(current, strict=True)
        with torch.no_grad():
            cumulative_kl = sum(weighted_kl(selected_logits(model, row), anchor, row["weights"]).item()
                                for row, anchor in zip(rows, anchors))
        values = {"hvp_cumulative_kl": cumulative_kl, "hvp_fisher_quadratic": quadratic,
                  "hvp_spectral_bound": 0.5 * power["lambda_estimate"] * norm.item() ** 2,
                  "hvp_displacement_norm": norm.item()}
    values.update({"hvp_lambda_max": power["lambda_estimate"], "hvp_residual": power["residual"],
                   "hvp_iterations": power["iterations"], "hvp_converged": 1.0})
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("Non-finite K/Q/B")
    if values["hvp_cumulative_kl"] < -1e-10 or values["hvp_fisher_quadratic"] < -1e-8:
        raise RuntimeError("Negative KL or Fisher energy beyond numerical tolerance")
    report = {"anchor_update": args.anchor_update, "age_after": args.age,
              "optimizer_step": args.anchor_update + args.age, "context_path": str(args.context),
              "parameter_count": sum(p.numel() for p in parameters), "parameter_dtype": "float32",
              "saved_tensors": "activations_cpu_weights_and_direction_gpu",
              "gradient_values_released_before_second_backward": True,
              "kl_dtype": "float64", "aggregation": "prompt_equal_then_position_equal",
              "vocabulary": "full", "prompts": len(rows), "positions": sum(len(r["indices"]) for r in rows),
              "maximum_prefix_length": max(r["input_ids"].shape[-1] for r in rows),
              "seconds": time.perf_counter() - started, "memory": memory(args.gpus), "metrics": values}
    if args.age == 0:
        report["power"] = power
    if offload_check is not None:
        report["gpu_offload_self_check"] = offload_check
    write_json(args.output, report)
    print(json.dumps({"hvp_completed": report}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-update", type=int, required=True)
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261005)
    arguments = parser.parse_args()
    if arguments.steps < 1 or arguments.tolerance <= 0 or arguments.age < 0:
        parser.error("Invalid measurement budget, tolerance or age")
    run(arguments)
