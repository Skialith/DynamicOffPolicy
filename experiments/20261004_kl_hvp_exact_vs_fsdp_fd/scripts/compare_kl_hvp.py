"""Fixed-anchor Fisher probe: layer-parallel exact AD versus sharded gradient FD.

This standalone diagnostic deliberately creates no optimizer and never trains.
Run exact first, then run fd with torchrun --nproc_per_node=2 using its reference.
Large .pt references stay on the server; JSON contains only diagnostic scalars.
"""

import argparse
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_model(args):
    torch.manual_seed(20261004)
    if args.model == "tiny":
        config = Qwen3Config(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, tie_word_embeddings=True, attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config)
    else:
        model = Qwen3ForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float32,
            attn_implementation="eager", local_files_only=True,
        )
    return model.float().eval()


def make_batch(args):
    if args.model == "tiny":
        generator = torch.Generator().manual_seed(20261004)
        tokens = torch.randint(0, 128, (2, 8), generator=generator)
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return dict(tokenizer([
        "Compute 17 + 28. Answer:", "Compute 7 times 9. Answer:",
        "Solve x + 3 = 11. Answer:", "What is one half of 36? Answer:",
    ], padding=True, return_tensors="pt"))


def move_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [move_tree(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    return value


def place_layers(model):
    """GPU-only transfers preserve AD; no inference dispatch/offload hooks."""
    model.to("cuda:0")

    def transfer_hook(device):
        def hook(module, inputs, kwargs):
            return move_tree(inputs, device), move_tree(kwargs, device)
        return hook

    midpoint = len(model.model.layers) // 2
    for index, layer in enumerate(model.model.layers):
        device = torch.device("cuda", int(index >= midpoint))
        layer.to(device)
        layer.register_forward_pre_hook(transfer_hook(device), with_kwargs=True)
    model.model.norm.to("cuda:1")
    model.model.norm.register_forward_pre_hook(transfer_hook("cuda:1"), with_kwargs=True)
    # The tied embedding/head parameter must remain on the same GPU.
    model.lm_head.register_forward_pre_hook(transfer_hook("cuda:0"), with_kwargs=True)


def logits(model, batch):
    return model(**batch, use_cache=False, logits_to_keep=2).logits.float()


def kl_loss(output, anchor_logp):
    # The anchor gradient is mathematically zero. FP32 softmax normalization
    # error can be amplified by a large model Jacobian into a nonzero gradient.
    return (anchor_logp.exp() * (anchor_logp - output.double().log_softmax(-1))).sum(-1).mean()


def vector_dot(left, right, distributed=False):
    device = left[0].device
    value = sum((x.float() * y.float()).sum(dtype=torch.float64).to(device)
                for x, y in zip(left, right))
    if distributed:
        dist.all_reduce(value)
    return value


def vector_norm(values, distributed=False):
    return vector_dot(values, values, distributed).sqrt()


def normalize(values, distributed=False):
    norm = vector_norm(values, distributed)
    if not torch.isfinite(norm) or norm.item() <= 0:
        raise RuntimeError("Non-finite or zero FVP/direction; not a valid eigenvalue estimate")
    return [value / norm.to(value) for value in values]


def synchronize(distributed=False):
    devices = [int(os.environ["LOCAL_RANK"])] if distributed else [0, 1]
    for device in devices:
        torch.cuda.synchronize(device)


def power_iteration(product, initial, steps, tolerance, distributed=False):
    direction = normalize(initial, distributed)
    history = []
    synchronize(distributed)
    started = time.perf_counter()
    for iteration in range(1, steps + 1):
        result = product(direction)
        eigenvalue = vector_dot(direction, result, distributed)
        residual = vector_norm([
            w - eigenvalue.to(w) * v for w, v in zip(result, direction)
        ], distributed) / vector_norm(result, distributed).clamp_min(1e-30)
        entry = {"iteration": iteration, "rayleigh": eigenvalue.item(),
                 "relative_residual": residual.item()}
        history.append(entry)
        if not distributed or dist.get_rank() == 0:
            emit({"power": entry})
        if residual.item() <= tolerance:
            break
        if iteration < steps:
            direction = normalize(result, distributed)
    synchronize(distributed)
    return direction, {"history": history, "seconds": time.perf_counter() - started,
                       "converged": history[-1]["relative_residual"] <= tolerance}


def random_direction(count, seed):
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(count, generator=generator)
    return value / value.double().norm().float()


def split_vector(value, parameters):
    sizes = [parameter.numel() for parameter in parameters]
    return [part.reshape(parameter.shape).to(parameter.device)
            for part, parameter in zip(value.split(sizes), parameters)]


def flatten_cpu(values):
    return torch.cat([value.detach().reshape(-1).cpu() for value in values])


def environment():
    return {"utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(index) for index in range(2)]}


def run_exact(args):
    model = load_model(args)
    names = [name for name, _ in model.named_parameters()]
    count = sum(parameter.numel() for parameter in model.parameters())
    batch_cpu = make_batch(args)
    place_layers(model)
    parameters = list(model.parameters())
    batch = move_tree(batch_cpu, "cuda:0")
    with torch.no_grad():
        anchor_logp = logits(model, batch).double().log_softmax(-1).detach()
    direction_cpu = random_direction(count, 20261005)
    direction = split_vector(direction_cpu, parameters)
    other = split_vector(random_direction(count, 20261006), parameters)

    def product(values):
        loss = kl_loss(logits(model, batch), anchor_logp)
        gradient = torch.autograd.grad(loss, parameters, create_graph=True)
        # Sum on GPU 0 through differentiable cross-device copies.
        contraction = sum((g * v).sum().to("cuda:0") for g, v in zip(gradient, values))
        return list(torch.autograd.grad(contraction, parameters))

    initial_loss = kl_loss(logits(model, batch), anchor_logp)
    anchor_grad = torch.autograd.grad(initial_loss, parameters)
    grad_norm = vector_norm(anchor_grad).item()
    del anchor_grad, initial_loss
    product(direction)  # Warm up both the first- and second-backward kernels.
    for device in (0, 1):
        torch.cuda.reset_peak_memory_stats(device)
    synchronize()
    started = time.perf_counter()
    reference_product = product(direction)
    synchronize()
    product_seconds = time.perf_counter() - started
    other_product = product(other)
    cross1 = vector_dot(other, reference_product).item()
    cross2 = vector_dot(direction, other_product).item()
    symmetry = abs(cross1 - cross2) / max(
        vector_norm(reference_product).item(), vector_norm(other_product).item(), 1e-30,
    )
    energy = vector_dot(direction, reference_product).item()
    torch.save({"direction": direction_cpu, "product": flatten_cpu(reference_product),
                "batch": batch_cpu, "anchor_logp": anchor_logp.cpu(), "names": names,
                "count": count}, args.output / "reference.pt")
    del reference_product, other_product, other, direction_cpu
    eigenvector, power = power_iteration(product, direction, args.steps, args.tolerance)
    torch.save(flatten_cpu(eigenvector), args.output / "eigenvector.pt")
    summary = {"route": "layer_parallel_exact_ad", "environment": environment(),
               "model": args.model, "parameters": count, "optimizer_updates": 0,
               "anchor_gradient_norm": grad_norm, "random_direction_energy": energy,
               "symmetry_scaled_error": symmetry, "fvp_seconds": product_seconds,
               "power": power, "peak_allocated_bytes": [
                   torch.cuda.max_memory_allocated(device) for device in (0, 1)],
               "peak_reserved_bytes": [
                   torch.cuda.max_memory_reserved(device) for device in (0, 1)]}
    write_json(args.output / "exact.json", summary)
    emit(summary)
    if grad_norm > 1e-4 or energy <= 0 or symmetry > 1e-4:
        raise RuntimeError("Exact AD anchor/PSD/symmetry check failed")


def run_fd(args):
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    reference = torch.load(args.output / "reference.pt", mmap=True, weights_only=True)
    model = load_model(args)
    assert [name for name, _ in model.named_parameters()] == reference["names"]
    model = FSDP(model.to(device), device_id=device, use_orig_params=False)
    batch = move_tree(reference["batch"], device)
    anchor_logp = reference["anchor_logp"].to(device)
    # A normal backward initializes FSDP and frees unsharded buffers. An anchor
    # no_grad forward can leave a stale full buffer when later modifying shards.
    initial_output = logits(model, batch)
    anchor_error = (initial_output.double().log_softmax(-1) - anchor_logp).abs().max().item()
    kl_loss(initial_output, anchor_logp).backward()
    del initial_output
    model.zero_grad(set_to_none=True)
    parameters = list(model.parameters())
    assert len(parameters) == 1, "This root-wrapper probe expects one flat shard"
    parameter = parameters[0]
    assert list(parameter._fqns) == reference["names"], "FSDP flatten order differs"
    shard_size = math.ceil(reference["count"] / world)
    assert parameter.numel() == shard_size
    base = parameter.detach().clone()
    valid = max(0, min(shard_size, reference["count"] - rank * shard_size))

    def shard(value):
        result = torch.zeros_like(base)
        result[:valid].copy_(value[rank * shard_size:rank * shard_size + valid])
        return result

    direction = [shard(reference["direction"])]
    exact_product = [shard(reference["product"])]
    top_direction = [shard(torch.load(args.output / "eigenvector.pt", mmap=True, weights_only=True))]
    with (args.output / "exact.json").open(encoding="utf-8") as stream:
        exact_summary = json.load(stream)
    exact_lambda = exact_summary["power"]["history"][-1]["rayleigh"]
    exact_residual = exact_summary["power"]["history"][-1]["relative_residual"]
    del reference

    def gradient(values, offset):
        try:
            with torch.no_grad():
                parameter.copy_(base + offset * values[0])
            model.zero_grad(set_to_none=True)
            output = logits(model, batch)
            loss = kl_loss(output, anchor_logp)
            loss.backward()
            result = parameter.grad.detach().clone()
            return result, output.detach().double().log_softmax(-1), loss.detach().item()
        finally:
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                parameter.copy_(base)

    def product(values, epsilon):
        plus, plus_logp, plus_kl = gradient(values, epsilon)
        minus, minus_logp, minus_kl = gradient(values, -epsilon)
        return [(plus - minus) / (2 * epsilon)], {
            "logp_plus_minus_max": (plus_logp - minus_logp).abs().max().item(),
            "plus_kl": plus_kl, "minus_kl": minus_kl,
        }

    torch.cuda.reset_peak_memory_stats(device)
    sweep = []
    for epsilon in args.epsilons:
        synchronize(True)
        started = time.perf_counter()
        result, perturbation = product(direction, epsilon)
        synchronize(True)
        seconds = time.perf_counter() - started
        relative = vector_norm([result[0] - exact_product[0]], True) / vector_norm(exact_product, True)
        cosine = vector_dot(result, exact_product, True) / (
            vector_norm(result, True) * vector_norm(exact_product, True)
        ).clamp_min(1e-30)
        top_result, _ = product(top_direction, epsilon)
        rayleigh = vector_dot(top_direction, top_result, True).item()
        top_residual = vector_norm([
            top_result[0] - rayleigh * top_direction[0]
        ], True).item() / max(vector_norm(top_result, True).item(), 1e-30)
        entry = {"epsilon": epsilon, "fvp_relative_error": relative.item(),
                 "cosine": cosine.item(), "fvp_seconds": seconds,
                 "random_direction_energy": vector_dot(direction, result, True).item(),
                 "at_exact_eigenvector_rayleigh": rayleigh,
                 "at_exact_eigenvector_lambda_relative_error": abs(rayleigh - exact_lambda) / abs(exact_lambda),
                 "at_exact_eigenvector_residual": top_residual, **perturbation}
        sweep.append(entry)
        if rank == 0:
            emit({"epsilon_sweep": entry})
    # Calibrate on both a diffuse random direction and the high-curvature one.
    best = min(sweep, key=lambda entry: max(
        entry["fvp_relative_error"], entry["at_exact_eigenvector_lambda_relative_error"],
        max(0, entry["at_exact_eigenvector_residual"] - exact_residual),
    ))
    epsilon = best["epsilon"]
    del exact_product, top_direction, result, top_result
    other = [shard(random_direction(sum(parameter._numels), 20261006))]
    first_product, _ = product(direction, epsilon)
    second_product, _ = product(other, epsilon)
    symmetry = abs(vector_dot(other, first_product, True).item()
                   - vector_dot(direction, second_product, True).item()) / max(
                       vector_norm(first_product, True).item(), vector_norm(second_product, True).item(), 1e-30)
    del other, first_product, second_product
    _, power = power_iteration(lambda values: product(values, epsilon)[0],
                               direction, args.steps, args.tolerance, True)
    _, restored_logp, _ = gradient(direction, 0.0)
    restored_error = (parameter.detach() - base).abs().max().item()
    restored_output_error = (restored_logp - anchor_logp).abs().max().item()
    memory = torch.tensor([torch.cuda.max_memory_allocated(device),
                           torch.cuda.max_memory_reserved(device)], device=device, dtype=torch.int64)
    memories = [torch.empty_like(memory) for _ in range(world)]
    dist.all_gather(memories, memory)
    eigenvalue_difference = abs(power["history"][-1]["rayleigh"] - exact_lambda) / abs(exact_lambda)
    passed = (anchor_error < 2e-4 and restored_output_error < 2e-4
              and restored_error == 0 and best["fvp_relative_error"] < 0.02
              and best["random_direction_energy"] > 0 and best["logp_plus_minus_max"] > 0
              and symmetry < 0.02 and best["at_exact_eigenvector_lambda_relative_error"] < 0.02
              and power["converged"] and exact_summary["power"]["converged"]
              and eigenvalue_difference < 0.02)
    passed_all = torch.tensor(int(passed), device=device)
    dist.all_reduce(passed_all, op=dist.ReduceOp.MIN)
    passed = bool(passed_all.item())
    if rank == 0:
        summary = {"route": "fsdp_shard_gradient_central_difference", "environment": environment(),
                   "model": args.model, "world_size": world, "optimizer_updates": 0,
                   "anchor_logp_max_error": anchor_error, "epsilon_sweep": sweep,
                   "selected_epsilon": epsilon, "symmetry_scaled_error": symmetry,
                   "restored_parameter_max_error": restored_error,
                   "restored_logp_max_error": restored_output_error,
                   "power": power, "exact_power_converged": exact_summary["power"]["converged"],
                   "eigenvalue_relative_difference": eigenvalue_difference,
                   "peak_allocated_bytes": [item[0].item() for item in memories],
                   "peak_reserved_bytes": [item[1].item() for item in memories],
                   "checks_passed": passed}
        write_json(args.output / "fd.json", summary)
        emit(summary)
    dist.destroy_process_group()
    if not passed:
        raise RuntimeError("FSDP finite-difference numerical checks failed; see fd.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", choices=["exact", "fd"])
    parser.add_argument("--model", choices=["tiny", "qwen06"], default="tiny")
    parser.add_argument("--model-path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001])
    args = parser.parse_args()
    if args.model == "qwen06" and not args.model_path:
        parser.error("qwen06 requires --model-path")
    if args.steps < 1 or args.tolerance <= 0 or any(value <= 0 for value in args.epsilons):
        parser.error("steps, tolerance and epsilons must be positive")
    if torch.cuda.device_count() != 2:
        parser.error("Expose exactly two GPUs with CUDA_VISIBLE_DEVICES")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    if args.route == "exact":
        run_exact(args)
    else:
        run_fd(args)


if __name__ == "__main__":
    main()
