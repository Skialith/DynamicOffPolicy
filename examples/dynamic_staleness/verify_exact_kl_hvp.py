"""Full-parameter frozen-KL HVP, on a separate layer-parallel Qwen3 model.

No FSDP, optimizer, forward-mode AD, finite differences, or inference offload.
Keep only the current direction and product; scalar diagnostics use chunks.
"""

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, Qwen3ForCausalLM

CHUNK = 1 << 20
PROMPTS = [
    "Compute 17 + 28. Answer:", "Compute 7 times 9. Answer:",
    "Solve x + 3 = 11. Answer:", "What is one half of 36? Answer:",
]


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def move_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    return value


def place_layers(model, gpus):
    """Move from CPU directly to each GPU, without gathering the model on GPU 0."""
    def transfer_hook(device):
        def hook(module, inputs, kwargs):
            return move_tree(inputs, device), move_tree(kwargs, device)
        return hook

    model.model.embed_tokens.to("cuda:0")
    model.model.rotary_emb.to("cuda:0")
    for index, layer in enumerate(model.model.layers):
        device = torch.device("cuda", index * gpus // len(model.model.layers))
        layer.to(device)
        layer.register_forward_pre_hook(transfer_hook(device), with_kwargs=True)
    last_device = torch.device("cuda", gpus - 1)
    model.model.norm.to(last_device)
    model.model.norm.register_forward_pre_hook(transfer_hook(last_device), with_kwargs=True)
    head_device = torch.device("cuda", 0) if model.config.tie_word_embeddings else last_device
    model.lm_head.to(head_device)
    model.lm_head.register_forward_pre_hook(transfer_hook(head_device), with_kwargs=True)


def vector_dot(left, right):
    total = torch.zeros((), dtype=torch.float64, device=left[0].device)
    for x, y in zip(left, right):
        x, y = x.reshape(-1), y.reshape(-1)
        local = torch.zeros((), dtype=torch.float64, device=x.device)
        for start in range(0, x.numel(), CHUNK):
            local += (x[start:start + CHUNK] * y[start:start + CHUNK]).sum(
                dtype=torch.float64,
            )
        total += local.to(total.device)
    return total


def residual_norm(direction, product, eigenvalue):
    total = torch.zeros_like(eigenvalue)
    for v, w in zip(direction, product):
        v, w = v.reshape(-1), w.reshape(-1)
        scale = eigenvalue.to(w)
        local = torch.zeros((), dtype=torch.float64, device=w.device)
        for start in range(0, v.numel(), CHUNK):
            error = w[start:start + CHUNK] - scale * v[start:start + CHUNK]
            local += error.square().sum(dtype=torch.float64)
        total += local.to(total.device)
    return total.sqrt()


def normalize_in_place(values, norm):
    if not torch.isfinite(norm) or norm.item() <= 0:
        raise RuntimeError("Non-finite or zero direction/product")
    for value in values:
        value.div_(norm.to(value))


def frozen_kl(logits, anchor_logp):
    return (anchor_logp.exp() * (anchor_logp - logits.double().log_softmax(-1))).sum(-1).mean()


def exact_product(loss, parameters, direction):
    gradient = torch.autograd.grad(loss, parameters, create_graph=True)
    contraction = sum((g * v).sum().to(parameters[0].device)
                      for g, v in zip(gradient, direction))
    result = list(torch.autograd.grad(contraction, parameters))
    return result


def validate_report(report, require_convergence):
    if not report["logic_passed"]:
        raise RuntimeError("The fixed anchor changed during the probe")
    if require_convergence and not report["converged"]:
        raise RuntimeError("Reached the iteration limit without satisfying the residual tolerance")


def synchronize(gpus):
    for device in range(gpus):
        torch.cuda.synchronize(device)


def memory(gpus):
    return [{"gpu": device, "allocated": torch.cuda.memory_allocated(device),
             "peak_allocated": torch.cuda.max_memory_allocated(device),
             "reserved": torch.cuda.memory_reserved(device),
             "peak_reserved": torch.cuda.max_memory_reserved(device)}
            for device in range(gpus)]


def run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    if torch.cuda.device_count() != args.gpus:
        raise RuntimeError(f"Expected {args.gpus} visible GPUs, got {torch.cuda.device_count()}")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {"utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
                   "python": platform.python_version(), "torch": torch.__version__,
                   "transformers": transformers.__version__, "cuda": torch.version.cuda,
                   "gpus": [torch.cuda.get_device_name(i) for i in range(args.gpus)],
                   "model_path": str(args.model_path), "seed": args.seed,
                   "steps": args.steps, "tolerance": args.tolerance,
                   "require_convergence": args.require_convergence,
                   "parameter_dtype": "float32", "kl_dtype": "float64",
                   "attention": "eager", "optimizer_updates": 0, "fsdp": False,
                   "prompts": PROMPTS, "positions_per_prompt": 2, "vocabulary": "full"}
    write_json(args.output / "environment.json", environment)
    emit({"stage": "loading", "environment": environment})
    started = time.perf_counter()
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float32, attn_implementation="eager",
        local_files_only=True,
    ).eval()
    model.gradient_checkpointing_disable()
    place_layers(model, args.gpus)
    parameters = list(model.parameters())
    assert all(p.requires_grad and p.dtype == torch.float32 for p in parameters)
    assert all(p.device.type == "cuda" for p in parameters)
    parameter_versions = [p._version for p in parameters]
    count = sum(p.numel() for p in parameters)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = move_tree(dict(tokenizer(PROMPTS, padding=True, return_tensors="pt")), "cuda:0")
    assert (batch["attention_mask"][:, -2:] == 1).all()
    write_json(args.output / "batch.json", move_tree(batch, "cpu")["input_ids"].tolist())

    def output():
        return model(**batch, use_cache=False, logits_to_keep=2).logits

    with torch.no_grad():
        anchor_logp = output().double().log_softmax(-1).detach()
    initial_loss = frozen_kl(output(), anchor_logp)
    anchor_gradient = torch.autograd.grad(initial_loss, parameters)
    anchor_gradient_norm = vector_dot(anchor_gradient, anchor_gradient).sqrt().item()
    del anchor_gradient, initial_loss
    if not math.isfinite(anchor_gradient_norm) or anchor_gradient_norm > 1e-4:
        raise RuntimeError(f"Anchor gradient not near zero: {anchor_gradient_norm}")
    emit({"stage": "anchor_checked", "count": count,
          "anchor_gradient_norm": anchor_gradient_norm, "memory": memory(args.gpus)})
    generators = [torch.Generator(device=f"cuda:{i}").manual_seed(args.seed + i)
                  for i in range(args.gpus)]
    direction = [torch.randn(p.shape, dtype=p.dtype, device=p.device,
                             generator=generators[p.device.index]) for p in parameters]
    normalize_in_place(direction, vector_dot(direction, direction).sqrt())
    synchronize(args.gpus)
    history = []
    for iteration in range(1, args.steps + 1):
        step_started = time.perf_counter()
        product = exact_product(frozen_kl(output(), anchor_logp), parameters, direction)
        synchronize(args.gpus)
        hvp_seconds = time.perf_counter() - step_started
        eigenvalue = vector_dot(direction, product)
        product_norm = vector_dot(product, product).sqrt()
        if not torch.isfinite(product_norm) or product_norm.item() <= 0:
            raise RuntimeError("HVP is non-finite or zero")
        relative_residual = (residual_norm(direction, product, eigenvalue) / product_norm).item()
        if not math.isfinite(relative_residual) or eigenvalue.item() <= 0:
            raise RuntimeError("Non-finite residual or non-positive Fisher Rayleigh quotient")
        entry = {"iteration": iteration, "rayleigh": eigenvalue.item(),
                 "relative_residual": relative_residual, "hvp_seconds": hvp_seconds,
                 "memory": memory(args.gpus)}
        history.append(entry)
        emit({"power": entry})
        if relative_residual <= args.tolerance or iteration == args.steps:
            break
        direction = product
        normalize_in_place(direction, product_norm)
        del product
    with torch.no_grad():
        anchor_difference = (output().double().log_softmax(-1) - anchor_logp).abs().max().item()
    unchanged = parameter_versions == [p._version for p in parameters]
    report = {"count": count, "anchor_gradient_norm": anchor_gradient_norm,
              "anchor_logp_max_difference": anchor_difference, "parameters_unchanged": unchanged,
              "history": history, "lambda_estimate": history[-1]["rayleigh"],
              "converged": history[-1]["relative_residual"] <= args.tolerance,
              "logic_passed": unchanged and anchor_difference == 0.0,
              "seconds_including_load": time.perf_counter() - started,
              "memory": memory(args.gpus)}
    write_json(args.output / "exact.json", report)
    emit({"completed": report})
    validate_report(report, args.require_convergence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--require-convergence", action="store_true",
                        help="Fail after saving diagnostics if the residual tolerance was not met")
    arguments = parser.parse_args()
    if arguments.gpus < 1 or arguments.steps < 1 or arguments.tolerance <= 0:
        parser.error("gpus, steps and tolerance must be positive")
    run(arguments)
