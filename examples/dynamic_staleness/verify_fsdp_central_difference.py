"""LAST JVP IMPLEMENTATION FALLBACK: parameter perturbations with logits central differences. Paused on 2026-10-07; not a finite post-update log-prob KL proxy."""

# """Probe frozen-anchor central differences through an ordinary legacy-FSDP forward.

# This is intentionally independent of the training pipeline. It checks that a
# sharded anchor can be restored temporarily, perturbed along one real AdamW
# update, scored, and restored without using torch.func or second derivatives.
# """

# import argparse
# import json
# import os
# from contextlib import nullcontext

# import torch
# import torch.distributed as dist
# import torch.nn.functional as F
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp import MixedPrecision
# from transformers import AutoModelForCausalLM, AutoTokenizer


# PROMPTS = [
#     "Prove that the sum of two even integers is even.",
#     "Compute the derivative of x squared plus three x.",
#     "What is the capital of France?",
#     "Explain why the harmonic series diverges.",
#     "Solve the equation two x plus five equals seventeen.",
#     "Give a short definition of entropy.",
#     "List the first five prime numbers.",
#     "Why is the sky blue during the day?",
# ]


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", required=True)
#     parser.add_argument("--lambdas", default="1,4,16,64,256")
#     parser.add_argument("--learning-rate", type=float, default=1e-6)
#     parser.add_argument("--max-length", type=int, default=64)
#     parser.add_argument("--forward-dtype", choices=("bf16", "fp32"), default="bf16")
#     return parser.parse_args()


# def local_parameters(model):
#     return {name: parameter for name, parameter in model.named_parameters()}


# def local_shard(parameter):
#     return getattr(parameter, "_local_shard", parameter)


# @torch.no_grad()
# def force_reshard(model):
#     for module in model.modules():
#         if isinstance(module, FSDP) and module._handle is not None:
#             module._handle.reshard(free_unsharded_flat_param=True)


# def cpu_snapshot(parameters):
#     return {name: local_shard(parameter).detach().cpu().clone() for name, parameter in parameters.items()}


# @torch.no_grad()
# def install(parameters, values):
#     for name, parameter in parameters.items():
#         target = local_shard(parameter)
#         target.copy_(values[name].to(device=target.device, dtype=target.dtype))


# @torch.no_grad()
# def install_anchor_plus_direction(parameters, anchor, direction, scale):
#     for name, parameter in parameters.items():
#         target = local_shard(parameter)
#         value = anchor[name].to(device=target.device, dtype=target.dtype)
#         tangent = direction[name].to(device=target.device, dtype=target.dtype)
#         target.copy_(value + scale * tangent)


# def forward_context(forward_dtype):
#     if forward_dtype == "bf16":
#         return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
#     return nullcontext()


# @torch.no_grad()
# def selected_logits(model, batch, lengths, forward_dtype):
#     was_training = model.training
#     try:
#         model.eval()
#         with forward_context(forward_dtype):
#             output = model(
#                 input_ids=batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 use_cache=False,
#             )
#         rows = torch.arange(batch["input_ids"].shape[0], device=batch["input_ids"].device)
#         logits = output.logits[rows, lengths - 1].float().cpu()
#         del output
#         return logits
#     finally:
#         model.train(was_training)


# def global_l2_from_cpu(values):
#     device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
#     squared = sum(value.double().square().sum() for value in values.values())
#     tensor = torch.tensor(float(squared), device=device, dtype=torch.float64)
#     dist.all_reduce(tensor)
#     return tensor.sqrt().item()


# def global_restore_error(parameters, expected):
#     device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
#     maximum = torch.zeros((), device=device, dtype=torch.float64)
#     for name, parameter in parameters.items():
#         error = (local_shard(parameter).detach().cpu() - expected[name]).abs().max().double()
#         maximum = torch.maximum(maximum, error.to(device))
#     dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
#     return maximum.item()


# def centered_directional_geometry(anchor_logits, plus_logits, minus_logits, scale):
#     anchor = anchor_logits.double()
#     probability = torch.softmax(anchor, dim=-1)
#     derivative = (plus_logits.double() - minus_logits.double()) / (2.0 * scale)
#     derivative -= (probability * derivative).sum(dim=-1, keepdim=True)
#     energy = 0.5 * (probability * derivative.square()).sum(dim=-1).mean()
#     rms = derivative.square().mean().sqrt()
#     return derivative, energy.item(), rms.item()


# def fisher_weighted_comparison(anchor_logits, derivative, reference):
#     probability = torch.softmax(anchor_logits.double(), dim=-1)
#     norm = (probability * derivative.square()).sum(dim=-1).mean()
#     reference_norm = (probability * reference.square()).sum(dim=-1).mean()
#     difference_norm = (probability * (derivative - reference).square()).sum(dim=-1).mean()
#     inner = (probability * derivative * reference).sum(dim=-1).mean()
#     relative = (difference_norm / reference_norm.clamp_min(1e-30)).sqrt()
#     cosine = inner / (norm * reference_norm).sqrt().clamp_min(1e-30)
#     return relative.item(), cosine.item()


# def main():
#     args = parse_args()
#     dist.init_process_group("nccl")
#     rank = dist.get_rank()
#     local_rank = int(os.environ["LOCAL_RANK"])
#     device = torch.device("cuda", local_rank)
#     torch.cuda.set_device(device)
#     torch.manual_seed(20260920)
#     if args.forward_dtype == "fp32":
#         torch.set_float32_matmul_precision("highest")
#         torch.backends.cuda.matmul.allow_tf32 = False

#     tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"
#     encoded = tokenizer(
#         PROMPTS,
#         padding=True,
#         truncation=True,
#         max_length=args.max_length,
#         return_tensors="pt",
#     )
#     lengths = encoded["attention_mask"].sum(dim=-1).to(device)
#     batch = {key: value.to(device) for key, value in encoded.items()}

#     base_model = AutoModelForCausalLM.from_pretrained(
#         args.model,
#         local_files_only=True,
#         torch_dtype=torch.float32,
#         low_cpu_mem_usage=True,
#     ).to(device)
#     mixed_precision = None
#     if args.forward_dtype == "bf16":
#         mixed_precision = MixedPrecision(
#             param_dtype=torch.bfloat16,
#             reduce_dtype=torch.float32,
#             buffer_dtype=torch.bfloat16,
#         )
#     model = FSDP(
#         base_model,
#         device_id=device,
#         use_orig_params=False,
#         mixed_precision=mixed_precision,
#     )
#     del base_model
#     parameters = local_parameters(model)
#     anchor = cpu_snapshot(parameters)
#     anchor_logits = selected_logits(model, batch, lengths, args.forward_dtype)

#     model.train()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
#     optimizer.zero_grad(set_to_none=True)
#     labels = batch["input_ids"].clone()
#     labels[batch["attention_mask"] == 0] = -100
#     with forward_context(args.forward_dtype):
#         output = model(
#             input_ids=batch["input_ids"],
#             attention_mask=batch["attention_mask"],
#             use_cache=False,
#         )
#         shift_logits = output.logits[:, :-1].float()
#         shift_labels = labels[:, 1:]
#         loss = F.cross_entropy(
#             shift_logits.reshape(-1, shift_logits.shape[-1]),
#             shift_labels.reshape(-1),
#             ignore_index=-100,
#         )
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad(set_to_none=True)
#     del output, shift_logits, shift_labels

#     current = cpu_snapshot(parameters)
#     direction = {name: current[name] - anchor[name] for name in parameters}
#     current_logits = selected_logits(model, batch, lengths, args.forward_dtype)
#     update_norm = global_l2_from_cpu(direction)
#     parameter_norm = global_l2_from_cpu(anchor)
#     results = []
#     previous_derivative = None

#     for scale in [float(value) for value in args.lambdas.split(",")]:
#         try:
#             force_reshard(model)
#             install_anchor_plus_direction(parameters, anchor, direction, scale)
#             dist.barrier()
#             plus_logits = selected_logits(model, batch, lengths, args.forward_dtype)
#             force_reshard(model)
#             install_anchor_plus_direction(parameters, anchor, direction, -scale)
#             dist.barrier()
#             minus_logits = selected_logits(model, batch, lengths, args.forward_dtype)
#         finally:
#             force_reshard(model)
#             install(parameters, current)
#             dist.barrier()

#         derivative, energy, directional_rms = centered_directional_geometry(
#             anchor_logits, plus_logits, minus_logits, scale
#         )
#         row = {
#             "lambda": scale,
#             "fisher_step_energy": energy,
#             "directional_rms": directional_rms,
#             "plus_minus_max_abs": (plus_logits - minus_logits).abs().max().item(),
#             "restore_max_abs": global_restore_error(parameters, current),
#         }
#         if previous_derivative is not None:
#             flat = derivative.flatten()
#             previous_flat = previous_derivative.flatten()
#             denominator = previous_flat.norm().clamp_min(1e-30)
#             row["relative_l2_to_previous"] = ((flat - previous_flat).norm() / denominator).item()
#             row["cosine_to_previous"] = F.cosine_similarity(flat, previous_flat, dim=0).item()
#             fisher_relative, fisher_cosine = fisher_weighted_comparison(
#                 anchor_logits, derivative, previous_derivative
#             )
#             row["fisher_relative_l2_to_previous"] = fisher_relative
#             row["fisher_cosine_to_previous"] = fisher_cosine
#         results.append(row)
#         previous_derivative = derivative

#     restore_error = global_restore_error(parameters, current)
#     if restore_error != 0.0:
#         raise RuntimeError(f"Parameters were not restored exactly: max_abs={restore_error}")
#     if not all(row["restore_max_abs"] == 0.0 for row in results):
#         raise RuntimeError("At least one virtual perturbation failed exact parameter restoration")

#     if rank == 0:
#         print(json.dumps({
#             "status": "FSDP_CENTRAL_DIFFERENCE_OK",
#             "world_size": dist.get_world_size(),
#             "model": args.model,
#             "prompts": len(PROMPTS),
#             "learning_rate": args.learning_rate,
#             "forward_dtype": args.forward_dtype,
#             "training_loss": loss.detach().item(),
#             "parameter_norm": parameter_norm,
#             "update_norm": update_norm,
#             "relative_update_norm": update_norm / parameter_norm,
#             "anchor_to_current_max_abs_logit": (anchor_logits - current_logits).abs().max().item(),
#             "results": results,
#         }, indent=2))

#     dist.destroy_process_group()


# if __name__ == "__main__":
#     main()


if __name__ == "__main__":
    raise SystemExit('Parameter-perturbation logits central-difference JVP is paused; retained as the last JVP implementation fallback.')
