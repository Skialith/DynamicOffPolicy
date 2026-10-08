"""Historical matrix-free Fisher-vector-product comparison through legacy FSDP.

The combined probe is paused: its logits central-difference JVP is retained
in comments as the last JVP implementation fallback. The KL-gradient
central-difference helpers remain as historical reference code.

The original probe compared two ordinary FSDP forward/backward routes:

1. central-difference the logits JVP, then apply the logits Fisher and a VJP;
2. central-difference gradients of frozen-anchor KL directly.

The second route bypasses JVP entirely because the Hessian of
KL(p_anchor || p_theta) at the anchor is the Fisher matrix.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(8, 16)
        self.activation = torch.nn.SiLU()
        self.second = torch.nn.Linear(16, 5)

    def forward(self, inputs):
        return self.second(self.activation(self.first(inputs)))


def distributed_dot(left, right):
    value = sum((x.double() * y.double()).sum() for x, y in zip(left, right))
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def distributed_norm(values):
    return distributed_dot(values, values).sqrt()


def normalize(values):
    norm = distributed_norm(values)
    if norm.item() == 0:
        raise RuntimeError("Cannot normalize a zero parameter direction")
    return [value / norm.to(value) for value in values]


def copy_parameters(parameters, values):
    with torch.no_grad():
        for parameter, value in zip(parameters, values):
            parameter.copy_(value)


# LAST JVP IMPLEMENTATION FALLBACK (2026-10-07): parameter-perturbation logits central-difference JVP; paused.
# def central_difference_jvp(model, parameters, base, direction, inputs, epsilon):
#     try:
#         copy_parameters(
#             parameters,
#             [value + epsilon * tangent for value, tangent in zip(base, direction)],
#         )
#         with torch.no_grad():
#             plus = model(inputs).float()
#         copy_parameters(
#             parameters,
#             [value - epsilon * tangent for value, tangent in zip(base, direction)],
#         )
#         with torch.no_grad():
#             minus = model(inputs).float()
#     finally:
#         copy_parameters(parameters, base)
#     return (plus - minus) / (2 * epsilon)


def cross_entropy_gradient(model, parameters, base, direction, inputs, probabilities, offset):
    """Return the gradient at base + offset * direction.

    The omitted entropy of the frozen anchor is constant in the parameters, so
    this has the same gradient as KL(p_anchor || p_theta).
    """
    try:
        copy_parameters(
            parameters,
            [value + offset * tangent for value, tangent in zip(base, direction)],
        )
        model.zero_grad(set_to_none=True)
        logits = model(inputs).float()
        loss = -(probabilities * logits.log_softmax(-1)).sum(-1).mean()
        loss.backward()
        gradients = [
            torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters
        ]
    finally:
        model.zero_grad(set_to_none=True)
        copy_parameters(parameters, base)
    return gradients


def gradient_difference_fvp(model, parameters, base, direction, inputs, probabilities, epsilon):
    plus = cross_entropy_gradient(
        model, parameters, base, direction, inputs, probabilities, epsilon,
    )
    minus = cross_entropy_gradient(
        model, parameters, base, direction, inputs, probabilities, -epsilon,
    )
    return [(left - right) / (2 * epsilon) for left, right in zip(plus, minus)]


# LAST JVP IMPLEMENTATION FALLBACK (2026-10-07): parameter-perturbation logits central-difference JVP; paused.
# def fisher_vector_product(model, parameters, base, direction, inputs, probabilities, epsilon):
#     logits_jvp = central_difference_jvp(
#         model, parameters, base, direction, inputs, epsilon,
#     )
#     mean_jvp = (probabilities * logits_jvp).sum(-1, keepdim=True)
#     logits_cotangent = (probabilities * (logits_jvp - mean_jvp)).detach()
#     output_energy = (logits_jvp * logits_cotangent).sum(-1).mean().double()

#     model.zero_grad(set_to_none=True)
#     logits = model(inputs).float()
#     (logits * logits_cotangent).sum(-1).mean().backward()
#     result = [
#         torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
#         for parameter in parameters
#     ]
#     model.zero_grad(set_to_none=True)
#     return result, output_energy


def relative_error(left, right):
    denominator = torch.maximum(left.abs(), right.abs()).clamp_min(1e-12)
    return (left - right).abs() / denominator


def vector_relative_error(left, right):
    difference = [x - y for x, y in zip(left, right)]
    denominator = torch.maximum(distributed_norm(left), distributed_norm(right)).clamp_min(1e-12)
    return distributed_norm(difference) / denominator


# HISTORICAL combined probe: paused because its logits-JVP comparison route is inactive; KL-gradient differences above are retained.
# def main():
#     dist.init_process_group("nccl")
#     rank = dist.get_rank()
#     device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
#     torch.cuda.set_device(device)
#     torch.manual_seed(20260924)
#     model = FSDP(ToyModel().to(device), device_id=device, use_orig_params=False)
#     model.eval()
#     inputs = torch.randn(3, 8, device=device)

#     with torch.no_grad():
#         anchor_logits = model(inputs).float()
#         probabilities = anchor_logits.softmax(-1)
#     parameters = list(model.parameters())
#     base = [parameter.detach().clone() for parameter in parameters]

#     generator = torch.Generator(device=device)
#     generator.manual_seed(1000 + rank)
#     first = normalize([
#         torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
#         for value in base
#     ])
#     second = normalize([
#         torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
#         for value in base
#     ])
#     epsilon = float(os.environ.get("FVP_EPSILON", "0.01"))

#     first_jvp = central_difference_jvp(model, parameters, base, first, inputs, epsilon)
#     refined_jvp = central_difference_jvp(model, parameters, base, first, inputs, epsilon / 2)
#     finite_difference_relative = (
#         (first_jvp - refined_jvp).norm()
#         / refined_jvp.norm().clamp_min(1e-12)
#     ).double()
#     dist.all_reduce(finite_difference_relative, op=dist.ReduceOp.MAX)

#     fisher_first, output_energy = fisher_vector_product(
#         model, parameters, base, first, inputs, probabilities, epsilon,
#     )
#     fisher_second, _ = fisher_vector_product(
#         model, parameters, base, second, inputs, probabilities, epsilon,
#     )
#     gradient_fisher_first = gradient_difference_fvp(
#         model, parameters, base, first, inputs, probabilities, epsilon,
#     )
#     gradient_fisher_second = gradient_difference_fvp(
#         model, parameters, base, second, inputs, probabilities, epsilon,
#     )
#     parameter_energy = distributed_dot(first, fisher_first)
#     gradient_parameter_energy = distributed_dot(first, gradient_fisher_first)
#     energy_relative = relative_error(parameter_energy, output_energy)
#     gradient_energy_relative = relative_error(gradient_parameter_energy, output_energy)
#     fvp_route_relative = vector_relative_error(fisher_first, gradient_fisher_first)
#     cross_forward = distributed_dot(first, fisher_second)
#     cross_reverse = distributed_dot(second, fisher_first)
#     symmetry_relative = relative_error(cross_forward, cross_reverse)
#     gradient_cross_forward = distributed_dot(first, gradient_fisher_second)
#     gradient_cross_reverse = distributed_dot(second, gradient_fisher_first)
#     gradient_symmetry_relative = relative_error(gradient_cross_forward, gradient_cross_reverse)

#     direction = first
#     eigenvalue = torch.zeros((), device=device, dtype=torch.float64)
#     residual = torch.full((), float("inf"), device=device, dtype=torch.float64)
#     for _ in range(4):
#         fisher_direction = gradient_difference_fvp(
#             model, parameters, base, direction, inputs, probabilities, epsilon,
#         )
#         eigenvalue = distributed_dot(direction, fisher_direction)
#         residual_values = [
#             product - eigenvalue.to(product) * value
#             for product, value in zip(fisher_direction, direction)
#         ]
#         residual = distributed_norm(residual_values).double()
#         direction = normalize(fisher_direction)

#     restored_error = max(
#         (parameter.detach() - value).abs().max().double()
#         for parameter, value in zip(parameters, base)
#     )
#     dist.all_reduce(restored_error, op=dist.ReduceOp.MAX)
#     checks = {
#         "finite_difference_relative": finite_difference_relative,
#         "energy_relative": energy_relative,
#         "gradient_energy_relative": gradient_energy_relative,
#         "fvp_route_relative": fvp_route_relative,
#         "symmetry_relative": symmetry_relative,
#         "gradient_symmetry_relative": gradient_symmetry_relative,
#         "restored_error": restored_error,
#     }
#     limits = {
#         "finite_difference_relative": 5e-3,
#         "energy_relative": 5e-4,
#         "gradient_energy_relative": 5e-3,
#         "fvp_route_relative": 5e-3,
#         "symmetry_relative": 5e-3,
#         "gradient_symmetry_relative": 5e-3,
#         "restored_error": 0.0,
#     }
#     failures = [
#         f"{name}={checks[name].item():.6g} limit={limit:.6g}"
#         for name, limit in limits.items()
#         if checks[name].item() > limit
#     ]
#     if parameter_energy.item() <= 0:
#         failures.append(f"parameter_energy={parameter_energy.item():.6g} must be positive")
#     if not torch.isfinite(eigenvalue) or not torch.isfinite(residual):
#         failures.append("power iteration produced a non-finite value")
#     if failures:
#         raise RuntimeError("FSDP Fisher-vector-product probe failed: " + "; ".join(failures))

#     if rank == 0:
#         print(
#             "FSDP_FISHER_VECTOR_PRODUCT_OK "
#             f"world_size={dist.get_world_size()} "
#             f"epsilon={epsilon:.6g} "
#             f"finite_difference_relative={finite_difference_relative.item():.6g} "
#             f"energy_relative={energy_relative.item():.6g} "
#             f"gradient_energy_relative={gradient_energy_relative.item():.6g} "
#             f"fvp_route_relative={fvp_route_relative.item():.6g} "
#             f"symmetry_relative={symmetry_relative.item():.6g} "
#             f"gradient_symmetry_relative={gradient_symmetry_relative.item():.6g} "
#             f"power_eigenvalue={eigenvalue.item():.6g} "
#             f"power_residual={residual.item():.6g}"
#         )
#     dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit("Combined FSDP FVP probe is paused with its logits-difference JVP route; standalone KL-gradient comparison remains in experiments/20261004_kl_hvp_exact_vs_fsdp_fd/scripts/compare_kl_hvp.py")
