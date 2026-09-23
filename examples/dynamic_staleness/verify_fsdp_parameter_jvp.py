"""GPU integration probe for torch.func parameter JVP through legacy FSDP."""

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


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.manual_seed(20260919)
    use_orig_params = os.environ.get("FSDP_USE_ORIG_PARAMS", "0") == "1"
    model = FSDP(ToyModel().to(device), device_id=device, use_orig_params=use_orig_params)
    model.eval()
    inputs = torch.randn(3, 8, device=device)

    # Match the real training path: FSDP has already completed a normal
    # forward before the post-update geometry measurement starts.
    with torch.no_grad():
        reference = model(inputs)

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    torch.manual_seed(1000 + rank)
    tangents = {name: torch.randn_like(parameter) for name, parameter in parameters.items()}

    def forward(parameter_values):
        return torch.func.functional_call(
            model, (parameter_values, buffers), (inputs,), strict=True,
        )

    primal, derivative = torch.func.jvp(forward, (parameters,), (tangents,))
    epsilon = 1e-3
    plus = {name: parameter + epsilon * tangents[name] for name, parameter in parameters.items()}
    minus = {name: parameter - epsilon * tangents[name] for name, parameter in parameters.items()}
    finite_difference = (forward(plus) - forward(minus)) / (2 * epsilon)
    max_abs = (derivative - finite_difference).abs().max()
    scale = finite_difference.abs().max().clamp_min(1e-8)
    relative = max_abs / scale
    if not torch.allclose(primal, reference, rtol=0, atol=1e-6):
        raise RuntimeError("FSDP functional-call primal does not match the module output")
    if relative.item() > 5e-3:
        raise RuntimeError(f"FSDP JVP mismatch: max_abs={max_abs.item()} relative={relative.item()}")
    dist.all_reduce(relative, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            "FSDP_PARAMETER_JVP_OK "
            f"world_size={dist.get_world_size()} "
            f"use_orig_params={use_orig_params} "
            f"max_relative_error={relative.item():.6g}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
