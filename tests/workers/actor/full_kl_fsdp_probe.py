"""Run via torchrun inside a four-GPU Slurm allocation, not pytest/login."""
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=19)
        self.embedding = torch.nn.Embedding(19, 16)
        self.dropout = torch.nn.Dropout(0.2)
        self.head = torch.nn.Linear(16, 19)

    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids).cumsum(dim=1)
        return SimpleNamespace(logits=self.head(self.dropout(hidden)))


def run(measure):
    rank = dist.get_rank()
    device = torch.cuda.current_device()
    torch.manual_seed(173)
    model = FSDP(TinyCausalModel().cuda(), device_id=device, sync_module_states=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = model
    actor.dynamic_batch_dp_group = dist.group.WORLD
    actor.use_ulysses_sp = actor.use_fused_kernels = actor.use_prefix_grouper = False
    actor.device_name = "cuda"
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]] * 4)
    positions = torch.full((4, 2), -1, dtype=torch.long)
    weights = torch.zeros((4, 2), dtype=torch.float64)
    # Rank zero has NO selected contexts; the others have 1, 2, 3 rows.
    positions[:rank] = torch.tensor([0, 1])
    weights[:rank] = 1 / 12
    data = DataProto(batch=TensorDict({
        "input_ids": ids, "attention_mask": torch.ones_like(ids),
        "position_ids": torch.arange(6).expand_as(ids), "responses": ids[:, -2:],
        "kl_positions": positions, "kl_weights": weights,
    }, batch_size=[4]))
    torch.manual_seed(91 + rank)
    if measure:
        anchor = actor._score_full_kl(data)
        repeated = actor._score_full_kl(data)
        self_kl = actor._reduce_full_kl(anchor, anchor, repeated, weights[positions >= 0])
        assert self_kl["adjacent_kl"] < 1e-10 and self_kl["context_count"] == 12
    for _ in range(4):
        optimizer.zero_grad()
        loss = model(input_ids=ids.cuda()).logits.square().mean()
        loss.backward()
        optimizer.step()
        if measure:
            current = actor._score_full_kl(data)
            assert model.training
            result = actor._reduce_full_kl(anchor, anchor, current, weights[positions >= 0])
            assert result["adjacent_kl"] >= 0
    state = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    rng = torch.cuda.get_rng_state()
    del model, optimizer, actor
    return state, rng


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 4
    control, control_rng = run(False)
    measured, measured_rng = run(True)
    torch.testing.assert_close(control, measured, atol=0, rtol=0)
    assert torch.equal(control_rng, measured_rng)
    if dist.get_rank() == 0:
        print("FULL_KL_FSDP_INVARIANCE_PASSED: uneven/empty ranks, weights and RNG unchanged")
    dist.destroy_process_group()
