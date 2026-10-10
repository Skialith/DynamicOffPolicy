"""Four-GPU FSDP integration: old contexts, empty ranks, no derivatives, unchanged updates/RNG."""
import os
import math
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=19)
        self.embedding = torch.nn.Embedding(19, 16)
        self.dropout = torch.nn.Dropout(.2)
        self.head = torch.nn.Linear(16, 19)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.head(self.dropout(self.embedding(input_ids).cumsum(dim=1))))


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
    actor.config = SimpleNamespace(full_kl_delayed_horizon=8, full_kl_delayed_anchor_every_rollouts=1)
    actor.delayed_kl_anchors = []
    torch.manual_seed(91 + rank)
    records = {}
    for cycle in range(2):
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]] * 4) + cycle
        positions = torch.full((4, 2), -1, dtype=torch.long)
        weights = torch.zeros((4, 2), dtype=torch.float64)
        positions[:rank] = torch.tensor([0, 1])
        weights[:rank] = 1 / 12
        data = DataProto.from_dict(tensors={
            "input_ids": ids, "attention_mask": torch.ones_like(ids),
            "position_ids": torch.arange(6).expand_as(ids), "responses": ids[:, -2:],
            "kl_positions": positions, "kl_weights": weights,
        })
        metadata = {"optimizer_step_start": cycle * 4, "full_kl_rollout_step": cycle + 1, "full_kl_reuse_n": 4}
        if measure:
            anchor = actor._score_full_kl(data)
            actor._capture_delayed_kl(data, anchor, cycle * 4, metadata, 0.)
            if cycle == 0:
                original = actor.delayed_kl_anchors[0]
        for age in range(1, 5):
            optimizer.zero_grad()
            loss = model(input_ids=ids.cuda()).logits.square().mean()
            loss.backward()
            optimizer.step()
            if measure:
                current = actor._score_full_kl(data)
                direct = actor._reduce_full_kl(anchor, anchor, current, weights[positions >= 0])
                records.update(actor._measure_delayed_kl(metadata, cycle * 4 + age, direct))
                assert model.training
    if measure:
        key = "delayed_kl/start_0000/update_0008/"
        current = actor._score_full_kl(DataProto.from_dict(tensors=original.contexts))
        totals = original.local_totals(current).cuda()
        dist.all_reduce(totals)
        assert abs(records[key + "cumulative_kl"] - totals[0].item()) < 1e-12
        assert records[key + "anchor_age"] == 8 and records[key + "extra_forward"] == 1
        assert records[key + "context_count"] == 12 and math.isclose(records[key + "weight_sum"], 1., abs_tol=1e-8)
        assert [item.anchor_update for item in actor.delayed_kl_anchors] == [4]
        assert len({key.rsplit("/", 1)[0] for key in records}) == 12
        assert not any(name.endswith(("layerwise_fisher", "measure_training_fisher")) for name in sys.modules)
    state = torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])
    rng = torch.cuda.get_rng_state()
    del model, optimizer, actor
    return state, rng


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 4
    control, control_rng = run(False)
    with ExitStack() as stack:
        for target in ("torch.func.jvp", "torch.func.vjp", "torch.autograd.grad"):
            stack.enter_context(patch(target, side_effect=AssertionError(f"Unexpected measurement derivative: {target}")))
        measured, measured_rng = run(True)
    torch.testing.assert_close(control, measured, atol=0, rtol=0)
    assert torch.equal(control_rng, measured_rng)
    if dist.get_rank() == 0:
        print("DELAYED_KL_FSDP_PROBE_PASSED: ages 1..8, empty rank, no JVP/VJP/HVP, unchanged updates/RNG")
    dist.destroy_process_group()
