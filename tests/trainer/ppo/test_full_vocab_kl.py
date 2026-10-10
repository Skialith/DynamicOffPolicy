import ast
import math
import random
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from verl.trainer.ppo.full_vocab_kl import (
    full_vocabulary_kl,
    make_update_scheduler, preserve_rng_state, request_seed, score_context_row,
    select_rollout_positions,
)


def test_exact_kl_direction_self_and_chunking():
    p = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]], dtype=torch.float64)
    q = torch.tensor([[0.4, 0.3, 0.3], [0.2, 0.5, 0.3]], dtype=torch.float64)
    expected = (p * (p.log() - q.log())).sum(-1)
    assert torch.allclose(full_vocabulary_kl(p.log(), q.log(), 1), expected, atol=1e-12)
    assert torch.equal(full_vocabulary_kl(p.log(), p.log()), torch.zeros(2, dtype=torch.float64))
    assert not torch.allclose(full_vocabulary_kl(q.log(), p.log()), expected)
    with unittest.TestCase().assertRaises(ValueError):
        full_vocabulary_kl(p.log(), q[:1].log())




def test_full_kl_reduction_matches_direct_weighted_kl():
    root = Path(__file__).parents[3]
    tree = ast.parse((root / "verl/workers/actor/dp_actor.py").read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_reduce_full_kl")
    namespace = {"torch": torch, "math": math, "full_vocabulary_kl": full_vocabulary_kl, "get_device_id": lambda: "cpu"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "actor_full_kl_reduction", "exec"), namespace)
    actor = SimpleNamespace(dynamic_batch_dp_group=None)
    anchor = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]], dtype=torch.float64).log()
    previous = torch.tensor([[0.2, 0.2, 0.6], [0.3, 0.5, 0.2]], dtype=torch.float64).log()
    current = torch.tensor([[0.3, 0.2, 0.5], [0.2, 0.4, 0.4]], dtype=torch.float64).log()
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    with patch("torch.distributed.all_reduce"):
        direct = namespace["_reduce_full_kl"](actor, previous, anchor, current, weights)
    for key, p, q in (
        ("adjacent_kl", previous, current), ("cumulative_kl", anchor, current),
        ("previous_cumulative_kl", anchor, previous),
    ):
        expected = (full_vocabulary_kl(p, q) * weights).sum().item()
        assert math.isclose(direct[key], expected, rel_tol=1e-12)
    assert direct["context_count"] == 2 and direct["weight_sum"] == 1


def test_n4_n8_have_identical_actual_lrs():
    sequences = []
    for n in (4, 8):
        param = torch.nn.Parameter(torch.ones(1))
        optim = torch.optim.SGD([param], lr=1e-6)
        scheduler = make_update_scheduler(optim, 10)
        used = []
        for _ in range(96 // n):
            for _ in range(n):
                used.append(optim.param_groups[0]["lr"])
                param.grad = torch.ones_like(param)
                optim.step()
                scheduler.step()
        sequences.append(used)
        assert scheduler.last_epoch == 96
        assert np.allclose(used, [1e-6 * min(s / 10, 1) for s in range(1, 97)], rtol=1e-12, atol=0)
    assert sequences[0] == sequences[1]


def test_positions_short_responses_prompt_weights_and_rng():
    mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0]])
    state = np.random.get_state()
    positions, weights = select_rollout_positions(mask, ["a", "a", "b", "c"], 64, 8, 10)
    selected_rows = torch.where((positions >= 0).any(-1))[0]
    assert len(selected_rows) == 2
    assert abs(weights.sum().item() - 1) < 1e-12
    for row in selected_rows:
        assert abs(weights[row].sum().item() - 0.5) < 1e-12
        assert mask[row, positions[row][positions[row] >= 0]].all()
    assert np.array_equal(state[1], np.random.get_state()[1])
    p2, w2 = select_rollout_positions(mask, ["a", "a", "b", "c"], 64, 8, 10)
    assert torch.equal(positions, p2) and torch.equal(weights, w2)


def test_rng_preservation_and_request_streams():
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    with preserve_rng_state():
        random.random()
        np.random.rand()
        torch.rand(5)
    assert before[0] == random.getstate()
    assert np.array_equal(before[1][1], np.random.get_state()[1])
    assert torch.equal(before[2], torch.get_rng_state())
    train = [request_seed(1, i) for i in range(100)]
    evaluation = [request_seed(1, i, evaluation=True, update=48) for i in range(100)]
    assert set(train).isdisjoint(evaluation)


def test_response_logit_alignment_left_padding_and_causal_scoring():
    class Toy(torch.nn.Module):
        def forward(self, input_ids, attention_mask, position_ids, use_cache):
            assert input_ids.tolist() == [[2, 3, 4, 5]]
            assert position_ids.tolist() == [[0, 1, 2, 3]]
            logits = input_ids.unsqueeze(-1) * torch.arange(6)
            return SimpleNamespace(logits=logits.float())

    row = {
        "input_ids": torch.tensor([0, 0, 2, 3, 4, 5]),
        "attention_mask": torch.tensor([0, 0, 1, 1, 1, 1]),
        "position_ids": torch.tensor([0, 0, 0, 1, 2, 3]),
        "responses": torch.tensor([4, 5]), "kl_positions": torch.tensor([0, 1, -1]),
    }
    got = score_context_row(Toy(), row, "cpu")
    expected = torch.log_softmax(torch.tensor([[3], [4]]) * torch.arange(6).float(), -1)
    assert torch.equal(got, expected)


def test_full_kl_forward_has_dp_max_alignment_and_outer_step_is_guarded():
    root = Path(__file__).parents[3]
    tree = ast.parse((root / "verl/workers/actor/dp_actor.py").read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_score_full_kl")
    text = ast.unparse(method)
    assert "ReduceOp.MAX" in text and "group=self.dynamic_batch_dp_group" in text
    assert "preserve_rng_state" in text
    worker = (root / "verl/workers/fsdp_workers.py").read_text()
    assert 'if self.config.actor.optim.get("step_unit", "rollout") == "rollout":' in worker


def test_actual_optimizer_method_aborts_nonfinite_without_advancing_scheduler():
    root = Path(__file__).parents[3]
    tree = ast.parse((root / "verl/workers/actor/dp_actor.py").read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_optimizer_step")
    namespace = {"torch": torch, "FSDP": (), "FSDPModule": (), "DTensor": ()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "actor_optimizer_step", "exec"), namespace)
    model = torch.nn.Linear(1, 1, bias=False)
    optim = torch.optim.SGD(model.parameters(), lr=1e-6)
    scheduler = make_update_scheduler(optim, 10)
    actor = SimpleNamespace(
        actor_module=model, actor_optimizer=optim, actor_lr_scheduler=scheduler,
        config=SimpleNamespace(grad_clip=1), scaler=None, update_scheduler=True,
    )
    model.weight.grad = torch.ones_like(model.weight)
    namespace["_optimizer_step"](actor)
    assert scheduler.last_epoch == 1
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    with patch("torch.distributed.get_rank", return_value=0), unittest.TestCase().assertRaises(FloatingPointError):
        namespace["_optimizer_step"](actor)
    assert scheduler.last_epoch == 1 and torch.equal(before, model.weight)


def test_trainer_context_serialization_and_per_update_records():
    from verl import DataProto

    root = Path(__file__).parents[3]
    tree = ast.parse((root / "verl/trainer/ppo/ray_trainer.py").read_text())
    names = {"_prepare_full_kl", "_append_full_kl", "_log_full_kl_by_update"}
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch, "np": np, "os": os, "json": json, "select_rollout_positions": select_rollout_positions}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "kl_trainer_methods", "exec"), namespace)
    ids = torch.tensor([[1, 2, 3, 4]] * 8)
    batch = DataProto.from_dict(tensors={
        "input_ids": ids, "attention_mask": torch.ones_like(ids),
        "position_ids": torch.arange(4).expand_as(ids), "responses": ids[:, -2:],
        "response_mask": torch.ones((8, 2)),
    }, non_tensors={"uid": np.array([str(i // 2) for i in range(8)], dtype=object)})
    actor_config = {
        "full_kl_num_prompts": 64, "full_kl_positions_per_response": 8, "full_kl_seed": 10,
    }
    with tempfile.TemporaryDirectory(prefix="full-kl-test-") as output:
        trainer = SimpleNamespace(
            optimizer_steps=0, global_steps=1, _optimizer_updates_per_rollout=lambda: 4,
            config=SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(**actor_config, get=actor_config.get)),
                trainer=SimpleNamespace(default_local_dir=output, experiment_name="test"),
                data=SimpleNamespace(seed=1),
            ),
        )
        namespace["_prepare_full_kl"](trainer, batch)
        saved = torch.load(Path(output) / "kl_contexts/start_0000.pt", weights_only=True)
        assert len(saved["input_ids"]) == 4 and saved["anchor_update"] == 0
        trainer.optimizer_steps = 4
        metrics = {
            f"full_kl/age_{age:02d}/{key}": value
            for age in range(4) for key, value in {
                "optimizer_step": age + 1, "update_applied": 1, "adjacent_kl": 0.1, "cumulative_kl": 0.2,
                "grad_norm": 0.3, "lr_used": 1e-6,
            }.items()
        }
        records = namespace["_append_full_kl"](trainer, metrics, 4)
        written = [json.loads(line) for line in (Path(output) / "kl_updates.jsonl").read_text().splitlines()]
        assert records == written
        assert [row["optimizer_step"] for row in records] == [1, 2, 3, 4]
        assert all(row["anchor_update"] == 0 and row["vocabulary"] == "full" for row in records)
        calls = []
        logger = SimpleNamespace(log=lambda **kwargs: calls.append(kwargs))
        namespace["_log_full_kl_by_update"](logger, records)
        assert [call["step"] for call in calls] == [1, 2, 3, 4]
        assert all(call["backend"] == ["tensorboard"] for call in calls)
        assert set(calls[0]["data"]) == {
            "full_kl_by_update/adjacent_kl", "full_kl_by_update/cumulative_kl",
            "full_kl_by_update/grad_norm", "full_kl_by_update/lr_used", "full_kl_by_update/policy_age",
        }


if __name__ == "__main__":
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(fn) for name, fn in list(globals().items()) if name.startswith("test_")
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
