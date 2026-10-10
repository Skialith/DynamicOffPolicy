"""CPU regressions for retained context/distribution alignment and route isolation."""
import ast
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.delayed_anchor_kl import DelayedKLAnchor, validate_delayed_kl_config
from verl.trainer.ppo.full_vocab_kl import full_vocabulary_kl, score_context_row


def actor_methods():
    source = Path(__file__).parents[3] / "verl/workers/actor/dp_actor.py"
    names = {"_capture_delayed_kl", "_measure_delayed_kl"}
    methods = [node for node in ast.walk(ast.parse(source.read_text()))
               if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch, "time": time, "math": math, "DataProto": DataProto,
                 "DelayedKLAnchor": DelayedKLAnchor, "get_device_id": lambda: "cpu"}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def contexts(selected=True):
    ids = torch.tensor([[0, 2, 3, 4, 5], [0, 6, 7, 8, 9]])
    return DataProto.from_dict(tensors={
        "input_ids": ids, "attention_mask": (ids != 0).long(),
        "position_ids": torch.tensor([[0, 0, 1, 2, 3]] * 2), "responses": ids[:, -2:],
        "kl_positions": torch.tensor([[0, 1], [-1, -1]]) if selected else torch.full((2, 2), -1),
        "kl_weights": torch.tensor([[0.5, 0.5], [0., 0.]], dtype=torch.float64) if selected else torch.zeros((2, 2)),
    })


class DelayedKLTests(unittest.TestCase):
    def test_derivative_implementations_and_config_are_removed(self):
        root = Path(__file__).parents[3]
        for path in ("examples/dynamic_staleness/layerwise_fisher.py",
                     "examples/dynamic_staleness/measure_training_fisher.py",
                     "verl/utils/fisher_measurement.py"):
            self.assertFalse((root / path).exists())
        actor = (root / "verl/workers/actor/dp_actor.py").read_text()
        worker = (root / "verl/workers/fsdp_workers.py").read_text()
        config = (root / "verl/trainer/config/actor/actor.yaml").read_text()
        for text in (actor, worker, config):
            self.assertNotIn("full_kl_hvp", text)
            self.assertNotIn("full_kl_jvp", text)
        names = {node.name for node in ast.walk(ast.parse(
            (root / "verl/trainer/ppo/full_vocab_kl.py").read_text()))
            if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("actual_update_squared_norm", names)
        self.assertNotIn("full_vocabulary_jvp_frozen_geometry", names)

    def test_configuration_rejects_derivative_routes_and_bad_horizon(self):
        config = dict(full_kl_delayed_measurement=True, full_kl_measurement=True,
                      full_kl_actor_measurement=True, full_kl_delayed_horizon=8,
                      full_kl_delayed_anchor_every_rollouts=1)
        validate_delayed_kl_config(config, 4)
        for changes in ({"full_kl_hvp_measurement": True}, {"full_kl_jvp_measurement": True},
                        {"full_kl_actor_measurement": False}, {"full_kl_delayed_horizon": 4},
                        {"full_kl_delayed_anchor_every_rollouts": 0}):
            with self.assertRaises(ValueError):
                validate_delayed_kl_config(config | changes, 4)

    def test_cache_is_selected_cpu_copy_and_matches_direct_kl(self):
        data = contexts()
        old = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]], requires_grad=True).log()
        anchor = DelayedKLAnchor.capture(data.batch, old, 0, 1, 4, 0.2)
        self.assertEqual(len(anchor.contexts["input_ids"]), 1)
        self.assertFalse(anchor.log_probs.requires_grad)
        data.batch["input_ids"][0, 1] = 99
        self.assertEqual(anchor.contexts["input_ids"][0, 1].item(), 2)
        new = torch.tensor([[0.4, 0.3, 0.3], [0.2, 0.5, 0.3]]).log()
        expected = (old.detach().exp().double() * (old.detach().double() - new.double())).sum(-1).mean()
        self.assertAlmostEqual(anchor.local_totals(new)[0].item(), expected.item(), places=7)
        self.assertEqual(anchor.local_totals(new)[1:].tolist(), [1., 2.])
        self.assertEqual(anchor.local_totals(anchor.log_probs)[0].item(), 0.)
        self.assertGreater(anchor.cache_bytes, old.numel() * old.element_size())

    def test_empty_rank_keeps_dummy_and_zero_weight(self):
        anchor = DelayedKLAnchor.capture(contexts(False).batch, torch.empty((0, 3)), 0, 1, 4, 0.)
        self.assertEqual(len(anchor.contexts["input_ids"]), 1)
        self.assertEqual(anchor.local_totals(torch.empty((0, 3))).tolist(), [0., 0., 0.])

    def test_cross_rollout_old_context_forward_age_and_release(self):
        methods = actor_methods()
        actor = SimpleNamespace(config=SimpleNamespace(full_kl_delayed_horizon=8,
                               full_kl_delayed_anchor_every_rollouts=1), delayed_kl_anchors=[],
                               dynamic_batch_dp_group=None)
        data = contexts()
        old = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]]).log()
        metadata = {"optimizer_step_start": 0, "full_kl_rollout_step": 1, "full_kl_reuse_n": 4}
        with patch("torch.distributed.all_reduce"), patch("torch.func.jvp", side_effect=AssertionError("JVP")), \
                patch("torch.func.vjp", side_effect=AssertionError("VJP")), \
                patch("torch.autograd.grad", side_effect=AssertionError("measurement gradient")):
            methods["_capture_delayed_kl"](actor, data, old, 0, metadata, .2)
            direct = {"cumulative_kl": .01, "weight_sum": 1., "context_count": 2.}
            first = methods["_measure_delayed_kl"](actor, metadata, 4, direct)
            self.assertEqual(first["delayed_kl/start_0000/update_0004/extra_forward"], 0)
            metadata |= {"optimizer_step_start": 4, "full_kl_rollout_step": 2}
            newer = torch.tensor([[0.4, 0.3, 0.3], [0.2, 0.5, 0.3]]).log()
            seen = []
            def score(cached):
                self.assertFalse(torch.is_grad_enabled())
                seen.append(cached.batch["input_ids"].clone())
                return newer
            actor._score_full_kl = score
            result = methods["_measure_delayed_kl"](actor, metadata, 8, direct)
            key = "delayed_kl/start_0000/update_0008/"
            self.assertEqual(result[key + "anchor_age"], 8)
            self.assertEqual(result[key + "extra_forward"], 1)
            self.assertAlmostEqual(result[key + "cumulative_kl"], full_vocabulary_kl(old, newer).mean().item())
            self.assertTrue(torch.equal(seen[0], data.batch["input_ids"][:1]))
            self.assertEqual(actor.delayed_kl_anchors, [])

    def test_interval_skips_capture_and_serialized_age_is_not_policy_age(self):
        methods = actor_methods()
        actor = SimpleNamespace(config=SimpleNamespace(full_kl_delayed_anchor_every_rollouts=2), delayed_kl_anchors=[])
        methods["_capture_delayed_kl"](actor, contexts(), None, 4, {"full_kl_rollout_step": 2}, .1)
        self.assertEqual(actor.delayed_kl_anchors, [])
        source = Path(__file__).parents[3] / "verl/trainer/ppo/ray_trainer.py"
        node = next(node for node in ast.walk(ast.parse(source.read_text()))
                    if isinstance(node, ast.FunctionDef) and node.name == "_append_delayed_kl")
        namespace = {"os": __import__("os"), "json": json, "np": np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as output:
            trainer = SimpleNamespace(optimizer_steps=8, global_steps=2, config=SimpleNamespace(
                data=SimpleNamespace(seed=1), trainer=SimpleNamespace(default_local_dir=output, experiment_name="probe")))
            metrics = {"delayed_kl/start_0000/update_0008/anchor_age": 8.,
                       "delayed_kl/start_0000/update_0008/cumulative_kl": .02}
            records = namespace["_append_delayed_kl"](trainer, metrics, 4)
            record = records[0]
            self.assertEqual((record["anchor_update"], record["behavior_anchor_update"], record["policy_age"]), (0, 4, 3))
            self.assertEqual(record["training_path"], "fixed_n4")
            self.assertEqual(json.loads((Path(output)/"delayed_kl_updates.jsonl").read_text()), record)

    def test_causal_positions_still_use_preceding_full_vocab_logits(self):
        class Model(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=input_ids.unsqueeze(-1) * torch.arange(11).float())
        anchor = DelayedKLAnchor.capture(contexts().batch, torch.zeros((2, 11)), 0, 1, 4, 0.)
        scored = score_context_row(Model(), {key: value[0] for key, value in anchor.contexts.items()}, "cpu")
        expected = (torch.tensor([[3], [4]]) * torch.arange(11).float()).log_softmax(-1)
        torch.testing.assert_close(scored, expected)


if __name__ == "__main__":
    unittest.main()
