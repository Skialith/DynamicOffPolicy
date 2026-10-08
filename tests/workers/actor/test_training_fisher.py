"""CPU checks for the training-to-independent-Fisher integration."""

import gc
import json
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from examples.dynamic_staleness import measure_training_fisher as measure
from examples.dynamic_staleness.verify_training_fisher import validate
from verl.trainer.ppo.full_vocab_kl import actual_update_squared_norm, score_context_row


class TrainingFisherTest(unittest.TestCase):
    def test_causal_context_selection(self):
        payload = {
            "input_ids": torch.tensor([[0, 3, 4, 5, 6, 2]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1]]),
            "position_ids": torch.tensor([[0, 0, 1, 2, 3, 4]]),
            "responses": torch.tensor([[5, 6, 2]]),
            "kl_positions": torch.tensor([[0, 2, -1]]),
            "kl_weights": torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64),
        }
        row, = measure.context_rows(payload)
        torch.testing.assert_close(row["input_ids"], torch.tensor([[3, 4, 5, 6]]))
        torch.testing.assert_close(row["indices"], torch.tensor([1, 3]))
        torch.manual_seed(1)
        model = self.tiny_model().eval()
        with torch.no_grad():
            expected = model(input_ids=torch.tensor([[3, 4, 5, 6, 2]]), use_cache=False).logits[0, [1, 3]]
            actual = measure.selected_logits(model, row)
            legacy = score_context_row(model, {key: value[0] for key, value in payload.items()}, "cpu")
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(actual.log_softmax(-1), legacy, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(row["weights"], payload["kl_weights"][0, :2], atol=0, rtol=0)
        payload["kl_weights"] *= 0.9
        with self.assertRaisesRegex(ValueError, "sum to one"):
            measure.context_rows(payload)

    @staticmethod
    def tiny_model():
        return Qwen3ForCausalLM(Qwen3Config(
            vocab_size=7, hidden_size=8, intermediate_size=12, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4,
            max_position_embeddings=16, tie_word_embeddings=False,
            attn_implementation="eager",
        ))

    def test_fp32_snapshot_roundtrip(self):
        original = self.tiny_model().eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchor.pt"
            torch.save(original.state_dict(), path)
            state = torch.load(path, mmap=True, weights_only=True)
            copy = Qwen3ForCausalLM.from_pretrained(
                None, config=original.config, state_dict=state,
                torch_dtype=torch.float32, attn_implementation="eager",
            ).eval()
            for name, value in copy.state_dict().items():
                torch.testing.assert_close(value, original.state_dict()[name], atol=0, rtol=0)
            anchor = dict(original.named_parameters())
            current = {name: value.detach() + 0.001 for name, value in anchor.items()}
            direction = measure.displacement_direction(list(copy.named_parameters()), state, current)
            for delta, (name, _) in zip(direction, copy.named_parameters()):
                torch.testing.assert_close(delta, current[name] - state[name], atol=0, rtol=0)
            # Includes embedding, LM head and normalization parameters.
            ids = torch.tensor([[1, 2, 3]])
            reference = copy(input_ids=ids, use_cache=False).logits.detach().double().log_softmax(-1)
            loss = measure.weighted_kl(copy(input_ids=ids, use_cache=False).logits[0], reference[0], torch.full((3,), 1/3, dtype=torch.float64))
            result = measure.exact_product(loss, list(copy.parameters()), direction)
            self.assertTrue(all(torch.isfinite(value).all() for value in result))

    def test_kb_only_skips_displacement_fvp(self):
        torch.manual_seed(29)
        model = self.tiny_model().eval()
        anchor = {name: value.clone() for name, value in model.state_dict().items()}
        current = {name: value + 0.002 * torch.randn_like(value) for name, value in anchor.items()}
        payload = {"anchor_update": 0,
                   "input_ids": torch.tensor([[0, 3, 4, 5, 6, 2]]),
                   "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1]]),
                   "position_ids": torch.tensor([[0, 0, 1, 2, 3, 4]]),
                   "responses": torch.tensor([[5, 6, 2]]),
                   "kl_positions": torch.tensor([[0, 2, -1]]),
                   "kl_weights": torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)}
        row, = measure.context_rows(payload)
        with torch.no_grad():
            logp = measure.selected_logits(model, row).double().log_softmax(-1)
        with tempfile.TemporaryDirectory() as directory:
            cycle = Path(directory)
            model.config.to_json_file(cycle / "config.json")
            for name, value in (("anchor", anchor), ("current", current), ("anchor_logp", [logp]),
                                ("contexts", payload)):
                torch.save(value, cycle / f"{name}.pt")
            power = {"lambda_estimate": 3.0, "residual": 1e-4, "iterations": 12, "converged": True}
            (cycle / "power.json").write_text(json.dumps(power))
            args = SimpleNamespace(cycle=cycle, context=cycle / "contexts.pt", output=cycle / "report.json",
                                   anchor_update=0, age=1, gpus=4, skip_quadratic=True, vjp_gpu_activations=False, layer_inputs_gpu=False)
            with patch.object(torch.cuda, "device_count", return_value=4), \
                 patch.object(measure, "place_layers"), patch.object(measure, "memory", return_value={}), \
                 patch.object(measure, "make_layerwise_product", side_effect=AssertionError("Unexpected Q FVP")):
                measure.run(args)
            report = json.loads(args.output.read_text())
        self.assertFalse(report["quadratic_measured"])
        self.assertNotIn("hvp_fisher_quadratic", report["metrics"])
        model.load_state_dict(current)
        with torch.no_grad():
            # Include the full response suffix, as in the old actor KL scorer.
            logits = model(input_ids=torch.tensor([[3, 4, 5, 6, 2]]), use_cache=False).logits[0, [1, 3]]
            expected = measure.weighted_kl(logits, logp, row["weights"]).item()
        self.assertAlmostEqual(report["metrics"]["hvp_cumulative_kl"], expected, places=10)
        norm_squared = sum((current[name].double() - value.double()).square().sum().item()
                           for name, value in anchor.items())
        self.assertAlmostEqual(report["metrics"]["hvp_spectral_bound"], 1.5 * norm_squared, places=7)

    def test_weighted_products_and_quadratic_match_dense_fisher(self):
        model = torch.nn.Module()
        model.theta = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.7]))
        matrices = [torch.tensor([[0.2, -0.3, 0.1], [0.5, 0.1, -0.2], [-0.1, 0.4, 0.7]]),
                    torch.tensor([[0.3, 0.2, -0.1], [-0.4, 0.1, 0.2], [0.1, -0.6, 0.5]])]
        rows = [{"matrix": matrix, "weights": torch.tensor([weight], dtype=torch.float64)}
                for matrix, weight in zip(matrices, [0.3, 0.7])]
        def logits(value, row):
            return (row["matrix"] @ (value + value.square())).unsqueeze(0)
        anchors = [logits(model.theta, row).detach().double().log_softmax(-1) for row in rows]
        fisher = torch.zeros((3, 3), dtype=torch.float64)
        for row, anchor in zip(rows, anchors):
            jacobian = torch.autograd.functional.jacobian(lambda value: logits(value, row), model.theta).squeeze(0).double()
            p = anchor.exp().squeeze(0)
            fisher += row["weights"][0] * jacobian.T @ (torch.diag(p) - torch.outer(p, p)) @ jacobian
        displacement = torch.tensor([0.003, -0.007, 0.01])
        norm = displacement.norm().double()
        direction = [displacement / norm.float()]
        with patch.object(measure, "selected_logits", lambda module, row: logits(module.theta, row)):
            result = measure.make_product(model, rows, anchors)(direction)
        torch.testing.assert_close(result[0].double(), fisher @ direction[0].double(), rtol=1e-5, atol=1e-7)
        quadratic = 0.5 * norm.square() * measure.vector_dot(direction, result)
        expected = 0.5 * displacement.double() @ fisher @ displacement.double()
        torch.testing.assert_close(quadratic, expected, rtol=1e-5, atol=1e-10)

    def test_cpu_saved_graph_matches_resident_qwen_hvp(self):
        torch.manual_seed(17)
        model = self.tiny_model().eval()
        row = {"input_ids": torch.tensor([[1, 2, 3, 4]]),
               "position_ids": torch.tensor([[0, 1, 2, 3]]),
               "indices": torch.tensor([1, 3]), "weights": torch.tensor([0.5, 0.5], dtype=torch.float64)}
        anchor = measure.selected_logits(model, row).detach().double().log_softmax(-1)
        parameters = list(model.parameters())
        direction = [torch.randn_like(parameter) for parameter in parameters]
        resident = measure.exact_product(
            measure.weighted_kl(measure.selected_logits(model, row), anchor, row["weights"]),
            parameters, direction,
        )
        measure.checkpoint_decoder_layers(model)
        offloaded = measure.make_product(model, [row], [anchor])(direction)
        self.assertTrue(all(not module.training for module in model.modules()))
        for expected, actual in zip(resident, offloaded):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_layerwise_fisher_matches_double_backward_all_parameters(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                torch.manual_seed(37)
                config = self.tiny_model().config
                config.num_hidden_layers = 2
                config.layer_types = ["full_attention"] * 2
                config.tie_word_embeddings = tied
                model = Qwen3ForCausalLM(config).eval()
                row = {"input_ids": torch.tensor([[1, 2, 3, 4, 2, 1]]),
                       "position_ids": torch.arange(6).unsqueeze(0), "indices": torch.tensor([1, 4, 5]),
                       "weights": torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64)}
                anchor = measure.selected_logits(model, row).detach().double().log_softmax(-1)
                parameters = list(model.parameters())
                versions = [parameter._version for parameter in parameters]
                direction = [torch.randn_like(parameter) for parameter in parameters]
                expected = measure.exact_product(
                    measure.weighted_kl(measure.selected_logits(model, row), anchor, row["weights"]),
                    parameters, direction,
                )
                stages = []
                actual = measure.layerwise_fisher_product(model, row, anchor, direction, progress=stages.append)
                for (name, _), reference, value in zip(model.named_parameters(), expected, actual):
                    torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5, msg=name)
                torch.testing.assert_close(measure.vector_dot(direction, actual),
                                           measure.vector_dot(direction, expected), atol=1e-6, rtol=1e-5)
                self.assertEqual(stages[-1], "row_derivatives_done")
                self.assertIn("jvp_layer_1", stages)
                self.assertEqual(versions, [parameter._version for parameter in parameters])
                self.assertTrue(all(not module.training for module in model.modules()))
                # A non-anchor target exercises a nonzero first derivative too.
                target = (measure.selected_logits(model, row).detach().double() + 0.05 *
                          torch.arange(model.config.vocab_size)).log_softmax(-1)
                expected_gradient = torch.autograd.grad(
                    measure.weighted_kl(measure.selected_logits(model, row), target, row["weights"]), parameters,
                )
                gradient = measure.layerwise_kl_gradient(model, row, target)
                for reference, value in zip(expected_gradient, gradient):
                    torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5)

    def test_layerwise_weighted_aggregation_matches_whole_model(self):
        torch.manual_seed(17)
        model = self.tiny_model().eval()
        rows = [{"input_ids": torch.tensor([tokens]),
                 "position_ids": torch.arange(len(tokens)).unsqueeze(0),
                 "indices": torch.tensor([len(tokens) - 1]),
                 "weights": torch.tensor([weight], dtype=torch.float64)}
                for tokens, weight in [([1, 2, 3], 0.3), ([4, 3, 2, 1], 0.7)]]
        anchors = [measure.selected_logits(model, row).detach().double().log_softmax(-1) for row in rows]
        direction = [torch.randn_like(parameter) for parameter in model.parameters()]
        expected = measure.make_product(model, rows, anchors)(direction)
        for cpu_offload in (True, False):
            with self.subTest(vjp_cpu_offload=cpu_offload):
                if cpu_offload:
                    actual = measure.make_layerwise_product(model, rows, anchors)(direction)
                else:
                    with patch.object(measure, "offload_saved_activations",
                                      side_effect=AssertionError("GPU VJP must not use CPU graph hooks")):
                        actual = measure.make_layerwise_product(
                            model, rows, anchors, vjp_cpu_offload=False)(direction)
                for reference, value in zip(expected, actual):
                    torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5)

    def test_gpu_inputs_and_direct_accumulation_match_weighted_exact_hvp(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                torch.manual_seed(43)
                config = self.tiny_model().config
                config.tie_word_embeddings = tied
                model = Qwen3ForCausalLM(config).eval()
                parameters = list(model.parameters())
                direction = [torch.randn_like(parameter) for parameter in parameters]
                rows = [{"input_ids": torch.tensor([tokens]),
                         "position_ids": torch.arange(len(tokens)).unsqueeze(0),
                         "indices": torch.tensor([len(tokens) - 1]),
                         "weights": torch.tensor([weight], dtype=torch.float64)}
                        for tokens, weight in [([1, 2, 3], 0.3), ([4, 3, 2, 1], 0.7)]]
                anchors = [measure.selected_logits(model, row).detach().double().log_softmax(-1) for row in rows]
                expected = measure.make_product(model, rows, anchors)(direction)
                actual = measure.make_layerwise_product(model, rows, anchors, False, False)(direction)
                for reference, value in zip(expected, actual):
                    torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5)
                initial = [torch.randn_like(parameter) for parameter in parameters]
                accumulator = [value.clone() for value in initial]
                for row, anchor in zip(rows, anchors):
                    returned = measure.layerwise_fisher_product(
                        model, row, anchor, direction, layer_inputs_cpu_offload=False,
                        accumulator=accumulator)
                    self.assertIs(returned, accumulator)
                for start, reference, value in zip(initial, expected, accumulator):
                    torch.testing.assert_close(value, start + reference, atol=1e-6, rtol=1e-5)

    def test_layerwise_releases_prefix_mask_without_cyclic_gc(self):
        from examples.dynamic_staleness import layerwise_fisher
        model = self.tiny_model().eval()
        row = {"input_ids": torch.tensor([[1, 2, 3, 4]]),
               "position_ids": torch.arange(4).unsqueeze(0), "indices": torch.tensor([1, 3]),
               "weights": torch.tensor([0.5, 0.5], dtype=torch.float64)}
        anchor = measure.selected_logits(model, row).detach().double().log_softmax(-1)
        direction = [torch.randn_like(parameter) for parameter in model.parameters()]
        references = []
        original = layerwise_fisher.create_causal_mask
        def tracked_mask(*args, **kwargs):
            mask = original(*args, **kwargs)
            references.append(weakref.ref(mask))
            return mask
        gc.collect()
        enabled = gc.isenabled()
        gc.disable()
        try:
            with patch.object(layerwise_fisher, "create_causal_mask", side_effect=tracked_mask):
                result = layerwise_fisher.layerwise_fisher_product(
                    model, row, anchor, direction, layer_inputs_cpu_offload=False)
            del result
            self.assertTrue(references)
            self.assertTrue(all(reference() is None for reference in references))
        finally:
            if enabled:
                gc.enable()
            gc.collect()

    def test_synthetic_long_prefix_product_matches_exact_hvp(self):
        torch.manual_seed(41)
        model = self.tiny_model().eval()
        direction = [torch.randn_like(parameter) for parameter in model.parameters()]
        length = 16
        row = {"input_ids": torch.arange(1, length + 1).remainder(model.config.vocab_size).unsqueeze(0),
               "position_ids": torch.arange(length).unsqueeze(0), "indices": torch.arange(length - 8, length),
               "weights": torch.full((8,), 1 / 8, dtype=torch.float64)}
        anchor = measure.selected_logits(model, row).detach().double().log_softmax(-1)
        expected = measure.exact_product(
            measure.weighted_kl(measure.selected_logits(model, row), anchor, row["weights"]),
            list(model.parameters()), direction,
        )
        expected_norm = measure.vector_dot(expected, expected).sqrt().item()
        with patch.object(measure, "memory", return_value={}):
            report = measure.verify_long_prefix(model, direction, length, 4, vjp_cpu_offload=False)
        self.assertTrue(report["passed"])
        self.assertEqual((report["prefix_length"], report["contexts"], report["positions_per_context"]), (16, 2, 8))
        self.assertAlmostEqual(report["fvp_norm"], expected_norm, delta=1e-6)

    def test_checkpoint_reduces_saved_graph_and_matches_hvp(self):
        torch.manual_seed(17)
        model = self.tiny_model().eval()
        row = {"input_ids": torch.tensor([[1, 2, 3, 4] * 8]),
               "position_ids": torch.arange(32).unsqueeze(0),
               "indices": torch.tensor([15, 31]), "weights": torch.tensor([0.5, 0.5], dtype=torch.float64)}
        anchor = measure.selected_logits(model, row).detach().double().log_softmax(-1)
        parameters = list(model.parameters())
        direction = [torch.randn_like(parameter) for parameter in parameters]

        class Saved:
            def __init__(self, tensor, stats):
                self.tensor = tensor.detach()
                self.stats = stats
                self.size = tensor.numel() * tensor.element_size()
                stats["live"] += self.size
                stats["peak"] = max(stats["peak"], stats["live"])

            def __del__(self):
                self.stats["live"] -= self.size

        def evaluate():
            stats = {"live": 0, "peak": 0}
            with torch.autograd.graph.saved_tensors_hooks(
                lambda tensor: Saved(tensor, stats), lambda packed: packed.tensor,
            ):
                loss = measure.weighted_kl(measure.selected_logits(model, row), anchor, row["weights"])
                result = measure.exact_product(loss, parameters, direction)
            del loss
            self.assertEqual(stats["live"], 0)
            return result, stats["peak"]

        resident, original_peak = evaluate()
        measure.checkpoint_decoder_layers(model)
        recomputed, checkpoint_peak = evaluate()
        for expected, actual in zip(resident, recomputed):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertLess(checkpoint_peak, original_peak)
        # A fresh model has no legacy checkpoint wrapper inside torch.func.jvp.
        layerwise_model = self.tiny_model().eval()
        layerwise_model.load_state_dict(model.state_dict())
        stats = {"live": 0, "peak": 0}
        def saved_context():
            return torch.autograd.graph.saved_tensors_hooks(
                lambda tensor: Saved(tensor, stats), lambda packed: packed.tensor,
            )
        layerwise = measure.layerwise_fisher_product(
            layerwise_model, row, anchor, direction, saved_context=saved_context,
        )
        self.assertEqual(stats["live"], 0)
        self.assertLess(stats["peak"], checkpoint_peak)
        for expected, actual in zip(resident, layerwise):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        print(json.dumps({"checkpoint_saved_graph_bytes": {
            "resident_peak": original_peak, "checkpoint_peak": checkpoint_peak,
            "layerwise_peak": stats["peak"],
        }}))

    def test_actual_update_norm_and_power_convergence(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.0]))
        before = [parameter.detach().clone()]
        with torch.no_grad():
            parameter.add_(torch.tensor([1e-5, -0.003, 0.02]))
        expected = (parameter.detach().double() - before[0].double()).square().sum()
        torch.testing.assert_close(actual_update_squared_norm([parameter], before, chunk_size=2), expected)
        direction = [torch.tensor([1.0, 1.0])]
        def product(values):
            return [torch.tensor([3.0, 1.0]) * values[0]]
        result = measure.power_iteration(product, direction, 50, 1e-5)
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["lambda_estimate"], 3.0, places=5)
        unlimited = measure.power_iteration(product, [torch.ones(2)], 0, 1e-5)
        self.assertTrue(unlimited["converged"])
        self.assertAlmostEqual(unlimited["lambda_estimate"], result["lambda_estimate"], places=5)
        unfinished = measure.power_iteration(product, [torch.ones(2)], 1, 1e-8)
        self.assertFalse(unfinished["converged"])

    def test_gate_rejects_missing_or_unconverged_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hvp_diagnostics/start_0000").mkdir(parents=True)
            metrics = {"hvp_cumulative_kl": 0.01, "hvp_fisher_quadratic": 0.011,
                       "hvp_lambda_max": 3.0, "hvp_displacement_norm": 0.1,
                       "hvp_spectral_bound": 0.015, "hvp_update_norm": 0.1,
                       "hvp_residual": 1e-4}
            record = dict(metrics, optimizer_step=1, anchor_update=0, age_after=1, reuse_n=1)
            report = {"metrics": metrics, "parameter_count": 8_190_735_360,
                      "parameter_dtype": "float32", "kl_dtype": "float64", "vocabulary": "full",
                      "prompts": 64, "positions": 512, "anchor_update": 0, "age_after": 1,
                      "optimizer_step": 1}
            initial = {"metrics": metrics, "power": {"converged": True, "residual": 1e-4}}
            (root / "kl_updates.jsonl").write_text(json.dumps(record) + "\n")
            (root / "hvp_diagnostics/start_0000/age_01.json").write_text(json.dumps(report))
            initial_path = root / "hvp_diagnostics/start_0000/age_00.json"
            initial_path.write_text(json.dumps(initial))
            self.assertTrue(validate(root, 1, 1, 64, 1e-3)["passed"])
            kb_metrics = {name: value for name, value in metrics.items() if name != "hvp_fisher_quadratic"}
            kb_record = dict(record)
            del kb_record["hvp_fisher_quadratic"]
            kb_report = dict(report, metrics=kb_metrics, quadratic_measured=False)
            (root / "kl_updates.jsonl").write_text(json.dumps(kb_record) + "\n")
            (root / "hvp_diagnostics/start_0000/age_01.json").write_text(json.dumps(kb_report))
            self.assertTrue(validate(root, 1, 1, 64, 1e-3, compute_quadratic=False)["passed"])
            with self.assertRaisesRegex(RuntimeError, "Missing"):
                validate(root, 1, 1, 64, 1e-3)
            kb_record["hvp_spectral_bound"] *= 2
            kb_report["metrics"]["hvp_spectral_bound"] *= 2
            (root / "kl_updates.jsonl").write_text(json.dumps(kb_record) + "\n")
            (root / "hvp_diagnostics/start_0000/age_01.json").write_text(json.dumps(kb_report))
            with self.assertRaisesRegex(RuntimeError, "Incorrect spectral bound"):
                validate(root, 1, 1, 64, 1e-3, compute_quadratic=False)
            (root / "kl_updates.jsonl").write_text(json.dumps(record) + "\n")
            (root / "hvp_diagnostics/start_0000/age_01.json").write_text(json.dumps(report))
            initial["power"]["converged"] = False
            initial_path.write_text(json.dumps(initial))
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                validate(root, 1, 1, 64, 1e-3)
            del record["hvp_update_norm"]
            (root / "kl_updates.jsonl").write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(RuntimeError, "Missing"):
                validate(root, 1, 1, 64, 1e-3)


if __name__ == "__main__":
    unittest.main()
