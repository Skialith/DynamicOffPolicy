"""CPU checks for the training-to-independent-Fisher integration."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from examples.dynamic_staleness import measure_training_fisher as measure
from examples.dynamic_staleness.verify_training_fisher import validate
from verl.trainer.ppo.full_vocab_kl import actual_update_squared_norm


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
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
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
        offloaded = measure.make_product(model, [row], [anchor])(direction)
        for expected, actual in zip(resident, offloaded):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

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
