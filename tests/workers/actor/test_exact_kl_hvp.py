"""Small CPU checks for the independent full-parameter HVP probe."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SCRIPT = Path(__file__).resolve().parents[3] / "examples/dynamic_staleness/verify_exact_kl_hvp.py"
SPEC = importlib.util.spec_from_file_location("exact_kl_hvp_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ExactHvpTest(unittest.TestCase):
    def test_strict_convergence_validation(self):
        unfinished = {"logic_passed": True, "converged": False}
        probe.validate_report(unfinished, require_convergence=False)
        with self.assertRaisesRegex(RuntimeError, "iteration limit"):
            probe.validate_report(unfinished, require_convergence=True)
        probe.validate_report({"logic_passed": True, "converged": True}, require_convergence=True)
        with self.assertRaisesRegex(RuntimeError, "anchor changed"):
            probe.validate_report({"logic_passed": False, "converged": True}, require_convergence=True)

    def test_chunked_scalars_and_normalization(self):
        direction = [torch.tensor([1.0, -2.0, 3.0]), torch.tensor([4.0, 5.0])]
        product = [torch.tensor([2.0, 3.0, 4.0]), torch.tensor([-1.0, 2.0])]
        with patch.object(probe, "CHUNK", 2):
            self.assertEqual(probe.vector_dot(direction, product).item(), 14.0)
            scale = torch.tensor(0.7, dtype=torch.float64)
            expected = torch.cat([w - scale.to(w) * v
                                  for v, w in zip(direction, product)]).double().norm()
            torch.testing.assert_close(probe.residual_norm(direction, product, scale), expected,
                                       atol=1e-6, rtol=1e-6)
            probe.normalize_in_place(direction, probe.vector_dot(direction, direction).sqrt())
            self.assertAlmostEqual(probe.vector_dot(direction, direction).item(), 1.0, places=6)

    def test_kl_hvp_matches_dense_fisher(self):
        theta = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.7]))
        matrix = torch.tensor([[0.1, 0.2, -0.3], [0.4, -0.2, 0.8],
                               [-0.6, 0.3, 0.1], [0.2, 0.7, 0.5]])

        def logits(value):
            return (matrix @ (value + value.square())).unsqueeze(0)

        anchor = logits(theta).detach().double().log_softmax(-1)
        loss = probe.frozen_kl(logits(theta), anchor)
        anchor_gradient, = torch.autograd.grad(loss, [theta])
        self.assertLess(anchor_gradient.norm().item(), 1e-6)
        direction = [torch.tensor([0.3, -0.5, 0.8])]
        before = theta.detach().clone()
        product, = probe.exact_product(probe.frozen_kl(logits(theta), anchor), [theta], direction)
        jacobian = torch.autograd.functional.jacobian(logits, theta).squeeze(0).double()
        probabilities = anchor.exp().squeeze(0)
        softmax_fisher = torch.diag(probabilities) - torch.outer(probabilities, probabilities)
        fisher = jacobian.T @ softmax_fisher @ jacobian
        expected = fisher @ direction[0].double()
        torch.testing.assert_close(product.double(), expected, atol=1e-7, rtol=1e-5)
        self.assertGreater(probe.vector_dot(direction, [product]).item(), 0)
        torch.testing.assert_close(theta.detach(), before, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
