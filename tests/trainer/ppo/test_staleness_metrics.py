import math

import pytest
import torch

from verl.trainer.ppo.staleness_metrics import StalenessMetricAccumulator, optimizer_updates_per_rollout


def test_optimizer_updates_per_rollout_uses_prompt_batch_units():
    assert optimizer_updates_per_rollout(1024, 256, 1) == 4
    assert optimizer_updates_per_rollout(2048, 256, 1) == 8
    assert optimizer_updates_per_rollout(4096, 256, 1) == 16


def test_optimizer_updates_per_rollout_rejects_partial_minibatch():
    with pytest.raises(ValueError, match="must be divisible"):
        optimizer_updates_per_rollout(1000, 256, 1)


def test_staleness_metrics_are_token_weighted_and_masked():
    ratios = torch.tensor([[0.5, 1.0, 2.0], [0.7, 1.3, 4.0]], dtype=torch.float64)
    log_prob = ratios.log()
    old_log_prob = torch.zeros_like(log_prob)
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    advantages = torch.tensor([[-1.0, 1.0, 1.0], [-1.0, -1.0, 0.0]])

    accumulator = StalenessMetricAccumulator.create("cpu")
    accumulator.update(
        log_prob,
        old_log_prob,
        response_mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages,
    )
    metrics = accumulator.finalize()

    valid_ratios = ratios[response_mask.bool()]
    expected_ess = valid_ratios.sum().square() / (valid_ratios.numel() * valid_ratios.square().sum())
    assert metrics["token_count"] == 5
    assert metrics["finite_fraction"] == 1
    assert metrics["ratio_mean"] == pytest.approx(valid_ratios.mean().item())
    assert metrics["clipfrac_low"] == pytest.approx(2 / 5)
    assert metrics["clipfrac_high"] == pytest.approx(2 / 5)
    assert metrics["clipfrac_total"] == pytest.approx(4 / 5)
    assert metrics["in_clip_fraction"] == pytest.approx(1 / 5)
    assert metrics["pg_clipfrac"] == pytest.approx(3 / 5)
    assert metrics["ratio_second_moment"] == pytest.approx(valid_ratios.square().mean().item())
    assert metrics["ess_fraction"] == pytest.approx(expected_ess.item())
    assert metrics["sampled_kl"] == pytest.approx(-valid_ratios.log().mean().item())
    expected_sequence_deviation = valid_ratios.log().abs().sum().item() / 2
    assert metrics["sequence_abs_log_ratio_sum_mean"] == pytest.approx(expected_sequence_deviation)
    assert metrics["ratio_p50"] == pytest.approx(1.0, rel=0.02)


def test_staleness_metrics_report_nonfinite_tokens():
    log_prob = torch.tensor([[0.0, math.inf]])
    old_log_prob = torch.zeros_like(log_prob)
    response_mask = torch.ones_like(log_prob)

    accumulator = StalenessMetricAccumulator.create("cpu")
    accumulator.update(log_prob, old_log_prob, response_mask, clip_ratio_low=0.2, clip_ratio_high=0.2)
    metrics = accumulator.finalize()

    assert metrics["token_count"] == 2
    assert metrics["finite_fraction"] == pytest.approx(0.5)
    assert metrics["ratio_mean"] == pytest.approx(1.0)
    assert metrics["ess_fraction"] == pytest.approx(1.0)
