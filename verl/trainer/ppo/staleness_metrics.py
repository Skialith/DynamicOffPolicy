# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

import torch


_HISTOGRAM_MIN = -4.0
_HISTOGRAM_MAX = 4.0
_HISTOGRAM_BINS = 801


def optimizer_updates_per_rollout(train_prompt_batch_size: int, mini_prompt_batch_size: int, ppo_epochs: int) -> int:
    """Return the number of synchronized optimizer updates after one rollout."""
    if train_prompt_batch_size <= 0 or mini_prompt_batch_size <= 0 or ppo_epochs <= 0:
        raise ValueError("batch sizes and ppo_epochs must be positive")
    if train_prompt_batch_size % mini_prompt_batch_size != 0:
        raise ValueError(
            f"train_prompt_batch_size ({train_prompt_batch_size}) must be divisible by "
            f"mini_prompt_batch_size ({mini_prompt_batch_size})"
        )
    return train_prompt_batch_size // mini_prompt_batch_size * ppo_epochs


@dataclass
class StalenessMetricAccumulator:
    """Token-weighted statistics for current-policy versus behavior-policy log probabilities."""

    totals: torch.Tensor
    histogram: torch.Tensor
    log_ratio_min: torch.Tensor
    log_ratio_max: torch.Tensor

    @classmethod
    def create(cls, device: torch.device | str | int) -> "StalenessMetricAccumulator":
        return cls(
            totals=torch.zeros(14, dtype=torch.float64, device=device),
            histogram=torch.zeros(_HISTOGRAM_BINS, dtype=torch.float64, device=device),
            log_ratio_min=torch.tensor(torch.inf, dtype=torch.float64, device=device),
            log_ratio_max=torch.tensor(-torch.inf, dtype=torch.float64, device=device),
        )

    @torch.no_grad()
    def update(
        self,
        log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        response_mask: torch.Tensor,
        clip_ratio_low: float,
        clip_ratio_high: float,
        advantages: torch.Tensor | None = None,
    ) -> None:
        mask = response_mask.bool()
        self.totals[0] += mask.sum(dtype=torch.float64)

        log_ratio_matrix = log_prob.detach().float() - old_log_prob.detach().float()
        finite_mask = mask & torch.isfinite(log_ratio_matrix)
        log_ratio = log_ratio_matrix.masked_select(mask)
        finite_log_ratio = log_ratio[torch.isfinite(log_ratio)].to(torch.float64)
        if finite_log_ratio.numel() == 0:
            return

        ratio = torch.exp(torch.clamp(finite_log_ratio, min=-20.0, max=20.0))
        self.totals[1] += finite_log_ratio.numel()
        self.totals[2] += finite_log_ratio.sum()
        self.totals[3] += finite_log_ratio.abs().sum()
        self.totals[4] += finite_log_ratio.square().sum()
        self.totals[5] += ratio.sum()
        self.totals[6] += ratio.square().sum()
        self.totals[7] += (ratio < 1.0 - clip_ratio_low).sum(dtype=torch.float64)
        self.totals[8] += (ratio > 1.0 + clip_ratio_high).sum(dtype=torch.float64)
        self.totals[9] += (finite_log_ratio < _HISTOGRAM_MIN).sum(dtype=torch.float64)
        self.totals[10] += (finite_log_ratio > _HISTOGRAM_MAX).sum(dtype=torch.float64)
        if advantages is not None:
            finite_advantage_mask = finite_mask & torch.isfinite(advantages)
            valid_ratio = torch.exp(
                torch.clamp(log_ratio_matrix.masked_select(finite_advantage_mask), min=-20.0, max=20.0)
            )
            valid_advantage = advantages.masked_select(finite_advantage_mask)
            active_clip = ((valid_advantage > 0) & (valid_ratio > 1.0 + clip_ratio_high)) | (
                (valid_advantage < 0) & (valid_ratio < 1.0 - clip_ratio_low)
            )
            self.totals[11] += active_clip.sum(dtype=torch.float64)

        valid_response = finite_mask.any(dim=-1)
        absolute_log_ratio_per_response = torch.where(
            finite_mask, log_ratio_matrix.abs(), torch.zeros_like(log_ratio_matrix)
        ).sum(dim=-1)
        self.totals[12] += valid_response.sum(dtype=torch.float64)
        self.totals[13] += absolute_log_ratio_per_response.masked_select(valid_response).sum(dtype=torch.float64)

        histogram_values = finite_log_ratio.clamp(min=_HISTOGRAM_MIN, max=_HISTOGRAM_MAX)
        self.histogram += torch.histc(
            histogram_values,
            bins=_HISTOGRAM_BINS,
            min=_HISTOGRAM_MIN,
            max=_HISTOGRAM_MAX,
        ).to(torch.float64)
        self.log_ratio_min = torch.minimum(self.log_ratio_min, finite_log_ratio.min())
        self.log_ratio_max = torch.maximum(self.log_ratio_max, finite_log_ratio.max())

    @torch.no_grad()
    def synchronize(self) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        torch.distributed.all_reduce(self.totals, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(self.histogram, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(self.log_ratio_min, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(self.log_ratio_max, op=torch.distributed.ReduceOp.MAX)

    def _log_ratio_quantile(self, quantile: float) -> float:
        count = int(self.histogram.sum().item())
        if count == 0:
            return float("nan")
        rank = max(1, int(torch.ceil(torch.tensor(quantile * count)).item()))
        index = int(torch.searchsorted(self.histogram.cumsum(dim=0), rank).item())
        width = (_HISTOGRAM_MAX - _HISTOGRAM_MIN) / _HISTOGRAM_BINS
        return _HISTOGRAM_MIN + (index + 0.5) * width

    def finalize(self) -> dict[str, float]:
        total_count = float(self.totals[0].item())
        finite_count = float(self.totals[1].item())
        if finite_count == 0:
            return {
                "token_count": total_count,
                "finite_fraction": 0.0,
                "sampled_kl": float("nan"),
            }

        log_ratio_mean = float(self.totals[2].item() / finite_count)
        log_ratio_variance = max(float(self.totals[4].item() / finite_count) - log_ratio_mean**2, 0.0)
        ratio_mean = float(self.totals[5].item() / finite_count)
        ratio_variance = max(float(self.totals[6].item() / finite_count) - ratio_mean**2, 0.0)
        ratio_square_sum = float(self.totals[6].item())
        ess_fraction = float(self.totals[5].item()) ** 2 / (finite_count * ratio_square_sum)

        metrics = {
            "token_count": total_count,
            "finite_fraction": finite_count / total_count if total_count else 0.0,
            "sampled_kl": -log_ratio_mean,
            "log_ratio_mean": log_ratio_mean,
            "log_ratio_abs_mean": float(self.totals[3].item() / finite_count),
            "log_ratio_std": log_ratio_variance**0.5,
            "ratio_mean": ratio_mean,
            "ratio_second_moment": float(self.totals[6].item() / finite_count),
            "ratio_std": ratio_variance**0.5,
            "ratio_min": float(torch.exp(self.log_ratio_min.clamp(min=-20.0, max=20.0)).item()),
            "ratio_max": float(torch.exp(self.log_ratio_max.clamp(min=-20.0, max=20.0)).item()),
            "clipfrac_low": float(self.totals[7].item() / finite_count),
            "clipfrac_high": float(self.totals[8].item() / finite_count),
            "clipfrac_total": float((self.totals[7] + self.totals[8]).item() / finite_count),
            "in_clip_fraction": 1.0 - float((self.totals[7] + self.totals[8]).item() / finite_count),
            "pg_clipfrac": float(self.totals[11].item() / finite_count),
            "ess_fraction": ess_fraction,
            "histogram_overflow_fraction": float((self.totals[9] + self.totals[10]).item() / finite_count),
            "sequence_abs_log_ratio_sum_mean": float(self.totals[13].item() / self.totals[12].item()),
        }
        for quantile in (0.05, 0.5, 0.95, 0.99):
            suffix = f"p{int(quantile * 100):02d}"
            metrics[f"ratio_{suffix}"] = float(torch.exp(torch.tensor(self._log_ratio_quantile(quantile))).item())
        return metrics
