#!/usr/bin/env python3
"""Validate the N=1 control run and summarize its age-one KL geometry."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = EXPERIMENT_ROOT / "raw" / "n1_job167543" / "kl_updates.jsonl"
TABLE_ROOT = EXPERIMENT_ROOT / "tables"
SUMMARY_PATH = TABLE_ROOT / "n1_control_validation.json"
PER_UPDATE_PATH = TABLE_ROOT / "n1_control_per_update.csv"


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def max_abs(rows: list[dict], field: str) -> float:
    return max(abs(float(row[field])) for row in rows)


def approximation_summary(rows: list[dict], predicted: str, exact: str) -> dict[str, float]:
    ratios = [float(row[predicted]) / float(row[exact]) for row in rows]
    signed_relative = [ratio - 1.0 for ratio in ratios]
    absolute_relative = [abs(value) for value in signed_relative]
    absolute_log_error = [abs(math.log(ratio)) for ratio in ratios]
    return {
        "median_predicted_over_exact": statistics.median(ratios),
        "mean_signed_relative_error": statistics.fmean(signed_relative),
        "median_signed_relative_error": statistics.median(signed_relative),
        "median_absolute_relative_error": statistics.median(absolute_relative),
        "p90_absolute_relative_error": quantile(absolute_relative, 0.90),
        "p95_absolute_relative_error": quantile(absolute_relative, 0.95),
        "max_absolute_relative_error": max(absolute_relative),
        "typical_error_factor": math.exp(statistics.median(absolute_log_error)),
    }


def main() -> None:
    rows = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 96:
        raise AssertionError(f"expected 96 N=1 updates, found {len(rows)}")
    if [row["optimizer_step"] for row in rows] != list(range(1, 97)):
        raise AssertionError("optimizer steps are incomplete")
    if any(row["reuse_n"] != 1 or row["age_after"] != 1 or row["policy_age"] != 0 for row in rows):
        raise AssertionError("N=1 age fields are inconsistent")
    if any(row["adjacent_kl"] <= 0 or row["cumulative_kl"] <= 0 for row in rows):
        raise AssertionError("exact KL values must be positive")
    if any(row["update_norm"] <= 0 for row in rows):
        raise AssertionError("update_norm must be positive")

    exact_kl = [float(row["cumulative_kl"]) for row in rows]
    max_adjacent_cumulative_difference = max(
        abs(float(row["adjacent_kl"]) - float(row["cumulative_kl"])) for row in rows
    )
    max_fd_step_current_difference = max(
        abs(float(row["frozen_fisher_fd_step"]) - float(row["frozen_fisher_fd_current"]))
        for row in rows
    )

    summary = {
        "analysis_scope": "N=1 control: measurement closure and age-one finite-difference accuracy",
        "source": str(INPUT_PATH.relative_to(EXPERIMENT_ROOT)),
        "job_id": 167543,
        "records": len(rows),
        "optimizer_step_range": [1, 96],
        "data_integrity": {
            "unique_context_sets": len({row["context_set_id"] for row in rows}),
            "context_count_range": [
                min(int(row["context_count"]) for row in rows),
                max(int(row["context_count"]) for row in rows),
            ],
            "weight_sum_range": [
                min(float(row["weight_sum"]) for row in rows),
                max(float(row["weight_sum"]) for row in rows),
            ],
            "aggregation": sorted({row["aggregation"] for row in rows}),
            "direction": sorted({row["direction"] for row in rows}),
            "positive_update_norm_count": sum(float(row["update_norm"]) > 0 for row in rows),
        },
        "control_identities": {
            "max_abs_adjacent_minus_cumulative_kl": max_adjacent_cumulative_difference,
            "max_abs_previous_cumulative_kl": max_abs(rows, "previous_cumulative_kl"),
            "max_abs_kl_three_point_cross": max_abs(rows, "kl_three_point_cross"),
            "max_abs_kl_three_point_error": max_abs(rows, "kl_three_point_error"),
            "max_abs_self_kl": max_abs(rows, "self_kl"),
            "max_abs_frozen_fisher_fd_cumulative": max_abs(rows, "frozen_fisher_fd_cumulative"),
            "max_abs_frozen_fisher_fd_cross": max_abs(rows, "frozen_fisher_fd_cross"),
            "max_abs_frozen_fisher_fd_closure_error": max_abs(
                rows, "frozen_fisher_fd_closure_error"
            ),
            "max_abs_frozen_fisher_fd_step_minus_current": max_fd_step_current_difference,
            "rho_eff_defined_count": sum(bool(row["rho_eff_defined"]) for row in rows),
            "frozen_fisher_fd_cosine_defined_count": sum(
                bool(row["frozen_fisher_fd_cosine_defined"]) for row in rows
            ),
        },
        "exact_kl_distribution": {
            "min": min(exact_kl),
            "median": statistics.median(exact_kl),
            "p95": quantile(exact_kl, 0.95),
            "max": max(exact_kl),
        },
        "finite_difference_accuracy": {
            "step_vs_adjacent_kl": approximation_summary(
                rows, "frozen_fisher_fd_step", "adjacent_kl"
            ),
            "current_vs_cumulative_kl": approximation_summary(
                rows, "frozen_fisher_fd_current", "cumulative_kl"
            ),
        },
    }

    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "optimizer_step",
        "context_count",
        "adjacent_kl",
        "cumulative_kl",
        "frozen_fisher_fd_step",
        "frozen_fisher_fd_current",
        "fd_step_relative_error",
        "fd_current_relative_error",
        "grad_norm",
        "update_norm",
    ]
    with PER_UPDATE_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "optimizer_step": row["optimizer_step"],
                    "context_count": int(row["context_count"]),
                    "adjacent_kl": row["adjacent_kl"],
                    "cumulative_kl": row["cumulative_kl"],
                    "frozen_fisher_fd_step": row["frozen_fisher_fd_step"],
                    "frozen_fisher_fd_current": row["frozen_fisher_fd_current"],
                    "fd_step_relative_error": (
                        row["frozen_fisher_fd_step"] / row["adjacent_kl"] - 1.0
                    ),
                    "fd_current_relative_error": (
                        row["frozen_fisher_fd_current"] / row["cumulative_kl"] - 1.0
                    ),
                    "grad_norm": row["grad_norm"],
                    "update_norm": row["update_norm"],
                }
            )

    print(f"Validated {len(rows)} N=1 updates; wrote {SUMMARY_PATH.name} and {PER_UPDATE_PATH.name}.")


if __name__ == "__main__":
    main()
