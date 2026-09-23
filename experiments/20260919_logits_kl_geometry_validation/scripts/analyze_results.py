#!/usr/bin/env python3
"""Summarize the completed N=4 setting of the logits-geometry KL validation.

The script reads only the archived JSONL files under raw/ and writes compact,
reproducible tables under tables/.  It intentionally produces no figures.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = EXPERIMENT_ROOT / "raw"
TABLE_ROOT = EXPERIMENT_ROOT / "tables"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0:
        return float("nan")
    return sum(a * b for a, b in zip(centered_x, centered_y)) / denominator


def typical_factor(predicted: list[float], observed: list[float]) -> float:
    errors = [abs(math.log(pred / obs)) for pred, obs in zip(predicted, observed)]
    return math.exp(statistics.median(errors))


def validate_probe(rows: list[dict]) -> None:
    if len(rows) != 4:
        raise AssertionError(f"probe must contain 4 updates, found {len(rows)}")
    if [row["optimizer_step"] for row in rows] != [1, 2, 3, 4]:
        raise AssertionError("probe optimizer steps are incomplete")
    if [row["age_after"] for row in rows] != [1, 2, 3, 4]:
        raise AssertionError("probe ages are incomplete")
    validate_numeric_fields(rows)


def validate_formal(rows: list[dict]) -> None:
    if len(rows) != 96:
        raise AssertionError(f"formal run must contain 96 updates, found {len(rows)}")
    if [row["optimizer_step"] for row in rows] != list(range(1, 97)):
        raise AssertionError("formal optimizer steps are incomplete")
    for rollout_index in range(24):
        block = rows[4 * rollout_index:4 * rollout_index + 4]
        if [row["age_after"] for row in block] != [1, 2, 3, 4]:
            raise AssertionError(f"rollout {rollout_index + 1} has invalid ages")
        if {row["anchor_update"] for row in block} != {4 * rollout_index}:
            raise AssertionError(f"rollout {rollout_index + 1} has invalid anchor")
        if len({row["context_set_id"] for row in block}) != 1:
            raise AssertionError(f"rollout {rollout_index + 1} changed contexts")
    validate_numeric_fields(rows)


def validate_numeric_fields(rows: list[dict]) -> None:
    required = {
        "adjacent_kl", "cumulative_kl", "previous_cumulative_kl", "grad_norm",
        "update_norm", "kl_three_point_error", "rho_eff",
        "frozen_fisher_fd_cosine", "frozen_fisher_fd_closure_error",
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise AssertionError(f"missing fields at update {row.get('optimizer_step')}: {missing}")
        if row["direction"] != "older_to_newer" or row["vocabulary"] != "full":
            raise AssertionError("unexpected KL direction or vocabulary aggregation")
        if row["update_applied"] != 1.0 or row["update_norm"] <= 0:
            raise AssertionError("an optimizer update was not applied")
        if row["adjacent_kl"] <= 0 or row["cumulative_kl"] <= 0:
            raise AssertionError("KL values must be positive")
        if abs(row["weight_sum"] - 1.0) > 1e-8 or abs(row["self_kl"]) > 1e-9:
            raise AssertionError("KL aggregation checks failed")
        if abs(row["kl_three_point_error"]) > 1e-9:
            raise AssertionError("KL three-point identity failed")
        if abs(row["frozen_fisher_fd_closure_error"]) > 1e-9:
            raise AssertionError("finite-difference Fisher closure failed")
        expected_defined = float(row["age_after"] > 1)
        if row["rho_eff_defined"] != expected_defined:
            raise AssertionError("rho_eff defined flag does not match age")
        if row["frozen_fisher_fd_cosine_defined"] != expected_defined:
            raise AssertionError("Fisher cosine defined flag does not match age")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def alignment_rows(rows: list[dict]) -> list[dict]:
    result = []
    groups: list[tuple[str, list[dict]]] = [
        (str(age), [row for row in rows if row["age_after"] == age])
        for age in (2, 3, 4)
    ]
    groups.append(("all", [row for row in rows if row["age_after"] > 1]))
    for label, subset in groups:
        rho = [row["rho_eff"] for row in subset]
        fisher = [row["frozen_fisher_fd_cosine"] for row in subset]
        difference = [left - right for left, right in zip(rho, fisher)]
        result.append({
            "age_after": label,
            "n": len(subset),
            "rho_eff_median": statistics.median(rho),
            "frozen_fisher_fd_cosine_median": statistics.median(fisher),
            "difference_mean_bias": statistics.fmean(difference),
            "difference_median": statistics.median(difference),
            "difference_mae": statistics.fmean(abs(value) for value in difference),
            "difference_max_abs": max(abs(value) for value in difference),
            "pearson_correlation": pearson(rho, fisher),
        })
    return result


def quadratic_rows(rows: list[dict]) -> list[dict]:
    result = []
    groups: list[tuple[str, list[dict]]] = [
        (str(age), [row for row in rows if row["age_after"] == age])
        for age in (1, 2, 3, 4)
    ]
    groups.append(("all", rows))
    for label, subset in groups:
        adjacent_approx = [row["frozen_fisher_fd_step"] for row in subset]
        adjacent_exact = [row["adjacent_kl"] for row in subset]
        cumulative_approx = [row["frozen_fisher_fd_current"] for row in subset]
        cumulative_exact = [row["cumulative_kl"] for row in subset]
        result.append({
            "age_after": label,
            "n": len(subset),
            "adjacent_approx_over_exact_median": statistics.median(
                pred / obs for pred, obs in zip(adjacent_approx, adjacent_exact)
            ),
            "adjacent_typical_error_factor": typical_factor(adjacent_approx, adjacent_exact),
            "cumulative_approx_over_exact_median": statistics.median(
                pred / obs for pred, obs in zip(cumulative_approx, cumulative_exact)
            ),
            "cumulative_typical_error_factor": typical_factor(cumulative_approx, cumulative_exact),
        })
    return result


def conversion_rows(rows: list[dict]) -> list[dict]:
    definitions = {
        "raw_gradient": lambda row: row["adjacent_kl"] / (
            row["lr_used"] * row["grad_norm"]
        ) ** 2,
        "adamw_update": lambda row: row["adjacent_kl"] / row["update_norm"] ** 2,
    }
    result = []
    for basis, compute in definitions.items():
        groups: list[tuple[str, list[dict]]] = [("all", rows)]
        groups.extend(
            (str(age), [row for row in rows if row["age_after"] == age])
            for age in (1, 2, 3, 4)
        )
        for label, subset in groups:
            values = [compute(row) for row in subset]
            q25 = quantile(values, 0.25)
            q75 = quantile(values, 0.75)
            median = statistics.median(values)
            result.append({
                "basis": basis,
                "age_after": label,
                "n": len(values),
                "coefficient_median": median,
                "coefficient_q25": q25,
                "coefficient_q75": q75,
                "coefficient_iqr_factor": q75 / q25,
                "typical_factor_to_median": math.exp(
                    statistics.median(abs(math.log(value / median)) for value in values)
                ),
            })
    return result


def main() -> None:
    probe_path = RAW_ROOT / "n4_probe_job166277" / "kl_updates.jsonl"
    formal_path = RAW_ROOT / "n4_job166278" / "kl_updates.jsonl"
    probe = read_jsonl(probe_path)
    formal = read_jsonl(formal_path)
    validate_probe(probe)
    validate_formal(formal)

    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    alignment = alignment_rows(formal)
    quadratic = quadratic_rows(formal)
    conversion = conversion_rows(formal)
    write_csv(TABLE_ROOT / "formula_alignment_by_age.csv", alignment)
    write_csv(TABLE_ROOT / "local_quadratic_by_age.csv", quadratic)
    write_csv(TABLE_ROOT / "conversion_stability.csv", conversion)

    all_alignment = next(row for row in alignment if row["age_after"] == "all")
    all_quadratic = next(row for row in quadratic if row["age_after"] == "all")
    raw_gradient = next(
        row for row in conversion
        if row["basis"] == "raw_gradient" and row["age_after"] == "all"
    )
    adamw_update = next(
        row for row in conversion
        if row["basis"] == "adamw_update" and row["age_after"] == "all"
    )
    age_medians = {
        basis: [
            row["coefficient_median"] for row in conversion
            if row["basis"] == basis and row["age_after"] != "all"
        ]
        for basis in ("raw_gradient", "adamw_update")
    }
    summary = {
        "jobs": {
            "probe": {"job_id": 166277, "updates": len(probe)},
            "formal": {"job_id": 166278, "updates": len(formal), "rollouts": 24},
        },
        "validation": {
            "max_abs_kl_three_point_error": max(
                abs(row["kl_three_point_error"]) for row in formal
            ),
            "max_abs_frozen_fisher_fd_closure_error": max(
                abs(row["frozen_fisher_fd_closure_error"]) for row in formal
            ),
            "all_update_norm_positive": all(row["update_norm"] > 0 for row in formal),
        },
        "rho_alignment_age_2_to_4": all_alignment,
        "rho_signs_age_2_to_4": {
            "rho_eff_negative": sum(
                row["rho_eff"] < 0 for row in formal if row["age_after"] > 1
            ),
            "frozen_fisher_fd_cosine_negative": sum(
                row["frozen_fisher_fd_cosine"] < 0
                for row in formal if row["age_after"] > 1
            ),
            "total": 72,
        },
        "local_quadratic_all_updates": all_quadratic,
        "conversion_stability_all_updates": {
            "raw_gradient": raw_gradient,
            "adamw_update": adamw_update,
        },
        "conversion_age_median_max_over_min": {
            basis: max(values) / min(values) for basis, values in age_medians.items()
        },
    }
    with (TABLE_ROOT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
