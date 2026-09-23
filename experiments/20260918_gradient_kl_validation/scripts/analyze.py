#!/usr/bin/env python3
"""Retrospective gradient/KL checks using the archived full-vocabulary KL logs."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
TABLES = ROOT / "tables"


def load_run(n):
    path = RAW / f"n{n}" / "kl_updates.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 24 * n
    cycles = defaultdict(list)
    for step, row in enumerate(rows, 1):
        assert row["optimizer_step"] == step
        assert row["reuse_n"] == n
        assert row["optimizer_step"] == row["anchor_update"] + row["age_after"]
        assert row["policy_age"] == row["age_after"] - 1
        assert row["adjacent_kl"] > 0 and row["cumulative_kl"] > 0
        assert row["grad_norm"] > 0 and row["lr_used"] > 0
        assert row["direction"] == "older_to_newer"
        assert row["aggregation"] == "prompt_equal_then_position_equal"
        assert row["vocabulary"] == "full"
        cycles[row["rollout_step"]].append(row)
    assert sorted(cycles) == list(range(1, 25))
    for rollout, cycle in cycles.items():
        assert [r["age_after"] for r in cycle] == list(range(1, n + 1))
        assert {r["anchor_update"] for r in cycle} == {(rollout - 1) * n}
        assert len({r["context_set_id"] for r in cycle}) == 1
        assert len({r["context_count"] for r in cycle}) == 1
        assert math.isclose(cycle[0]["adjacent_kl"], cycle[0]["cumulative_kl"], abs_tol=1e-12)
    return rows, cycles


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score(actual, predicted):
    error = np.log(np.asarray(predicted) / np.asarray(actual))
    return {
        "n": len(error),
        "mean_abs_log_error": float(np.mean(np.abs(error))),
        "median_error_factor": float(np.exp(np.median(np.abs(error)))),
        "mean_log_bias": float(np.mean(error)),
        "underprediction_count": int(np.count_nonzero(error < 0)),
        "max_underprediction_factor": float(np.exp(max(0.0, -np.min(error)))),
    }


def conversion(n4_rows, n8_rows, n4_cycles):
    train_d = np.array([r["adjacent_kl"] for r in n4_rows])
    train_g = np.array([r["grad_norm"] for r in n4_rows])
    train_x = np.array([(r["lr_used"] * r["grad_norm"]) ** 2 for r in n4_rows])
    constant = float(np.median(train_d))
    coefficient = float(np.median(train_d / train_x))
    slope, intercept = np.polyfit(np.log(train_g), np.log(train_d), 1)

    # Preserve some temporal dependence by resampling four-cycle blocks.
    rng = np.random.default_rng(20260918)
    groups = list(n4_cycles.values())
    blocks = [groups[i:i + 4] for i in range(0, len(groups), 4)]
    boot_slopes = []
    for _ in range(1000):
        sampled = [r for i in rng.integers(0, len(blocks), len(blocks))
                   for cycle in blocks[i] for r in cycle]
        g = np.log([r["grad_norm"] for r in sampled])
        d = np.log([r["adjacent_kl"] for r in sampled])
        boot_slopes.append(np.polyfit(g, d, 1)[0])

    def predict(row):
        return {
            "constant": constant,
            "quadratic": coefficient * (row["lr_used"] * row["grad_norm"]) ** 2,
            "free_power": float(np.exp(intercept + slope * np.log(row["grad_norm"]))),
        }

    train_residual_q90 = {
        model: float(np.quantile([np.log(r["adjacent_kl"] / predict(r)[model]) for r in n4_rows], 0.9))
        for model in ("constant", "quadratic", "free_power")
    }
    all_rows = []
    for n, rows in ((4, n4_rows), (8, n8_rows)):
        for row in rows:
            segment = "train_n4" if n == 4 else ("n8_early" if row["age_after"] <= 4 else "n8_late_oracle")
            preds = predict(row)
            all_rows.append({
                "n": n,
                "rollout_step": row["rollout_step"],
                "optimizer_step": row["optimizer_step"],
                "age_after": row["age_after"],
                "segment": segment,
                "grad_norm": row["grad_norm"],
                "adjacent_kl": row["adjacent_kl"],
                "c_eff": row["adjacent_kl"] / (row["lr_used"] * row["grad_norm"]) ** 2,
                **{f"pred_{name}": value for name, value in preds.items()},
            })
    write_csv(TABLES / "local_conversion.csv", all_rows, list(all_rows[0]))
    coefficient_by_age = []
    for n in (4, 8):
        for age in range(1, n + 1):
            values = [r["c_eff"] for r in all_rows if r["n"] == n and r["age_after"] == age]
            coefficient_by_age.append({
                "n": n,
                "age_after": age,
                "median_c_eff": float(np.median(values)),
                "q25_c_eff": float(np.quantile(values, 0.25)),
                "q75_c_eff": float(np.quantile(values, 0.75)),
            })
    write_csv(TABLES / "coefficient_by_age.csv", coefficient_by_age, list(coefficient_by_age[0]))

    subsets = {
        "n4_fit": n4_rows,
        "n8_age_1_4": [r for r in n8_rows if r["age_after"] <= 4],
        "n8_age_5_8_oracle": [r for r in n8_rows if r["age_after"] >= 5],
        "n8_age_1_4_updates_1_96": [r for r in n8_rows if r["age_after"] <= 4 and r["optimizer_step"] <= 96],
        "n8_age_5_8_updates_1_96_oracle": [r for r in n8_rows if r["age_after"] >= 5 and r["optimizer_step"] <= 96],
    }
    scores = {}
    for subset, rows in subsets.items():
        coefficients = [r["adjacent_kl"] / (r["lr_used"] * r["grad_norm"]) ** 2 for r in rows]
        scores[subset] = {
            "observed_c_eff_median": float(np.median(coefficients)),
            "observed_c_eff_q25_q75": [float(v) for v in np.quantile(coefficients, [0.25, 0.75])],
        }
        for model in train_residual_q90:
            predicted = [predict(r)[model] for r in rows]
            upper = np.array(predicted) * np.exp(train_residual_q90[model])
            scores[subset][model] = {
                **score([r["adjacent_kl"] for r in rows], predicted),
                "upper_90_residual_coverage": float(np.mean([r["adjacent_kl"] <= u for r, u in zip(rows, upper)])),
            }
    return {
        "fit": {
            "constant_adjacent_kl": constant,
            "median_c_eff": coefficient,
            "log_gradient_slope": float(slope),
            "slope_four_cycle_block_bootstrap_95pct": [float(v) for v in np.quantile(boot_slopes, [0.025, 0.975])],
            "train_log_residual_q90": train_residual_q90,
        },
        "scores": scores,
    }


def accumulation(runs):
    derived = []
    by_age = []
    summaries = {}
    for n, (_, cycles) in runs.items():
        run_rows = []
        for cycle in cycles.values():
            previous_k = 0.0
            sum_d = 0.0
            for row in cycle:
                d, k = row["adjacent_kl"], row["cumulative_kl"]
                sum_d += d
                residual = k - previous_k - d if row["age_after"] >= 2 else 0.0
                rho = residual / (2 * math.sqrt(previous_k * d)) if row["age_after"] >= 2 else ""
                result = {
                    "n": n,
                    "rollout_step": row["rollout_step"],
                    "age_after": row["age_after"],
                    "adjacent_kl": d,
                    "cumulative_kl": k,
                    "sum_adjacent_kl": sum_d,
                    "sum_to_K_ratio": sum_d / k,
                    "C_eff": residual,
                    "rho_eff": rho,
                    "K_decreased": int(k < previous_k),
                }
                run_rows.append(result)
                derived.append(result)
                previous_k = k
        for age in range(2, n + 1):
            subset = [r for r in run_rows if r["age_after"] == age]
            by_age.append({
                "n": n,
                "age_after": age,
                "median_K_over_sum_d": float(np.median([1 / r["sum_to_K_ratio"] for r in subset])),
                "median_rho_eff": float(np.median([r["rho_eff"] for r in subset])),
                "positive_C_count": sum(r["C_eff"] > 0 for r in subset),
                "rho_outside_unit_count": sum(abs(r["rho_eff"]) > 1 for r in subset),
            })
        later = [r for r in run_rows if r["age_after"] >= 2]
        endpoints = [r for r in run_rows if r["age_after"] == n]
        summaries[f"n{n}"] = {
            "noninitial_updates": len(later),
            "positive_C_count": sum(r["C_eff"] > 0 for r in later),
            "negative_C_count": sum(r["C_eff"] < 0 for r in later),
            "rho_outside_unit_count": sum(abs(r["rho_eff"]) > 1 for r in later),
            "K_decreased_count": sum(r["K_decreased"] for r in later),
            "endpoint_median_K_over_sum_d": float(np.median([1 / r["sum_to_K_ratio"] for r in endpoints])),
            "endpoint_median_sum_error_factor": float(np.exp(np.median([abs(math.log(r["sum_to_K_ratio"])) for r in endpoints]))),
        }
    write_csv(TABLES / "cumulative_residuals.csv", derived, list(derived[0]))
    write_csv(TABLES / "accumulation_by_age.csv", by_age, list(by_age[0]))
    return summaries


def ridge_forecast(features, target, train_size, columns):
    train = features[:train_size, columns]
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std == 0] = 1
    design = np.column_stack([np.ones(len(features)), (features[:, columns] - mean) / std])
    penalty = np.diag([0] + [1] * len(columns))
    weights = np.linalg.solve(design[:train_size].T @ design[:train_size] + penalty,
                              design[:train_size].T @ target[:train_size])
    return design @ weights


def future_peaks(n8_cycles):
    rows = []
    for cycle in n8_cycles.values():
        early, late = cycle[:4], cycle[4:]
        rows.append({
            "rollout_step": cycle[0]["rollout_step"],
            "anchor_update": cycle[0]["anchor_update"],
            "early_peak": max(r["cumulative_kl"] for r in early),
            "early_mean_grad": float(np.mean([r["grad_norm"] for r in early])),
            "future_peak": max(r["cumulative_kl"] for r in late),
        })
    features = np.array([[r["anchor_update"], math.log(r["early_peak"]),
                          math.log(r["early_mean_grad"])] for r in rows])
    target = np.log([r["future_peak"] for r in rows])
    train_size = 16
    forecasts = {
        "progress_only": ridge_forecast(features, target, train_size, [0]),
        "progress_early_K": ridge_forecast(features, target, train_size, [0, 1]),
        "progress_early_K_grad": ridge_forecast(features, target, train_size, [0, 1, 2]),
    }
    diagnostics = {}
    for model, log_predictions in forecasts.items():
        # In-sample residual quantiles are diagnostic, not calibrated safety bounds.
        residual_q90 = float(np.quantile(target[:train_size] - log_predictions[:train_size], 0.9))
        upper = np.exp(log_predictions + residual_q90)
        predictions = np.exp(log_predictions)
        for i, row in enumerate(rows):
            row[f"pred_{model}"] = float(predictions[i])
            row[f"upper_{model}"] = float(upper[i])
        diagnostics[model] = {
            "train": score(np.exp(target[:train_size]), predictions[:train_size]),
            "holdout": {
                **score(np.exp(target[train_size:]), predictions[train_size:]),
                "upper_90_residual_coverage_count": int(np.count_nonzero(np.exp(target[train_size:]) <= upper[train_size:])),
            },
            "train_log_residual_q90": residual_q90,
        }
    for i, row in enumerate(rows):
        row["split"] = "fit_first_16" if i < train_size else "holdout_last_8"
        row["pred_early_peak_naive"] = row["early_peak"]
    diagnostics["early_peak_naive"] = {
        "holdout": score([r["future_peak"] for r in rows[train_size:]],
                         [r["early_peak"] for r in rows[train_size:]])
    }
    write_csv(TABLES / "future_peak_predictions.csv", rows, list(rows[0]))
    return {
        "fit_rollouts": list(range(1, 17)),
        "holdout_rollouts": list(range(17, 25)),
        "early_peak_below_future_count_all_24": sum(r["early_peak"] < r["future_peak"] for r in rows),
        "scores": diagnostics,
    }


def main():
    TABLES.mkdir(exist_ok=True)
    runs = {n: load_run(n) for n in (4, 8)}
    result = {
        "conversion": conversion(runs[4][0], runs[8][0], runs[4][1]),
        "accumulation": accumulation(runs),
        "future_peak": future_peaks(runs[8][1]),
    }
    (TABLES / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
