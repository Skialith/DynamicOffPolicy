#!/usr/bin/env python3
"""Analyze the four finite-log-prob KL approximation pairs for each N setting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


FAMILY_ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    1: {"directory": "n1_control", "job_directory": "job167543", "job_id": 167543},
    4: {"directory": "n4_main", "job_directory": "job166278", "job_id": 166278},
    16: {"directory": "n16_large_age", "job_directory": "job167220", "job_id": 167220},
    32: {"directory": "n32_large_age", "job_directory": "job167221", "job_id": 167221},
}
AGE_BANDS = ((1, 4), (5, 8), (9, 16), (17, 32))


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


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(centered_x, centered_y)) / denominator


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": quantile(values, 0.95),
        "max": max(values),
    }


def positive_approximation(rows: list[dict], predicted: str, exact: str) -> dict[str, float | int]:
    ratios = [float(row[predicted]) / float(row[exact]) for row in rows]
    signed_relative = [ratio - 1.0 for ratio in ratios]
    absolute_relative = [abs(value) for value in signed_relative]
    absolute_log_error = [abs(math.log(ratio)) for ratio in ratios]
    return {
        "points": len(rows),
        "median_predicted_over_exact": statistics.median(ratios),
        "mean_signed_relative_error": statistics.fmean(signed_relative),
        "median_signed_relative_error": statistics.median(signed_relative),
        "median_absolute_relative_error": statistics.median(absolute_relative),
        "p90_absolute_relative_error": quantile(absolute_relative, 0.90),
        "p95_absolute_relative_error": quantile(absolute_relative, 0.95),
        "max_absolute_relative_error": max(absolute_relative),
        "typical_error_factor": math.exp(statistics.median(absolute_log_error)),
    }


def valid_alignment_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if bool(row["rho_eff_defined"]) and bool(row["frozen_fisher_fd_cosine_defined"])
    ]


def normalized_cross_error(row: dict) -> float:
    scale = 2.0 * math.sqrt(float(row["previous_cumulative_kl"]) * float(row["adjacent_kl"]))
    if scale <= 0:
        raise ValueError("cross-term normalization requires positive KL scale")
    return (float(row["frozen_fisher_fd_cross"]) - float(row["kl_three_point_cross"])) / scale


def sign_agreement(exact: list[float], predicted: list[float]) -> tuple[int, float | None]:
    pairs = [(a, b) for a, b in zip(exact, predicted) if a != 0 and b != 0]
    if not pairs:
        return 0, None
    matches = sum((a > 0) == (b > 0) for a, b in pairs)
    return len(pairs), matches / len(pairs)


def cross_approximation(rows: list[dict]) -> dict[str, float | int | bool | None]:
    valid = valid_alignment_rows(rows)
    if not valid:
        return {"applicable": False, "points": 0, "reason": "no nonzero cumulative direction"}
    exact = [float(row["kl_three_point_cross"]) for row in valid]
    predicted = [float(row["frozen_fisher_fd_cross"]) for row in valid]
    raw_error = [prediction - target for prediction, target in zip(predicted, exact)]
    normalized_error = [normalized_cross_error(row) for row in valid]
    comparable, agreement = sign_agreement(exact, predicted)
    return {
        "applicable": True,
        "points": len(valid),
        "exact_median": statistics.median(exact),
        "predicted_median": statistics.median(predicted),
        "raw_mean_absolute_error": statistics.fmean(abs(value) for value in raw_error),
        "raw_max_absolute_error": max(abs(value) for value in raw_error),
        "median_absolute_normalized_error": statistics.median(abs(value) for value in normalized_error),
        "p95_absolute_normalized_error": quantile([abs(value) for value in normalized_error], 0.95),
        "max_absolute_normalized_error": max(abs(value) for value in normalized_error),
        "pearson": pearson(exact, predicted),
        "sign_comparable_points": comparable,
        "sign_agreement": agreement,
    }


def cosine_approximation(rows: list[dict]) -> dict[str, float | int | bool | None]:
    valid = valid_alignment_rows(rows)
    if not valid:
        return {"applicable": False, "points": 0, "reason": "no nonzero cumulative direction"}
    exact = [float(row["rho_eff"]) for row in valid]
    predicted = [float(row["frozen_fisher_fd_cosine"]) for row in valid]
    errors = [prediction - target for prediction, target in zip(predicted, exact)]
    comparable, agreement = sign_agreement(exact, predicted)
    return {
        "applicable": True,
        "points": len(valid),
        "rho_eff_median": statistics.median(exact),
        "frozen_fisher_fd_cosine_median": statistics.median(predicted),
        "mean_absolute_error": statistics.fmean(abs(value) for value in errors),
        "median_absolute_error": statistics.median(abs(value) for value in errors),
        "p95_absolute_error": quantile([abs(value) for value in errors], 0.95),
        "max_absolute_error": max(abs(value) for value in errors),
        "pearson": pearson(exact, predicted),
        "sign_comparable_points": comparable,
        "sign_agreement": agreement,
    }


def read_and_validate(n: int, subexperiment_root: Path, job_directory: str) -> list[dict]:
    path = subexperiment_root / "raw" / job_directory / "kl_updates.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 96:
        raise AssertionError(f"N={n}: expected 96 updates, found {len(rows)}")
    if [int(row["optimizer_step"]) for row in rows] != list(range(1, 97)):
        raise AssertionError(f"N={n}: optimizer steps are incomplete")
    if any(int(row["reuse_n"]) != n for row in rows):
        raise AssertionError(f"N={n}: reuse_n mismatch")
    expected_rollouts = 96 // n
    for rollout_index in range(expected_rollouts):
        block = rows[rollout_index * n:(rollout_index + 1) * n]
        if [int(row["age_after"]) for row in block] != list(range(1, n + 1)):
            raise AssertionError(f"N={n}: rollout {rollout_index + 1} has invalid ages")
        if len({int(row["anchor_update"]) for row in block}) != 1:
            raise AssertionError(f"N={n}: rollout {rollout_index + 1} changed anchor")
    positive_fields = (
        "adjacent_kl",
        "cumulative_kl",
        "frozen_fisher_fd_step",
        "frozen_fisher_fd_current",
    )
    if any(float(row[field]) <= 0 for row in rows for field in positive_fields):
        raise AssertionError(f"N={n}: positive KL quantity is non-positive")
    expected_valid = len(rows) - expected_rollouts
    if len(valid_alignment_rows(rows)) != expected_valid:
        raise AssertionError(f"N={n}: unexpected number of defined alignment records")
    return rows


def summarize_subset(rows: list[dict]) -> dict[str, object]:
    result: dict[str, object] = {
        "points": len(rows),
        "comparison_1_step_vs_adjacent": positive_approximation(
            rows, "frozen_fisher_fd_step", "adjacent_kl"
        ),
        "comparison_2_current_vs_cumulative": positive_approximation(
            rows, "frozen_fisher_fd_current", "cumulative_kl"
        ),
        "comparison_3_cross": cross_approximation(rows),
        "comparison_4_cosine_vs_rho": cosine_approximation(rows),
    }
    return result


def summarize(n: int, rows: list[dict], spec: dict) -> dict[str, object]:
    exact_adjacent = [float(row["adjacent_kl"]) for row in rows]
    exact_cumulative = [float(row["cumulative_kl"]) for row in rows]
    age_bands = {}
    for lower, upper in AGE_BANDS:
        subset = [row for row in rows if lower <= int(row["age_after"]) <= upper]
        if subset:
            age_bands[f"{lower}-{min(upper, n)}"] = summarize_subset(subset)
    return {
        "n": n,
        "job_id": spec["job_id"],
        "records": len(rows),
        "rollouts": len(rows) // n,
        "age_range": [1, n],
        "analysis_unit_note": "Rows within a rollout share its anchor and context set; statistics are descriptive, not IID confidence estimates.",
        "data_integrity": {
            "optimizer_step_range": [1, 96],
            "unique_context_sets": len({row["context_set_id"] for row in rows}),
            "context_count_range": [
                min(int(row["context_count"]) for row in rows),
                max(int(row["context_count"]) for row in rows),
            ],
            "max_abs_self_kl": max(abs(float(row["self_kl"])) for row in rows),
            "max_abs_kl_three_point_error": max(
                abs(float(row["kl_three_point_error"])) for row in rows
            ),
            "max_abs_frozen_fisher_fd_closure_error": max(
                abs(float(row["frozen_fisher_fd_closure_error"])) for row in rows
            ),
        },
        "exact_kl_distribution": {
            "adjacent_kl": distribution(exact_adjacent),
            "cumulative_kl": distribution(exact_cumulative),
        },
        **summarize_subset(rows),
        "age_bands": age_bands,
    }


def per_update_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        defined = bool(row["rho_eff_defined"]) and bool(row["frozen_fisher_fd_cosine_defined"])
        output.append(
            {
                "optimizer_step": int(row["optimizer_step"]),
                "rollout_step": int(row["rollout_step"]),
                "age_after": int(row["age_after"]),
                "adjacent_kl": row["adjacent_kl"],
                "frozen_fisher_fd_step": row["frozen_fisher_fd_step"],
                "step_relative_error": row["frozen_fisher_fd_step"] / row["adjacent_kl"] - 1.0,
                "cumulative_kl": row["cumulative_kl"],
                "frozen_fisher_fd_current": row["frozen_fisher_fd_current"],
                "current_relative_error": (
                    row["frozen_fisher_fd_current"] / row["cumulative_kl"] - 1.0
                ),
                "kl_three_point_cross": row["kl_three_point_cross"],
                "frozen_fisher_fd_cross": row["frozen_fisher_fd_cross"],
                "cross_normalized_error": normalized_cross_error(row) if defined else "",
                "rho_eff": row["rho_eff"] if defined else "",
                "frozen_fisher_fd_cosine": row["frozen_fisher_fd_cosine"] if defined else "",
                "cosine_minus_rho": (
                    row["frozen_fisher_fd_cosine"] - row["rho_eff"] if defined else ""
                ),
            }
        )
    return output


def write_tables(subexperiment_root: Path, summary: dict, rows: list[dict]) -> None:
    table_root = subexperiment_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    (table_root / "kl_approximation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    update_rows = per_update_rows(rows)
    with (table_root / "kl_approximation_per_update.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(update_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(update_rows)

    fieldnames = [
        "age_after",
        "points",
        "adjacent_kl_median",
        "fd_step_ratio_median",
        "fd_step_median_abs_relative_error",
        "fd_step_p95_abs_relative_error",
        "cumulative_kl_median",
        "fd_current_ratio_median",
        "fd_current_median_abs_relative_error",
        "fd_current_p95_abs_relative_error",
        "cross_points",
        "cross_median_abs_normalized_error",
        "cross_p95_abs_normalized_error",
        "cross_sign_agreement",
        "rho_cosine_mean_abs_error",
        "rho_cosine_p95_abs_error",
        "rho_cosine_pearson",
        "rho_cosine_sign_agreement",
    ]
    with (table_root / "kl_approximation_by_age.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for age in sorted({int(row["age_after"]) for row in rows}):
            subset = [row for row in rows if int(row["age_after"]) == age]
            step = positive_approximation(subset, "frozen_fisher_fd_step", "adjacent_kl")
            current = positive_approximation(subset, "frozen_fisher_fd_current", "cumulative_kl")
            cross = cross_approximation(subset)
            cosine = cosine_approximation(subset)
            writer.writerow(
                {
                    "age_after": age,
                    "points": len(subset),
                    "adjacent_kl_median": statistics.median(
                        float(row["adjacent_kl"]) for row in subset
                    ),
                    "fd_step_ratio_median": step["median_predicted_over_exact"],
                    "fd_step_median_abs_relative_error": step["median_absolute_relative_error"],
                    "fd_step_p95_abs_relative_error": step["p95_absolute_relative_error"],
                    "cumulative_kl_median": statistics.median(
                        float(row["cumulative_kl"]) for row in subset
                    ),
                    "fd_current_ratio_median": current["median_predicted_over_exact"],
                    "fd_current_median_abs_relative_error": current[
                        "median_absolute_relative_error"
                    ],
                    "fd_current_p95_abs_relative_error": current["p95_absolute_relative_error"],
                    "cross_points": cross["points"],
                    "cross_median_abs_normalized_error": (
                        cross.get("median_absolute_normalized_error", "")
                    ),
                    "cross_p95_abs_normalized_error": (
                        cross.get("p95_absolute_normalized_error", "")
                    ),
                    "cross_sign_agreement": cross.get("sign_agreement", ""),
                    "rho_cosine_mean_abs_error": cosine.get("mean_absolute_error", ""),
                    "rho_cosine_p95_abs_error": cosine.get("p95_absolute_error", ""),
                    "rho_cosine_pearson": cosine.get("pearson", ""),
                    "rho_cosine_sign_agreement": cosine.get("sign_agreement", ""),
                }
            )


def identity_line(ax, x: list[float], y: list[float], logarithmic: bool = False) -> None:
    lower = min(x + y)
    upper = max(x + y)
    if logarithmic:
        lower *= 0.9
        upper *= 1.1
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        padding = max((upper - lower) * 0.08, abs(upper) * 0.02, 1e-12)
        lower -= padding
        upper += padding
    ax.plot([lower, upper], [lower, upper], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)


def plot_parity(n: int, rows: list[dict], subexperiment_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    figure_root = subexperiment_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    ages = [int(row["age_after"]) for row in rows]
    norm = Normalize(vmin=1, vmax=max(2, n))
    cmap = "viridis"

    comparisons = [
        ("adjacent_kl", "frozen_fisher_fd_step", "Step energy vs adjacent KL", True),
        ("cumulative_kl", "frozen_fisher_fd_current", "Current energy vs cumulative KL", True),
    ]
    for ax, (exact_field, predicted_field, title, logarithmic) in zip(axes[0], comparisons):
        exact = [float(row[exact_field]) for row in rows]
        predicted = [float(row[predicted_field]) for row in rows]
        ax.scatter(exact, predicted, c=ages, cmap=cmap, norm=norm, s=24, alpha=0.78)
        identity_line(ax, exact, predicted, logarithmic=logarithmic)
        ax.set_title(title)
        ax.set_xlabel("Exact full-vocabulary KL")
        ax.set_ylabel("Finite-difference quadratic")
        ax.grid(alpha=0.2)

    valid = valid_alignment_rows(rows)
    if valid:
        exact_cross = [float(row["kl_three_point_cross"]) for row in valid]
        predicted_cross = [float(row["frozen_fisher_fd_cross"]) for row in valid]
        valid_ages = [int(row["age_after"]) for row in valid]
        axes[1, 0].scatter(
            exact_cross, predicted_cross, c=valid_ages, cmap=cmap, norm=norm, s=24, alpha=0.78
        )
        identity_line(axes[1, 0], exact_cross, predicted_cross)
        axes[1, 0].set_title("Finite-difference cross vs exact KL cross")
        axes[1, 0].set_xlabel("Exact three-point KL cross")
        axes[1, 0].set_ylabel("Finite-difference cross")
        axes[1, 0].grid(alpha=0.2)

        rho = [float(row["rho_eff"]) for row in valid]
        cosine = [float(row["frozen_fisher_fd_cosine"]) for row in valid]
        axes[1, 1].scatter(rho, cosine, c=valid_ages, cmap=cmap, norm=norm, s=24, alpha=0.78)
        identity_line(axes[1, 1], rho, cosine)
        axes[1, 1].set_title("Finite-difference cosine vs effective rho")
        axes[1, 1].set_xlabel("rho_eff")
        axes[1, 1].set_ylabel("Frozen-Fisher finite-difference cosine")
        axes[1, 1].grid(alpha=0.2)
    else:
        for ax, title in zip(
            axes[1],
            ("Cross-term comparison", "Cosine vs rho_eff"),
        ):
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "Not applicable for N=1:\nno pre-existing cumulative direction",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title(title)

    if n > 1:
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=axes.ravel().tolist(),
            shrink=0.78,
            pad=0.02,
        )
        colorbar.set_label("age_after")
    fig.suptitle(f"N={n}: finite-log-prob geometry against exact KL", fontsize=15)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_root / f"kl_approximation_2x2.{suffix}", dpi=220)
    plt.close(fig)


def plot_errors_by_age(n: int, rows: list[dict], subexperiment_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = subexperiment_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    age_values = sorted({int(row["age_after"]) for row in rows})

    series = [
        (
            axes[0, 0],
            [abs(float(row["frozen_fisher_fd_step"]) / float(row["adjacent_kl"]) - 1.0) * 100 for row in rows],
            "Step vs adjacent KL",
            "Absolute relative error (%)",
        ),
        (
            axes[0, 1],
            [abs(float(row["frozen_fisher_fd_current"]) / float(row["cumulative_kl"]) - 1.0) * 100 for row in rows],
            "Current vs cumulative KL",
            "Absolute relative error (%)",
        ),
    ]
    ages = [int(row["age_after"]) for row in rows]
    for ax, values, title, ylabel in series:
        ax.scatter(ages, values, color="#0072B2", alpha=0.32, s=20)
        medians = [statistics.median(v for a, v in zip(ages, values) if a == age) for age in age_values]
        ax.plot(age_values, medians, color="#D55E00", marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("age_after")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    valid = valid_alignment_rows(rows)
    if valid:
        valid_ages = [int(row["age_after"]) for row in valid]
        cross_errors = [abs(normalized_cross_error(row)) for row in valid]
        cosine_errors = [
            abs(float(row["frozen_fisher_fd_cosine"]) - float(row["rho_eff"])) for row in valid
        ]
        for ax, values, title, ylabel in (
            (
                axes[1, 0],
                cross_errors,
                "Cross-term approximation",
                "Absolute normalized cross error",
            ),
            (
                axes[1, 1],
                cosine_errors,
                "Cosine vs rho_eff",
                "Absolute alignment error",
            ),
        ):
            ax.scatter(valid_ages, values, color="#009E73", alpha=0.38, s=20)
            valid_age_values = sorted(set(valid_ages))
            medians = [
                statistics.median(v for a, v in zip(valid_ages, values) if a == age)
                for age in valid_age_values
            ]
            ax.plot(valid_age_values, medians, color="#CC79A7", marker="o", linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("age_after")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
    else:
        for ax, title in zip(axes[1], ("Cross-term approximation", "Cosine vs rho_eff")):
            ax.axis("off")
            ax.text(0.5, 0.5, "Not applicable for N=1", ha="center", va="center", fontsize=12)
            ax.set_title(title)

    fig.suptitle(f"N={n}: approximation error by within-rollout age", fontsize=15)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_root / f"kl_approximation_error_by_age.{suffix}", dpi=220)
    plt.close(fig)


def analyze_one(n: int, make_plots: bool) -> None:
    spec = SPECS[n]
    subexperiment_root = FAMILY_ROOT / spec["directory"]
    rows = read_and_validate(n, subexperiment_root, spec["job_directory"])
    summary = summarize(n, rows, spec)
    write_tables(subexperiment_root, summary, rows)
    if make_plots:
        plot_parity(n, rows, subexperiment_root)
        plot_errors_by_age(n, rows, subexperiment_root)
    print(
        f"N={n}: validated {len(rows)} updates across {len(rows) // n} rollouts; "
        f"wrote tables{' and figures' if make_plots else ''}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", choices=("1", "4", "16", "32", "all"), default="all")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    selected = sorted(SPECS) if args.n == "all" else [int(args.n)]
    for n in selected:
        analyze_one(n, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
