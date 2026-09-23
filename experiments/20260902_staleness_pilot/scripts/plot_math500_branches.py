#!/usr/bin/env python3
"""Plot the three post-anchor MATH500 capability trajectories."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_TABLE = EXPERIMENT_DIR / "tables" / "math500_greedy_avg1.csv"
OUTPUT_STEM = EXPERIMENT_DIR / "figures" / "math500_branches_by_optimizer_step"
METRIC = "val-core/math_sis/acc/mean@1"
EXPECTED_BRANCH_STEPS = {
    4: list(range(104, 197, 4)),
    8: list(range(108, 197, 8)),
    16: list(range(116, 197, 16)),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return sorted(rows, key=lambda row: row["optimizer_step"])


def load_series() -> dict[int, list[tuple[int, float]]]:
    anchor_rows = load_jsonl(EXPERIMENT_DIR / "raw" / "anchor" / "eval_metrics.jsonl")
    if [row["optimizer_step"] for row in anchor_rows] != list(range(0, 101, 4)):
        raise ValueError("shared anchor evaluation steps are incomplete")
    anchor = anchor_rows[-1]
    if anchor["optimizer_step"] != 100 or anchor[METRIC] != 0.832:
        raise ValueError("unexpected update-100 anchor")

    series = {}
    for reuse_n, expected_steps in EXPECTED_BRANCH_STEPS.items():
        rows = load_jsonl(
            EXPERIMENT_DIR / "raw" / f"n{reuse_n}" / "eval_metrics.jsonl"
        )
        if [row["optimizer_step"] for row in rows] != expected_steps:
            raise ValueError(f"N={reuse_n}: unexpected branch evaluation steps")
        if any(row["reuse_n"] != reuse_n for row in rows):
            raise ValueError(f"N={reuse_n}: inconsistent reuse_n field")

        values = [(100, 100.0 * anchor[METRIC])]
        values.extend((row["optimizer_step"], 100.0 * row[METRIC]) for row in rows)
        if not all(0.0 <= accuracy <= 100.0 for _, accuracy in values):
            raise ValueError(f"N={reuse_n}: accuracy must be a percentage")
        series[reuse_n] = values

    starts = {values[0] for values in series.values()}
    if starts != {(100, 83.2)}:
        raise ValueError(f"branches do not share the update-100 anchor: {starts}")
    return series


def write_table(series: dict[int, list[tuple[int, float]]]) -> None:
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_TABLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reuse_n", "optimizer_step", "accuracy"])
        for reuse_n in (4, 8, 16):
            for step, accuracy_percent in series[reuse_n]:
                writer.writerow([reuse_n, step, f"{accuracy_percent / 100.0:.3f}"])


def main() -> None:
    series = load_series()
    write_table(series)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
        }
    )

    styles = {
        4: {"color": "#0072B2", "marker": "o"},
        8: {"color": "#D55E00", "marker": "s"},
        16: {"color": "#009E73", "marker": "^"},
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for reuse_n in (4, 8, 16):
        values = series[reuse_n]
        ax.plot(
            [step for step, _ in values],
            [accuracy for _, accuracy in values],
            label=f"N = {reuse_n}",
            linewidth=1.6,
            markersize=4.2,
            markeredgewidth=0.8,
            markerfacecolor="white",
            **styles[reuse_n],
        )

    ax.set_xlim(98, 198)
    ax.set_ylim(81, 88)
    ax.set_xticks([100, 116, 132, 148, 164, 180, 196])
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("MATH500 Accuracy (%)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=3, handlelength=2.2)

    fig.tight_layout(pad=0.6)
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
