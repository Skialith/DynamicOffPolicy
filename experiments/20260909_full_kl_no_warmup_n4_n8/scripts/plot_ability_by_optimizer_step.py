#!/usr/bin/env python3
"""Plot MATH500 sampled Avg@1 against real optimizer updates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = EXPERIMENT_DIR / "figures" / "ability_by_optimizer_step.png"
EXPECTED_ROLLOUTS = 24
METRIC = "val-core/math_sis/acc/mean@1"


def load_evaluations(reuse_n: int) -> list[dict]:
    path = EXPERIMENT_DIR / "raw" / f"n{reuse_n}" / "eval_metrics.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: row["rollout_step"])

    expected_rollout_steps = list(range(EXPECTED_ROLLOUTS + 1))
    if [row["rollout_step"] for row in rows] != expected_rollout_steps:
        raise ValueError(f"N={reuse_n}: rollout checkpoints are incomplete")
    if any(row["optimizer_step"] != row["rollout_step"] * reuse_n for row in rows):
        raise ValueError(f"N={reuse_n}: optimizer-step alignment is inconsistent")
    return rows


def main() -> None:
    evaluations = {n: load_evaluations(n) for n in (4, 8)}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(11.8, 5.4), constrained_layout=False)
    styles = {
        4: {
            "color": "#0072B2",
            "marker": "o",
            "markerfacecolor": "none",
        },
        8: {
            "color": "#D55E00",
            "marker": "s",
            "markerfacecolor": "#D55E00",
        },
    }
    for reuse_n in (4, 8):
        rows = evaluations[reuse_n]
        ax.plot(
            [row["optimizer_step"] for row in rows],
            [row[METRIC] for row in rows],
            linewidth=1.6,
            markersize=5.0,
            markeredgewidth=1.0,
            label=f"N={reuse_n}",
            **styles[reuse_n],
        )

    all_accuracy = [row[METRIC] for rows in evaluations.values() for row in rows]
    max_step = max(row["optimizer_step"] for rows in evaluations.values() for row in rows)
    ax.set_xlim(-4, max_step + 4)
    ax.set_xticks(range(0, max_step + 1, 16))
    ax.set_ylim(max(0, min(all_accuracy) - 0.04), min(1, max(all_accuracy) + 0.04))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.7)
    ax.set_axisbelow(True)

    ax.set_xlabel("Completed optimizer updates (real update step)")
    ax.set_ylabel("MATH500 sampled Avg@1")
    ax.set_title(
        "MATH500 capability by optimizer update step",
        loc="left",
        pad=20,
    )
    ax.text(
        0,
        1.015,
        "At equal steps, both branches have completed the same number of optimizer updates.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#555555",
        fontsize=9.5,
    )
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.text(
        0.5,
        0.02,
        "Direct N=4/N=8 comparison is available through step 96; beyond that only N=8 was measured. "
        "Raw sampled Avg@1; no smoothing or baseline correction.",
        ha="center",
        va="bottom",
        color="#555555",
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.85, bottom=0.18)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, facecolor="white")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
