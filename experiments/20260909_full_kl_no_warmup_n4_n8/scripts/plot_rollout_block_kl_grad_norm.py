#!/usr/bin/env python3
"""Plot cumulative KL and gradient norm on aligned rollout-block axes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATHS = {
    reuse_n: EXPERIMENT_DIR
    / "figures"
    / f"rollout_block_cumulative_kl_grad_norm_n{reuse_n}.png"
    for reuse_n in (4, 8)
}
EXPECTED_ROLLOUTS = 24


def load_kl_updates(reuse_n: int) -> list[dict]:
    path = EXPERIMENT_DIR / "raw" / f"n{reuse_n}" / "kl_updates.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    rows.sort(key=lambda row: (row["rollout_step"], row["age_after"]))
    expected_ages = list(range(1, reuse_n + 1))
    for rollout_step in range(1, EXPECTED_ROLLOUTS + 1):
        cycle = [row for row in rows if row["rollout_step"] == rollout_step]
        ages = [row["age_after"] for row in cycle]
        if ages != expected_ages:
            raise ValueError(
                f"N={reuse_n}, rollout {rollout_step}: expected ages "
                f"{expected_ages}, found {ages}"
            )
        if any(row["reuse_n"] != reuse_n for row in cycle):
            raise ValueError(f"N={reuse_n}, rollout {rollout_step}: reuse_n mismatch")
        if any("cumulative_kl" not in row or "grad_norm" not in row for row in cycle):
            raise ValueError(
                f"N={reuse_n}, rollout {rollout_step}: required metric missing"
            )
    return rows


def plot_branch(
    ax: plt.Axes,
    rows: list[dict],
    *,
    reuse_n: int,
    value_key: str,
    scale: float,
    color: str,
    marker: str,
    marker_face: str,
    zorder: int,
) -> None:
    by_rollout: dict[int, list[dict]] = {}
    for row in rows:
        by_rollout.setdefault(row["rollout_step"], []).append(row)

    for rollout_step, cycle in by_rollout.items():
        x = [
            (rollout_step - 1) * reuse_n + row["age_after"]
            for row in cycle
        ]
        y = [row[value_key] * scale for row in cycle]
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.2,
            marker=marker,
            markersize=3.8,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            markeredgewidth=0.85,
            zorder=zorder,
        )


def add_rollout_boundaries(ax: plt.Axes, reuse_n: int) -> None:
    for boundary in range(
        reuse_n,
        EXPECTED_ROLLOUTS * reuse_n,
        reuse_n,
    ):
        ax.axvline(boundary + 0.5, color="#D0D0D0", linewidth=0.65, zorder=0)


def save_branch_plot(
    rows: list[dict],
    *,
    reuse_n: int,
    color: str,
    marker: str,
    marker_face: str,
    kl_ylim: tuple[float, float],
    grad_ylim: tuple[float, float],
) -> None:
    fig, (kl_ax, grad_ax) = plt.subplots(
        2,
        1,
        figsize=(18, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12},
        constrained_layout=False,
    )

    for ax, value_key, scale in (
        (kl_ax, "cumulative_kl", 1_000.0),
        (grad_ax, "grad_norm", 1.0),
    ):
        plot_branch(
            ax,
            rows,
            reuse_n=reuse_n,
            value_key=value_key,
            scale=scale,
            color=color,
            marker=marker,
            marker_face=marker_face,
            zorder=2,
        )
        add_rollout_boundaries(ax, reuse_n)
        ax.grid(axis="y", color="#E4E4E4", linewidth=0.7)
        ax.set_axisbelow(True)

    x_max = EXPECTED_ROLLOUTS * reuse_n
    kl_ax.set_xlim(0.5, x_max + 0.5)
    kl_ax.set_ylim(*kl_ylim)
    grad_ax.set_ylim(*grad_ylim)

    centers = [
        (rollout_step - 1) * reuse_n + (reuse_n + 1) / 2
        for rollout_step in range(1, EXPECTED_ROLLOUTS + 1)
    ]
    grad_ax.set_xticks(centers, labels=range(1, EXPECTED_ROLLOUTS + 1))

    kl_ax.set_ylabel("Cumulative full-vocabulary KL\nto rollout anchor (×10⁻³)")
    grad_ax.set_ylabel("Gradient norm at update")
    grad_ax.set_xlabel(
        f"Rollout cycle (each block contains age_after = 1–{reuse_n})"
    )
    kl_ax.set_title("Cumulative KL after the aligned update", loc="left", pad=8)
    grad_ax.set_title("Gradient norm driving the aligned update", loc="left", pad=8)

    fig.suptitle(
        f"N={reuse_n}: cumulative KL and gradient norm across 24 rollout cycles",
        x=0.065,
        y=0.975,
        ha="left",
        fontsize=15,
    )
    fig.text(
        0.065,
        0.935,
        "At each slot, the lower panel shows the gradient norm for that update; "
        "the upper panel shows cumulative KL measured immediately after it. "
        "Y-axis limits match the other branch figure.",
        ha="left",
        va="top",
        color="#555555",
        fontsize=9.5,
    )
    fig.text(
        0.5,
        0.02,
        "Lines connect updates only within a rollout cycle; each new cycle resets the KL anchor.",
        ha="center",
        va="bottom",
        color="#555555",
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.89, bottom=0.12)

    output_path = OUTPUT_PATHS[reuse_n]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)
    print(output_path)


def main() -> None:
    rows_by_n = {reuse_n: load_kl_updates(reuse_n) for reuse_n in (4, 8)}
    all_rows = rows_by_n[4] + rows_by_n[8]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    kl_ylim = (
        0.0,
        max(row["cumulative_kl"] for row in all_rows) * 1_000 * 1.10,
    )
    grad_ylim = (0.0, max(row["grad_norm"] for row in all_rows) * 1.10)

    save_branch_plot(
        rows_by_n[4],
        reuse_n=4,
        color="#0072B2",
        marker="o",
        marker_face="none",
        kl_ylim=kl_ylim,
        grad_ylim=grad_ylim,
    )
    save_branch_plot(
        rows_by_n[8],
        reuse_n=8,
        color="#D55E00",
        marker="s",
        marker_face="#D55E00",
        kl_ylim=kl_ylim,
        grad_ylim=grad_ylim,
    )


if __name__ == "__main__":
    main()
