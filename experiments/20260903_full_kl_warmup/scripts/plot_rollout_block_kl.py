#!/usr/bin/env python3
"""Plot historical rollout-block trajectories and their per-cycle KL peaks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = EXPERIMENT_DIR / "raw" / "raw_kl_snapshot.json"
OUTPUT_PATH = EXPERIMENT_DIR / "figures" / "rollout_block_cumulative_kl.png"
PEAK_OUTPUT_PATH = EXPERIMENT_DIR / "figures" / "rollout_peak_cumulative_kl.png"
TOTAL_UPDATES = 96
MAX_ROLLOUTS = 24
SLOTS_PER_ROLLOUT = 8


def load_runs() -> dict[int, list[dict]]:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    runs = {run["n"]: run["rows"] for run in snapshot["runs"]}
    if set(runs) != {4, 8}:
        raise ValueError(f"Expected N=4 and N=8 runs, found {sorted(runs)}")

    for reuse_n, rows in runs.items():
        rows.sort(key=lambda row: (row["rollout_step"], row["age_after"]))
        expected_rollouts = TOTAL_UPDATES // reuse_n
        for rollout_step in range(1, expected_rollouts + 1):
            cycle = [row for row in rows if row["rollout_step"] == rollout_step]
            ages = [row["age_after"] for row in cycle]
            if ages != list(range(1, reuse_n + 1)):
                raise ValueError(
                    f"N={reuse_n}, rollout {rollout_step}: invalid ages {ages}"
                )
        if len(rows) != TOTAL_UPDATES:
            raise ValueError(f"N={reuse_n}: expected 96 updates, found {len(rows)}")
    return runs


def plot_branch(
    ax: plt.Axes,
    rows: list[dict],
    *,
    color: str,
    marker: str,
    marker_face: str,
    zorder: int,
) -> None:
    rollout_steps = sorted({row["rollout_step"] for row in rows})
    for rollout_step in rollout_steps:
        cycle = [row for row in rows if row["rollout_step"] == rollout_step]
        x = [
            (rollout_step - 1) * SLOTS_PER_ROLLOUT + row["age_after"]
            for row in cycle
        ]
        y = [row["cumulative_kl"] * 1_000 for row in cycle]
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.25,
            marker=marker,
            markersize=4.0,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            markeredgewidth=0.9,
            zorder=zorder,
        )


def rollout_peaks(rows: list[dict]) -> tuple[list[int], list[float]]:
    peaks: dict[int, float] = {}
    for row in rows:
        rollout_step = row["rollout_step"]
        value = row["cumulative_kl"] * 1_000
        peaks[rollout_step] = max(peaks.get(rollout_step, 0.0), value)
    rollout_steps = sorted(peaks)
    return rollout_steps, [peaks[step] for step in rollout_steps]


def save_peak_plot(runs: dict[int, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.4), constrained_layout=False)
    n4_x, n4_y = rollout_peaks(runs[4])
    n8_x, n8_y = rollout_peaks(runs[8])
    ax.plot(
        n4_x,
        n4_y,
        color="#0072B2",
        linewidth=1.6,
        marker="o",
        markersize=5.0,
        markerfacecolor="none",
        markeredgewidth=1.0,
        label="N=4: max over age_after 1–4",
    )
    ax.plot(
        n8_x,
        n8_y,
        color="#D55E00",
        linewidth=1.6,
        marker="s",
        markersize=5.0,
        label="N=8: max over age_after 1–8",
    )

    ax.set_xlim(0.5, MAX_ROLLOUTS + 0.5)
    ax.set_xticks(range(1, MAX_ROLLOUTS + 1))
    ax.set_ylim(0, max(n4_y + n8_y) * 1.10)
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Rollout cycle")
    ax.set_ylabel("Peak cumulative full-vocabulary KL (×10⁻³)")
    ax.set_title(
        "Historical 96-update experiment: within-rollout peak cumulative KL",
        loc="left",
        pad=20,
    )
    ax.text(
        0,
        1.015,
        "N=4 has 24 rollout peaks; N=8 has 12. Each uses max over its full post-update age range.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#555555",
        fontsize=9.5,
    )
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.text(
        0.5,
        0.02,
        "Rollout index aligns collection cycles; N=8 reaches update 96 at cycle 12 and N=4 at cycle 24.",
        ha="center",
        va="bottom",
        color="#555555",
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.85, bottom=0.18)
    fig.savefig(PEAK_OUTPUT_PATH, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    runs = load_runs()
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

    fig, ax = plt.subplots(figsize=(18, 6.2), constrained_layout=False)
    n4_color = "#0072B2"
    n8_color = "#D55E00"
    plot_branch(
        ax,
        runs[8],
        color=n8_color,
        marker="s",
        marker_face=n8_color,
        zorder=2,
    )
    plot_branch(
        ax,
        runs[4],
        color=n4_color,
        marker="o",
        marker_face="none",
        zorder=3,
    )

    for boundary in range(
        SLOTS_PER_ROLLOUT,
        MAX_ROLLOUTS * SLOTS_PER_ROLLOUT,
        SLOTS_PER_ROLLOUT,
    ):
        ax.axvline(boundary + 0.5, color="#D0D0D0", linewidth=0.65, zorder=0)

    centers = [
        (rollout_step - 1) * SLOTS_PER_ROLLOUT + (SLOTS_PER_ROLLOUT + 1) / 2
        for rollout_step in range(1, MAX_ROLLOUTS + 1)
    ]
    ax.set_xticks(centers, labels=range(1, MAX_ROLLOUTS + 1))
    ax.set_xlim(0.5, MAX_ROLLOUTS * SLOTS_PER_ROLLOUT + 0.5)
    max_kl = max(row["cumulative_kl"] for rows in runs.values() for row in rows)
    ax.set_ylim(0, max_kl * 1_000 * 1.10)
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.7)
    ax.set_axisbelow(True)

    ax.set_xlabel(
        "Rollout cycle (each block contains age_after = 1–8; N=4 occupies slots 1–4)"
    )
    ax.set_ylabel("Cumulative full-vocabulary KL to rollout anchor (×10⁻³)")
    ax.set_title(
        "Historical 96-update experiment: within-rollout cumulative KL",
        loc="left",
        pad=20,
    )
    ax.text(
        0,
        1.015,
        "N=4 has 24 rollout cycles; N=8 has 12. Both runs use a 10-update warmup and "
        "their own rollout anchors and contexts.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#555555",
        fontsize=9.5,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=n4_color,
            marker="o",
            markerfacecolor="none",
            markeredgecolor=n4_color,
            linewidth=1.25,
            label="N=4 (24 cycles; age_after 1–4)",
        ),
        Line2D(
            [0],
            [0],
            color=n8_color,
            marker="s",
            markerfacecolor=n8_color,
            markeredgecolor=n8_color,
            linewidth=1.25,
            label="N=8 (12 cycles; age_after 1–8)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, ncol=2)

    fig.text(
        0.5,
        0.02,
        "Rollout index aligns collection cycles; N=8 reaches update 96 after cycle 12, "
        "while N=4 reaches it after cycle 24.",
        ha="center",
        va="bottom",
        color="#555555",
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.87, bottom=0.17)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, facecolor="white")
    plt.close(fig)
    save_peak_plot(runs)
    print(OUTPUT_PATH)
    print(PEAK_OUTPUT_PATH)


if __name__ == "__main__":
    main()
