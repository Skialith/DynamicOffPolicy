#!/usr/bin/env python3
"""Plot rollout-block trajectories and their per-cycle cumulative-KL peaks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_STEM = EXPERIMENT_DIR / "figures" / "rollout_block_cumulative_kl"
PEAK_OUTPUT_STEM = EXPERIMENT_DIR / "figures" / "rollout_peak_cumulative_kl"
EXPECTED_ROLLOUTS = 24
COLORS = {4: "#0072B2", 8: "#D55E00"}
MARKERS = {4: "o", 8: "s"}


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
    return rows


def plot_branch(
    ax: plt.Axes,
    rows: list[dict],
    *,
    reuse_n: int,
    zorder: int,
) -> None:
    by_rollout: dict[int, list[dict]] = {}
    for row in rows:
        by_rollout.setdefault(row["rollout_step"], []).append(row)

    for rollout_step, cycle in sorted(by_rollout.items()):
        # Normalize each rollout to one x-axis block so N=4 and N=8 cycles align.
        x = [
            (rollout_step - 1) + row["age_after"] / reuse_n
            for row in cycle
        ]
        y = [row["cumulative_kl"] * 1_000 for row in cycle]
        ax.plot(
            x,
            y,
            color=COLORS[reuse_n],
            linewidth=1.1,
            marker=MARKERS[reuse_n],
            markersize=3.0,
            markerfacecolor="white" if reuse_n == 4 else COLORS[reuse_n],
            markeredgecolor=COLORS[reuse_n],
            markeredgewidth=0.8,
            label=f"N={reuse_n}" if rollout_step == 1 else None,
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


def save_peak_plot(n4_rows: list[dict], n8_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    n4_x, n4_y = rollout_peaks(n4_rows)
    n8_x, n8_y = rollout_peaks(n8_rows)
    ax.plot(
        n4_x,
        n4_y,
        color=COLORS[4],
        linewidth=1.45,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label="N=4",
    )
    ax.plot(
        n8_x,
        n8_y,
        color=COLORS[8],
        linewidth=1.45,
        marker="s",
        markersize=4.2,
        label="N=8",
    )

    ax.set_xlim(0.5, EXPECTED_ROLLOUTS + 0.5)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_ylim(0, max(n4_y + n8_y) * 1.07)
    ax.grid(axis="y", color="#DDE3EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Rollout cycle")
    ax.set_ylabel("Peak cumulative KL (×10⁻³)")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.tight_layout(pad=0.6)
    fig.savefig(PEAK_OUTPUT_STEM.with_suffix(".png"), dpi=240)
    fig.savefig(PEAK_OUTPUT_STEM.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    n4_rows = load_kl_updates(4)
    n8_rows = load_kl_updates(8)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#8A94A3",
            "axes.labelcolor": "#263241",
            "xtick.color": "#455468",
            "ytick.color": "#455468",
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 3.7))

    plot_branch(
        ax,
        n8_rows,
        reuse_n=8,
        zorder=2,
    )
    plot_branch(
        ax,
        n4_rows,
        reuse_n=4,
        zorder=3,
    )

    for boundary in range(1, EXPECTED_ROLLOUTS):
        ax.axvline(boundary, color="#E3E7ED", linewidth=0.55, zorder=0)

    tick_cycles = [1, 4, 8, 12, 16, 20, 24]
    ax.set_xticks([cycle - 0.5 for cycle in tick_cycles], labels=tick_cycles)
    ax.set_xlim(0, EXPECTED_ROLLOUTS)

    max_kl = max(row["cumulative_kl"] for row in n4_rows + n8_rows) * 1_000
    ax.set_ylim(0, max_kl * 1.07)
    ax.grid(axis="y", color="#DDE3EB", linewidth=0.7)
    ax.set_axisbelow(True)

    ax.set_xlabel("Rollout cycle")
    ax.set_ylabel("Cumulative KL to rollout anchor (×10⁻³)")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("N=4"), labels.index("N=8")]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="upper right",
        frameon=False,
        ncol=2,
    )
    fig.tight_layout(pad=0.6)

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=240)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    plt.close(fig)
    save_peak_plot(n4_rows, n8_rows)
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))
    print(PEAK_OUTPUT_STEM.with_suffix(".png"))
    print(PEAK_OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
