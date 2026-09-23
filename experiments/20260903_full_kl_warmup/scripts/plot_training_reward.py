#!/usr/bin/env python3
"""Plot warmup-experiment rollout reward against the generating policy update."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = EXPERIMENT_DIR / "raw" / "raw_training_reward_snapshot.json"
OUTPUT_STEM = EXPERIMENT_DIR / "figures" / "08_training_reward_vs_optimizer_step"

STYLE = {
    4: {"color": "#0072B2", "marker": "o", "markerfacecolor": "white"},
    8: {"color": "#D55E00", "marker": "s", "markerfacecolor": "#D55E00"},
}


def main() -> None:
    snapshot = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    runs = {run["n"]: run["rows"] for run in snapshot["runs"]}

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

    fig, ax = plt.subplots(figsize=(6.4, 3.7))

    for n in (4, 8):
        rows = runs[n]
        style = STYLE[n]
        ax.plot(
            [row["anchor_update"] for row in rows],
            [row["reward_mean"] for row in rows],
            color=style["color"],
            linewidth=1.45,
            marker=style["marker"],
            markersize=4.2,
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=0.9,
            label=f"N={n}",
            zorder=3 if n == 4 else 2,
        )

    ax.set_xlim(-1.5, 96)
    ax.set_ylim(0.10, 0.42)
    ax.set_xticks(range(0, 97, 8))
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("Optimizer step at rollout generation")
    ax.set_ylabel("Mean training reward")
    ax.grid(axis="y", color="#DDE3EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout(pad=0.6)

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=240)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    plt.close(fig)

    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
