#!/usr/bin/env python3
"""Plot N=4 and N=8 cumulative KL on aligned optimizer steps."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT = EXPERIMENT_DIR / "raw" / "raw_kl_snapshot.json"
FIGURES = EXPERIMENT_DIR / "figures"
COLORS = {4: "#0072B2", 8: "#D55E00"}
MARKERS = {4: "o", 8: "s"}


def main() -> None:
    runs = {
        run["n"]: run["rows"]
        for run in json.loads(SNAPSHOT.read_text(encoding="utf-8"))["runs"]
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#8A94A3",
            "axes.labelcolor": "#263241",
            "xtick.color": "#455468",
            "ytick.color": "#455468",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.axvspan(0.5, 10.5, color="#D9DEE7", alpha=0.45, zorder=0)
    for boundary in range(8, 96, 8):
        ax.axvline(boundary + 0.5, color="#AAB3BF", linewidth=0.85, zorder=1)
    for midpoint in range(4, 96, 8):
        ax.axvline(
            midpoint + 0.5,
            color="#C7CED8",
            linewidth=0.7,
            linestyle=(0, (3, 3)),
            zorder=1,
        )

    for n in (4, 8):
        cycles: dict[int, list[dict]] = {}
        for row in runs[n]:
            cycles.setdefault(row["anchor_update"], []).append(row)

        for cycle_index, (anchor, rows) in enumerate(sorted(cycles.items())):
            rows.sort(key=lambda row: row["optimizer_step"])
            assert [row["optimizer_step"] for row in rows] == list(
                range(anchor + 1, anchor + n + 1)
            )
            ax.plot(
                [row["optimizer_step"] for row in rows],
                [row["cumulative_kl"] for row in rows],
                color=COLORS[n],
                linewidth=1.2,
                marker=MARKERS[n],
                markersize=3.1,
                markerfacecolor="white" if n == 4 else COLORS[n],
                markeredgewidth=0.9,
                label=f"N={n}" if cycle_index == 0 else None,
                zorder=3 if n == 4 else 2,
            )

    ax.set_xlim(0, 97)
    ax.set_xticks(range(0, 97, 8))
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cumulative KL to rollout anchor (nats)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.grid(axis="y", color="#DDE3EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    output_stem = FIGURES / "07_cumulative_kl_vs_optimizer_step_aligned"
    fig.savefig(output_stem.with_suffix(".png"), dpi=240)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
    print(output_stem.with_suffix(".png"))
    print(output_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
