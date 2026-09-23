"""Align each rollout cycle at age 1, retaining all measured cumulative KL values."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
RUNS = json.loads((RAW / "raw_kl_snapshot.json").read_text())["runs"]
PROGRESS = Normalize(vmin=0, vmax=96)
COLORS = LinearSegmentedColormap.from_list(
    "training_progress", ["#86BFE0", "#3E8CBD", "#16568A", "#082F5B"]
)
KL_TOP = max(r["cumulative_kl"] for run in RUNS for r in run["rows"]) * 1.10

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": "#334155",
    "axes.edgecolor": "#94A3B8",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})

for index, run in enumerate(RUNS, start=5):
    n = run["n"]
    cycles = {}
    for row in run["rows"]:
        cycles.setdefault(row["anchor_update"], []).append(row)
    assert len(cycles) == 96 // n
    assert sorted(cycles) == list(range(0, 96, n))

    fig, ax = plt.subplots(figsize=(10.8, 6.1))
    fig.subplots_adjust(left=0.105, right=0.79, top=0.78, bottom=0.24)
    fig.text(0.105, 0.925, f"Cumulative KL by rollout age | N={n}",
             fontsize=17, weight="bold", color="#0F172A")
    fig.text(0.105, 0.865,
             f"{len(cycles)} rollout cycles  |  96 optimizer updates  |  job {run['job_id']}",
             fontsize=11, color="#475569")

    for anchor, rows in sorted(cycles.items()):
        rows.sort(key=lambda r: r["age_after"])
        assert [r["age_after"] for r in rows] == list(range(1, n + 1))
        assert len({r["context_set_id"] for r in rows}) == 1
        assert all(r["optimizer_step"] == anchor + r["age_after"] for r in rows)
        assert all(math.isfinite(r["cumulative_kl"]) and r["cumulative_kl"] >= 0 for r in rows)
        ax.plot([r["age_after"] for r in rows], [r["cumulative_kl"] for r in rows],
                color=COLORS(PROGRESS(anchor)), marker="o", markersize=3.4,
                linewidth=1.5, alpha=0.92)

    ax.set(xlim=(0.85, n + 0.15), ylim=(0, KL_TOP),
           xlabel="Updates since rollout (post-update age)",
           ylabel="KL from rollout-start model (nats)")
    ax.set_xticks(range(1, n + 1))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    ax.set_axisbelow(True)

    color_axis = fig.add_axes([0.835, 0.24, 0.022, 0.54])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=PROGRESS, cmap=COLORS), cax=color_axis)
    colorbar.set_ticks([0, 16, 32, 48, 64, 80, 96])
    colorbar.set_label("Rollout-start gradient step", labelpad=13)
    colorbar.outline.set_visible(False)
    fig.text(0.105, 0.105,
             "One line per rollout cycle; light = earlier, dark = later. Raw values, no smoothing.\n"
             "Prefixes are fixed within a line and change between lines; both plots share KL and color scales.\n"
             "Post-update age = gradient step - rollout-start step. Warmup occurs at gradient steps 1-10.",
             fontsize=9, color="#64748B", linespacing=1.6, va="center")

    stem = f"{index:02d}_cumulative_kl_by_age_n{n}"
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=180)
    plt.close(fig)
    print(f"N={n}: validated and plotted {len(cycles)} cycles, {sum(map(len, cycles.values()))} points.")
