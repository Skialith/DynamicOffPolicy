"""Plot the two completed KL runs from the preserved, read-only JSON snapshot."""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
for directory in (TABLES, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)
SNAPSHOT = json.loads((RAW / "raw_kl_snapshot.json").read_text())
RUNS = SNAPSHOT["runs"]
COLORS = {4: "#2563A6", 8: "#DE7523"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.labelcolor": "#334155",
    "axes.edgecolor": "#94A3B8",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})

for run in RUNS:
    rows = run["rows"]
    n = run["n"]
    assert len(rows) == 96
    assert [r["optimizer_step"] for r in rows] == list(range(1, 97))
    assert len({r["context_set_id"] for r in rows}) == 96 // n
    for r in rows:
        s = r["optimizer_step"]
        assert r["reuse_n"] == n and r["update_applied"] == 1
        assert r["anchor_update"] == ((s - 1) // n) * n
        assert r["age_after"] == s - r["anchor_update"]
        assert r["policy_age"] == r["age_after"] - 1
        assert math.isclose(r["lr_used"], 1e-6 * min(s / 10, 1), rel_tol=1e-9)
        assert r["self_kl"] <= 1e-6
        for key in ("adjacent_kl", "cumulative_kl", "grad_norm"):
            assert math.isfinite(r[key]) and r[key] >= 0
        if r["age_after"] == 1:
            assert math.isclose(r["adjacent_kl"], r["cumulative_kl"], abs_tol=1e-12)

fields = [
    "job_id", "reuse_n", "seed", "optimizer_step", "rollout_step",
    "anchor_update", "policy_age", "age_after", "context_set_id",
    "adjacent_kl", "cumulative_kl", "grad_norm", "lr_used", "context_count",
    "self_kl", "direction", "aggregation", "vocabulary", "temperature",
]
with (TABLES / "kl_and_grad_updates.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for run in RUNS:
        for r in run["rows"]:
            writer.writerow({k: run["job_id"] if k == "job_id" else r[k] for k in fields})

all_rows = [r for run in RUNS for r in run["rows"]]
kl_top = max(r[k] for r in all_rows for k in ("adjacent_kl", "cumulative_kl")) * 1.10
grad_top = max(r["grad_norm"] for r in all_rows) * 1.10


def new_plot(title, ylabel, top, footer, scientific=False):
    fig, ax = plt.subplots(figsize=(11.8, 4.65))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.82, bottom=0.23)
    fig.text(0.095, 0.93, title, fontsize=16, weight="bold", color="#0F172A")
    ax.axvspan(0.5, 10.5, color="#CBD5E1", alpha=0.36, zorder=0)
    ax.text(5.5, 0.98, "Warmup", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color="#64748B")
    ax.set(xlim=(0, 97), ylim=(0, top), xlabel="Gradient step (actual optimizer update)", ylabel=ylabel)
    ax.set_xticks([0, 16, 32, 48, 64, 80, 96])
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    ax.set_axisbelow(True)
    if scientific:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    fig.text(0.095, 0.065, footer, fontsize=9, color="#64748B", linespacing=1.55)
    return fig, ax


def finish(fig, ax, name):
    ax.legend(loc="upper right", frameon=False, ncol=2)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


fig, ax = new_plot(
    "Adjacent-model KL across training", "Full-vocabulary KL (nats)", kl_top,
    "Each point compares the model before and after one optimizer update; raw values, no smoothing.\n"
    "Measurement prefixes are fixed within each rollout cycle and reselected at the next cycle.",
    scientific=True,
)
for run in RUNS:
    n, rows = run["n"], run["rows"]
    ax.plot([r["optimizer_step"] for r in rows], [r["adjacent_kl"] for r in rows],
            color=COLORS[n], marker="o", markersize=2.7, linewidth=1.25,
            label=f"N={n} | job {run['job_id']}")
finish(fig, ax, "01_adjacent_kl_vs_gradient_step")

fig, ax = new_plot(
    "KL from the current rollout-start model", "Full-vocabulary KL (nats)", kl_top,
    "Each segment compares one rollout-start model with its successive post-update models.\n"
    "Lines stop at cycle boundaries: both the anchor model and measurement prefixes change.",
    scientific=True,
)
for run in RUNS:
    n, rows = run["n"], run["rows"]
    for start in range(0, len(rows), n):
        cycle = rows[start:start + n]
        ax.plot([r["optimizer_step"] for r in cycle], [r["cumulative_kl"] for r in cycle],
                color=COLORS[n], marker="o", markersize=2.7, linewidth=1.4,
                label=f"N={n} | job {run['job_id']}" if start == 0 else None)
finish(fig, ax, "02_rollout_anchor_kl_vs_gradient_step")

for index, run in enumerate(RUNS, start=3):
    n, rows = run["n"], run["rows"]
    fig, ax = new_plot(
        f"Gradient norm across training | N={n}", "Logged gradient norm", grad_top,
        "One point per actual optimizer update; raw values, no smoothing.\n"
        "N=4 and N=8 use the same axis ranges; shading marks the first 10 warmup updates.",
    )
    ax.plot([r["optimizer_step"] for r in rows], [r["grad_norm"] for r in rows],
            color=COLORS[n], marker="o", markersize=2.7, linewidth=1.25,
            label=f"N={n} | job {run['job_id']}")
    finish(fig, ax, f"{index:02d}_grad_norm_n{n}_vs_gradient_step")

print("Validated 192 update records; wrote four PNG/PDF figures and one CSV.")
