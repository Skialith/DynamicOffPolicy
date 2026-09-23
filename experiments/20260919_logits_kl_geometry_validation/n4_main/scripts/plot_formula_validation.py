#!/usr/bin/env python3
"""Plot the N=4 finite-difference KL validation as a 2x2 figure.

The script reads the archived formal-run JSONL only and writes publication-ready
PNG and PDF files under figures/.  It deliberately uses the theoretical y=x
reference rather than fitting a regression line.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = EXPERIMENT_ROOT / "raw" / "job166278" / "kl_updates.jsonl"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
OUTPUT_STEM = FIGURE_ROOT / "kl_formula_validation_2x2"

AGE_COLORS = {
    1: "#0072B2",
    2: "#E69F00",
    3: "#009E73",
    4: "#CC79A7",
}
ADJACENT_COLOR = "#0072B2"
CUMULATIVE_COLOR = "#D55E00"


def read_rows() -> list[dict]:
    with INPUT_PATH.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 96:
        raise AssertionError(f"expected 96 updates, found {len(rows)}")
    if [row["optimizer_step"] for row in rows] != list(range(1, 97)):
        raise AssertionError("optimizer steps are incomplete")
    if any(row["age_after"] not in AGE_COLORS for row in rows):
        raise AssertionError("unexpected age_after value")
    return rows


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def typical_error(predicted: np.ndarray, exact: np.ndarray) -> float:
    return math.exp(float(np.median(np.abs(np.log(predicted / exact))))) - 1.0


def parity_statistics(predicted: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    relative = predicted / exact - 1.0
    return {
        "median_ratio": float(np.median(predicted / exact)),
        "typical_error": typical_error(predicted, exact),
        "p95_absolute_error": float(np.quantile(np.abs(relative), 0.95)),
        "max_absolute_error": float(np.max(np.abs(relative))),
    }


def padded_log_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate(arrays)
    return float(values.min() / 1.12), float(values.max() * 1.12)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def plot_parity(
    ax: plt.Axes,
    rows: list[dict],
    exact_key: str,
    predicted_key: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    exact = np.asarray([row[exact_key] for row in rows], dtype=float)
    predicted = np.asarray([row[predicted_key] for row in rows], dtype=float)
    stats = parity_statistics(predicted, exact)
    limits = padded_log_limits(exact, predicted)

    ax.plot(limits, limits, color="0.2", linewidth=1.2, zorder=1, label=r"$y=x$")
    for age in AGE_COLORS:
        selected = [index for index, row in enumerate(rows) if row["age_after"] == age]
        ax.scatter(
            exact[selected],
            predicted[selected],
            s=30,
            color=AGE_COLORS[age],
            alpha=0.78,
            edgecolors="white",
            linewidths=0.45,
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.text(
        0.04,
        0.96,
        "\n".join(
            [
                r"$n=96$",
                f"median ratio = {stats['median_ratio']:.4f}",
                f"typical error = {format_percent(stats['typical_error'])}",
                f"95th |error| = {format_percent(stats['p95_absolute_error'])}",
                f"max |error| = {format_percent(stats['max_absolute_error'])}",
            ]
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        linespacing=1.25,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.92, "pad": 4},
    )


def main() -> None:
    rows = read_rows()
    adjacent_exact = np.asarray([row["adjacent_kl"] for row in rows], dtype=float)
    adjacent_approx = np.asarray(
        [row["frozen_fisher_fd_step"] for row in rows], dtype=float
    )
    cumulative_exact = np.asarray([row["cumulative_kl"] for row in rows], dtype=float)
    cumulative_approx = np.asarray(
        [row["frozen_fisher_fd_current"] for row in rows], dtype=float
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "0.25",
            "axes.labelcolor": "0.12",
            "xtick.color": "0.2",
            "ytick.color": "0.2",
            "grid.color": "0.88",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 7.5), constrained_layout=True)
    figure.suptitle(
        "Finite-difference reconstruction of full-vocabulary KL\n"
        "N=4, 96 optimizer updates",
        fontsize=13,
        fontweight="bold",
    )

    ax_a, ax_b, ax_c, ax_d = axes.flat
    plot_parity(
        ax_a,
        rows,
        "adjacent_kl",
        "frozen_fisher_fd_step",
        "Adjacent KL",
        r"Measured $d_a$",
        r"Finite-difference $Q_{\mathrm{step}}$",
    )
    add_panel_label(ax_a, "a")

    age_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5.5,
            markerfacecolor=AGE_COLORS[age],
            markeredgecolor="white",
            label=f"age {age}",
        )
        for age in AGE_COLORS
    ]
    identity_handle = Line2D([0], [0], color="0.2", linewidth=1.2, label=r"$y=x$")
    ax_a.legend(
        handles=[identity_handle, *age_handles],
        loc="lower right",
        fontsize=7.7,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.82",
        ncol=2,
        columnspacing=0.8,
        handlelength=1.5,
    )

    plot_parity(
        ax_b,
        rows,
        "cumulative_kl",
        "frozen_fisher_fd_current",
        "Cumulative KL from rollout anchor",
        r"Measured $K(a)$",
        r"Finite-difference $Q_{\mathrm{current}}$",
    )
    add_panel_label(ax_b, "b")

    adjacent_residual = 100.0 * (adjacent_approx / adjacent_exact - 1.0)
    cumulative_residual = 100.0 * (cumulative_approx / cumulative_exact - 1.0)
    adjacent_band = np.quantile(adjacent_residual, [0.025, 0.975])
    cumulative_band = np.quantile(cumulative_residual, [0.025, 0.975])
    ax_c.axhspan(
        adjacent_band[0],
        adjacent_band[1],
        color=ADJACENT_COLOR,
        alpha=0.08,
        linewidth=0,
    )
    ax_c.axhspan(
        cumulative_band[0],
        cumulative_band[1],
        color=CUMULATIVE_COLOR,
        alpha=0.08,
        linewidth=0,
    )
    ax_c.axhline(0.0, color="0.2", linewidth=1.0, zorder=1)
    ax_c.scatter(
        adjacent_exact,
        adjacent_residual,
        s=24,
        marker="o",
        color=ADJACENT_COLOR,
        alpha=0.68,
        edgecolors="white",
        linewidths=0.4,
        label="Adjacent",
        zorder=2,
    )
    ax_c.scatter(
        cumulative_exact,
        cumulative_residual,
        s=27,
        marker="^",
        color=CUMULATIVE_COLOR,
        alpha=0.68,
        edgecolors="white",
        linewidths=0.4,
        label="Cumulative",
        zorder=2,
    )
    residual_extent = max(np.max(np.abs(adjacent_residual)), np.max(np.abs(cumulative_residual)))
    ax_c.set_xscale("log")
    ax_c.set_ylim(-1.18 * residual_extent, 1.18 * residual_extent)
    ax_c.set_title("Signed relative error", loc="left", fontsize=10.5, fontweight="bold")
    ax_c.set_xlabel("Measured KL")
    ax_c.set_ylabel(r"$100\,(Q/D-1)$ (%)")
    ax_c.text(
        0.04,
        0.96,
        "Central 95% residual band\n"
        f"adjacent [{adjacent_band[0]:+.2f}, {adjacent_band[1]:+.2f}]%\n"
        f"cumulative [{cumulative_band[0]:+.2f}, {cumulative_band[1]:+.2f}]%",
        transform=ax_c.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        linespacing=1.25,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.92, "pad": 4},
    )
    ax_c.legend(
        loc="lower right",
        fontsize=8,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.82",
    )
    add_panel_label(ax_c, "c")

    direction_rows = [row for row in rows if row["age_after"] > 1]
    rho_eff = np.asarray([row["rho_eff"] for row in direction_rows], dtype=float)
    rho_fd = np.asarray(
        [row["frozen_fisher_fd_cosine"] for row in direction_rows], dtype=float
    )
    low = float(min(rho_eff.min(), rho_fd.min()))
    high = float(max(rho_eff.max(), rho_fd.max()))
    padding = 0.08 * (high - low)
    direction_limits = (low - padding, high + padding)
    ax_d.plot(direction_limits, direction_limits, color="0.2", linewidth=1.2, zorder=1)
    for age in (2, 3, 4):
        selected = [
            index for index, row in enumerate(direction_rows) if row["age_after"] == age
        ]
        ax_d.scatter(
            rho_eff[selected],
            rho_fd[selected],
            s=30,
            color=AGE_COLORS[age],
            alpha=0.78,
            edgecolors="white",
            linewidths=0.45,
            zorder=2,
        )
    difference = rho_eff - rho_fd
    ax_d.set_xlim(direction_limits)
    ax_d.set_ylim(direction_limits)
    ax_d.set_aspect("equal", adjustable="box")
    ax_d.set_title("Cross-term direction", loc="left", fontsize=10.5, fontweight="bold")
    ax_d.set_xlabel(r"KL-derived $\rho_{\mathrm{eff}}$")
    ax_d.set_ylabel(r"Finite-difference cosine $\rho_{\mathrm{FD}}$")
    ax_d.text(
        0.04,
        0.96,
        "\n".join(
            [
                r"$n=72$ (ages 2--4)",
                f"Pearson r = {pearson(rho_eff, rho_fd):.4f}",
                f"MAE = {statistics.fmean(abs(value) for value in difference):.5f}",
                f"max |difference| = {np.max(np.abs(difference)):.5f}",
            ]
        ),
        transform=ax_d.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        linespacing=1.25,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.92, "pad": 4},
    )
    add_panel_label(ax_d, "d")

    for ax in axes.flat:
        ax.grid(True, which="major")
        ax.grid(False, which="minor")
        ax.tick_params(which="both", direction="out", length=3)

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
