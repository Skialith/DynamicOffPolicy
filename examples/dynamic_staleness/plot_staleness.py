#!/usr/bin/env python3
"""Plot and summarize per-optimizer-update staleness JSONL files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


METRICS = [
    "update_sampled_kl",
    "update_clipfrac_total",
    "update_ess_fraction",
    "update_ratio_second_moment",
    "update_sequence_abs_log_ratio_sum_mean",
    "sampled_kl",
    "pg_clipfrac",
    "clipfrac_total",
    "ess_fraction",
    "ratio_second_moment",
    "sequence_abs_log_ratio_sum_mean",
    "rollout_sampled_kl",
    "rollout_clipfrac_total",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inputs use LABEL=PATH syntax, for example N8=outputs/.../staleness_metrics.jsonl"
    )
    parser.add_argument("inputs", nargs="+")
    parser.add_argument(
        "--eval-inputs",
        nargs="*",
        default=[],
        metavar="LABEL=PATH",
        help="Optional eval_metrics.jsonl files using the same LABEL=PATH syntax",
    )
    parser.add_argument("--output", type=Path, default=Path("staleness_dashboard.png"))
    parser.add_argument("--branch-step", type=int, default=100)
    return parser.parse_args()


def read_input(specification: str) -> pd.DataFrame:
    if "=" not in specification:
        raise ValueError(f"Expected LABEL=PATH, got: {specification}")
    label, raw_path = specification.split("=", 1)
    path = Path(raw_path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"No metric records in {path}")
    frame = frame.sort_values("optimizer_step").drop_duplicates("optimizer_step", keep="last")
    frame["run"] = label
    frame["source"] = str(path.resolve())
    return frame


def save_summary(frame: pd.DataFrame, output: Path) -> Path:
    available = [metric for metric in METRICS if metric in frame]
    grouped = frame.groupby(["run", "policy_age"], as_index=False)[available]
    median = grouped.median().rename(columns={metric: f"{metric}_median" for metric in available})
    q90 = grouped.quantile(0.9).rename(columns={metric: f"{metric}_q90" for metric in available})
    summary = median.merge(q90, on=["run", "policy_age"])
    summary_path = output.with_name(f"{output.stem}_by_age.csv")
    summary.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    args = parse_args()
    frames = [read_input(specification) for specification in args.inputs]
    frame = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined_path = args.output.with_name(f"{args.output.stem}_records.csv")
    frame.to_csv(combined_path, index=False)
    summary_path = save_summary(frame, args.output)
    eval_frame = None
    eval_records_path = None
    if args.eval_inputs:
        eval_frame = pd.concat([read_input(specification) for specification in args.eval_inputs], ignore_index=True)
        eval_records_path = args.output.with_name(f"{args.output.stem}_eval_records.csv")
        eval_frame.to_csv(eval_records_path, index=False)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/dynamic-staleness-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dynamic-staleness-cache")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required: python -m pip install matplotlib") from error

    figure, axes = plt.subplots(4, 2, figsize=(14, 15), sharex=True)
    axes = axes.ravel()
    panels = [
        ("update_sampled_kl", "Update-only sampled KL(anchor || current)", False),
        ("update_clipfrac_total", "Update-only clip-window rejection", False),
        ("sampled_kl", "PPO sampled KL(old-eval || current-train)", False),
        ("rollout_sampled_kl", "End-to-end KL(rollout || current)", False),
        ("update_ratio_second_moment", "Update-only ratio second moment", True),
    ]
    for axis, (metric, title, log_scale) in zip(axes, panels):
        if metric not in frame:
            axis.set_visible(False)
            continue
        for label, run in frame.groupby("run", sort=False):
            run = run.sort_values("optimizer_step")
            axis.plot(run["optimizer_step"], run[metric], linewidth=1.1, alpha=0.8, label=label)
        axis.axvline(args.branch_step, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if log_scale:
            axis.set_yscale("log")

    ratio_axis = axes[5]
    if {"update_ratio_p05", "update_ratio_p95"}.issubset(frame.columns):
        for label, run in frame.groupby("run", sort=False):
            run = run.sort_values("optimizer_step")
            x = run["optimizer_step"].to_numpy()
            lower = run["update_ratio_p05"].to_numpy(dtype=float)
            upper = run["update_ratio_p95"].to_numpy(dtype=float)
            line = ratio_axis.plot(x, run["update_ratio_mean"], linewidth=1.0, label=label)[0]
            ratio_axis.fill_between(x, lower, upper, color=line.get_color(), alpha=0.15)
        ratio_axis.axhline(1.0, color="black", linewidth=0.8)
        ratio_axis.axvline(args.branch_step, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ratio_axis.set_title("Update-only ratio mean and p05-p95 envelope")
        ratio_axis.grid(alpha=0.25)
    else:
        ratio_axis.set_visible(False)

    eval_axis = axes[6]
    if eval_frame is not None:
        accuracy_metrics = [
            column
            for column in eval_frame.columns
            if column.startswith("val-core/") and column.endswith("/acc/mean@1")
        ]
        if len(accuracy_metrics) != 1:
            raise ValueError(f"Expected one Avg@1 accuracy metric, found: {accuracy_metrics}")
        accuracy_metric = accuracy_metrics[0]
        for label, run in eval_frame.groupby("run", sort=False):
            run = run.sort_values("optimizer_step")
            eval_axis.plot(
                run["optimizer_step"],
                run[accuracy_metric],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                label=label,
            )
        eval_axis.axvline(args.branch_step, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        eval_axis.set_title("MATH500 Avg@1 accuracy")
        eval_axis.set_ylim(0.0, 1.0)
        eval_axis.grid(alpha=0.25)
    else:
        eval_axis.set_visible(False)
    axes[7].set_visible(False)

    for axis in axes:
        if axis.get_visible():
            axis.set_xlabel("optimizer update")
    handles_by_label = {}
    for axis in axes:
        if axis.get_visible():
            handles, labels = axis.get_legend_handles_labels()
            handles_by_label.update(zip(labels, handles, strict=True))
    labels = list(handles_by_label)
    handles = [handles_by_label[label] for label in labels]
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")
    print(f"wrote {combined_path}")
    print(f"wrote {summary_path}")
    if eval_records_path is not None:
        print(f"wrote {eval_records_path}")


if __name__ == "__main__":
    main()
