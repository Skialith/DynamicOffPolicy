"""Fail a Slurm integration gate unless every real update has aligned K/Q/B."""

import argparse
import json
import math
from pathlib import Path


FIELDS = ("hvp_cumulative_kl", "hvp_fisher_quadratic", "hvp_spectral_bound",
          "hvp_lambda_max", "hvp_displacement_norm", "hvp_update_norm", "hvp_residual")


def validate(run_dir, updates, reuse_n, prompts, tolerance, scratch=None, compute_quadratic=True):
    if updates < 1 or updates % reuse_n:
        raise ValueError("Expected an integral number of rollout cycles")
    with (run_dir / "kl_updates.jsonl").open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if len(records) != updates:
        raise RuntimeError(f"Expected {updates} updates, found {len(records)}")
    diagnostics = run_dir / "hvp_diagnostics"
    fields = FIELDS if compute_quadratic else tuple(name for name in FIELDS if name != "hvp_fisher_quadratic")
    expected_reports = updates + updates // reuse_n
    if len(list(diagnostics.glob("start_*/age_*.json"))) != expected_reports:
        raise RuntimeError("Incomplete HVP reports")
    for index, record in enumerate(records, 1):
        anchor = ((index - 1) // reuse_n) * reuse_n
        age = index - anchor
        if (record["optimizer_step"], record["anchor_update"], record["age_after"], record["reuse_n"]) != (index, anchor, age, reuse_n):
            raise RuntimeError(f"Incorrect anchor/update coordinates at {index}")
        if not all(name in record and math.isfinite(record[name]) for name in fields):
            raise RuntimeError(f"Missing or non-finite K/Q/B at {index}")
        with (diagnostics / f"start_{anchor:04d}" / f"age_{age:02d}.json").open() as stream:
            report = json.load(stream)
        if report["parameter_count"] != 8_190_735_360 or report["parameter_dtype"] != "float32" or report["kl_dtype"] != "float64":
            raise RuntimeError("Incorrect full-parameter model or measurement precision")
        if report["vocabulary"] != "full" or report["prompts"] != prompts or report["positions"] < prompts:
            raise RuntimeError("Incorrect measurement vocabulary or context count")
        if (report["anchor_update"], report["age_after"], report["optimizer_step"]) != (anchor, age, index):
            raise RuntimeError("HVP report and training update do not match")
        if report.get("quadratic_measured", True) != compute_quadratic:
            raise RuntimeError("Incorrect quadratic measurement mode")
        if not compute_quadratic and "hvp_fisher_quadratic" in record:
            raise RuntimeError("Unexpected Q metric in K/B-only mode")
        for name in fields:
            if name != "hvp_update_norm" and not math.isclose(report["metrics"][name], record[name], rel_tol=1e-10, abs_tol=1e-12):
                raise RuntimeError(f"Training and HVP metrics disagree: {name}")
        if record["hvp_lambda_max"] <= 0 or not 0 <= record["hvp_residual"] <= tolerance:
            raise RuntimeError("Unconverged Fisher spectral estimate")
        if record["hvp_cumulative_kl"] < -1e-10 or record.get("hvp_fisher_quadratic", 0.0) < -1e-8:
            raise RuntimeError("Negative KL/Fisher quadratic beyond rounding tolerance")
        if record["hvp_displacement_norm"] <= 0 or record["hvp_update_norm"] <= 0:
            raise RuntimeError("No actual optimizer displacement")
        expected_bound = 0.5 * record["hvp_lambda_max"] * record["hvp_displacement_norm"] ** 2
        if not math.isclose(expected_bound, record["hvp_spectral_bound"], rel_tol=1e-8):
            raise RuntimeError("Incorrect spectral bound formula")
        if age == 1:
            if not math.isclose(record["hvp_update_norm"], record["hvp_displacement_norm"], rel_tol=1e-5):
                raise RuntimeError("Sharded update norm disagrees with full-state displacement")
            with (diagnostics / f"start_{anchor:04d}" / "age_00.json").open() as stream:
                initial = json.load(stream)
            if not initial["power"]["converged"] or initial["power"]["residual"] > tolerance:
                raise RuntimeError("Anchor spectrum did not converge")
            if initial["metrics"]["hvp_lambda_max"] != record["hvp_lambda_max"]:
                raise RuntimeError("Spectrum was not frozen at the rollout anchor")
    if list(run_dir.glob("global_step_*")):
        raise RuntimeError("Unexpected persistent training checkpoint")
    if scratch is not None and scratch.exists() and any(scratch.iterdir()):
        raise RuntimeError("Transient measurement snapshots were not removed")
    return {"passed": True, "updates": updates, "reuse_n": reuse_n,
            "rollout_cycles": updates // reuse_n, "hvp_reports": expected_reports,
            "quadratic_measured": compute_quadratic,
            "residual_tolerance": tolerance, "no_persistent_checkpoint": True,
            "transient_snapshots_removed": scratch is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--reuse-n", type=int, required=True)
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--skip-quadratic", action="store_true")
    args = parser.parse_args()
    result = validate(args.run_dir, args.updates, args.reuse_n, args.prompts, args.tolerance, args.scratch,
                      compute_quadratic=not args.skip_quadratic)
    with (args.run_dir / "hvp_validation.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"hvp_validation": result}), flush=True)
