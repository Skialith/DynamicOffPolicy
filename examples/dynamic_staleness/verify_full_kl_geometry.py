#!/usr/bin/env python3
"""Verify the per-update outputs needed by the N=4 Fisher-alignment experiment."""

import argparse
import json
import math
from pathlib import Path


FIELDS = (
    "update_norm",
    "previous_cumulative_kl",
    "kl_three_point_cross",
    "kl_three_point_error",
    "rho_eff",
    "rho_eff_defined",
    "frozen_fisher_fd_cumulative",
    "frozen_fisher_fd_step",
    "frozen_fisher_fd_current",
    "frozen_fisher_fd_cross",
    "frozen_fisher_fd_cosine",
    "frozen_fisher_fd_cosine_defined",
    "frozen_fisher_fd_closure_error",
)

JVP_FIELDS = (
    "jvp_seconds",
    "jvp_anchor_functional_kl",
    "jvp_anchor_repeat_kl",
    "frozen_fisher_jvp_cumulative",
    "frozen_fisher_jvp_step",
    "frozen_fisher_jvp_current",
    "frozen_fisher_jvp_cross",
    "frozen_fisher_jvp_cosine",
    "frozen_fisher_jvp_cosine_defined",
    "frozen_fisher_jvp_closure_error",
    "frozen_fisher_jvp_residual_previous",
    "frozen_fisher_jvp_residual_current",
    "frozen_fisher_jvp_residual_increment",
    "frozen_fisher_jvp_reconstruction_error",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--require-jvp", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.output_dir / "kl_updates.jsonl").read_text().splitlines()]
    assert len(rows) == args.updates
    for step, row in enumerate(rows, 1):
        assert row["optimizer_step"] == step
        assert all(field in row and math.isfinite(row[field]) for field in FIELDS)
        assert row["update_norm"] > 0
        assert abs(row["kl_three_point_error"]) < 1e-9
        assert abs(row["frozen_fisher_fd_closure_error"]) < 1e-9
        assert row["frozen_fisher_fd_step"] > 0
        if row["age_after"] == 1:
            assert row["rho_eff_defined"] == 0
            assert row["frozen_fisher_fd_cosine_defined"] == 0
        else:
            assert row["rho_eff_defined"] == 1
            assert row["frozen_fisher_fd_cosine_defined"] == 1
            assert abs(row["frozen_fisher_fd_cosine"]) <= 1 + 1e-8
        if args.require_jvp:
            assert all(field in row and math.isfinite(row[field]) for field in JVP_FIELDS)
            assert row["jvp_seconds"] > 0
            assert row["jvp_anchor_functional_kl"] < 1e-6
            assert row["jvp_anchor_repeat_kl"] < 1e-6
            assert abs(row["frozen_fisher_jvp_closure_error"]) < 1e-9
            assert row["frozen_fisher_jvp_step"] > 0
            assert abs(
                row["frozen_fisher_jvp_residual_increment"]
                - row["frozen_fisher_jvp_reconstruction_error"]
            ) < 1e-10
            if row["frozen_fisher_jvp_cosine_defined"]:
                assert abs(row["frozen_fisher_jvp_cosine"]) <= 1 + 1e-8
    print(f"FULL_KL_GEOMETRY_PASSED rows={len(rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
