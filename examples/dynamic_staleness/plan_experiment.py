#!/usr/bin/env python3
"""Print exact optimizer-update/rollout-step conversions for the pilot."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-prompts", type=int, default=256)
    parser.add_argument("--responses", type=int, default=8)
    parser.add_argument("--anchor-updates", type=int, default=100)
    parser.add_argument("--target-updates", type=int, nargs="+", default=[196, 292])
    parser.add_argument("--reuse-n", type=int, nargs="+", default=[4, 8, 16])
    args = parser.parse_args()

    anchor_n = args.reuse_n[0]
    if args.anchor_updates % anchor_n:
        raise SystemExit("anchor updates must be divisible by anchor N")
    anchor_rollouts = args.anchor_updates // anchor_n

    print(
        "phase\tN\tstart_update\ttarget_update\touter_steps_added\tabsolute_outer_target"
        "\tprompts_per_rollout\tresponses_per_optimizer_update"
    )
    print(
        f"anchor\t{anchor_n}\t0\t{args.anchor_updates}\t{anchor_rollouts}\t{anchor_rollouts}"
        f"\t{args.mini_prompts * anchor_n}\t{args.mini_prompts * args.responses}"
    )
    for target in args.target_updates:
        for reuse_n in args.reuse_n:
            remaining = target - args.anchor_updates
            if remaining <= 0 or remaining % reuse_n:
                print(f"branch\t{reuse_n}\t{args.anchor_updates}\t{target}\tINVALID\tINVALID")
                continue
            added = remaining // reuse_n
            print(
                f"branch\t{reuse_n}\t{args.anchor_updates}\t{target}\t{added}\t{anchor_rollouts + added}"
                f"\t{args.mini_prompts * reuse_n}\t{args.mini_prompts * args.responses}"
            )


if __name__ == "__main__":
    main()
