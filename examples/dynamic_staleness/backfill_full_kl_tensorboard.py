"""Add optimizer-update TensorBoard scalars from an existing full-KL JSONL."""

import argparse
import json
import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


FIELDS = ("adjacent_kl", "cumulative_kl", "grad_norm", "lr_used", "policy_age")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in (args.run / "kl_updates.jsonl").read_text().splitlines()]
    steps = [int(record["optimizer_step"]) for record in records]
    if steps != list(range(1, len(records) + 1)):
        raise ValueError(f"Expected consecutive optimizer steps 1..{len(records)}, got {steps}")
    if any(not math.isfinite(float(record[field])) for record in records for field in FIELDS):
        raise ValueError("Cannot write non-finite full-KL metrics to TensorBoard")

    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    marker = args.tensorboard_dir / ".full_kl_by_update_backfilled.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous != {"source": str(args.run.resolve()), "records": len(records), "fields": list(FIELDS)}:
            raise ValueError(f"Backfill marker does not match requested source: {marker}")
        print(f"Already backfilled: {marker}")
        return

    with SummaryWriter(str(args.tensorboard_dir)) as writer:
        for record in records:
            for field in FIELDS:
                writer.add_scalar(f"full_kl_by_update/{field}", record[field], int(record["optimizer_step"]))
    with marker.open("x", encoding="utf-8") as handle:
        json.dump({"source": str(args.run.resolve()), "records": len(records), "fields": list(FIELDS)}, handle)
    print(f"Backfilled {len(records)} optimizer updates into {args.tensorboard_dir}")


if __name__ == "__main__":
    main()
