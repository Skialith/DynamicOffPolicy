#!/usr/bin/env python3
"""Materialize a deterministic flat prompt schedule for matched-N pilots."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch


def build_schedule(
    input_path: Path,
    output_path: Path,
    optimizer_steps: int,
    prompts_per_update: int,
    seed: int,
    virtual_copies: int,
) -> None:
    source = pq.read_table(input_path)
    unique_rows = source.num_rows
    schedule_rows = optimizer_steps * prompts_per_update
    virtual_rows = unique_rows * virtual_copies
    if schedule_rows > virtual_rows:
        raise ValueError(
            f"schedule requires {schedule_rows} rows, but {virtual_copies} virtual copies "
            f"provide only {virtual_rows}"
        )

    # The original expanded Parquet was 100 consecutive copies of the same
    # ordered 17,917 rows.  RandomSampler shuffles those virtual row positions;
    # position % unique_rows maps the shuffled prefix back to the unique table.
    generator = torch.Generator()
    generator.manual_seed(seed)
    virtual_positions = torch.randperm(virtual_rows, generator=generator)[:schedule_rows]
    source_indices = (virtual_positions % unique_rows).numpy()
    schedule = source.take(pa.array(source_indices))

    metadata = dict(schedule.schema.metadata or {})
    metadata.update(
        {
            b"dynamic_staleness.schedule_seed": str(seed).encode(),
            b"dynamic_staleness.optimizer_steps": str(optimizer_steps).encode(),
            b"dynamic_staleness.prompts_per_update": str(prompts_per_update).encode(),
            b"dynamic_staleness.unique_rows": str(unique_rows).encode(),
            b"dynamic_staleness.virtual_copies": str(virtual_copies).encode(),
        }
    )
    schedule = schedule.replace_schema_metadata(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    pq.write_table(schedule, temporary_path, compression="zstd")
    temporary_path.replace(output_path)
    print(
        f"wrote seed={seed} schedule with {schedule_rows} rows "
        f"({optimizer_steps} updates) to {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=292)
    parser.add_argument("--prompts-per-update", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--virtual-copies", type=int, default=100)
    args = parser.parse_args()
    build_schedule(
        args.input,
        args.output,
        args.optimizer_steps,
        args.prompts_per_update,
        args.seed,
        args.virtual_copies,
    )


if __name__ == "__main__":
    main()
