#!/usr/bin/env python3
"""Convert the pinned MATH-500 JSONL into SIS-format validation Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SIS_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def convert(input_path: Path, output_path: Path) -> None:
    records = []
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            for key in ("problem", "answer", "subject", "level", "unique_id"):
                if key not in item:
                    raise ValueError(f"line {line_number} is missing {key!r}")
            records.append(
                {
                    "data_source": "math_sis",
                    "prompt": [
                        {"role": "system", "content": SIS_SYSTEM_PROMPT},
                        {"role": "user", "content": str(item["problem"])},
                    ],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": str(item["answer"]),
                    },
                    "extra_info": {
                        "index": str(item["unique_id"]),
                        "subject": str(item["subject"]),
                        "level": int(item["level"]),
                    },
                }
            )

    if len(records) != 500:
        raise ValueError(f"expected exactly 500 MATH-500 records, got {len(records)}")
    unique_ids = {record["extra_info"]["index"] for record in records}
    if len(unique_ids) != len(records):
        raise ValueError("MATH-500 unique_id values are not unique")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    pq.write_table(pa.Table.from_pylist(records), temporary_path, compression="zstd")
    temporary_path.replace(output_path)
    print(f"wrote {len(records)} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
