#!/usr/bin/env python3
"""Build the deduplicated SIS-format GRPO training Parquet.

The downloaded DAPO-Math-17K asset on ParaCloud contains 100 consecutive
copies of the same 17,917 keyed examples.  This converter verifies that index
sequence before retaining one copy, extracts each raw question from the
original Answer:-style prompt, and writes the prompt format used by SIS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SIS_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
ORIGINAL_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
ORIGINAL_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def extract_question(prompt: list[dict[str, str]], index: str) -> str:
    if len(prompt) != 1 or prompt[0].get("role") != "user":
        raise ValueError(f"unexpected source prompt structure for {index}")
    content = prompt[0].get("content", "")
    if not content.startswith(ORIGINAL_PREFIX) or not content.endswith(ORIGINAL_SUFFIX):
        raise ValueError(f"source prompt does not match the expected DAPO template for {index}")
    question = content[len(ORIGINAL_PREFIX) : -len(ORIGINAL_SUFFIX)]
    if not question.strip():
        raise ValueError(f"empty question for {index}")
    return question


def verify_repeated_index_sequence(parquet_file: pq.ParquetFile, expected_unique: int) -> int:
    """Verify that every copy has the same ordered UUID sequence."""
    total_rows = parquet_file.metadata.num_rows
    if total_rows % expected_unique:
        raise ValueError(f"{total_rows} rows is not divisible by expected_unique={expected_unique}")

    base_ids: list[str] = []
    position = 0
    for batch in parquet_file.iter_batches(batch_size=65_536, columns=["extra_info"]):
        ids = batch.column(0).field("index").to_pylist()
        for item_id in ids:
            item_id = str(item_id)
            if position < expected_unique:
                base_ids.append(item_id)
            elif item_id != base_ids[position % expected_unique]:
                raise ValueError(
                    "the source is not consecutive identical copies: "
                    f"row {position} has index {item_id!r}, expected "
                    f"{base_ids[position % expected_unique]!r}"
                )
            position += 1

    if len(base_ids) != expected_unique or len(set(base_ids)) != expected_unique:
        raise ValueError("the first source copy does not contain the expected number of unique indices")
    return total_rows // expected_unique


def read_first_copy(parquet_file: pq.ParquetFile, row_count: int) -> pa.Table:
    batches = []
    remaining = row_count
    for batch in parquet_file.iter_batches(batch_size=min(65_536, row_count)):
        if remaining <= 0:
            break
        if batch.num_rows > remaining:
            batch = batch.slice(0, remaining)
        batches.append(batch)
        remaining -= batch.num_rows
    if remaining:
        raise ValueError(f"source ended with {remaining} rows still required")
    return pa.Table.from_batches(batches)


def convert(input_path: Path, output_path: Path, expected_unique: int) -> None:
    parquet_file = pq.ParquetFile(input_path)
    copies = verify_repeated_index_sequence(parquet_file, expected_unique)
    source_rows = read_first_copy(parquet_file, expected_unique).to_pylist()

    records = []
    for source in source_rows:
        index = str(source["extra_info"]["index"])
        question = extract_question(source["prompt"], index)
        records.append(
            {
                "data_source": "math_sis",
                "prompt": [
                    {"role": "system", "content": SIS_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(source["reward_model"]["ground_truth"]),
                },
                "extra_info": {
                    "index": index,
                    "source": "DAPO-Math-17K",
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    pq.write_table(pa.Table.from_pylist(records), temporary_path, compression="zstd")
    temporary_path.replace(output_path)
    print(
        f"verified {copies} identical index-sequence copies; "
        f"wrote {len(records)} unique SIS-format records to {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-unique", type=int, default=17_917)
    args = parser.parse_args()
    convert(args.input, args.output, args.expected_unique)


if __name__ == "__main__":
    main()
