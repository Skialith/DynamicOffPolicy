#!/usr/bin/env python3
"""Preflight the Python environment and optional experiment assets."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from packaging.version import Version


REQUIRED_PACKAGES = {
    "torch": (Version("2.7"), None),
    "vllm": (Version("0.8.5"), Version("0.12.0")),
    "ray": (Version("2.41"), None),
    "tensordict": (Version("0.8"), Version("0.10.0")),
    "transformers": (Version("4.51"), None),
    "flash-attn": (Version("2.7"), None),
    "codetiming": (Version("1.4"), None),
    "hydra-core": (Version("1.3"), None),
    "pyarrow": (Version("19"), None),
    "math-verify": (Version("0.9"), None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--require-assets", action="store_true")
    parser.add_argument("--require-gpus", type=int, default=0)
    return parser.parse_args()


def check_versions() -> list[str]:
    failures = []
    for package, (minimum, maximum) in REQUIRED_PACKAGES.items():
        try:
            installed = Version(importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"missing package: {package}")
            continue
        print(f"{package:14s} {installed}")
        if installed < minimum or (maximum is not None and installed > maximum):
            failures.append(f"{package}=={installed} is outside [{minimum}, {maximum or 'unbounded'}]")
        if package == "tensordict" and installed.release[:2] == (0, 9):
            failures.append("tensordict 0.9.x is excluded by verl v0.7.1")
    return failures


def check_assets(model: Path | None, train_file: Path | None, required: bool) -> list[str]:
    failures = []
    if model is not None:
        config_path = model / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            print(f"model          {model} ({config.get('model_type', 'unknown')})")
        elif required:
            failures.append(f"missing model config: {config_path}")
        else:
            print(f"model          not present yet: {model}")

    if train_file is not None:
        if train_file.exists():
            import pyarrow.parquet as pq

            schema_names = set(pq.read_schema(train_file).names)
            expected = {"prompt", "data_source", "reward_model"}
            missing = expected - schema_names
            print(f"train parquet  {train_file}")
            if missing:
                failures.append(f"training parquet lacks columns: {sorted(missing)}")
        elif required:
            failures.append(f"missing training parquet: {train_file}")
        else:
            print(f"train parquet  not present yet: {train_file}")
    return failures


def main() -> int:
    args = parse_args()
    failures = check_versions()

    import torch
    import verl

    resolved_repo = args.repo_root.resolve()
    resolved_verl = Path(verl.__file__).resolve()
    print(f"verl source     {resolved_verl}")
    if resolved_repo not in resolved_verl.parents:
        failures.append(f"verl resolves outside experiment clone: {resolved_verl}")

    gpu_count = torch.cuda.device_count()
    print(f"CUDA devices    {gpu_count}")
    if gpu_count < args.require_gpus:
        failures.append(f"need {args.require_gpus} visible CUDA devices, found {gpu_count}")
    for index in range(gpu_count):
        print(f"cuda:{index}         {torch.cuda.get_device_name(index)}")

    failures.extend(check_assets(args.model, args.train_file, args.require_assets))
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print("preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
