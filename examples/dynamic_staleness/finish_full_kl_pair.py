"""The last finishing training job computes endpoint KL in its own allocation."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, choices=(4, 8), required=True)
    args = parser.parse_args()
    args.pair_dir.mkdir(parents=True, exist_ok=True)
    ready = args.pair_dir / f"ready_n{args.n}.json"
    temporary = ready.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump({"run": str(args.run.resolve()), "job_id": os.environ.get("SLURM_JOB_ID")}, handle)
    os.replace(temporary, ready)
    if not all((args.pair_dir / f"ready_n{n}.json").is_file() for n in (4, 8)):
        print("Training artifacts verified; the other job will compute endpoint KL after it finishes.")
        return
    try:
        (args.pair_dir / "endpoint.lock").mkdir()
    except FileExistsError:
        print("Endpoint KL has already been claimed by the other job.")
        return
    runs = [json.loads((args.pair_dir / f"ready_n{n}.json").read_text())["run"] for n in (4, 8)]
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("endpoint_full_kl.py")),
        "--run-n4", runs[0], "--run-n8", runs[1], "--output", str(args.pair_dir / "endpoint_kl.json"),
    ], check=True)


if __name__ == "__main__":
    main()
