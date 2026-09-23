"""CPU-only artifact validation; does not load a model or initialize CUDA."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--updates", type=int, default=96)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--eval-steps", default="0,48,96")
    parser.add_argument("--eval-count", type=int, default=500)
    parser.add_argument("--no-endpoint", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / "kl_updates.jsonl").read_text().splitlines()]
    assert len(rows) == args.updates
    for step, row in enumerate(rows, 1):
        assert row["optimizer_step"] == step and row["update_applied"] == 1
        assert row["reuse_n"] == args.n and row["policy_age"] == (step - 1) % args.n
        assert row["anchor_update"] == (step - 1) // args.n * args.n
        assert math.isclose(row["lr_used"], 1e-6 * min(step / 10, 1), rel_tol=1e-9)
        assert row["context_count"] > 0 and math.isclose(row["weight_sum"], 1, abs_tol=1e-8)
        assert row["self_kl"] <= 1e-6
        assert all(math.isfinite(row[k]) and row[k] >= 0 for k in ("adjacent_kl", "cumulative_kl", "grad_norm"))
        if row["policy_age"] == 0:
            assert math.isclose(row["adjacent_kl"], row["cumulative_kl"], abs_tol=1e-12)
    assert len(list((args.run / "kl_contexts").glob("start_*.pt"))) == args.updates // args.n
    if args.eval_steps:
        expected = [int(x) for x in args.eval_steps.split(",")]
        evaluations = [json.loads(line) for line in (args.run / "eval_metrics.jsonl").read_text().splitlines()]
        assert [row["optimizer_step"] for row in evaluations] == expected
        for row in evaluations:
            assert row["n"] == 1 and row["do_sample"] and row["temperature"] == 1.0 and row["top_p"] == 0.95
        for step in expected:
            answers = [json.loads(line) for line in (
                args.run / "eval_generations" / f"optimizer_step_{step:04d}.jsonl"
            ).read_text().splitlines()]
            assert len(answers) == args.eval_count and all("sampling_seed" in item for item in answers)
    if not args.no_endpoint:
        manifest = json.loads((args.run / "run_manifest.json").read_text())
        endpoint = Path(manifest["endpoint_model"])
        assert (endpoint / "config.json").is_file()
        assert list(endpoint.glob("*.safetensors"))
        assert not list(endpoint.parent.glob("optim*"))
    print(f"PASS: N={args.n}, {args.updates} updates, full-vocab KL/LR/eval/artifacts verified: {args.run}")


if __name__ == "__main__":
    main()
