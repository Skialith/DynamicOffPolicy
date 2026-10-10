"""Check forward-only cross-rollout artifacts without deriving research conclusions."""
import argparse
import json
import math
from pathlib import Path

import yaml


def verify(run_dir, updates, reuse_n, horizon):
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    actor = config["actor_rollout_ref"]["actor"]
    assert actor["full_kl_delayed_measurement"] and actor["full_kl_actor_measurement"]
    assert not any(key.startswith(("full_kl_jvp", "full_kl_hvp")) for key in actor)
    assert actor["full_kl_delayed_horizon"] == horizon
    assert actor["full_kl_delayed_anchor_every_rollouts"] == 1
    ordinary = [json.loads(line) for line in (run_dir / "kl_updates.jsonl").read_text().splitlines()]
    delayed = [json.loads(line) for line in (run_dir / "delayed_kl_updates.jsonl").read_text().splitlines()]
    assert [row["optimizer_step"] for row in ordinary] == list(range(1, updates + 1))
    pairs = [(row["anchor_update"], row["optimizer_step"]) for row in delayed]
    expected = {(anchor, step) for anchor in range(0, updates, reuse_n)
                for step in range(anchor + 1, min(updates, anchor + horizon) + 1)}
    assert len(pairs) == len(set(pairs)) and set(pairs) == expected
    for row in delayed:
        step, anchor = row["optimizer_step"], row["anchor_update"]
        behavior = ((step - 1) // reuse_n) * reuse_n
        assert row["anchor_age"] == step - anchor
        assert row["behavior_anchor_update"] == behavior and row["policy_age"] == step - behavior - 1
        assert row["context_set_id"] == f"start_{anchor:04d}" and row["training_path"] == f"fixed_n{reuse_n}"
        assert row["vocabulary"] == "full" and row["model_precision"] == "bf16_actor"
        assert math.isclose(row["weight_sum"], 1., abs_tol=1e-8) and row["cumulative_kl"] >= 0
        assert row["anchor_cache_bytes"] > 0
        assert row["extra_forward"] == int(anchor != behavior)
        assert row["extra_forward_and_kl_seconds"] >= 0
    assert not (run_dir / "hvp_diagnostics").exists()
    assert not any(key.startswith(("hvp_", "jvp_", "frozen_fisher_")) for row in ordinary for key in row)
    return {"updates": len(ordinary), "delayed_records": len(delayed), "forward_only": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--reuse-n", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    result = verify(args.run_dir, args.updates, args.reuse_n, args.horizon)
    with (args.run_dir / "delayed_kl_validation.json").open("x") as handle:
        json.dump(result | {"passed": True}, handle, indent=2)
    print("DELAYED_KL_TRAINING_PROBE_PASSED", json.dumps(result))
