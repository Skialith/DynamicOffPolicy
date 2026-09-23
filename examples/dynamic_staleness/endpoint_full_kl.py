"""Compare two completed runs on a shared mixture of saved rollout contexts.

Run on an allocated GPU, never the login node. Models are loaded sequentially.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.trainer.ppo.full_vocab_kl import full_vocabulary_kl, score_context_row


def sample_contexts(run_dir, count, rng):
    files = sorted((run_dir / "kl_contexts").glob("start_*.pt"))
    candidates = []
    for path in files:
        data = torch.load(path, map_location="cpu", weights_only=True)
        candidates.extend((path, i) for i in range(len(data["input_ids"])))
    if len(candidates) < count:
        raise ValueError(f"Need {count} saved prompt contexts in {run_dir}, found {len(candidates)}")
    rows = []
    for item in rng.choice(len(candidates), count, replace=False):
        path, index = candidates[item]
        data = torch.load(path, map_location="cpu", weights_only=True)
        row = {key: data[key][index] for key in (
            "input_ids", "attention_mask", "position_ids", "responses", "kl_positions"
        )}
        row["source"] = f"{path}:{index}"
        rows.append(row)
    return rows


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-n4", type=Path, required=True)
    parser.add_argument("--run-n8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts-per-branch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Endpoint scoring requires a Slurm GPU allocation")
    manifests = [json.loads((run / "run_manifest.json").read_text()) for run in (args.run_n4, args.run_n8)]
    for expected_n, run, manifest in zip((4, 8), (args.run_n4, args.run_n8), manifests):
        if manifest["reuse_n"] != expected_n:
            raise ValueError("Endpoint branch N mismatch")
        records = [json.loads(line) for line in (run / "kl_updates.jsonl").read_text().splitlines()]
        if len(records) != manifest["target_updates"] or records[-1]["optimizer_step"] != manifest["target_updates"]:
            raise ValueError(f"Training is incomplete: {run}")
    if manifests[0]["target_updates"] != manifests[1]["target_updates"]:
        raise ValueError("Compare endpoints at the same optimizer-update count")
    tokenizers = [AutoTokenizer.from_pretrained(m["endpoint_model"], local_files_only=True) for m in manifests]
    if tokenizers[0].get_vocab() != tokenizers[1].get_vocab():
        raise ValueError("Endpoint vocabularies do not align")
    rng = np.random.default_rng(args.seed)
    contexts = sample_contexts(args.run_n4, args.prompts_per_branch, rng)
    contexts += sample_contexts(args.run_n8, args.prompts_per_branch, rng)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".contexts.pt").open("xb") as handle:
        torch.save(contexts, handle)
    logps = []
    for manifest in manifests:
        model = AutoModelForCausalLM.from_pretrained(
            manifest["endpoint_model"], torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", local_files_only=True,
        ).to("cuda:0").eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logps.append(torch.cat([score_context_row(model, row, "cuda:0") for row in contexts]))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    weights = torch.cat([
        torch.full((int((row["kl_positions"] >= 0).sum()),),
                   1 / (len(contexts) * int((row["kl_positions"] >= 0).sum())), dtype=torch.float64)
        for row in contexts
    ])
    forward = full_vocabulary_kl(logps[0], logps[1])
    reverse = full_vocabulary_kl(logps[1], logps[0])
    result = {
        "n4_to_n8": float((forward * weights).sum()), "n8_to_n4": float((reverse * weights).sum()),
        "context_count": len(weights), "prompt_count": len(contexts), "seed": args.seed,
        "optimizer_step": manifests[0]["target_updates"],
        "aggregation": "equal_branch_equal_prompt_equal_position", "temperature": 1.0,
        "vocabulary": "full", "models": [m["endpoint_model"] for m in manifests],
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
