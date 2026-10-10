"""Passive, full-vocabulary KL on selected real-rollout contexts.

No generation, policy loss, or global random generator is used by selection.
Vocabulary sums are exact (up to floating point); only contexts are sampled.
"""

import random
from contextlib import contextmanager

import numpy as np
import torch


def make_update_scheduler(optimizer, warmup_steps):
    """The first update uses peak/warmup; update warmup uses peak."""
    if warmup_steps < 0:
        raise ValueError("Optimizer-update warmup requires an explicit nonnegative step count")
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda completed: min((completed + 1) / max(1, warmup_steps), 1.0)
    )


@contextmanager
def preserve_rng_state(cuda_devices=()):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(cuda_devices)):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def request_seed(seed, index, *, evaluation=False, update=0):
    """Disjoint, per-request sampling streams, independent of server scheduling."""
    offset = 1_000_000_000 + update * 100_000 if evaluation else 0
    return (int(seed) * 1_000_003 + offset + int(index)) % 2_147_483_647


def select_rollout_positions(response_mask, uids, num_prompts, positions_per_response, seed):
    """One uniformly selected valid response per selected prompt group."""
    if num_prompts < 1 or positions_per_response < 1:
        raise ValueError("KL sample counts must be positive")
    mask = response_mask.detach().cpu().bool()
    groups = {}
    for row, uid in enumerate(uids):
        if mask[row].any():
            groups.setdefault(str(uid), []).append(row)
    if not groups:
        raise ValueError("No valid response contexts for KL measurement")
    rng = np.random.default_rng(seed)
    group_rows = list(groups.values())
    count = min(num_prompts, len(group_rows))
    positions = torch.full((len(mask), positions_per_response), -1, dtype=torch.long)
    weights = torch.zeros_like(positions, dtype=torch.float64)
    for group in rng.choice(len(group_rows), count, replace=False):
        row = int(rng.choice(group_rows[group]))
        valid = torch.where(mask[row])[0].numpy()
        selected = np.sort(rng.choice(valid, min(len(valid), positions_per_response), replace=False))
        positions[row, :len(selected)] = torch.as_tensor(selected)
        weights[row, :len(selected)] = 1.0 / (count * len(selected))
    return positions, weights


def full_vocabulary_kl(log_p, log_q, chunk_positions=8):
    """Per-context KL(P||Q); CPU caches, FP64 renormalization and vocabulary sum."""
    if log_p.shape != log_q.shape or log_p.ndim != 2:
        raise ValueError("KL requires aligned [contexts, vocabulary] log probabilities")
    result = []
    for start in range(0, len(log_p), chunk_positions):
        p = torch.log_softmax(log_p[start:start + chunk_positions].double(), dim=-1)
        q = torch.log_softmax(log_q[start:start + chunk_positions].double(), dim=-1)
        kl = (p.exp() * (p - q)).sum(-1)
        if not torch.isfinite(kl).all() or (kl < -1e-10).any():
            raise FloatingPointError("Invalid full-vocabulary KL")
        result.append(kl.clamp_min(0))
    return torch.cat(result) if result else torch.empty(0, dtype=torch.float64, device=log_p.device)


def score_context_row(model, row, device):
    """Score selected response positions in one unpadded, causal forward.

Single-sequence forwards support the existing remove-padding attention patch.
The logit immediately BEFORE each response token predicts that token.
"""
    valid = row["attention_mask"].bool()
    ids = row["input_ids"][valid].unsqueeze(0).to(device)
    position_ids = row["position_ids"][valid].unsqueeze(0).to(device)
    selected = row["kl_positions"][row["kl_positions"] >= 0]
    prompt_width = row["input_ids"].numel() - row["responses"].numel()
    absolute = prompt_width + selected - 1
    # Count real tokens up to each predictive position, removing left padding.
    indices = valid.long().cumsum(0)[absolute] - 1
    if (indices < 0).any() or not valid[absolute].all():
        raise ValueError("KL position does not have a valid causal prefix")
    output = model(input_ids=ids, attention_mask=None, position_ids=position_ids, use_cache=False)
    logits = output.logits[0, indices.to(device), :]
    log_probs = torch.log_softmax(logits.float(), dim=-1).cpu()
    del output, logits
    return log_probs
