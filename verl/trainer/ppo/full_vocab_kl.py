"""Passive, full-vocabulary KL on selected real-rollout contexts.

No generation, policy loss, or global random generator is used by selection.
Vocabulary sums are exact (up to floating point); only contexts are sampled.
"""

import math
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


def full_vocabulary_frozen_geometry(log_anchor, log_previous, log_current, chunk_positions=8):
    """Per-context KL cross term and frozen-anchor finite-difference Fisher geometry.

    The KL cross term is exact.  The Fisher quantities use centered finite log-probability
    differences under the anchor distribution; they approach the parameter-space Fisher
    quadratic forms when all policies are local to the anchor.
    """
    if log_anchor.shape != log_previous.shape or log_anchor.shape != log_current.shape:
        raise ValueError("Frozen geometry requires aligned policy scores")
    if log_anchor.ndim != 2:
        raise ValueError("Frozen geometry requires [contexts, vocabulary] policy scores")
    outputs = []
    for start in range(0, len(log_anchor), chunk_positions):
        anchor = torch.log_softmax(log_anchor[start:start + chunk_positions].double(), dim=-1)
        previous = torch.log_softmax(log_previous[start:start + chunk_positions].double(), dim=-1)
        current = torch.log_softmax(log_current[start:start + chunk_positions].double(), dim=-1)
        p_anchor, p_previous = anchor.exp(), previous.exp()

        cumulative = previous - anchor
        step = current - previous
        cumulative -= (p_anchor * cumulative).sum(-1, keepdim=True)
        step -= (p_anchor * step).sum(-1, keepdim=True)
        total = cumulative + step
        outputs.append(torch.stack([
            ((p_anchor - p_previous) * (previous - current)).sum(-1),
            0.5 * (p_anchor * cumulative.square()).sum(-1),
            0.5 * (p_anchor * step.square()).sum(-1),
            0.5 * (p_anchor * total.square()).sum(-1),
            (p_anchor * cumulative * step).sum(-1),
        ], dim=-1))
    if not outputs:
        return torch.empty((0, 5), dtype=torch.float64, device=log_anchor.device)
    result = torch.cat(outputs)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Invalid frozen-anchor geometry")
    return result


def full_vocabulary_jvp_frozen_geometry(log_anchor, cumulative_logits_jvp, step_logits_jvp, chunk_positions=8):
    """Frozen-anchor Fisher geometry from exact parameter-to-logit JVPs.

    A logit directional derivative becomes a score directional derivative after
    centering under the anchor policy.  The returned columns are cumulative,
    step, current quadratic energy, and the cumulative/step cross term.
    """
    if log_anchor.shape != cumulative_logits_jvp.shape or log_anchor.shape != step_logits_jvp.shape:
        raise ValueError("JVP geometry requires aligned [contexts, vocabulary] tensors")
    if log_anchor.ndim != 2:
        raise ValueError("JVP geometry requires [contexts, vocabulary] tensors")
    outputs = []
    for start in range(0, len(log_anchor), chunk_positions):
        anchor = torch.log_softmax(log_anchor[start:start + chunk_positions].double(), dim=-1)
        p_anchor = anchor.exp()
        cumulative = cumulative_logits_jvp[start:start + chunk_positions].double().clone()
        step = step_logits_jvp[start:start + chunk_positions].double().clone()
        cumulative -= (p_anchor * cumulative).sum(-1, keepdim=True)
        step -= (p_anchor * step).sum(-1, keepdim=True)
        total = cumulative + step
        outputs.append(torch.stack([
            0.5 * (p_anchor * cumulative.square()).sum(-1),
            0.5 * (p_anchor * step.square()).sum(-1),
            0.5 * (p_anchor * total.square()).sum(-1),
            (p_anchor * cumulative * step).sum(-1),
        ], dim=-1))
    if not outputs:
        return torch.empty((0, 4), dtype=torch.float64, device=log_anchor.device)
    result = torch.cat(outputs)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Invalid frozen-anchor JVP geometry")
    return result


def adamw_update_squared_norm(optimizer, lrs_used, chunk_size=16_777_216):
    """Reconstruct the local squared parameter displacement after a torch AdamW step.

    This avoids storing a second model copy.  `lrs_used` must contain the learning rate
    applied by the just-completed step because the scheduler may already have advanced.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("Update-norm reconstruction currently supports torch.optim.AdamW only")
    if len(lrs_used) != len(optimizer.param_groups):
        raise ValueError("One applied learning rate is required for each optimizer group")
    squared_norm = None
    for group, lr_used in zip(optimizer.param_groups, lrs_used):
        if group.get("differentiable", False) or group.get("capturable", False):
            raise ValueError("Update-norm reconstruction requires ordinary non-capturable AdamW")
        beta1, beta2 = group["betas"]
        weight_decay, eps = group["weight_decay"], group["eps"]
        decay = 1.0 - float(lr_used) * weight_decay
        if decay <= 0:
            raise ValueError("AdamW multiplicative decay must be positive")
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = optimizer.state[parameter]
            step_number = int(state["step"].item())
            exp_avg = state["exp_avg"].detach().reshape(-1)
            variance = state["max_exp_avg_sq"] if group["amsgrad"] else state["exp_avg_sq"]
            variance = variance.detach().reshape(-1)
            parameter_after = parameter.detach().reshape(-1)
            bias_correction1 = 1.0 - beta1 ** step_number
            bias_correction2_sqrt = math.sqrt(1.0 - beta2 ** step_number)
            step_size = float(lr_used) / bias_correction1
            for start in range(0, parameter_after.numel(), chunk_size):
                stop = min(start + chunk_size, parameter_after.numel())
                denominator = variance[start:stop].sqrt() / bias_correction2_sqrt + eps
                adam_delta = step_size * exp_avg[start:stop] / denominator
                update = (-float(lr_used) * weight_decay * parameter_after[start:stop] - adam_delta) / decay
                value = update.double().square().sum()
                squared_norm = value if squared_norm is None else squared_norm + value
    if squared_norm is None:
        raise ValueError("AdamW step had no parameters with gradients")
    return squared_norm


def adamw_parameter_updates(optimizer, lrs_used, chunk_size=16_777_216):
    """Reconstruct each local AdamW parameter displacement after a completed step.

    This is intentionally separate from the low-memory norm-only path: callers
    enabling parameter JVPs need the actual local sharded tangent tensors.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("Update reconstruction currently supports torch.optim.AdamW only")
    if len(lrs_used) != len(optimizer.param_groups):
        raise ValueError("One applied learning rate is required for each optimizer group")
    updates = {}
    for group, lr_used in zip(optimizer.param_groups, lrs_used):
        if group.get("differentiable", False) or group.get("capturable", False):
            raise ValueError("Update reconstruction requires ordinary non-capturable AdamW")
        beta1, beta2 = group["betas"]
        weight_decay, eps = group["weight_decay"], group["eps"]
        decay = 1.0 - float(lr_used) * weight_decay
        if decay <= 0:
            raise ValueError("AdamW multiplicative decay must be positive")
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = optimizer.state[parameter]
            step_number = int(state["step"].item())
            exp_avg = state["exp_avg"].detach().reshape(-1)
            variance = state["max_exp_avg_sq"] if group["amsgrad"] else state["exp_avg_sq"]
            variance = variance.detach().reshape(-1)
            parameter_after = parameter.detach().reshape(-1)
            bias_correction1 = 1.0 - beta1 ** step_number
            bias_correction2_sqrt = math.sqrt(1.0 - beta2 ** step_number)
            step_size = float(lr_used) / bias_correction1
            update = torch.empty_like(parameter_after)
            for start in range(0, parameter_after.numel(), chunk_size):
                stop = min(start + chunk_size, parameter_after.numel())
                denominator = variance[start:stop].sqrt() / bias_correction2_sqrt + eps
                adam_delta = step_size * exp_avg[start:stop] / denominator
                update[start:stop] = (
                    -float(lr_used) * weight_decay * parameter_after[start:stop] - adam_delta
                ) / decay
            updates[id(parameter)] = update.view_as(parameter)
    if not updates:
        raise ValueError("AdamW step had no parameters with gradients")
    return updates


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


def score_context_row_parameter_jvp(model, row, device, parameters, tangents, buffers):
    """Score one row at fixed parameters and return its parameter-to-logit JVP."""
    valid = row["attention_mask"].bool()
    ids = row["input_ids"][valid].unsqueeze(0).to(device)
    position_ids = row["position_ids"][valid].unsqueeze(0).to(device)
    selected = row["kl_positions"][row["kl_positions"] >= 0]
    prompt_width = row["input_ids"].numel() - row["responses"].numel()
    absolute = prompt_width + selected - 1
    indices = valid.long().cumsum(0)[absolute] - 1
    if (indices < 0).any() or not valid[absolute].all():
        raise ValueError("KL position does not have a valid causal prefix")
    indices = indices.to(device)

    def forward(parameter_values):
        output = torch.func.functional_call(
            model,
            (parameter_values, buffers),
            (),
            {
                "input_ids": ids,
                "attention_mask": None,
                "position_ids": position_ids,
                "use_cache": False,
            },
            strict=True,
        )
        return output.logits[0, indices, :].float()

    logits, logits_jvp = torch.func.jvp(forward, (parameters,), (tangents,))
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.detach().cpu(), logits_jvp.detach().float().cpu()
