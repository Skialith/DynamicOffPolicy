"""CPU anchor caches and direct KL; no parameter derivatives or model snapshots."""

from dataclasses import dataclass

import torch

from verl.trainer.ppo.full_vocab_kl import full_vocabulary_kl


CONTEXT_KEYS = ("input_ids", "attention_mask", "position_ids", "responses", "kl_positions", "kl_weights")


def validate_delayed_kl_config(config, reuse_n=None):
    if config.get("full_kl_jvp_measurement", False) or config.get("full_kl_hvp_measurement", False):
        raise ValueError("JVP/HVP measurement has been removed from this branch")
    if not config.get("full_kl_delayed_measurement", False):
        return
    if not config.get("full_kl_measurement", False) or not config.get("full_kl_actor_measurement", True):
        raise ValueError("Delayed-anchor KL requires direct actor full-vocabulary KL")
    for key in ("full_kl_delayed_horizon", "full_kl_delayed_anchor_every_rollouts"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if reuse_n is not None and config.get("full_kl_delayed_horizon") <= reuse_n:
        raise ValueError("Delayed KL horizon must extend beyond one rollout")


@dataclass
class DelayedKLAnchor:
    anchor_update: int
    rollout_step: int
    reuse_n: int
    contexts: dict[str, torch.Tensor]
    log_probs: torch.Tensor
    cache_seconds: float
    forward_seconds: float

    @classmethod
    @torch.no_grad()
    def capture(cls, batch, log_probs, anchor_update, rollout_step, reuse_n, forward_seconds):
        selected = torch.where((batch["kl_positions"] >= 0).any(-1))[0]
        # Empty ranks still need a dummy row to join every FSDP forward collective.
        rows = selected if len(selected) else torch.zeros(1, dtype=torch.long, device=selected.device)
        contexts = {key: batch[key][rows].detach().to("cpu", copy=True) for key in CONTEXT_KEYS}
        cached = log_probs.detach().cpu()
        weights = contexts["kl_weights"][contexts["kl_positions"] >= 0]
        if cached.ndim != 2 or len(cached) != len(weights):
            raise ValueError("Anchor distributions must align with the selected context positions")
        return cls(anchor_update, rollout_step, reuse_n, contexts, cached, 0.0, forward_seconds)

    @property
    def weights(self):
        return self.contexts["kl_weights"][self.contexts["kl_positions"] >= 0]

    @property
    def cache_bytes(self):
        return sum(t.numel() * t.element_size() for t in (*self.contexts.values(), self.log_probs))

    @torch.no_grad()
    def local_totals(self, current):
        values = full_vocabulary_kl(self.log_probs, current)
        weights = self.weights.double()
        return torch.stack(((values * weights).sum(), weights.sum(), weights.new_tensor(len(weights))))
