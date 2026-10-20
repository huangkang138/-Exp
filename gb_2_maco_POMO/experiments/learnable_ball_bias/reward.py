"""POMO length objective plus a bounded, sampled 2-opt diversity signal."""

from __future__ import annotations

import random
from collections import Counter

import torch

from gb_2_maco_POMO.runner import prepare_tsp
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical


def diversity_signal(outcome, instances, *, routes_per_instance: int,
                     tolerance_pct: float, rng: random.Random):
    """Only selected routes receive a diversity signal; others stay unassessed."""
    if routes_per_instance < 2 or tolerance_pct < 0:
        raise ValueError("diversity budget must be >=2 and tolerance nonnegative")
    batch, pomo = outcome.lengths.shape
    values = torch.zeros_like(outcome.lengths)
    selected = torch.zeros_like(outcome.lengths, dtype=torch.bool)
    diagnostics = []
    for index, item in enumerate(instances):
        chosen = sorted(rng.sample(range(pomo), min(pomo, routes_per_instance)))
        matrix = prepare_tsp(item.path).distance_matrix
        routes = outcome.routes[index, chosen].detach().cpu().tolist()
        refined = [two_opt(route, matrix) for route in routes]
        best = min(result.length for result in refined)
        quality = [result.length <= best * (1 + tolerance_pct / 100)
                   for result in refined]
        counts = Counter(canonical(result.route) for result, good in zip(refined, quality)
                         if good)
        for position, result, good in zip(chosen, refined, quality):
            selected[index, position] = True
            if good:
                values[index, position] = 1 / counts[canonical(result.route)]
        diagnostics.append({"instance": item.path.name, "evaluated": len(chosen),
                            "eligible": sum(quality), "unique_eligible": len(counts),
                            "reference_2opt_length": best})
    return values, selected, diagnostics


def policy_loss(outcome, *, diversity_weight: float = 0,
                diversity: torch.Tensor | None = None,
                assessed: torch.Tensor | None = None):
    """Center length over all POMO starts; center diversity only over assessed starts.

    The latter is a set-dependent auxiliary policy signal, not an unbiased
    gradient of an exact route-coverage objective. Rewards carry no graph.
    """
    logp = outcome.joint_log_probs
    if logp is None or not torch.isfinite(logp).all():
        raise RuntimeError("sampled rollout needs finite joint log probabilities")
    lengths = outcome.lengths.detach()
    reward = -lengths
    advantage = reward - reward.mean(dim=1, keepdim=True)
    loss = -(advantage * logp).mean()
    if diversity_weight:
        if diversity is None or assessed is None or diversity.shape != lengths.shape:
            raise ValueError("diversity signal and assessment mask required")
        if not torch.isfinite(diversity).all() or not assessed.any(dim=1).all():
            raise RuntimeError("invalid diversity signal")
        count = assessed.sum(dim=1, keepdim=True)
        mean = (diversity.detach() * assessed).sum(dim=1, keepdim=True) / count
        diversity_advantage = (diversity.detach() - mean) * assessed
        scale = lengths.mean().detach()
        # Normalize by assessed count, not all POMO starts. Unassessed routes
        # are excluded rather than treated as low-quality duplicates.
        loss = loss - diversity_weight * scale * (
            (diversity_advantage * logp).sum(dim=1) / count.squeeze(1)).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite policy loss")
    return loss
