"""Fixed-budget validation for opt-guided training; no model updates."""

from __future__ import annotations

import statistics
from pathlib import Path

import torch

from ..attention import POMOPolicy
from .batch_rollout import rollout_batch
from .data import DatasetInstance
from .opt_guided_feedback import selected_feedback


@torch.no_grad()
def validate(
    policy: POMOPolicy, instances: list[DatasetInstance], device: torch.device,
    references: dict[Path, int], tolerance: float, epoch: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """All N greedy starts choose best checkpoint; eight fixed starts get full 2-opt."""
    policy.eval()
    raw_bests = []
    opt_bests = []
    unique_good = []
    route_rows = []
    for instance in instances:
        outcome = rollout_batch(policy, [instance], device, sample_actions=False)
        raw_bests.append(float(outcome.lengths.min().item()))
        count = instance.city_count
        indices = [tuple((i * count) // min(8, count) for i in range(min(8, count)))]
        item = selected_feedback(outcome, [instance], indices, references, tolerance)[0]
        opt_bests.append(min(item.refined_lengths))
        good = {cycle for cycle, length in zip(item.refined_cycles, item.refined_lengths)
                if length <= item.reference_length * (1 + tolerance)}
        unique_good.append(len(good))
        for start, raw, refined, cycle in zip(item.selected_indices, item.raw_lengths,
                                               item.refined_lengths, item.refined_cycles):
            route_rows.append({
                "epoch": epoch, "instance": str(instance.path), "pomo_start": start,
                "reference": item.reference_length, "raw_length": raw,
                "two_opt_length": refined, "good": int(cycle in good),
                "canonical_two_opt_cycle": ",".join(map(str, cycle)),
            })
    return {
        "validation_mean_best_raw_length": statistics.mean(raw_bests),
        "validation_mean_best_two_opt_length": statistics.mean(opt_bests),
        "validation_mean_unique_good_cycles": statistics.mean(unique_good),
    }, route_rows
