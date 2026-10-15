"""Detached full 2-opt feedback for sampled POMO tours.

The route identities and 2-opt distances are CPU data. Policy gradients flow
only through the log probabilities of the original sampled actions.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ..pomo import canonical_cycle
from ..runner import prepare_tsp
from .batch_rollout import BatchRollout
from .data import DatasetInstance
from .local_search_2opt import two_opt


@dataclass(frozen=True)
class InstanceFeedback:
    selected_indices: tuple[int, ...]
    raw_lengths: tuple[int, ...]
    refined_lengths: tuple[int, ...]
    refined_cycles: tuple[tuple[int, ...], ...]
    reference_length: int
    quality_rewards: tuple[float, ...]
    coverage_rewards: tuple[int, ...]
    pass_count: int
    unique_count: int
    two_opt_seconds: float


def heuristic_reference(distance: list[list[int]]) -> int:
    """Fixed NN(start=0) + complete 2-opt; independent of policy samples."""
    remaining = set(range(1, len(distance)))
    route = [0]
    while remaining:
        current = route[-1]
        chosen = min(remaining, key=lambda city: (distance[current][city], city))
        route.append(chosen)
        remaining.remove(chosen)
    return two_opt(route, distance).length


def coverage_marginals(
    cycles: tuple[tuple[int, ...], ...], lengths: tuple[int, ...],
    reference: int, tolerance: float,
) -> tuple[tuple[int, ...], int]:
    """Leave-one-out set coverage with a fixed action-independent threshold."""
    if reference <= 0 or not 0 <= tolerance <= 0.005:
        raise ValueError("reference or tolerance invalid")
    passed = tuple(length <= reference * (1 + tolerance) for length in lengths)
    counts = Counter(cycle for cycle, ok in zip(cycles, passed) if ok)
    return tuple(int(ok and counts[cycle] == 1)
                 for cycle, ok in zip(cycles, passed)), sum(passed)


def selected_feedback(
    rollout: BatchRollout, instances: list[DatasetInstance],
    indices: list[tuple[int, ...]], references: dict[Path, int], tolerance: float,
) -> list[InstanceFeedback]:
    if rollout.routes is None or len(rollout.routes) != len(instances):
        raise ValueError("rollout did not retain sampled routes")
    outputs = []
    for batch_id, instance in enumerate(instances):
        selected = indices[batch_id]
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("selected POMO indices must be unique and nonempty")
        reference = references[instance.path]
        problem = prepare_tsp(instance.path)
        start = time.perf_counter()
        original = tuple(rollout.routes[batch_id][index] for index in selected)
        refined = tuple(two_opt(route, problem.distance_matrix) for route in original)
        elapsed = time.perf_counter() - start
        raw_lengths = tuple(int(rollout.lengths[batch_id, index].item()) for index in selected)
        refined_lengths = tuple(item.length for item in refined)
        if any(after > before for before, after in zip(raw_lengths, refined_lengths)):
            raise RuntimeError("2-opt length increased")
        cycles = tuple(canonical_cycle(item.route) for item in refined)
        marginals, pass_count = coverage_marginals(cycles, refined_lengths,
                                                    reference, tolerance)
        outputs.append(InstanceFeedback(
            selected, raw_lengths, refined_lengths, cycles, reference,
            tuple(-length / reference for length in refined_lengths), marginals,
            pass_count, len(set(cycles)), elapsed,
        ))
    return outputs


def select_indices(instances: list[DatasetInstance], rng: random.Random,
                   limit: int = 8) -> list[tuple[int, ...]]:
    """Draw starts before route sampling with a private RNG."""
    return [tuple(sorted(rng.sample(range(item.city_count), min(limit, item.city_count))))
            for item in instances]


def auxiliary_losses(log_probs: Tensor, feedback: list[InstanceFeedback]) -> tuple[Tensor, Tensor]:
    quality_terms = []
    coverage_terms = []
    for batch_id, item in enumerate(feedback):
        lp = log_probs[batch_id, list(item.selected_indices)]
        q = lp.new_tensor(item.quality_rewards)
        # The sample-mean baseline includes each route's own reward. The
        # k/(k-1) correction restores the leave-one-out REINFORCE scale.
        if len(item.selected_indices) > 1:
            quality_terms.append(-((q - q.mean()).detach() * lp).mean()
                                 * len(item.selected_indices) / (len(item.selected_indices) - 1))
        else:
            quality_terms.append(lp.sum() * 0)
        # Other starts are conditionally independent given the instance and
        # selected indices. G(S)-G(S\i) removes an action-independent baseline
        # for each score_i; no group-mean baseline is applied to this term.
        coverage_terms.append(-(lp * lp.new_tensor(item.coverage_rewards).detach()).sum())
    return torch.stack(quality_terms).mean(), torch.stack(coverage_terms).mean()
