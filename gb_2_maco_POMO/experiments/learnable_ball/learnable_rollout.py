"""Vectorized POMO routes for the learnable ball-and-city policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from gb_2_maco_POMO.attention import build_city_features
from gb_2_maco_POMO.runner import prepare_tsp
from gb_2_maco_POMO.experiments.shared.data import DatasetInstance
from gb_2_maco_POMO.experiments.learnable_ball.learnable_policy import LearnableBallPolicy


@dataclass
class LearnedRollout:
    lengths: Tensor                 # [B,P] closed tours, original integer distances
    joint_log_probs: Tensor | None  # [B,P] sum of ball + conditional city logP
    routes: Tensor                  # [B,P,N] every city once


def rollout_batch(policy: LearnableBallPolicy, instances: list[DatasetInstance],
                  device: torch.device, *, sample: bool) -> LearnedRollout:
    if not instances or any(item.city_count != instances[0].city_count for item in instances):
        raise ValueError("micro batch must have one common positive city count")
    prepared = [prepare_tsp(item.path) for item in instances]
    batch, n = len(prepared), instances[0].city_count
    if any(len(problem.cities) != n for problem in prepared):
        raise ValueError("manifest city count mismatch")
    features = torch.cat([
        build_city_features(problem.cities, problem.granular_balls, problem.city_to_ball)
        for problem in prepared
    ], dim=0).to(device)
    distance = torch.tensor([problem.distance_matrix for problem in prepared],
                            dtype=torch.float32, device=device)
    mapping = torch.tensor([problem.city_to_ball for problem in prepared],
                           dtype=torch.long, device=device)
    _, encoded = policy.encode(features)
    first = torch.arange(n, device=device).expand(batch, -1)
    current = first
    routes = torch.empty((batch, n, n), dtype=torch.long, device=device)
    routes[:, :, 0] = first
    available = torch.ones((batch, n, n), dtype=torch.bool, device=device)
    available.scatter_(2, first[..., None], False)
    log_probs = []
    for step in range(1, n):
        current, joint_logp, _ = policy.choose_step(
            encoded, first, current, available, mapping, sample=sample)
        if not available.gather(2, current[..., None]).all().item():
            raise RuntimeError("already visited city selected")
        routes[:, :, step] = current
        available = available.scatter(2, current[..., None], False)
        if sample:
            log_probs.append(joint_logp)
    expected = torch.arange(n, device=device).expand(batch, n, n)
    if not torch.equal(routes.sort(dim=-1).values, expected):
        raise RuntimeError("route is missing or repeating cities")
    batch_ids = torch.arange(batch, device=device)[:, None, None]
    lengths = distance[batch_ids, routes, routes.roll(-1, dims=-1)].sum(dim=-1)
    return LearnedRollout(lengths, torch.stack(log_probs).sum(dim=0) if sample else None,
                          routes)
