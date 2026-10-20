"""POMO training routes with the unchanged negative closed-tour reward."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from gb_2_maco_POMO.attention import build_city_features
from gb_2_maco_POMO.experiments.shared.data import DatasetInstance
from gb_2_maco_POMO.runner import prepare_tsp

from .model import JointBallCityPolicy


@dataclass
class JointRollout:
    lengths: Tensor
    joint_log_probs: Tensor | None
    routes: Tensor


def rollout_batch(policy: JointBallCityPolicy, instances: list[DatasetInstance],
                  device: torch.device, *, sample: bool) -> JointRollout:
    if not instances or len({item.city_count for item in instances}) != 1:
        raise ValueError("microbatch must have one city count")
    problems = [prepare_tsp(item.path) for item in instances]
    batch, n = len(problems), instances[0].city_count
    features = torch.cat([build_city_features(
        p.cities, p.granular_balls, p.city_to_ball) for p in problems], dim=0).to(device)
    matrix = torch.tensor([p.distance_matrix for p in problems],
                          dtype=torch.float32, device=device)
    normalized = matrix / matrix.amax(dim=(1, 2), keepdim=True).clamp_min(1)
    mapping = torch.tensor([p.city_to_ball for p in problems],
                           dtype=torch.long, device=device)
    with torch.no_grad():
        _, encoded = policy.encode(features)
    first = torch.arange(n, device=device).expand(batch, -1)
    current = first
    routes = torch.empty((batch, n, n), dtype=torch.long, device=device)
    routes[:, :, 0] = first
    available = torch.ones((batch, n, n), dtype=torch.bool, device=device)
    available.scatter_(2, first[..., None], False)
    logps = []
    for step in range(1, n):
        current, joint_logp, _, _ = policy.choose_joint_step(
            encoded, first, current, available, mapping, normalized, sample=sample)
        routes[:, :, step] = current
        available = available.scatter(2, current[..., None], False)
        if sample:
            logps.append(joint_logp)
    if not torch.equal(routes.sort(dim=-1).values,
                       torch.arange(n, device=device).expand_as(routes)):
        raise RuntimeError("joint training produced an invalid tour")
    batch_ids = torch.arange(batch, device=device)[:, None, None]
    lengths = matrix[batch_ids, routes, routes.roll(-1, dims=-1)].sum(dim=-1)
    return JointRollout(lengths, torch.stack(logps).sum(dim=0) if sample else None,
                        routes)
