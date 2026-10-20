"""Original two-stage action selection with separate ball/city log probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from gb_2_maco_POMO.attention import build_city_features
from gb_2_maco_POMO.experiments.shared.data import DatasetInstance
from gb_2_maco_POMO.runner import prepare_tsp

from .reward import decompose


@dataclass
class RewardRollout:
    lengths: Tensor
    inter_lengths: Tensor
    intra_lengths: Tensor
    crossings: Tensor
    routes: Tensor
    ball_log_probs: Tensor | None
    city_log_probs: Tensor | None
    joint_log_probs: Tensor | None


def rollout_batch(policy, instances: list[DatasetInstance], device: torch.device,
                  *, sample: bool) -> RewardRollout:
    if not instances or len({item.city_count for item in instances}) != 1:
        raise ValueError("microbatch requires a common city count")
    prepared = [prepare_tsp(item.path) for item in instances]
    batch, n = len(prepared), instances[0].city_count
    if any(len(problem.cities) != n for problem in prepared):
        raise RuntimeError("manifest city count mismatch")
    features = torch.cat([
        build_city_features(p.cities, p.granular_balls, p.city_to_ball)
        for p in prepared], dim=0).to(device)
    distance = torch.tensor([p.distance_matrix for p in prepared],
                            dtype=torch.float32, device=device)
    mapping = torch.tensor([p.city_to_ball for p in prepared],
                           dtype=torch.long, device=device)
    _, encoded = policy.encode(features)
    first = torch.arange(n, device=device).expand(batch, -1)
    current = first
    routes = torch.empty((batch, n, n), dtype=torch.long, device=device)
    routes[:, :, 0] = first
    available = torch.ones((batch, n, n), dtype=torch.bool, device=device)
    available.scatter_(2, first[..., None], False)
    ball_logs, city_logs = [], []
    for step in range(1, n):
        # Identical logits, Categorical draws, masks and order to
        # LearnableBallPolicy.choose_step; only the log probabilities are split.
        ball_logits, city_logits = policy.action_logits(
            encoded, first, current, available, mapping)
        ball_dist = torch.distributions.Categorical(logits=ball_logits)
        chosen_ball = ball_dist.sample() if sample else ball_logits.argmax(dim=-1)
        city_mask = available & (mapping[:, None, :] == chosen_ball[..., None])
        conditional_logits = city_logits.masked_fill(~city_mask, -torch.inf)
        city_dist = torch.distributions.Categorical(logits=conditional_logits)
        current = city_dist.sample() if sample else conditional_logits.argmax(dim=-1)
        if not available.gather(2, current[..., None]).all().item():
            raise RuntimeError("sampled a visited city")
        routes[:, :, step] = current
        available = available.scatter(2, current[..., None], False)
        if sample:
            ball_logs.append(ball_dist.log_prob(chosen_ball))
            city_logs.append(city_dist.log_prob(current))
    if not torch.equal(routes.sort(dim=-1).values,
                       torch.arange(n, device=device).expand_as(routes)):
        raise RuntimeError("illegal sampled route")
    total, inter, intra, crossings = decompose(routes, distance, mapping)
    ball_log = torch.stack(ball_logs).sum(dim=0) if sample else None
    city_log = torch.stack(city_logs).sum(dim=0) if sample else None
    return RewardRollout(total, inter, intra, crossings, routes, ball_log,
                         city_log, ball_log + city_log if sample else None)
