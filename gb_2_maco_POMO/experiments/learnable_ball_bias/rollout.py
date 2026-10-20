"""Batch rollout with the existing encoder and the optional geometric ball bias."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from gb_2_maco_POMO.attention import build_city_features
from gb_2_maco_POMO.runner import PreparedTSP, prepare_tsp
from gb_2_maco_POMO.experiments.shared.data import DatasetInstance
from .model import BallBiasPolicy


@dataclass
class Rollout:
    lengths: Tensor
    joint_log_probs: Tensor | None
    routes: Tensor


def geometry_batch(problems: list[PreparedTSP], device: torch.device):
    """Normalize geometry exactly once per instance; pad absent ball IDs."""
    max_balls = max(len(p.granular_balls) for p in problems)
    positions, centers, adjacency, mappings, features = [], [], [], [], []
    for problem in problems:
        features.append(build_city_features(problem.cities, problem.granular_balls,
                                            problem.city_to_ball))
        # The first two encoder features already use the normalized city frame.
        xy = features[-1][0, :, :2]
        center = torch.zeros(max_balls, 2)
        for ball_id in range(len(problem.granular_balls)):
            first_city = problem.city_to_ball.index(ball_id)
            center[ball_id] = features[-1][0, first_city, 2:4]
        adjacent = torch.zeros(max_balls, max_balls, dtype=torch.bool)
        for source, neighbors in problem.ball_adjacency.items():
            for target in neighbors:
                adjacent[source, target] = True
        positions.append(xy)
        centers.append(center)
        adjacency.append(adjacent)
        mappings.append(problem.city_to_ball)
    return (torch.cat(features, 0).to(device),
            torch.tensor(mappings, dtype=torch.long, device=device),
            (torch.stack(positions).to(device), torch.stack(centers).to(device),
             torch.stack(adjacency).to(device)))


def rollout_batch(model: BallBiasPolicy, instances: list[DatasetInstance],
                  device: torch.device, *, sample: bool) -> Rollout:
    if not instances or len({item.city_count for item in instances}) != 1:
        raise ValueError("batch must contain instances with one city count")
    problems = [prepare_tsp(item.path) for item in instances]
    batch, n = len(problems), instances[0].city_count
    if any(len(p.cities) != n for p in problems):
        raise ValueError("manifest city count mismatch")
    features, mapping, geometry = geometry_batch(problems, device)
    distance = torch.tensor([p.distance_matrix for p in problems],
                            dtype=torch.float32, device=device)
    _, encoded = model.encode(features)
    first = torch.arange(n, device=device).expand(batch, -1)
    current = first
    routes = torch.empty((batch, n, n), dtype=torch.long, device=device)
    routes[:, :, 0] = first
    available = torch.ones((batch, n, n), dtype=torch.bool, device=device)
    available.scatter_(2, first[..., None], False)
    log_probs = []
    for step in range(1, n):
        current, logp, _ = model.choose_step(
            encoded, first, current, available, mapping, sample=sample, geometry=geometry)
        routes[:, :, step] = current
        available = available.scatter(2, current[..., None], False)
        if sample:
            log_probs.append(logp)
    if not torch.equal(routes.sort(dim=-1).values,
                       torch.arange(n, device=device).expand_as(routes)):
        raise RuntimeError("invalid tour")
    index = torch.arange(batch, device=device)[:, None, None]
    lengths = distance[index, routes, routes.roll(-1, dims=-1)].sum(-1)
    return Rollout(lengths, torch.stack(log_probs).sum(0) if sample else None, routes)


@torch.inference_mode()
def generate_routes(model: BallBiasPolicy, problem: PreparedTSP, *, samples: int,
                    seed: int, device: torch.device,
                    starts: list[int] | None = None) -> list[list[int]]:
    """Historical order: one argmax round, then the configured sampling rounds."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model.eval()
    features, mapping, geometry = geometry_batch([problem], device)
    _, encoded = model.encode(features)
    n = len(problem.cities)
    starts = list(range(n)) if starts is None else starts
    first = torch.tensor(starts, dtype=torch.long, device=device)[None, :]
    generated = []
    for repeat in range(samples + 1):
        route = torch.empty((1, len(starts), n), dtype=torch.long, device=device)
        route[:, :, 0] = first
        current = first
        available = torch.ones((1, len(starts), n), dtype=torch.bool, device=device)
        available.scatter_(2, first[..., None], False)
        for step in range(1, n):
            current, _, _ = model.choose_step(
                encoded, first, current, available, mapping,
                sample=repeat > 0, geometry=geometry)
            route[:, :, step] = current
            available = available.scatter(2, current[..., None], False)
        if not torch.equal(route.sort(dim=-1).values,
                           torch.arange(n, device=device).expand_as(route)):
            raise RuntimeError("generated route is missing or repeating cities")
        generated.extend(route[0].cpu().tolist())
    return generated
