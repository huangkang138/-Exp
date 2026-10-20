"""Bounded 2-opt quality and distinct cross/within-ball structure reward."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor

from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical
from gb_2_maco_POMO.runner import prepare_tsp

from .settings import RewardSettings


@dataclass(frozen=True)
class Refined:
    route: tuple[int, ...]
    length: int
    checks: int


def bounded_two_opt(route: tuple[int, ...], distance: list[list[int]], max_checks: int) -> Refined:
    """Best improving swap within one fixed pair-check budget; no Opt-B."""
    path = list(canonical(list(route)))
    n = len(path)
    length = sum(distance[path[i]][path[(i + 1) % n]] for i in range(n))
    checks = 0
    while checks < max_checks:
        best_delta, best_pair = 0, None
        exhausted = False
        for i in range(n - 2):
            a, b = path[i], path[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c, d = path[j], path[(j + 1) % n]
                delta = distance[a][c] + distance[b][d] - distance[a][b] - distance[c][d]
                checks += 1
                if delta < best_delta:
                    best_delta, best_pair = delta, (i, j)
                if checks == max_checks:
                    exhausted = True
                    break
            if exhausted:
                break
        if best_pair is None:
            break
        i, j = best_pair
        path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
        length += best_delta
    verified = sum(distance[path[i]][path[(i + 1) % n]] for i in range(n))
    if verified != length:
        raise RuntimeError("bounded 2-opt length differs from benchmark distance")
    return Refined(canonical(path), length, checks)


def structure(route: tuple[int, ...], city_to_ball: list[int]):
    """Undirected edges, with within-ball edges retaining their ball ID."""
    cross, within = set(), set()
    for a, b in zip(route, route[1:] + route[:1]):
        edge = (min(a, b), max(a, b))
        if city_to_ball[a] != city_to_ball[b]:
            cross.add(edge)
        else:
            within.add((city_to_ball[a], *edge))
    return frozenset(cross), frozenset(within)


def _jaccard(left: frozenset, right: frozenset) -> float:
    union = left | right
    return len(left ^ right) / len(union) if union else 0.0


def novelty(refined: list[Refined], mapping: list[int], quality_ratio: float,
            cross_weight: float, within_weight: float) -> list[float]:
    """Mean distance to other distinct high-quality structures, no duplicate credit."""
    if not refined:
        return []
    best = min(item.length for item in refined)
    eligible = [i for i, item in enumerate(refined) if item.length <= quality_ratio * best + 1e-9]
    structures = {i: structure(refined[i].route, mapping) for i in eligible}
    multiplicity = Counter(structures.values())
    unique = sorted(multiplicity, key=lambda pair: (tuple(sorted(pair[0])), tuple(sorted(pair[1]))))
    values = [0.0] * len(refined)
    if len(unique) < 2:
        return values
    novelty_by_structure = {}
    for current in unique:
        others = [other for other in unique if other != current]
        score = sum(cross_weight * _jaccard(current[0], other[0]) +
                    within_weight * _jaccard(current[1], other[1]) for other in others) / len(others)
        novelty_by_structure[current] = score
    for index, key in structures.items():
        values[index] = novelty_by_structure[key] / multiplicity[key]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise RuntimeError("invalid structure novelty")
    return values


def standardized_advantage(rewards: Tensor) -> Tensor:
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    scale = centered.square().mean(dim=1, keepdim=True).sqrt()
    return torch.where(scale > 1e-8, centered / scale.clamp_min(1e-8),
                       torch.zeros_like(centered))


def loss(outcome, instances, reward: RewardSettings):
    if outcome.joint_log_probs is None or len(instances) != outcome.routes.shape[0]:
        raise ValueError("sampled J2 rollout and instance list differ")
    started = perf_counter()
    routes = outcome.routes.detach().cpu().tolist()
    refined_lengths, novelty_values = [], []
    total_checks, eligible_routes = 0, 0
    for item, case_routes in zip(instances, routes):
        problem = prepare_tsp(item.path)
        cache = {}
        case_refined = []
        for route in case_routes:
            key = canonical(route)
            if key not in cache:
                cache[key] = bounded_two_opt(key, problem.distance_matrix,
                                             reward.two_opt_max_pair_checks_per_route)
                total_checks += cache[key].checks
            case_refined.append(cache[key])
        best = min(result.length for result in case_refined)
        eligible_routes += sum(result.length <= reward.quality_ratio * best + 1e-9
                               for result in case_refined)
        refined_lengths.append([result.length for result in case_refined])
        novelty_values.append(novelty(case_refined, problem.city_to_ball, reward.quality_ratio,
                                      reward.cross_ball_novelty_weight,
                                      reward.within_ball_novelty_weight))
    device = outcome.lengths.device
    two_opt = torch.tensor(refined_lengths, dtype=outcome.lengths.dtype, device=device)
    diversity = torch.tensor(novelty_values, dtype=outcome.lengths.dtype, device=device)
    raw_adv = standardized_advantage(-outcome.lengths.detach())
    opt_adv = standardized_advantage(-two_opt)
    div_adv = standardized_advantage(diversity)
    combined = (reward.raw_advantage_weight * raw_adv +
                reward.two_opt_advantage_weight * opt_adv +
                reward.diversity_advantage_weight * div_adv)
    objective = -(combined.detach() * outcome.joint_log_probs).mean()
    if not torch.isfinite(objective).item() or not torch.isfinite(combined).all().item():
        raise RuntimeError("J2 reward or gradient objective is not finite")
    stats = {"raw_adv_abs_mean": float(raw_adv.abs().mean()),
             "two_opt_adv_abs_mean": float(opt_adv.abs().mean()),
             "diversity_adv_abs_mean": float(div_adv.abs().mean()),
             "eligible_routes": eligible_routes, "novelty_positive_routes": int((diversity > 0).sum()),
             "two_opt_pair_checks": total_checks,
             "two_opt_reward_seconds": perf_counter() - started,
             "raw_mean_length": float(outcome.lengths.detach().mean()),
             "two_opt_mean_length": float(two_opt.mean()),
             "novelty_mean": float(diversity.mean())}
    return objective, stats
