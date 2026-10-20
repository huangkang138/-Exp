"""Observed equal-length coverage reward; inference and local search are reused."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import torch

from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, equal_plateau
from gb_2_maco_POMO.runner import prepare_tsp


@dataclass
class RewardDiagnostics:
    selected: int = 0
    eligible: int = 0
    entrances: int = 0
    regions: int = 0
    nonzero_coverage_advantages: int = 0
    optb_checks: int = 0
    two_opt_checks: int = 0
    actual_optb_checks: int = 0
    actual_two_opt_checks: int = 0
    optb_cache_hits: int = 0
    two_opt_cache_hits: int = 0
    lower_reference_candidates: int = 0


def _edges(route):
    return {tuple(sorted((route[i], route[(i + 1) % len(route)]))) for i in range(len(route))}


def select_candidates(routes, lengths, limit=16):
    """Fixed quality-first and edge-diverse selection, independent of model group."""
    unique = {}
    for index, route in enumerate(routes):
        key = canonical(route)
        old = unique.get(key)
        if old is None or (lengths[index], index) < (lengths[old], old):
            unique[key] = index
    ordered = sorted(unique.values(), key=lambda i: (lengths[i], i))
    if len(ordered) <= limit:
        return ordered
    selected = ordered[: limit // 2]
    pool = ordered[limit // 2: max(limit * 4, len(ordered) // 2)]
    edge_sets = {i: _edges(routes[i]) for i in ordered}
    while len(selected) < limit and pool:
        # Restrict candidates to the better half, then maximize distance from
        # already selected edge sets. Stable ties use raw length and POMO index.
        chosen = max(pool, key=lambda i: (
            min(1 - len(edge_sets[i] & edge_sets[j]) / len(edge_sets[i] | edge_sets[j])
                for j in selected), -lengths[i], -i))
        selected.append(chosen)
        pool.remove(chosen)
    return selected


def standardized_quality(lengths, epsilon=1e-6):
    """Smaller lengths receive a larger within-instance advantage."""
    if not lengths:
        return []
    values = torch.as_tensor(lengths, dtype=torch.float64)
    std = values.std(unbiased=False)
    if std <= epsilon:
        return [0.0] * len(lengths)
    return ((values.mean() - values) / std).tolist()


def _observed_regions(sets_by_index, lengths_by_index):
    """Merge partially explored plateau sets only on observed overlap."""
    indices = list(sets_by_index)
    parent = {i: i for i in indices}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for at, i in enumerate(indices):
        for j in indices[at + 1:]:
            if lengths_by_index[i] == lengths_by_index[j] and sets_by_index[i] & sets_by_index[j]:
                parent[root(j)] = root(i)
    members = defaultdict(list)
    for i in indices:
        members[root(i)].append(i)
    return list(members.values())


def coverage_contributions(sets_by_index, lengths_by_index):
    """Split observed coverage among entrants and normalize its two scales."""
    if not sets_by_index:
        return {}, 0
    groups = _observed_regions(sets_by_index, lengths_by_index)
    region_credit = {}
    for group in groups:
        for i in group:
            region_credit[i] = 1 / len(group)
    frequency = Counter((lengths_by_index[i], key) for i, routes in sets_by_index.items() for key in routes)
    solution_credit = {i: sum(1 / frequency[lengths_by_index[i], key] for key in routes)
                       for i, routes in sets_by_index.items()}
    # Each branch sums to one across entrants. A repeated entry keeps its 1/m
    # share instead of being inflated back to one by max-normalization.
    region_scale = len(groups)
    solution_scale = len(frequency)
    values = {i: 0.7 * region_credit[i] / region_scale + 0.3 * solution_credit[i] / solution_scale
              for i in sets_by_index}
    return values, len(groups)


def compute_advantages(outcome, instances, references, mode, caches, *,
                       selected_limit=16, entrance_limit=8, optb_budget=8000):
    """Return detached per-route advantages and diagnostics for one batch."""
    lengths = outcome.lengths.detach().cpu().tolist()
    routes = outcome.routes.detach().cpu().tolist()
    advantage = torch.zeros_like(outcome.lengths)
    diagnostics = RewardDiagnostics()
    pending = {}
    for batch_index, item in enumerate(instances):
        route_list, raw_lengths = routes[batch_index], lengths[batch_index]
        chosen = select_candidates(route_list, raw_lengths, selected_limit)
        chosen_set = set(chosen)
        fallback = [i for i in range(len(route_list)) if i not in chosen_set]
        for i, score in zip(fallback, standardized_quality([raw_lengths[j] for j in fallback])):
            advantage[batch_index, i] = score
        distance = prepare_tsp(item.path).distance_matrix
        refined = {}
        for i in chosen:
            cache_key = (str(item.path), canonical(route_list[i]))
            if cache_key not in caches["two_opt"]:
                caches["two_opt"][cache_key] = two_opt(route_list[i], distance)
                diagnostics.actual_two_opt_checks += caches["two_opt"][cache_key].pair_checks
            else:
                diagnostics.two_opt_cache_hits += 1
            refined[i] = caches["two_opt"][cache_key]
            diagnostics.two_opt_checks += refined[i].pair_checks
        for i, score in zip(chosen, standardized_quality([refined[j].length for j in chosen])):
            advantage[batch_index, i] = score
        diagnostics.selected += len(chosen)
        if mode == "Q":
            continue
        reference = references[str(item.path)]
        cutoff = reference["length"] if reference["certified"] else reference["length"] * 1.005
        for result in refined.values():
            if result.length < reference["length"]:
                if reference["certified"]:
                    raise RuntimeError(f"route below certified optimal length: {item.path}")
                pending[str(item.path)] = min(pending.get(str(item.path), reference["length"]), result.length)
                diagnostics.lower_reference_candidates += 1
        eligible = [i for i in chosen if reference["length"] <= refined[i].length <= cutoff]
        # Equal-length plateau sets are generated from at most eight distinct
        # optimized entrances. An exhausted/partial result stays observational.
        eligible = sorted(eligible, key=lambda i: (refined[i].length, i))
        entrances = []
        seen = set()
        for i in eligible:
            key = canonical(refined[i].route)
            if key not in seen:
                entrances.append(i)
                seen.add(key)
            if len(entrances) == entrance_limit:
                break
        diagnostics.entrances += len(entrances)
        sets = {}
        entry_lengths = {}
        each_budget = optb_budget // max(1, len(entrances))
        for i in entrances:
            key = (str(item.path), canonical(refined[i].route), each_budget)
            if key not in caches["opt_b"]:
                caches["opt_b"][key] = equal_plateau([refined[i].route], distance, each_budget)
                diagnostics.actual_optb_checks += caches["opt_b"][key].checks
            else:
                diagnostics.optb_cache_hits += 1
            result = caches["opt_b"][key]
            diagnostics.optb_checks += result.checks
            for route_index in eligible:
                if canonical(refined[route_index].route) == canonical(refined[i].route):
                    sets[route_index] = set(result.routes)
                    entry_lengths[route_index] = result.best_length
        diagnostics.eligible += len(sets)
        credit, region_count = coverage_contributions(sets, entry_lengths)
        diagnostics.regions += region_count
        centered_mean = sum(credit.values()) / len(route_list)
        for i in range(len(route_list)):
            score = credit.get(i, 0.0) - centered_mean
            advantage[batch_index, i] += 0.5 * score
            diagnostics.nonzero_coverage_advantages += abs(score) > 1e-12
    return advantage.detach(), diagnostics, pending


def coverage_loss(outcome, advantage):
    logp = outcome.joint_log_probs
    if logp is None or advantage.shape != logp.shape or not torch.isfinite(logp).all():
        raise RuntimeError("invalid two-level policy log probabilities")
    if not torch.isfinite(advantage).all():
        raise RuntimeError("nonfinite advantage")
    return -(advantage * logp).mean()
