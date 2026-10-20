"""Identical validation for every epoch; incomplete truth uses a local proxy."""

from __future__ import annotations

from statistics import mean
from time import perf_counter

import torch

from gb_2_maco_POMO.runner import prepare_tsp
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, equal_plateau, tour_length
from gb_2_maco_POMO.experiments.learnable_ball_bias.rollout import generate_routes
from gb_2_maco_POMO.experiments.learnable_ball_bias.validation import certified_lengths, heuristic_reference


@torch.inference_mode()
def evaluate(model, instances, data_root, device, *, samples, optb_checks, seed):
    certified = certified_lengths(data_root)
    rows = []
    for index, item in enumerate(instances, 1):
        started = perf_counter()
        problem = prepare_tsp(item.path)
        distance = problem.distance_matrix
        raw = generate_routes(model, problem, samples=samples, seed=seed + index,
                              device=device)
        raw_lengths = [tour_length(route, distance) for route in raw]
        unique = {}
        for route in raw:
            unique.setdefault(canonical(route), route)
        refined = [two_opt(route, distance) for route in unique.values()]
        best = min([*raw_lengths, *(r.length for r in refined)])
        a_pool = {canonical(route): route for route, length in zip(raw, raw_lengths)
                  if length == best}
        a_pool.update({canonical(r.route): r.route for r in refined if r.length == best})
        b = equal_plateau(list(a_pool.values()), distance, optb_checks)
        proof = certified.get(item.path.name)
        if proof is not None and best < proof:
            raise RuntimeError('route below certified optimum')
        rows.append({'instance': item.path.name, 'city_count': item.city_count,
                     'raw_best': min(raw_lengths), 'two_opt_best': best,
                     'optb_best': b.best_length, 'a_routes': len(a_pool),
                     'b_routes': len(b.routes), 'optb_checks': b.checks,
                     'heuristic_reference': heuristic_reference(distance),
                     'certified_length': proof,
                     'certified_hit': int(proof is not None and best == proof),
                     'seconds': perf_counter() - started})
    if len(rows) != len(instances) or len(certified) != sum(r['certified_length'] is not None for r in rows):
        raise RuntimeError('validation coverage mismatch')
    known = [r for r in rows if r['certified_length'] is not None]
    metrics = {'cases': len(rows), 'certified_cases': len(known),
               'certified_hits': sum(r['certified_hit'] for r in rows),
               'mean_certified_gap_pct': (mean(100 * (r['two_opt_best'] /
                       r['certified_length'] - 1) for r in known) if known else None),
               'mean_relative_2opt_pct': mean(100 * (r['two_opt_best'] /
                       r['heuristic_reference'] - 1) for r in rows),
               'mean_relative_raw_pct': mean(100 * (r['raw_best'] /
                       r['heuristic_reference'] - 1) for r in rows),
               'mean_b_routes': mean(r['b_routes'] for r in rows),
               'total_seconds': sum(r['seconds'] for r in rows)}
    return rows, metrics


def select_checkpoints(records, rows_by_epoch, *, tolerance_pct):
    """Select without the final 25; never call local-best coverage true recall."""
    if not records:
        raise ValueError('no validation records')
    all_names = set(rows_by_epoch[records[0]['epoch']])
    if any(set(rows_by_epoch[r['epoch']]) != all_names for r in records):
        raise ValueError('validation instance set changed between epochs')
    certified_cases = {r['metrics']['certified_cases'] for r in records}
    if len(certified_cases) != 1:
        raise ValueError('certified coverage changed')
    if next(iter(certified_cases)):
        # Distances are certified only for this subset. No full tour-set Fbeta.
        best_hits = max(r['metrics']['certified_hits'] for r in records)
        eligible_hits = [r for r in records if r['metrics']['certified_hits'] == best_hits]
        minimum_gap = min(r['metrics']['mean_certified_gap_pct'] for r in eligible_hits)
        length = min(eligible_hits, key=lambda r: (
            r['metrics']['mean_certified_gap_pct'], r['metrics']['mean_relative_raw_pct'], r['epoch']))
        candidates = [r for r in eligible_hits if r['metrics']['mean_certified_gap_pct']
                      <= minimum_gap + tolerance_pct]
        multi = min(candidates, key=lambda r: (-r['metrics']['mean_b_routes'],
                                                r['metrics']['mean_certified_gap_pct'], r['epoch']))
        return length, multi, {'reference': 'certified subset', 'distance_hits': best_hits,
                               'minimum_gap_pct': minimum_gap}
    proxy_best = {name: min(rows_by_epoch[r['epoch']][name]['two_opt_best'] for r in records)
                  for name in all_names}
    for r in records:
        rows = rows_by_epoch[r['epoch']]
        r['selection_metrics'] = {
            'proxy_hits': sum(rows[name]['two_opt_best'] == proxy_best[name] for name in all_names),
            'proxy_gap_pct': mean(100 * (rows[name]['two_opt_best'] / proxy_best[name] - 1)
                                  for name in all_names),
            'proxy_b_routes': sum(rows[name]['b_routes'] for name in all_names
                                  if rows[name]['two_opt_best'] == proxy_best[name]),
            'mean_raw_length': mean(rows[name]['raw_best'] for name in all_names),
        }
    best_hits = max(r['selection_metrics']['proxy_hits'] for r in records)
    eligible_hits = [r for r in records if r['selection_metrics']['proxy_hits'] == best_hits]
    minimum_gap = min(r['selection_metrics']['proxy_gap_pct'] for r in records)
    length = min(eligible_hits, key=lambda r: (
        r['selection_metrics']['proxy_gap_pct'], r['selection_metrics']['mean_raw_length'], r['epoch']))
    candidates = [r for r in eligible_hits if r['selection_metrics']['proxy_gap_pct']
                  <= minimum_gap + tolerance_pct + 1e-12]
    if not candidates:
        # This can occur when the globally smallest mean gap has fewer proxy
        # hits. Retain the highest-hit gate and use its own minimum gap.
        minimum_gap = min(r['selection_metrics']['proxy_gap_pct'] for r in eligible_hits)
        candidates = [r for r in eligible_hits if r['selection_metrics']['proxy_gap_pct']
                      <= minimum_gap + tolerance_pct + 1e-12]
    multi = min(candidates, key=lambda r: (-r['selection_metrics']['proxy_b_routes'],
                                            r['selection_metrics']['proxy_gap_pct'], r['epoch']))
    return length, multi, {'reference': 'best 2-opt length observed on validation across epochs',
                           'distance_hits': best_hits, 'minimum_gap_pct': minimum_gap,
                           'proxy_best_by_case': proxy_best,
                           'note': 'Local proxy only; no certified recall or Fbeta'}
