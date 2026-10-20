"""Project-standard ordinary 2-opt/Opt-B with per-route provenance."""

from __future__ import annotations

import hashlib
import json

from gb_maco_POMO.experiments.compare import fbeta, load_ground_truth, read_optimum
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, equal_plateau, tour_length


def evaluate(prepared, routes: list[list[int]], solution, optb_checks: int) -> tuple[dict, list[dict], dict]:
    distance = prepared.distance_matrix
    n = len(distance)
    if not routes or any(len(route) != n or sorted(route) != list(range(n)) for route in routes):
        raise ValueError("routes must be complete city permutations")
    lengths = [tour_length(route, distance) for route in routes]
    seen = set()
    candidates = []
    for index in sorted(range(len(routes)), key=lambda i: (lengths[i], i)):
        key = canonical(routes[index])
        if key not in seen:
            candidates.append(index)
            seen.add(key)
    refined = {index: two_opt(routes[index], distance) for index in candidates}
    representative = {canonical(routes[index]): index for index in candidates}
    best = min([*lengths, *(item.length for item in refined.values())])
    a_pool = {canonical(route): route for route, length in zip(routes, lengths) if length == best}
    a_pool.update({canonical(item.route): item.route for item in refined.values() if item.length == best})
    b = equal_plateau(list(a_pool.values()), distance, optb_checks)
    # Labels enter only after route generation and both local-search stages.
    reference = read_optimum(solution)
    truth = load_ground_truth(solution, n)
    if min(lengths) < reference or best < reference or b.best_length < reference:
        raise RuntimeError("generated a distance below the benchmark reference")
    ah, bh = len(set(a_pool) & truth), len(set(b.routes) & truth)
    rows = []
    for index, route in enumerate(routes):
        endpoint = refined.get(index)
        source_index = representative[canonical(route)]
        rows.append({"route_index": index, "start": route[0], "raw_route": route,
            "raw_length": lengths[index], "two_opt_route": endpoint.route if endpoint else None,
            "two_opt_length": endpoint.length if endpoint else None,
            "two_opt_pair_checks": endpoint.pair_checks if endpoint else 0,
            "two_opt_status": "processed" if endpoint else "duplicate_cycle_skipped",
            "representative_route_index": source_index,
            "optb_seed": bool(endpoint and canonical(endpoint.route) in a_pool)})
    # Opt-B explores a plateau from the union of best 2-opt endpoints. It does
    # not expose a unique parent for each new tour, so record its seed pool and
    # case-wide pair-check count rather than inventing per-route attribution.
    provenance = {"a_best_routes": [list(key) for key in sorted(a_pool)],
        "b_best_routes": [list(key) for key in sorted(b.routes)],
        "optb_new_routes": [list(key) for key in sorted(set(b.routes) - set(a_pool))],
        "optb_seed_routes": [list(key) for key in sorted(a_pool)],
        "optb_pair_checks": b.checks, "optb_attempts": b.attempts}
    raw = json.dumps(routes, separators=(",", ":")).encode()
    record = {"raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_routes": len(routes),
        "raw_best": min(lengths), "raw_gap_pct": 100 * (min(lengths) / reference - 1),
        "a_best": best, "a_gap_pct": 100 * (best / reference - 1),
        "a_distance_hit": int(best == reference), "a_routes": len(a_pool), "a_hits": ah,
        "a_precision": ah / len(a_pool), "a_recall": ah / len(truth),
        "a_fbeta": fbeta(set(a_pool), truth),
        "b_best": b.best_length, "b_gap_pct": 100 * (b.best_length / reference - 1),
        "b_distance_hit": int(b.best_length == reference), "b_routes": len(b.routes),
        "b_hits": bh, "b_precision": bh / len(b.routes), "b_recall": bh / len(truth),
        "b_fbeta": fbeta(set(b.routes), truth),
        "a_checks": sum(item.pair_checks for item in refined.values()),
        "b_checks": b.checks, "b_expanded_routes": b.attempts,
        "reference_length": reference, "truth_count": len(truth)}
    return record, rows, provenance
