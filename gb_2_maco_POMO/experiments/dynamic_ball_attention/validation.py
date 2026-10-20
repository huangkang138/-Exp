"""Select checkpoints on the original 100-case validation split only."""

from __future__ import annotations

import statistics

import torch

from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import generate_learned_routes
from gb_2_maco_POMO.experiments.learnable_ball_bias.validation import (
    certified_lengths, heuristic_reference,
)
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, tour_length
from gb_2_maco_POMO.runner import prepare_tsp

from .config import DATA

VALIDATION_SEED = 20261018


@torch.inference_mode()
def evaluate(model, instances, device):
    """Fixed 1+8 all-city sampling then ordinary 2-opt; no test solutions used."""
    model.eval()
    certified = certified_lengths(DATA)
    rows = []
    for index, item in enumerate(instances, 1):
            problem = prepare_tsp(item.path)
            distance = problem.distance_matrix
            routes, _ = generate_learned_routes(
                model, problem, list(range(item.city_count)), 8, VALIDATION_SEED + index)
            raw = min(tour_length(route, distance) for route in routes)
            unique = {canonical(route): route for route in routes}
            refined = {canonical(result.route): result.length
                       for route in unique.values()
                       for result in (two_opt(route, distance),)}
            best = min(refined.values())
            optimum = certified.get(item.path.name)
            reference = optimum if optimum is not None else heuristic_reference(distance)
            if optimum is not None and best < optimum:
                raise RuntimeError(f"validation below certified optimum: {item.path}")
            rows.append({"instance": item.path.name, "city_count": item.city_count,
                         "distribution": item.distribution,
                         "reference_kind": "certified" if optimum is not None else "heuristic_upper_bound",
                         "reference_length": reference, "raw_best": raw,
                         "two_opt_best": best, "raw_relative_reference_pct":
                         100 * (raw / reference - 1), "two_opt_relative_reference_pct":
                         100 * (best / reference - 1),
                         "raw_unique": len(unique), "two_opt_unique": len(refined),
                         "two_opt_best_tours": sum(length == best for length in refined.values()),
                         "high_quality_tours": sum(length <= 1.01 * reference
                                                   for length in refined.values())})
    if len(rows) != 100:
        raise RuntimeError("validation split is incomplete")
    metrics = {"cases": len(rows), "certified_cases": sum(r["reference_kind"] == "certified"
                                                      for r in rows),
               "mean_raw_relative_reference_pct": statistics.mean(
                   r["raw_relative_reference_pct"] for r in rows),
               "mean_two_opt_relative_reference_pct": statistics.mean(
                   r["two_opt_relative_reference_pct"] for r in rows),
               "mean_two_opt_best_tours": statistics.mean(r["two_opt_best_tours"] for r in rows),
               "mean_high_quality_tours": statistics.mean(r["high_quality_tours"] for r in rows)}
    return rows, metrics


def choose(records):
    """Frozen selection: lower 2-opt reference gap, then raw gap, then earlier epoch."""
    return min(records, key=lambda record: (
        record["metrics"]["mean_two_opt_relative_reference_pct"],
        record["metrics"]["mean_raw_relative_reference_pct"], record["epoch"]))
