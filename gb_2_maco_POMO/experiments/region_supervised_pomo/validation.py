"""Independent 120-case validation and a fixed quality-gated selector."""

from __future__ import annotations

import statistics

import torch

from gb_maco_POMO.experiments.compare import fbeta
from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import generate_learned_routes
from gb_2_maco_POMO.experiments.learnable_ball_bias.validation import heuristic_reference
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, equal_plateau, tour_length
from gb_2_maco_POMO.runner import prepare_tsp

from .config import OPTB_CHECKS, SAMPLES_PER_START


@torch.inference_mode()
def evaluate(model, instances, labels, inference_seed):
    model.eval()
    rows = []
    for index, item in enumerate(instances, 1):
        problem = prepare_tsp(item.path)
        distance = problem.distance_matrix
        raw, _ = generate_learned_routes(
            model, problem, list(range(item.city_count)), SAMPLES_PER_START,
            inference_seed + index)
        lengths = [tour_length(route, distance) for route in raw]
        candidates = {}
        for i in sorted(range(len(raw)), key=lambda k: (lengths[k], k)):
            candidates.setdefault(canonical(raw[i]), raw[i])
        refined = [two_opt(route, distance) for route in candidates.values()]
        a_best = min([*lengths, *(x.length for x in refined)])
        a_routes = {canonical(route): route for route, length in zip(raw, lengths)
                    if length == a_best}
        a_routes.update({canonical(x.route): x.route for x in refined if x.length == a_best})
        b = equal_plateau(list(a_routes.values()), distance, OPTB_CHECKS)
        label = labels.get(item.path)
        certified = label.optimal_length if label else None
        reference = certified if certified is not None else heuristic_reference(distance)
        if certified is not None and b.best_length < certified:
            raise RuntimeError(f"below certified optimum: {item.path}")
        known = set(label.tours) if label else set()
        region_sets = [{label.tours[i] for i in region} for region in label.regions] if label else []
        a_known = set(a_routes) & known
        b_known = set(b.routes) & known
        a_regions = sum(bool(a_known & region) for region in region_sets)
        b_regions = sum(bool(b_known & region) for region in region_sets)
        full_fbeta = (fbeta(set(b.routes), known) if label and label.complete else None)
        rows.append({
            "instance": item.path.name, "city_count": item.city_count,
            "distribution": item.distribution, "source": "original" if index <= 100 else "new",
            "reference_kind": "certified" if label else "heuristic_upper_bound",
            "reference_length": reference,
            "raw_best": min(lengths), "two_opt_best": a_best, "optb_best": b.best_length,
            "raw_gap_pct": 100 * (min(lengths) / certified - 1) if label else None,
            "two_opt_gap_pct": 100 * (a_best / certified - 1) if label else None,
            "optb_gap_pct": 100 * (b.best_length / certified - 1) if label else None,
            "certified_hit": int(certified is not None and b.best_length == certified),
            "raw_unique": len(candidates), "raw_count": len(raw),
            "two_opt_routes": len(a_routes), "optb_routes": len(b.routes),
            "a_known_tours": len(a_known), "b_known_tours": len(b_known),
            "a_known_regions": a_regions, "b_known_regions": b_regions,
            "known_regions": len(region_sets), "known_tours": len(known),
            "full_optimal_set": int(bool(label and label.complete)),
            "full_fbeta": full_fbeta, "optb_checks": b.checks,
        })
    return rows, summarize(rows)


def summarize(rows):
    certified = [r for r in rows if r["reference_kind"] == "certified"]
    complete = [r for r in certified if r["full_optimal_set"]]
    if (len(rows), len(certified), len(complete)) != (120, 76, 31):
        raise RuntimeError("validation population changed")
    by_size = {}
    for size in sorted({r["city_count"] for r in rows}):
        group = [r for r in rows if r["city_count"] == size]
        proved = [r for r in group if r["reference_kind"] == "certified"]
        by_size[str(size)] = {"cases": len(group), "certified": len(proved),
                              "certified_hits": sum(r["certified_hit"] for r in proved),
                              "mean_certified_gap_pct": (statistics.mean(r["optb_gap_pct"] for r in proved)
                                                          if proved else None)}
    return {
        "cases": len(rows), "certified_cases": len(certified),
        "certified_hits": sum(r["certified_hit"] for r in certified),
        "mean_certified_gap_pct": statistics.mean(r["optb_gap_pct"] for r in certified),
        "mean_full_region_coverage": statistics.mean(
            r["b_known_regions"] / r["known_regions"] for r in complete),
        "mean_known_region_coverage": statistics.mean(
            r["b_known_regions"] / r["known_regions"] for r in certified),
        "mean_full_fbeta": statistics.mean(r["full_fbeta"] for r in complete),
        "known_tours_found": sum(r["b_known_tours"] for r in certified),
        "known_regions_found": sum(r["b_known_regions"] for r in certified),
        "known_regions_listed": sum(r["known_regions"] for r in certified),
        "by_city_count": by_size,
    }


def choose_checkpoint(records):
    """Protect epoch-0 distance quality, then rank complete-region coverage."""
    baseline = records[0]["metrics"]
    eligible = [record for record in records if
                record["metrics"]["certified_hits"] >= baseline["certified_hits"] - 1
                and record["metrics"]["mean_certified_gap_pct"]
                <= baseline["mean_certified_gap_pct"] + 0.5]
    if not eligible:
        return records[0]
    return max(eligible, key=lambda record: (
        record["metrics"]["mean_full_region_coverage"],
        record["metrics"]["mean_known_region_coverage"],
        record["metrics"]["certified_hits"],
        -record["metrics"]["mean_certified_gap_pct"], -record["epoch"]))
