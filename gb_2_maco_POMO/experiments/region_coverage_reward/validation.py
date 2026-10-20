"""Independent validation under one fixed inference and search protocol."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import torch

from gb_2_maco_POMO.experiments.learnable_ball_bias.rollout import generate_routes
from gb_2_maco_POMO.experiments.learnable_ball_bias.validation import heuristic_reference
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, equal_plateau, tour_length
from gb_2_maco_POMO.runner import prepare_tsp


def validation_references(root: Path, instances):
    names = {item.path.name for item in instances}
    proofs = {}
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "validation" or row["relative_path"].split("/")[-1] not in names:
                continue
            if row.get("multiopt_certified") == "1" and row.get("proven_optimal_length"):
                if not (root / row["proof_file"]).is_file():
                    raise RuntimeError(f"missing validation proof: {row['proof_file']}")
                proofs[Path(row["relative_path"]).name] = int(row["proven_optimal_length"])
    results = {}
    for item in instances:
        results[item.path.name] = {"certified": proofs.get(item.path.name),
                                   "heuristic": heuristic_reference(prepare_tsp(item.path).distance_matrix)}
    if len(results) != 100 or len(proofs) != 56:
        raise RuntimeError("expected 100 independent validation cases, 56 with proof")
    return results


def observed_components(routes, distance):
    """Count equal-swap connected components in the routes actually observed."""
    keys = set(routes)
    visited = set()
    components = 0
    for first in keys:
        if first in visited:
            continue
        components += 1
        stack = [first]
        visited.add(first)
        while stack:
            node = stack.pop()
            path = list(node)
            n = len(path)
            for i in range(n - 2):
                a, b = path[i], path[i + 1]
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    c, d = path[j], path[(j + 1) % n]
                    if distance[a][c] + distance[b][d] != distance[a][b] + distance[c][d]:
                        continue
                    neighbor = canonical(path[:i + 1] + list(reversed(path[i + 1:j + 1])) + path[j + 1:])
                    if neighbor in keys and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
    return components


@torch.inference_mode()
def evaluate(model, instances, references, device, *, seed, samples=8, optb_checks=50000):
    rows = []
    for case_index, item in enumerate(instances, 1):
        problem = prepare_tsp(item.path)
        distance = problem.distance_matrix
        raw = generate_routes(model, problem, samples=samples,
                              seed=seed + case_index, device=device)
        lengths = [tour_length(route, distance) for route in raw]
        candidates = {}
        for index in sorted(range(len(raw)), key=lambda i: (lengths[i], i)):
            candidates.setdefault(canonical(raw[index]), raw[index])
        refined = [two_opt(route, distance) for route in candidates.values()]
        best = min([*lengths, *(x.length for x in refined)])
        a_pool = {canonical(route): route for route, length in zip(raw, lengths) if length == best}
        a_pool.update({canonical(x.route): x.route for x in refined if x.length == best})
        b = equal_plateau(list(a_pool.values()), distance, optb_checks)
        reference = references[item.path.name]
        certified = reference["certified"]
        if certified is not None and b.best_length < certified:
            raise RuntimeError(f"below certified optimum: {item.path.name}")
        high_quality = (b.best_length == certified if certified is not None
                        else b.best_length <= reference["heuristic"])
        regions = observed_components(b.routes, distance) if high_quality else 0
        best_entries = [canonical(x.route) for x in refined if x.length == best]
        repeated_best_entries = len(best_entries) - len(set(best_entries))
        rows.append({"instance": item.path.name, "city_count": item.city_count,
                     "raw_best": min(lengths), "two_opt_best": best, "optb_best": b.best_length,
                     "reference_kind": "certified" if certified is not None else "fixed_heuristic",
                     "reference_length": certified if certified is not None else reference["heuristic"],
                     "certified_hit": int(certified is not None and b.best_length == certified),
                     "high_quality": int(high_quality), "a_routes": len(a_pool),
                     "b_routes": len(b.routes), "observed_regions": regions,
                     "repeated_best_entries": repeated_best_entries,
                     "optb_checks": b.checks})
    known = [r for r in rows if r["reference_kind"] == "certified"]
    if len(known) != 56:
        raise RuntimeError("certified validation count changed")
    mean = statistics.mean
    metrics = {
        "cases": len(rows), "certified_cases": len(known),
        "certified_hits": sum(r["certified_hit"] for r in known),
        "mean_certified_gap_pct": mean(100 * (r["two_opt_best"] / r["reference_length"] - 1)
                                       for r in known),
        "mean_raw_relative_pct": mean(100 * (r["raw_best"] / r["reference_length"] - 1)
                                      for r in rows),
        "mean_two_opt_relative_pct": mean(100 * (r["two_opt_best"] / r["reference_length"] - 1)
                                          for r in rows),
        "high_quality_cases": sum(r["high_quality"] for r in rows),
        "mean_certified_regions": mean(r["observed_regions"] if r["certified_hit"] else 0
                                       for r in known),
        "mean_certified_routes": mean(r["b_routes"] if r["certified_hit"] else 0
                                      for r in known),
        "mean_high_quality_regions": mean(r["observed_regions"] for r in rows),
        "mean_high_quality_routes": mean(r["b_routes"] if r["high_quality"] else 0
                                         for r in rows),
        "mean_repeated_best_entries": mean(r["repeated_best_entries"] for r in rows),
    }
    return rows, metrics


def choose_checkpoint(records, *, baseline, gap_tolerance_pct=0.5, allowed_hit_loss=1):
    """Fix the gate at epoch zero; maximize certified coverage within it."""
    eligible = [r for r in records if
                r["metrics"]["mean_certified_gap_pct"] <= baseline["mean_certified_gap_pct"] + gap_tolerance_pct
                and r["metrics"]["certified_hits"] >= baseline["certified_hits"] - allowed_hit_loss]
    if not eligible:
        return records[0]
    return max(eligible, key=lambda r: (
        r["metrics"]["mean_certified_regions"],
        r["metrics"]["mean_certified_routes"],
        r["metrics"]["certified_hits"],
        -r["metrics"]["mean_certified_gap_pct"], -r["epoch"]))
