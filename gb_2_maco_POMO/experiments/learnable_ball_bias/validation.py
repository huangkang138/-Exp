"""Independent validation; certified optimum and heuristic reference stay distinct."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from gb_2_maco_POMO.runner import prepare_tsp
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, equal_plateau, tour_length
from .rollout import generate_routes


def certified_lengths(data_root: Path) -> dict[str, int]:
    result = {}
    with (data_root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "validation" or not row.get("proven_optimal_length"):
                continue
            proof = row.get("proof_file", "")
            if row.get("multiopt_certified") != "1" or not proof or not (data_root / proof).is_file():
                raise ValueError(f"unverified reference for {row['relative_path']}")
            result[Path(row["relative_path"]).name] = int(row["proven_optimal_length"])
    return result


def heuristic_reference(distance):
    """Nearest neighbor then existing complete 2-opt; explicitly not an optimum."""
    remaining = set(range(1, len(distance)))
    route = [0]
    while remaining:
        city = min(remaining, key=lambda j: (distance[route[-1]][j], j))
        route.append(city)
        remaining.remove(city)
    return two_opt(route, distance).length


@torch.inference_mode()
def evaluate(model, instances, data_root: Path, device: torch.device, *,
             samples: int, optb_checks: int, seed: int):
    """Every candidate checkpoint gets exactly the same validation cases/budget."""
    certified = certified_lengths(data_root)
    rows = []
    for index, item in enumerate(instances, 1):
        problem = prepare_tsp(item.path)
        distance = problem.distance_matrix
        raw = generate_routes(model, problem, samples=samples, seed=seed + index,
                              device=device)
        raw_lengths = [tour_length(route, distance) for route in raw]
        candidates = {}
        for route in sorted(raw, key=lambda r: tour_length(r, distance)):
            candidates.setdefault(canonical(route), route)
        refined = [two_opt(route, distance) for route in candidates.values()]
        best = min([*raw_lengths, *(x.length for x in refined)])
        a_pool = {canonical(route): route for route, length in zip(raw, raw_lengths)
                  if length == best}
        a_pool.update({canonical(x.route): x.route for x in refined if x.length == best})
        b = equal_plateau(list(a_pool.values()), distance, optb_checks)
        proof = certified.get(item.path.name)
        if proof is not None and best < proof:
            raise RuntimeError(f"route below certified optimum: {item.path.name}")
        heuristic = heuristic_reference(distance)
        rows.append({"instance": item.path.name, "city_count": item.city_count,
                     "raw_best": min(raw_lengths), "two_opt_best": best,
                     "optb_best": b.best_length, "a_routes": len(a_pool),
                     "b_routes": len(b.routes), "optb_checks": b.checks,
                     "heuristic_reference": heuristic, "certified_length": proof,
                     "certified_hit": int(proof is not None and best == proof),
                     "certified_optb_routes": len(b.routes) if proof is not None
                     and b.best_length == proof else 0})
    known = [r for r in rows if r["certified_length"] is not None]
    if len(known) != len(certified):
        raise RuntimeError("not all certified validation cases were evaluated")
    mean = lambda values: sum(values) / len(values)
    metrics = {
        "cases": len(rows), "certified_cases": len(known),
        "certified_hits": sum(r["certified_hit"] for r in known),
        "mean_certified_gap_pct": (mean([100 * (r["two_opt_best"] /
                                           r["certified_length"] - 1) for r in known])
                                   if known else None),
        "mean_certified_optb_routes": (mean([r["certified_optb_routes"] for r in known])
                                       if known else None),
        "mean_heuristic_relative_2opt_pct": mean([100 * (r["two_opt_best"] /
                                                        r["heuristic_reference"] - 1)
                                                    for r in rows]),
        "mean_heuristic_relative_raw_pct": mean([100 * (r["raw_best"] /
                                                      r["heuristic_reference"] - 1)
                                                  for r in rows]),
    }
    return rows, metrics


def select_checkpoints(records, *, quality_gap_tolerance_pct: float,
                       allowed_hit_loss: int):
    """Rules are fixed before any final-test evaluation."""
    if not records:
        raise ValueError("no validation checkpoints")
    certified_counts = {r["metrics"].get("certified_cases", 1) for r in records}
    if certified_counts == {0}:
        # Ordinary MATSP has no proof: retain a length-based checkpoint only.
        length = min(records, key=lambda r: (
            r["metrics"]["mean_heuristic_relative_2opt_pct"],
            r["metrics"]["mean_heuristic_relative_raw_pct"], r["epoch"]))
        return length, length
    if 0 in certified_counts:
        raise ValueError("inconsistent certified validation coverage across checkpoints")
    def length_key(record):
        m = record["metrics"]
        return (m["mean_certified_gap_pct"], -m["certified_hits"],
                m["mean_heuristic_relative_2opt_pct"],
                m["mean_heuristic_relative_raw_pct"], record["epoch"])
    length = min(records, key=length_key)
    baseline = length["metrics"]
    eligible = [record for record in records if
                record["metrics"]["mean_certified_gap_pct"] <=
                baseline["mean_certified_gap_pct"] + quality_gap_tolerance_pct and
                record["metrics"]["certified_hits"] >=
                baseline["certified_hits"] - allowed_hit_loss]
    multi = min(eligible, key=lambda r: (
        -r["metrics"]["mean_certified_optb_routes"],
        -r["metrics"]["certified_hits"],
        r["metrics"]["mean_certified_gap_pct"], r["epoch"]))
    return length, multi
