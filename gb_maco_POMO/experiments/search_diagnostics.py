"""Trace route generation, full 2-opt convergence and Opt-B without changing search rules."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from uuid import uuid4

import torch

from ..pomo import search_routes
from ..runner import prepare_tsp
from .compare import BENCHMARKS, load_ground_truth, load_policy, read_optimum
from .compare_learnable_bias import (
    BASELINE, BASELINE_SUMMARY, BENCHMARK_DIR, PROJECT, canonical,
    equal_plateau, fbeta,
)
from .diagnose_2opt import _select_candidates
from .local_search_2opt import two_opt
from .search_diagnostics_optb import discovered_components, replay


RESULTS = PROJECT / "gb_maco_POMO/results"
SOURCES = {
    "fixed": {
        "checkpoint": RESULTS / "checkpoints/run_20261013_230742_828b99/best.pt",
        "sha256": "dbdfc5cb2faef0d7cbecb7c5c7cc77db16049e5d10e5d584a26e288f1bd88410",
        "comparison": BASELINE.parent,
    },
    "learned": {
        "checkpoint": RESULTS / "learnable_bias/checkpoints/run_20261014_223323_7de9e6/best.pt",
        "sha256": "a45a4231997e588440d863c667cb2ff88bdba316e8b6cacb234b052be8909039",
        "comparison": RESULTS / "learnable_bias/comparison/run_20261014_233707",
    },
    "finetuned": {
        "checkpoint": RESULTS / "bias_head_finetune/checkpoints/run_20261015_004237_9fc2bf/best.pt",
        "sha256": "a1610cd4e5cb6485f98bc17597006af8e122879be3b2d2c06407f05eacc27a25",
        "comparison": RESULTS / "bias_head_finetune/comparison/run_20261015_005043_1c12c2",
    },
}
TARGETED = (16, 18, 20, 21, 22, 23, 24, 25)
SEEDS = (42, 123, 2026)
PER_FIELDS = (
    "model", "instance", "seed", "source", "city_count", "reference_length", "truth_count",
    "raw_total", "raw_unique", "raw_duplicate_rate", "raw_length_min", "raw_length_mean",
    "raw_length_std", "edge_jaccard_mean", "edge_jaccard_std", "edge_pairs_sampled",
    "argmax_unique", "sampling_unique", "argmax_only", "sampling_only", "shared_raw",
    "multi_start_raw", "same_start_unique_mean", "post_unique", "post_optimal_unique",
    "post_true_optimal_unique", "post_basin_max", "post_basin_multi", "post_basin_max_share",
    "post_argmax_only_optimal", "post_sampling_only_optimal", "post_shared_optimal",
    "a_best", "a_start_count", "a_hits", "b_best", "b_routes", "b_hits",
    "b_new_routes", "b_new_hits", "b_fbeta", "b_checks", "b_budget_hit", "b_attempts",
    "discovered_components", "largest_component", "gap_pct",
)
MAP_FIELDS = (
    "model", "instance", "seed", "raw_index", "mode", "sampling_round", "pomo_start",
    "raw_route", "raw_length", "raw_unique_rank", "post_route", "post_length",
    "post_is_reference_length", "post_in_ground_truth", "post_basin_size",
)
OPTB_FIELDS = (
    "model", "instance", "seed", "seed_index", "seed_route", "global_first_discoveries",
    "global_new_discoveries", "global_checks", "budget_hit", "isolated_reachable",
    "isolated_new_beyond_all_starts", "isolated_overlap_mean",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route_json(key: tuple[int, ...]) -> str:
    return json.dumps(key, separators=(",", ":"))


def _edges(route: list[int]) -> frozenset[tuple[int, int]]:
    return frozenset(tuple(sorted((route[i], route[(i + 1) % len(route)])))
                     for i in range(len(route)))


def _edge_similarity(unique: dict[tuple[int, ...], list[int]], seed: int) -> tuple[float, float, int]:
    edges = [_edges(route) for route in unique.values()]
    n = len(edges)
    if n < 2:
        return 1.0, 0.0, 0
    possible = n * (n - 1) // 2
    if possible <= 2_000:
        pairs = combinations(range(n), 2)
        count = possible
    else:
        rng = random.Random(seed)
        sampled = set()
        while len(sampled) < 2_000:
            a, b = rng.sample(range(n), 2)
            sampled.add((min(a, b), max(a, b)))
        pairs = sorted(sampled)
        count = 2_000
    values = [len(edges[a] & edges[b]) / len(edges[a] | edges[b]) for a, b in pairs]
    return statistics.mean(values), statistics.pstdev(values), count


def _read_history(model: str) -> tuple[dict[str, dict[str, str]], dict]:
    folder = SOURCES[model]["comparison"]
    with (folder / "per_instance.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if len(rows) != 25 or len({row["instance"] for row in rows}) != 25:
        raise ValueError(f"{model}: historical comparison lacks 25 unique cases")
    if model == "fixed":
        valid = (summary["checkpoint_sha256"] == SOURCES[model]["sha256"] and
                 summary["device"] == "cpu" and summary["seed"] == 42 and
                 summary["samples_per_start"] == 8 and
                 summary["extra_pair_check_ceiling_per_instance"] == 50_000)
    else:
        valid = (summary["learnable_checkpoint_sha256"] == SOURCES[model]["sha256"] and
                 summary["device"] == "cpu" and summary["seed"] == 42 and
                 summary["inference"] == "per start 1 argmax + 8 sampling" and
                 summary["local_search"] == "full 2-opt; Opt-B 50000 additional checks/case")
    if not valid or _sha(SOURCES[model]["checkpoint"]) != SOURCES[model]["sha256"]:
        raise ValueError(f"{model}: checkpoint or historical inference protocol mismatch")
    return {row["instance"]: row for row in rows}, summary


def _routes(model: str, index: int, seed: int, problem, policies, output: Path,
            old: dict[str, str]) -> tuple[list[list[int]], str]:
    name = f"MSTSP{index}"
    if seed == 42:
        path = SOURCES[model]["comparison"] / "raw_routes" / f"{name}.json"
        raw = path.read_bytes()
        if model == "fixed" and _sha(path) != old["raw_sha256"]:
            raise ValueError(f"{model} {name}: historical raw SHA mismatch")
        routes = json.loads(raw)
        source = "historical"
    else:
        if model not in policies:
            policies[model], _ = load_policy(SOURCES[model]["checkpoint"], torch.device("cpu"))
        search = search_routes(policies[model], problem, inference_mode="mixed",
                               samples_per_start=8, seed=seed + index)
        routes = search.routes
        source = "new_inference"
        target = output / "new_raw_routes" / model / str(seed)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{name}.json").write_text(json.dumps(routes, separators=(",", ":")), encoding="utf-8")
    if len(routes) != len(problem.cities) * 9 or any(
            len(route) != len(problem.cities) or set(route) != set(range(len(problem.cities)))
            for route in routes):
        raise ValueError(f"{model} {name} seed={seed}: invalid raw routes")
    return routes, source


def _case(model: str, index: int, seed: int, output: Path, policies,
          history: dict[str, dict[str, str]]) -> tuple[dict, list[dict], list[dict]]:
    stem, n, _ = BENCHMARKS[index - 1]
    name = f"MSTSP{index}"
    problem = prepare_tsp(BENCHMARK_DIR / f"{stem}.tsp")
    old = history[name]
    routes, source = _routes(model, index, seed, problem, policies, output, old)
    distance = problem.distance_matrix
    lengths = [sum(distance[route[i]][route[(i + 1) % n]] for i in range(n))
               for route in routes]
    unique_raw = {canonical(route): route for route in routes}
    edge_mean, edge_std, edge_pairs = _edge_similarity(unique_raw, seed + index)
    raw_by_start: dict[tuple[int, ...], set[int]] = defaultdict(set)
    same_start = defaultdict(set)
    for k, route in enumerate(routes):
        key = canonical(route)
        raw_by_start[key].add(k % n)
        same_start[k % n].add(key)
    argmax = {canonical(route) for route in routes[:n]}
    sampling = {canonical(route) for route in routes[n:]}
    candidates = _select_candidates(routes, lengths, n, len(routes), all_generated=True)
    if len(candidates) != len(unique_raw):
        raise RuntimeError(f"{model} {name}: 2-opt candidate selection lost a route")
    refined = [two_opt(route, problem.distance_matrix) for route in candidates]
    mapping = {canonical(raw): result for raw, result in zip(candidates, refined)}
    post_keys = {canonical(result.route) for result in refined}
    basin = Counter(canonical(result.route) for result in refined)
    solution = BENCHMARK_DIR / f"{stem}.solution"
    reference, truth = read_optimum(solution), load_ground_truth(solution, n)
    post_optimal = {canonical(result.route) for result in refined if result.length == reference}
    post_truth = post_optimal & truth
    arg_post = {canonical(mapping[key].route) for key in argmax}
    sampling_post = {canonical(mapping[key].route) for key in sampling}
    best = min([*lengths, *(result.length for result in refined)])
    best_routes = [route for route, length in zip(routes, lengths) if length == best]
    best_routes.extend(result.route for result in refined if result.length == best)
    a_pool = {canonical(route): route for route in best_routes}
    trace = replay(list(a_pool.values()), problem.distance_matrix)
    b_keys = set(trace.routes)
    components = discovered_components(trace, problem.distance_matrix)
    if best < reference or len(b_keys) < len(a_pool):
        raise RuntimeError(f"{model} {name}: invalid optimal distance or Opt-B result")
    if seed == 42:
        for field, actual in (("a_best", best), ("a_hits", len(set(a_pool) & truth)),
                              ("b_best", best), ("b_hits", len(b_keys & truth)),
                              ("b_checks", trace.checks), ("raw_routes", len(routes)),
                              ("unique_candidates", len(candidates))):
            if int(old[field]) != actual:
                raise RuntimeError(f"{model} {name}: replayed {field} differs from historical result")
        if abs(float(old["b_fbeta"]) - fbeta(b_keys, truth)) > 1e-12:
            raise RuntimeError(f"{model} {name}: replayed Fbeta differs from historical result")
    row = {
        "model": model, "instance": name, "seed": seed, "source": source,
        "city_count": n, "reference_length": reference, "truth_count": len(truth),
        "raw_total": len(routes), "raw_unique": len(unique_raw),
        "raw_duplicate_rate": 1 - len(unique_raw) / len(routes),
        "raw_length_min": min(lengths), "raw_length_mean": statistics.mean(lengths),
        "raw_length_std": statistics.pstdev(lengths), "edge_jaccard_mean": edge_mean,
        "edge_jaccard_std": edge_std, "edge_pairs_sampled": edge_pairs,
        "argmax_unique": len(argmax), "sampling_unique": len(sampling),
        "argmax_only": len(argmax - sampling), "sampling_only": len(sampling - argmax),
        "shared_raw": len(argmax & sampling),
        "multi_start_raw": sum(len(starts) > 1 for starts in raw_by_start.values()),
        "same_start_unique_mean": statistics.mean(len(same_start[start]) for start in range(n)),
        "post_unique": len(post_keys), "post_optimal_unique": len(post_optimal),
        "post_true_optimal_unique": len(post_truth),
        "post_basin_max": max(basin.values()), "post_basin_multi": sum(v > 1 for v in basin.values()),
        "post_basin_max_share": max(basin.values()) / len(candidates),
        "post_argmax_only_optimal": len((arg_post - sampling_post) & truth),
        "post_sampling_only_optimal": len((sampling_post - arg_post) & truth),
        "post_shared_optimal": len((arg_post & sampling_post) & truth),
        "a_best": best, "a_start_count": len(a_pool), "a_hits": len(set(a_pool) & truth),
        "b_best": best, "b_routes": len(b_keys), "b_hits": len(b_keys & truth),
        "b_new_routes": len(b_keys - set(a_pool)),
        "b_new_hits": len((b_keys - set(a_pool)) & truth), "b_fbeta": fbeta(b_keys, truth),
        "b_checks": trace.checks, "b_budget_hit": trace.checks == 50_000,
        "b_attempts": trace.attempts, "discovered_components": len(components),
        "largest_component": components[0], "gap_pct": 100 * (best / reference - 1),
    }
    ranks = {canonical(route): rank for rank, route in enumerate(candidates)}
    map_rows = []
    for k, (route, length) in enumerate(zip(routes, lengths)):
        key = canonical(route)
        result = mapping[key]
        post = canonical(result.route)
        map_rows.append({
            "model": model, "instance": name, "seed": seed, "raw_index": k,
            "mode": "argmax" if k < n else "sampling",
            "sampling_round": 0 if k < n else k // n,
            "pomo_start": k % n, "raw_route": _route_json(key), "raw_length": length,
            "raw_unique_rank": ranks[key], "post_route": _route_json(post),
            "post_length": result.length, "post_is_reference_length": result.length == reference,
            "post_in_ground_truth": post in truth, "post_basin_size": basin[post],
        })
    isolated = []
    if seed == 42 and index in TARGETED:
        isolated = [set(replay([route], problem.distance_matrix).routes) for route in a_pool.values()]
    opt_rows = []
    for seed_index, key in enumerate(trace.seed_keys):
        global_new = sum(trace.root_by_key[found] == seed_index for found in b_keys - set(a_pool))
        overlap = None
        if isolated and len(isolated) > 1:
            comparisons = [len(isolated[seed_index] & other) / len(isolated[seed_index] | other)
                           for j, other in enumerate(isolated) if j != seed_index]
            overlap = statistics.mean(comparisons)
        opt_rows.append({
            "model": model, "instance": name, "seed": seed, "seed_index": seed_index,
            "seed_route": _route_json(key),
            "global_first_discoveries": sum(root == seed_index for root in trace.root_by_key.values()),
            "global_new_discoveries": global_new, "global_checks": trace.checks,
            "budget_hit": trace.checks == 50_000,
            "isolated_reachable": len(isolated[seed_index]) if isolated else "",
            "isolated_new_beyond_all_starts": len(isolated[seed_index] - set(a_pool)) if isolated else "",
            "isolated_overlap_mean": overlap if overlap is not None else "",
        })
    return row, map_rows, opt_rows


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    histories = {model: _read_history(model)[0] for model in SOURCES}
    output.mkdir(parents=True)
    (output / "sources.json").write_text(json.dumps({
        "models": {model: {"checkpoint": str(source["checkpoint"]),
                           "checkpoint_sha256": source["sha256"],
                           "historical_comparison": str(source["comparison"])}
                   for model, source in SOURCES.items()},
        "seed_rule": "torch seed = base seed + MSTSP index; same base seed across models",
        "historical_base_seed": 42, "new_base_seeds": [123, 2026],
        "targeted_new_seed_cases": list(TARGETED), "device": "cpu",
        "inference": "1 Argmax + 8 Sampling per POMO start",
        "local_search": "full ordinary 2-opt; original Opt-B, 50000 checks per instance",
        "edge_similarity": "mean Jaccard on all pairs or deterministic 2000-pair sample",
        "isolated_optb": "seed=42, targeted cases only; independent 50000-check replay per Opt-B start",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    policies = {}
    rows = []
    with (output / "route_mapping.csv").open("w", encoding="utf-8-sig", newline="") as mapping_stream, \
         (output / "optb_expansion.csv").open("w", encoding="utf-8-sig", newline="") as optb_stream:
        mapping_writer = csv.DictWriter(mapping_stream, fieldnames=MAP_FIELDS)
        optb_writer = csv.DictWriter(optb_stream, fieldnames=OPTB_FIELDS)
        mapping_writer.writeheader()
        optb_writer.writeheader()
        for model in SOURCES:
            for seed in SEEDS:
                indices = range(1, 26) if seed == 42 else TARGETED
                for index in indices:
                    row, mapping, expansion = _case(model, index, seed, output, policies,
                                                     histories[model])
                    rows.append(row)
                    mapping_writer.writerows(mapping)
                    optb_writer.writerows(expansion)
                    print(f"{model} seed={seed} {row['instance']}: raw {row['raw_unique']}/{row['raw_total']}, "
                          f"2-opt {row['post_unique']}, A {row['a_hits']}, B {row['b_hits']}", flush=True)
    with (output / "per_instance.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    from .search_diagnostics_report import write_report
    write_report(rows, output)
    print(f"Saved: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = RESULTS / "search_diagnostics"
    output = args.output_dir or root / f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    run(output)


if __name__ == "__main__":
    main()
