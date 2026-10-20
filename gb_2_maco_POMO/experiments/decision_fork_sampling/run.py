"""Run the seed-42 inference-only S0/S2/S3/S4 fork comparison.

Usage: python -m gb_2_maco_POMO.experiments.decision_fork_sampling.run --output-dir NEW_DIR
The output directory must not exist. No training code is imported or invoked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from gb_maco_POMO.experiments.compare import BENCHMARKS, load_ground_truth
from gb_2_maco_POMO.experiments.hierarchical import original25
from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import load_checkpoint
from gb_2_maco_POMO.experiments.region_supervised_pomo.config import INITIAL, INITIAL_SHA256
from gb_2_maco_POMO.experiments.region_supervised_pomo.data import _project_components
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, tour_length
from gb_2_maco_POMO.runner import prepare_tsp

from .routes import generate_case
from .report import write_report

PROJECT = Path(__file__).resolve().parents[3]
HISTORY = PROJECT / "gb_2_maco_POMO/results/learnable_ball_test25_20261015"
DEFAULT_OUTPUT = PROJECT / "gb_2_maco_POMO/results/decision_fork_sampling/seed_42"
GROUPS = ("S0", "S2", "S3", "S4")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv(path: Path, rows):
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _gate() -> dict:
    protocol = _json(HISTORY / "protocol.json")
    expected = {"trained_checkpoint_sha256": INITIAL_SHA256, "device": "cpu",
                "seed": 42, "starts": "all", "argmax": 1, "sampling": 8,
                "optb_extra_pair_checks_per_case": 50_000, "beta_squared": 0.3}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise RuntimeError("historical S0 protocol differs from the requested comparison")
    if Path(protocol["trained_checkpoint"]).resolve() != INITIAL.resolve():
        raise RuntimeError("historical S0 used a different checkpoint path")
    digest = hashlib.sha256(INITIAL.read_bytes()).hexdigest()
    if digest != INITIAL_SHA256:
        raise RuntimeError("A0 checkpoint SHA256 mismatch")
    if len(list((HISTORY / "learned_ball/cases").glob("MSTSP*.json"))) != 25:
        raise RuntimeError("historical S0 cases are incomplete")
    for index in range(1, 26):
        name = f"MSTSP{index}"
        for part in (f"learned_ball/raw_routes/{name}.json",
                     f"learned_ball/best_routes/{name}.json"):
            if not (HISTORY / part).is_file():
                raise FileNotFoundError(HISTORY / part)
    return protocol


def _endpoints(routes, distance, truth, regions):
    """Use ordinary project 2-opt on raw routes; preserve one row per generated route."""
    region_for = {tour: i for i, region in enumerate(regions) for tour in region}
    rows = []
    for index, route in enumerate(routes):
        result = two_opt(route, distance)
        endpoint = canonical(result.route)
        rows.append({"route_index": index, "start": route[0],
                     "raw_length": tour_length(route, distance),
                     "raw_canonical": json.dumps(canonical(route)),
                     "two_opt_length": result.length,
                     "two_opt_endpoint": json.dumps(endpoint),
                     "two_opt_true_optimal": int(endpoint in truth),
                     "two_opt_region": region_for.get(endpoint, "")})
    return rows


def _coverage(path: Path, truth, regions):
    saved = _json(path)
    a = {tuple(route) for route in saved["A"]["routes"]} & truth
    b = {tuple(route) for route in saved["B"]["routes"]} & truth
    return a, b, {i for i, region in enumerate(regions) if b & region}


def run(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    historical_protocol = _gate()
    original25.SEED = 42
    policy, _ = load_checkpoint(INITIAL)
    policy.eval()
    torch.set_num_threads(1)
    config = {"checkpoint": str(INITIAL.resolve()), "checkpoint_sha256": INITIAL_SHA256,
              "historical_s0": str(HISTORY.resolve()),
              "historical_protocol_sha256": hashlib.sha256((HISTORY / "protocol.json").read_bytes()).hexdigest(),
              "device": "cpu", "seed": 42, "case_seed": "42 + case index",
              "starts": "all cities", "routes_per_start": 9,
              "groups": {"S0": "1 argmax + 8 sampling",
                         "S2": "1 argmax + 4 sampling + 4 city forks",
                         "S3": "1 argmax + 4 sampling + 4 ball forks",
                         "S4": "1 argmax + 4 sampling + 2 city forks + 2 ball forks"},
              "reference_rounds": [1, 2, 3, 4],
              "fork_rule": "one legal alternative, probability-renormalized, then ordinary two-level sampling",
              "two_opt": historical_protocol["two_opt"],
              "optb_pair_checks_per_case": 50_000, "beta_squared": 0.3,
              "torch_version": torch.__version__}
    output.mkdir(parents=True)
    _save_json(output / "config.json", config)
    all_cases, all_endpoints, all_forks, comparisons, timing = [], [], [], [], []
    paper = original25._csv(original25.PAPER)
    for index, (stem, n, _) in enumerate(BENCHMARKS, 1):
        name = f"MSTSP{index}"
        prepared = prepare_tsp(original25.DATA / f"{stem}.tsp")
        if len(prepared.cities) != n:
            raise RuntimeError(f"{name}: benchmark city count mismatch")
        generated = generate_case(policy, prepared, 42 + index)
        historical_raw = (HISTORY / "learned_ball/raw_routes" / f"{name}.json").read_bytes()
        raw = json.dumps(generated.groups["S0"], separators=(",", ":")).encode("utf-8")
        if raw != historical_raw:
            raise RuntimeError(f"{name}: S0 raw routes differ; stop before comparing forks")
        old_row = _json(HISTORY / "learned_ball/cases" / f"{name}.json")
        if hashlib.sha256(raw).hexdigest() != old_row["raw_sha256"]:
            raise RuntimeError(f"{name}: S0 historical row/hash mismatch")
        _save_json(output / "reference_traces" / f"{name}.json", generated.reference_traces)
        group_inference = {"S0": generated.timing["s0_seconds"],
            "S2": generated.timing["common_seconds"] + sum(generated.timing["city_slot_seconds"]),
            "S3": generated.timing["common_seconds"] + sum(generated.timing["ball_slot_seconds"]),
            "S4": generated.timing["common_seconds"] +
                  sum(generated.timing["city_slot_seconds"][:2]) +
                  sum(generated.timing["ball_slot_seconds"][:2])}
        truth = load_ground_truth(original25.DATA / f"{stem}.solution", n)
        truth_tours = tuple(sorted(truth))
        regions = tuple({truth_tours[j] for j in part}
                        for part in _project_components(truth_tours, prepared.distance_matrix))
        by_group = {"S0": old_row}
        by_endpoint = {}
        by_cover = {}
        for group in GROUPS:
            routes = generated.groups[group]
            if len(routes) != 9 * n or any(sorted(route) != list(range(n)) for route in routes):
                raise RuntimeError(f"{name} {group}: invalid route count or permutation")
            if group != "S0":
                label_dir = output / group
                for folder in ("raw_routes", "best_routes"):
                    (label_dir / folder).mkdir(parents=True, exist_ok=True)
                def cached_generator(_policy, _prepared, starts, samples, seed, *, _routes=routes):
                    if starts != list(range(n)) or samples != 8 or seed != 42 + index:
                        raise RuntimeError("original25.run_case inference contract changed")
                    return _routes, None
                row = original25.run_case(policy, index, group, 0.0, output,
                                          float(paper[name]["paper_fbeta"]),
                                          route_generator=cached_generator)
                if row["b_checks"] > 50_000:
                    raise RuntimeError(f"{name} {group}: Opt-B budget exceeded")
                row["inference_seconds"] = group_inference[group]
                row["case_seconds"] = (row["prepare_seconds"] + group_inference[group] +
                                       row["postprocess_seconds"])
                by_group[group] = row
                _save_json(output / group / "cases" / f"{name}.json", row)
            endpoints = _endpoints(routes, prepared.distance_matrix, truth, regions)
            for endpoint in endpoints:
                all_endpoints.append({"instance": name, "group": group, **endpoint})
            by_endpoint[group] = {row["two_opt_endpoint"] for row in endpoints}
            best = (HISTORY / "learned_ball/best_routes" / f"{name}.json") if group == "S0" else (
                output / group / "best_routes" / f"{name}.json")
            by_cover[group] = _coverage(best, truth, regions)
            if len(by_cover[group][0]) != by_group[group]["a_hits"] or len(by_cover[group][1]) != by_group[group]["b_hits"]:
                raise RuntimeError(f"{name} {group}: saved true-optimal coverage differs from case metrics")
            all_cases.append({"group": group, **by_group[group],
                "raw_unique": len({canonical(route) for route in routes}),
                "two_opt_unique_endpoints": len(by_endpoint[group]),
                "measured_inference_seconds": group_inference[group],
                "regions_hit": len(by_cover[group][2]), "truth_regions": len(regions)})
        base_endpoints = by_endpoint["S0"]
        base_tours = by_cover["S0"][1]
        base_regions = by_cover["S0"][2]
        for group in GROUPS[1:]:
            tours = by_cover[group][1]
            areas = by_cover[group][2]
            comparisons.append({"instance": name, "group": group,
                "new_two_opt_endpoints": len(by_endpoint[group] - base_endpoints),
                "new_nonoptimal_two_opt_endpoints": sum(
                    tuple(json.loads(item)) not in truth
                    for item in by_endpoint[group] - base_endpoints),
                "new_true_optimal_tours": len(tours - base_tours),
                "lost_true_optimal_tours": len(base_tours - tours),
                "new_regions": len(areas - base_regions),
                "lost_regions": len(base_regions - areas)})
        endpoint_lookup = {(r["group"], r["route_index"]): r for r in all_endpoints
                           if r["instance"] == name}
        for record in generated.fork_records:
            slot, start, kind = record["slot"], record["start"], record["kind"]
            group = "S2" if kind == "city" else "S3"
            endpoint = endpoint_lookup[(group, (5 + slot) * n + start)]
            all_forks.append({"instance": name,
                "groups": f"{group},S4" if slot < 2 else group, **{
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in record.items()},
                "raw_length": endpoint["raw_length"],
                "two_opt_length": endpoint["two_opt_length"],
                "two_opt_endpoint": endpoint["two_opt_endpoint"],
                "two_opt_true_optimal": endpoint["two_opt_true_optimal"]})
        timing.append({"instance": name, "s0_inference_seconds": generated.timing["s0_seconds"],
                       "all_forks_seconds": generated.timing["fork_seconds"],
                       **{f"{group}_inference_seconds": group_inference[group] for group in GROUPS},
                       **{f"{group}_two_opt_seconds": by_group[group]["two_opt_seconds"]
                          for group in GROUPS},
                       **{f"{group}_optb_seconds": by_group[group]["optb_seconds"]
                          for group in GROUPS}})
        print(f"{name}: " + " ".join(f"{g}={by_group[g]['b_best']}/{by_group[g]['b_hits']}"
                                    for g in GROUPS), flush=True)
    _csv(output / "per_instance.csv", all_cases)
    _csv(output / "route_endpoints.csv", all_endpoints)
    _csv(output / "fork_records.csv", all_forks)
    _csv(output / "coverage_comparison.csv", comparisons)
    _csv(output / "timing.csv", timing)
    summaries = []
    for group in GROUPS:
        rows = [r for r in all_cases if r["group"] == group]
        summaries.append({"group": group,
            "distance_hits": sum(r["b_distance_hit"] for r in rows),
            "raw_gap_mean_pct": sum(r["raw_gap_pct"] for r in rows) / 25,
            "two_opt_tours": sum(r["a_hits"] for r in rows),
            "optb_tours": sum(r["b_hits"] for r in rows),
            "regions_hit": sum(r["regions_hit"] for r in rows),
            "mean_fbeta": sum(r["b_fbeta"] for r in rows) / 25,
            "raw_unique": sum(r["raw_unique"] for r in rows),
            "two_opt_unique_endpoints": sum(r["two_opt_unique_endpoints"] for r in rows),
            "fork_success": sum(r["status"] == "forked" and (
                group == "S4" and r["slot"] < 2 or
                group == "S2" and r["kind"] == "city" or
                group == "S3" and r["kind"] == "ball") for r in all_forks),
            "inference_seconds": sum(r[f"{group}_inference_seconds"] for r in timing),
            "postprocess_seconds": sum(r["two_opt_seconds"] + r["optb_seconds"] for r in rows)})
        summaries[-1]["total_seconds"] = (summaries[-1]["inference_seconds"] +
                                         summaries[-1]["postprocess_seconds"])
    _csv(output / "summary.csv", summaries)
    write_report(output, summaries, all_cases, comparisons, timing, all_forks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
