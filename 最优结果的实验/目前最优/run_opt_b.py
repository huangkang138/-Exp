"""Run the saved Soft POMO -> mixed routes -> full 2-opt -> Opt-B pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

from opt_b_search import canonical, equal_plateau, tour_length
from report import write_report


PROJECT = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT / "gb_maco_POMO/results/checkpoints/run_20261013_230742_828b99/best.pt"
BENCHMARKS_DIR = PROJECT.parent / "GB-MACO_PPSN2026-main/PPSN_GB-MACO/benchmark_MSTSP"
BASELINE_CSV = PROJECT / (
    "gb_maco_POMO/results/ablation/"
    "run_20261013_230742_828b99_F_1plus8_cpu_2opt_top10/"
    "budget_sweep/all_argmax_sampling_unique/per_instance.csv"
)
PAPER_CSV = PROJECT / "reproduction/config/paper_results.csv"
CHECKPOINT_SHA256 = "dbdfc5cb2faef0d7cbecb7c5c7cc77db16049e5d10e5d584a26e288f1bd88410"
SEED = 42
SAMPLES_PER_START = 8
EXTRA_CHECKS = 50_000
FIELDS = (
    "instance", "city_count", "seed", "raw_routes", "unique_candidates", "raw_sha256",
    "reference_length", "truth_count", "mixed_best", "a_best", "a_routes", "a_hits",
    "a_precision", "a_recall", "a_fbeta", "a_checks", "a_seconds",
    "b_best", "b_routes", "b_hits", "b_precision", "b_recall", "b_fbeta",
    "b_new_routes", "b_checks", "b_expanded_routes", "b_seconds",
    "prepare_seconds", "pomo_seconds", "case_seconds", "paper_fbeta",
)


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["instance"]: row for row in csv.DictReader(stream)}


def _metrics(routes: dict[tuple[int, ...], list[int]], truth, fbeta_fn) -> dict[str, object]:
    returned = set(routes)
    hits = len(returned & truth)
    return {
        "routes": len(returned), "hits": hits,
        "precision": hits / len(returned), "recall": hits / len(truth),
        "fbeta": fbeta_fn(returned, truth),
    }


def _check_old_baseline(name: str, row: dict[str, object], saved: dict[str, str]) -> None:
    for current, old in (
        ("a_best", "two_opt_best"), ("a_routes", "two_opt_best_route_count"),
        ("a_hits", "two_opt_optimal_hits"), ("unique_candidates", "selected_routes"),
    ):
        if int(row[current]) != int(saved[old]):
            raise RuntimeError(f"{name}: ordinary 2-opt differs from saved baseline ({current})")
    if abs(float(row["a_fbeta"]) - float(saved["two_opt_fbeta"])) > 1e-12:
        raise RuntimeError(f"{name}: ordinary 2-opt Fbeta differs from saved baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the validated Soft POMO + Opt-B experiment")
    parser.add_argument("--output-dir", type=Path,
                        help="new result directory; default: a timestamped folder under runs/")
    parser.add_argument("--raw-routes-dir", type=Path,
                        help="replay saved original routes without invoking POMO")
    parser.add_argument("--instances", nargs="+", type=int,
                        help="optional case numbers for a small check; default: all 25")
    args = parser.parse_args()
    indices = sorted(set(args.instances or range(1, 26)))
    if not indices or indices[0] < 1 or indices[-1] > 25:
        parser.error("instances must be in 1..25")
    output = args.output_dir or (Path(__file__).resolve().parent / "runs" /
                                 f"run_{datetime.now():%Y%m%d_%H%M%S_%f}")
    if output.exists():
        parser.error(f"result directory already exists: {output}")
    for path in (CHECKPOINT, BASELINE_CSV, PAPER_CSV):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    checkpoint_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if checkpoint_hash != CHECKPOINT_SHA256:
        parser.error("the validated Soft checkpoint changed; refusing a misleading comparison")

    sys.path.insert(0, str(PROJECT))
    from gb_maco_POMO.pomo import canonical_cycle, search_routes
    from gb_maco_POMO.runner import prepare_tsp
    from gb_maco_POMO.experiments.compare import (
        BENCHMARKS, fbeta, load_ground_truth, load_policy, read_optimum,
    )
    from gb_maco_POMO.experiments.diagnose_2opt import _select_candidates
    from gb_maco_POMO.experiments.local_search_2opt import two_opt

    for index in indices:
        stem, _, _ = BENCHMARKS[index - 1]
        for suffix in (".tsp", ".solution"):
            if not (BENCHMARKS_DIR / f"{stem}{suffix}").is_file():
                parser.error(f"missing benchmark input: {stem}{suffix}")
        if args.raw_routes_dir and not (args.raw_routes_dir / f"MSTSP{index}.json").is_file():
            parser.error(f"missing saved raw routes: MSTSP{index}.json")
    torch.set_num_threads(1)
    policy, state = load_policy(CHECKPOINT, torch.device("cpu"))
    if policy.candidate_mode != "soft":
        parser.error("checkpoint is not Soft POMO")
    baseline = _read_csv(BASELINE_CSV)
    paper = _read_csv(PAPER_CSV)
    output.mkdir(parents=True)
    (output / "raw_routes").mkdir()
    (output / "best_routes").mkdir()
    rows = []
    batch_started = perf_counter()
    for index in indices:
        name = f"MSTSP{index}"
        stem, city_count, _ = BENCHMARKS[index - 1]
        case_started = perf_counter()
        prepared = prepare_tsp(BENCHMARKS_DIR / f"{stem}.tsp")
        prepare_seconds = perf_counter() - case_started
        if len(prepared.cities) != city_count:
            raise RuntimeError(f"{name}: unexpected city count")
        if args.raw_routes_dir:
            raw = (args.raw_routes_dir / f"{name}.json").read_bytes()
            routes = json.loads(raw)
            lengths = [tour_length(route, prepared.distance_matrix) for route in routes]
            pomo_seconds = 0.0
        else:
            # POMO is called exactly once for this case. A and B share its tours.
            search = search_routes(policy, prepared, inference_mode="mixed",
                                   samples_per_start=SAMPLES_PER_START, seed=SEED + index)
            routes, lengths = search.routes, search.lengths
            pomo_seconds = search.elapsed_seconds
            raw = json.dumps(routes, separators=(",", ":")).encode("utf-8")
        if len(routes) != city_count * (1 + SAMPLES_PER_START):
            raise RuntimeError(f"{name}: unexpected number of original routes")
        (output / "raw_routes" / f"{name}.json").write_bytes(raw)
        candidates = _select_candidates(routes, lengths, city_count, len(routes),
                                        all_generated=True)
        if len(candidates) != len({canonical_cycle(route) for route in routes}):
            raise RuntimeError(f"{name}: candidate set lost a unique route")

        a_started = perf_counter()
        refined = [two_opt(route, prepared.distance_matrix) for route in candidates]
        a_seconds = perf_counter() - a_started
        best = min([*lengths, *(item.length for item in refined)])
        best_routes = [route for route, length in zip(routes, lengths) if length == best]
        best_routes.extend(item.route for item in refined if item.length == best)
        a_pool = {canonical(route): route for route in best_routes}

        b_started = perf_counter()
        b = equal_plateau(list(a_pool.values()), prepared.distance_matrix, EXTRA_CHECKS)
        b_seconds = perf_counter() - b_started
        if b.checks > EXTRA_CHECKS or any(
                canonical(route) != key or tour_length(route, prepared.distance_matrix) != b.best_length
                for key, route in b.routes.items()):
            raise RuntimeError(f"{name}: invalid Opt-B route or budget")

        # Evaluation is deliberately after both searches; truth never guides A/B.
        solution = BENCHMARKS_DIR / f"{stem}.solution"
        reference = read_optimum(solution)
        truth = load_ground_truth(solution, city_count)
        if min(best, b.best_length) < reference:
            raise RuntimeError(f"{name}: route shorter than the reference optimum")
        am = _metrics(a_pool, truth, fbeta)
        bm = _metrics(b.routes, truth, fbeta)
        row = {
            "instance": name, "city_count": city_count, "seed": SEED + index,
            "raw_routes": len(routes), "unique_candidates": len(candidates),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "reference_length": reference, "truth_count": len(truth),
            "mixed_best": min(lengths), "a_best": best,
            **{f"a_{key}": value for key, value in am.items()},
            "a_checks": sum(item.pair_checks for item in refined), "a_seconds": a_seconds,
            "b_best": b.best_length,
            **{f"b_{key}": value for key, value in bm.items()},
            "b_new_routes": len(set(b.routes) - set(a_pool)),
            "b_checks": b.checks, "b_expanded_routes": b.attempts,
            "b_seconds": b_seconds, "prepare_seconds": prepare_seconds,
            "pomo_seconds": pomo_seconds, "case_seconds": perf_counter() - case_started,
            "paper_fbeta": float(paper[name]["paper_fbeta"]),
        }
        _check_old_baseline(name, row, baseline[name])
        rows.append(row)
        (output / "best_routes" / f"{name}.json").write_text(
            json.dumps({
                "instance": name,
                "route_format": "zero-based city indices; return from last to first city",
                "A": {"best_length": best, "routes": [list(key) for key in sorted(a_pool)]},
                "B": {"best_length": b.best_length,
                      "routes": [list(key) for key in sorted(b.routes)]},
            }, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"{name}: A {am['routes']} -> B {bm['routes']} best tours; "
              f"Opt-B checks {b.checks}", flush=True)

    with (output / "per_instance.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "input_mode": "saved_raw_routes" if args.raw_routes_dir else "fresh_pomo_once_per_case",
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": state.get("epoch"), "device": "cpu", "seed": SEED,
        "samples_per_start": SAMPLES_PER_START,
        "extra_pair_check_ceiling_per_instance": EXTRA_CHECKS,
        "instances": len(rows), "batch_seconds": perf_counter() - batch_started,
        "groups": {group: {
            "optimal_distance_cases": sum(r[f"{group}_best"] == r["reference_length"] for r in rows),
            "mean_fbeta": sum(r[f"{group}_fbeta"] for r in rows) / len(rows),
            "optimal_hits": sum(r[f"{group}_hits"] for r in rows),
            "pair_checks": sum(r[f"{group}_checks"] for r in rows),
            "seconds": sum(r[f"{group}_seconds"] for r in rows),
        } for group in ("a", "b")},
        "paper_mean_fbeta": sum(r["paper_fbeta"] for r in rows) / len(rows),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(rows, summary, output / "comparison.md")
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
