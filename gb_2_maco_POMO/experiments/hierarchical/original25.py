"""Run the original 25 cases with the exact historical Soft-POMO search protocol.

The fixed-Soft group is a gate: every saved historical raw-route hash and A/B
metric must match before any hierarchical group starts. Truth is read only
after inference, ordinary 2-opt, and Opt-B have completed for each case.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

from gb_maco_POMO.experiments.compare import (
    BENCHMARKS, fbeta, load_ground_truth, read_optimum,
)
from gb_2_maco_POMO.runner import prepare_tsp
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, equal_plateau, tour_length
from gb_2_maco_POMO.experiments.hierarchical.selected_rollout import generate_routes, load_policy
from gb_2_maco_POMO.experiments.hierarchical.report_original25 import write_report

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT.parent / "GB-MACO_PPSN2026-main" / "PPSN_GB-MACO" / "benchmark_MSTSP"
CHECKPOINT = PROJECT / "gb_maco_POMO/results/checkpoints/run_20261013_230742_828b99/best.pt"
HISTORY = PROJECT / ("gb_maco_POMO/results/ablation/"
    "run_20261013_230742_828b99_F_1plus8_cpu_2opt_top10/budget_sweep/"
    "plateau_vs_perturb_abc_20261014")
PAPER = PROJECT / "reproduction/config/paper_results.csv"
MODES = (("fixed_soft", 0.0), ("hierarchical_0", 0.0),
         ("hierarchical_0p5", 0.5), ("hierarchical_1", 1.0),
         ("hierarchical_2", 2.0))
SAMPLES = 8
SEED = 42
OPT_B_CHECKS = 50_000
EXPECTED_CHECKPOINT = "dbdfc5cb2faef0d7cbecb7c5c7cc77db16049e5d10e5d584a26e288f1bd88410"
EXPECTED_BASELINE = {"a_hits": 250, "b_hits": 663, "b_fbeta": 0.8968434047253819}


def _csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["instance"]: row for row in rows}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(routes: dict[tuple[int, ...], list[int]], truth: set[tuple[int, ...]]):
    returned = set(routes)
    hits = len(returned & truth)
    return {"routes": len(returned), "hits": hits,
            "precision": hits / len(returned), "recall": hits / len(truth),
            "fbeta": fbeta(returned, truth)}


def run_case(policy, index: int, label: str, penalty: float, output: Path,
             paper_fbeta: float, route_generator=None):
    stem, n, _ = BENCHMARKS[index - 1]
    name = f"MSTSP{index}"
    started = perf_counter()
    prepared = prepare_tsp(DATA / f"{stem}.tsp")
    prepare_seconds = perf_counter() - started
    if len(prepared.cities) != n:
        raise RuntimeError(f"{name}: city count mismatch")
    mode = "fixed_soft" if label == "fixed_soft" else "hierarchical"
    infer_started = perf_counter()
    if route_generator is None:
        routes, scale = generate_routes(policy, prepared, list(range(n)), SAMPLES,
                                        SEED + index, mode=mode, penalty=penalty)
    else:
        routes, scale = route_generator(policy, prepared, list(range(n)), SAMPLES,
                                        SEED + index)
    infer_seconds = perf_counter() - infer_started
    if len(routes) != n * (1 + SAMPLES) or any(
        len(route) != n or set(route) != set(range(n)) for route in routes
    ):
        raise RuntimeError(f"{name}: illegal or missing POMO route")
    raw = json.dumps(routes, separators=(",", ":")).encode("utf-8")
    (output / label / "raw_routes" / f"{name}.json").write_bytes(raw)
    distance = prepared.distance_matrix
    lengths = [tour_length(route, distance) for route in routes]
    # Historical all_generated rule: length, then original route index.
    seen = set()
    candidates = []
    for route_index in sorted(range(len(routes)), key=lambda i: (lengths[i], i)):
        key = canonical(routes[route_index])
        if key not in seen:
            candidates.append(routes[route_index])
            seen.add(key)
    a_started = perf_counter()
    refined = [two_opt(route, distance) for route in candidates]
    a_seconds = perf_counter() - a_started
    best = min([*lengths, *(item.length for item in refined)])
    best_routes = [route for route, length in zip(routes, lengths) if length == best]
    best_routes.extend(item.route for item in refined if item.length == best)
    a_pool = {canonical(route): route for route in best_routes}
    b_started = perf_counter()
    b = equal_plateau(list(a_pool.values()), distance, OPT_B_CHECKS)
    b_seconds = perf_counter() - b_started
    if b.checks > OPT_B_CHECKS or any(
        canonical(route) != key or tour_length(route, distance) != b.best_length
        for key, route in b.routes.items()
    ):
        raise RuntimeError(f"{name}: invalid Opt-B result")
    # The .solution is used for evaluation only, after both search stages.
    solution = DATA / f"{stem}.solution"
    reference = read_optimum(solution)
    truth = load_ground_truth(solution, n)
    if min(lengths) < reference or best < reference or b.best_length < reference:
        raise RuntimeError(f"{name}: distance below historical reference")
    am, bm = _metrics(a_pool, truth), _metrics(b.routes, truth)
    row = {
        "instance": name, "mode": label,
        "lambda": "" if label in ("fixed_soft", "learned_ball") else penalty,
        "city_count": n, "seed": SEED + index, "starts": n,
        "samples_per_start": SAMPLES, "raw_routes": len(routes),
        "unique_candidates": len(candidates), "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "reference_length": reference, "truth_count": len(truth),
        "raw_best": min(lengths), "raw_gap_pct": 100 * (min(lengths) / reference - 1),
        "a_best": best, "a_gap_pct": 100 * (best / reference - 1),
        "a_distance_hit": int(best == reference),
        **{f"a_{key}": value for key, value in am.items()},
        "a_checks": sum(item.pair_checks for item in refined),
        "b_best": b.best_length, "b_gap_pct": 100 * (b.best_length / reference - 1),
        "b_distance_hit": int(b.best_length == reference),
        **{f"b_{key}": value for key, value in bm.items()},
        "b_new_routes": len(set(b.routes) - set(a_pool)),
        "b_checks": b.checks, "b_expanded_routes": b.attempts,
        "distance_scale": scale, "paper_fbeta": paper_fbeta,
        "prepare_seconds": prepare_seconds, "inference_seconds": infer_seconds,
        "two_opt_seconds": a_seconds, "optb_seconds": b_seconds,
        "postprocess_seconds": a_seconds + b_seconds,
        "case_seconds": perf_counter() - started,
    }
    (output / label / "best_routes" / f"{name}.json").write_text(
        json.dumps({"instance": name, "A": {"best_length": best,
                    "routes": [list(key) for key in sorted(a_pool)]},
                    "B": {"best_length": b.best_length,
                    "routes": [list(key) for key in sorted(b.routes)]}},
                   ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return row


def check_baseline(row: dict, saved: dict[str, str]) -> None:
    name = row["instance"]
    fields = {"raw_sha256": "raw_sha256", "unique_candidates": "unique_candidates",
              "reference_length": "reference_length", "truth_count": "truth_count",
              "a_best": "baseline_best", "a_routes": "baseline_routes",
              "a_hits": "baseline_hits", "b_best": "b_best",
              "b_routes": "b_routes", "b_hits": "b_hits", "b_checks": "b_checks"}
    for actual, historical in fields.items():
        if str(row[actual]) != str(saved[historical]):
            raise RuntimeError(f"{name}: historical baseline mismatch: {actual}={row[actual]} vs {saved[historical]}")
    for actual, historical in (("a_fbeta", "baseline_fbeta"), ("b_fbeta", "b_fbeta")):
        if abs(row[actual] - float(saved[historical])) > 1e-12:
            raise RuntimeError(f"{name}: historical Fbeta mismatch: {actual}")


def _save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    historical = _csv(HISTORY / "per_instance.csv")
    paper = _csv(PAPER)
    digest = _sha(CHECKPOINT)
    if digest != EXPECTED_CHECKPOINT or len(historical) != 25 or len(paper) != 25:
        raise RuntimeError("historical input/config differs from validated 25-case run")
    for index, (stem, n, _) in enumerate(BENCHMARKS, start=1):
        if not (DATA / f"{stem}.tsp").is_file() or not (DATA / f"{stem}.solution").is_file():
            raise FileNotFoundError(stem)
    protocol = {"benchmark": str(DATA), "checkpoint": str(CHECKPOINT),
                "checkpoint_sha256": digest, "historical_csv_sha256": _sha(HISTORY / "per_instance.csv"),
                "paper_csv_sha256": _sha(PAPER), "device": "cpu", "seed": SEED,
                "starts": "all cities", "argmax_per_start": 1,
                "samples_per_start": SAMPLES, "two_opt": "all unique raw cycles; full ordinary 2-opt",
                "optb_extra_pair_checks_per_case": OPT_B_CHECKS,
                "beta_squared": 0.3, "modes": [name for name, _ in MODES]}
    if output.exists():
        if not args.resume or json.loads((output / "protocol.json").read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("output exists or resume protocol differs")
    else:
        output.mkdir(parents=True)
        _save_json(output / "protocol.json", protocol)
    policy, loaded_digest = load_policy(CHECKPOINT, "cpu")
    if loaded_digest != digest:
        raise RuntimeError("checkpoint changed during load")
    all_rows = []
    for label, penalty in MODES:
        for folder in ("raw_routes", "best_routes", "cases"):
            (output / label / folder).mkdir(parents=True, exist_ok=True)
        rows = []
        for index in range(1, 26):
            name = f"MSTSP{index}"
            path = output / label / "cases" / f"{name}.json"
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
            else:
                row = run_case(policy, index, label, penalty, output,
                               float(paper[name]["paper_fbeta"]))
                _save_json(path, row)
            if label == "fixed_soft":
                check_baseline(row, historical[name])
            rows.append(row)
            print(f"{label} {index}/25: A={row['a_best']} hits={row['a_hits']} "
                  f"B={row['b_best']} hits={row['b_hits']}", flush=True)
        if label == "fixed_soft":
            if (sum(r["a_distance_hit"] for r in rows) != 25
                    or sum(r["a_hits"] for r in rows) != EXPECTED_BASELINE["a_hits"]
                    or sum(r["b_hits"] for r in rows) != EXPECTED_BASELINE["b_hits"]
                    or abs(sum(r["b_fbeta"] for r in rows) / 25 - EXPECTED_BASELINE["b_fbeta"]) > 1e-12):
                raise RuntimeError("historical 25-case aggregate baseline mismatch")
            _save_json(output / "baseline_gate.json", {"passed": True,
                "distance_hits": 25, "a_hits": 250, "b_hits": 663,
                "b_mean_fbeta": EXPECTED_BASELINE["b_fbeta"]})
        all_rows.extend(rows)
    with (output / "per_instance.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_report(all_rows, output)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
