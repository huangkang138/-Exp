"""Compare two saved Soft-POMO checkpoints on the same 130 certified TSP cases.

Each case is saved immediately. Re-run with --output-dir DIR --resume after an
interruption; completed cases are checked and skipped. This benchmark does not
have an exhaustive optimal-tour set, so recall/F-beta are not computed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

from selected_rollout import generate_routes, load_policy, selected_starts


PROJECT = Path(r"D:\Project\球粒计算\GB-MACO_python")
SAVED = PROJECT / "最优结果的实验"
DATA = PROJECT / "benachmark_STSP_Complex_Multiopt_Test"
BACKUPS = {
    "fixed_soft": SAVED / "固定Soft-POMO_代码与模型",
    "learnable_bias": SAVED / "旧数据可学习Bias-POMO_代码与模型",
}


def read_cases(limit: int) -> list[dict[str, str]]:
    with (DATA / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        cases = list(csv.DictReader(stream))
    if not 1 <= limit <= len(cases):
        raise ValueError(f"--limit must be 1..{len(cases)}")
    chosen = cases[:limit]  # Dataset manifest order: 32 cities upward.
    for case in chosen:
        if not (DATA / case["file"]).is_file():
            raise FileNotFoundError(DATA / case["file"])
    return chosen


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def protocol(args, cases: list[dict[str, str]]) -> dict[str, object]:
    return {
        "dataset": str(DATA),
        "manifest_sha256": hashlib.sha256((DATA / "manifest.csv").read_bytes()).hexdigest(),
        "cases": [case["name"] for case in cases],
        "device": args.device,
        "seed": args.seed,
        "max_starts": args.max_starts,
        "samples_per_start": args.samples,
        "max_2opt_candidates": args.max_2opt,
        "optb_extra_pair_checks": args.optb_checks,
        "candidate_rule": "unique raw cycles, ascending length, then canonical cycle; first max_2opt_candidates",
        "optimal_route_rule": "exact route length equals certified global optimal length",
        "excluded_metrics": "Recall/F-beta and paper F-beta: full optimal-tour set and matched paper results unavailable",
        "checkpoints": {
            name: hashlib.sha256((root / "checkpoints" / "best.pt").read_bytes()).hexdigest()
            for name, root in BACKUPS.items()
        },
    }


def prepare_output(args, cases):
    output = args.output_dir or (SAVED / "新验证集评估" /
                                  f"full_search_130_{datetime.now():%Y%m%d_%H%M%S}")
    output = output.resolve()
    expected = protocol(args, cases)
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"Output exists. Use --resume to continue: {output}")
        existing = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError("Resume protocol or checkpoint differs from saved run")
    else:
        output.mkdir(parents=True)
        atomic_json(output / "protocol.json", expected)
    return output


def modules_for_worker():
    # These unchanged implementations are reused, not rewritten for this study.
    sys.path.insert(0, str(PROJECT / "gb_maco_POMO" / "experiments"))
    sys.path.insert(0, str(SAVED / "目前最优"))
    from local_search_2opt import two_opt
    from opt_b_search import canonical, equal_plateau, tour_length
    from gb_maco_POMO.runner import prepare_tsp
    return two_opt, canonical, equal_plateau, tour_length, prepare_tsp


def run_case(case, index, policy, args, model_dir, ops):
    two_opt, canonical, equal_plateau, tour_length, prepare_tsp = ops
    name = case["name"]
    started = perf_counter()
    prepared = prepare_tsp(DATA / case["file"])
    city_count = int(case["n"])
    if len(prepared.cities) != city_count:
        raise ValueError(f"{name}: city count mismatch")
    starts = selected_starts(city_count, args.max_starts)
    seed = args.seed + index
    infer_started = perf_counter()
    routes = generate_routes(policy, prepared, starts, args.samples, seed)
    infer_seconds = perf_counter() - infer_started
    if len(routes) != len(starts) * (args.samples + 1):
        raise RuntimeError(f"{name}: wrong route count")
    distance = prepared.distance_matrix
    raw = {canonical(route): route for route in routes}
    lengths = {key: tour_length(route, distance) for key, route in raw.items()}
    mixed_best = min(lengths.values())
    ordered = sorted(raw, key=lambda key: (lengths[key], key))
    chosen = [raw[key] for key in ordered[: args.max_2opt]]

    opt_started = perf_counter()
    refined = [two_opt(route, distance) for route in chosen]
    opt_seconds = perf_counter() - opt_started
    a_best = min([mixed_best, *(result.length for result in refined)])
    a_routes = {key: route for key, route in raw.items() if lengths[key] == a_best}
    a_routes.update({canonical(result.route): result.route
                     for result in refined if result.length == a_best})

    optb_started = perf_counter()
    b = equal_plateau(list(a_routes.values()), distance, args.optb_checks)
    optb_seconds = perf_counter() - optb_started
    reference = int(case["exact_optimal_length"])
    if min(mixed_best, a_best, b.best_length) < reference:
        raise RuntimeError(f"{name}: a route is shorter than proven optimum")
    if b.checks > args.optb_checks or any(
        tour_length(route, distance) != b.best_length for route in b.routes.values()
    ):
        raise RuntimeError(f"{name}: invalid Opt-B output")

    # Keep generated routes for audit without inflating the CSV or changing search.
    raw_path = model_dir / "raw_routes" / f"{name}.json.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as stream:
        json.dump({"name": name, "routes": routes}, stream, separators=(",", ":"))
    atomic_json(model_dir / "best_routes" / f"{name}.json", {
        "name": name, "route_format": "zero-based indices; cyclic; rotation/reversal equivalent",
        "after_2opt": {"best_length": a_best, "routes": [list(key) for key in sorted(a_routes)]},
        "after_optb": {"best_length": b.best_length,
                       "routes": [list(key) for key in sorted(b.routes)]},
    })
    result = {
        "name": name, "file": case["file"], "tier": case["tier"],
        "shape": case["shape"], "n": city_count, "seed": seed,
        "starts": len(starts), "samples_per_start": args.samples,
        "raw_routes": len(routes), "raw_unique": len(raw),
        "two_opt_candidates": len(chosen),
        "reference_length": reference,
        "raw_best": mixed_best, "raw_gap_pct": 100 * (mixed_best / reference - 1),
        "a_best": a_best, "a_gap_pct": 100 * (a_best / reference - 1),
        "a_best_routes": len(a_routes),
        "a_proven_optimal_routes": len(a_routes) if a_best == reference else 0,
        "a_pair_checks": sum(item.pair_checks for item in refined),
        "b_best": b.best_length, "b_gap_pct": 100 * (b.best_length / reference - 1),
        "b_best_routes": len(b.routes),
        "b_proven_optimal_routes": len(b.routes) if b.best_length == reference else 0,
        "b_extra_checks": b.checks, "b_expanded_routes": b.attempts,
        "prepare_seconds": infer_started - started,
        "inference_seconds": infer_seconds, "two_opt_seconds": opt_seconds,
        "optb_seconds": optb_seconds, "total_seconds": perf_counter() - started,
    }
    atomic_json(model_dir / "cases" / f"{name}.json", result)
    return result


def worker(args, cases, output):
    root = BACKUPS[args.worker]
    policy, digest, epoch = load_policy(root, args.device)
    if digest != protocol(args, cases)["checkpoints"][args.worker]:
        raise ValueError("Checkpoint changed while run was starting")
    ops = modules_for_worker()
    model_dir = output / args.worker
    for child in ("cases", "raw_routes", "best_routes"):
        (model_dir / child).mkdir(parents=True, exist_ok=True)
    atomic_json(model_dir / "model.json", {
        "checkpoint": str(root / "checkpoints" / "best.pt"),
        "checkpoint_sha256": digest, "checkpoint_epoch": epoch,
    })
    for index, case in enumerate(cases, 1):
        existing = model_dir / "cases" / f"{case['name']}.json"
        if existing.exists():
            saved = json.loads(existing.read_text(encoding="utf-8"))
            if saved["file"] != case["file"] or saved["reference_length"] != int(case["exact_optimal_length"]):
                raise ValueError(f"Saved case mismatch: {case['name']}")
            print(f"{args.worker}: {index}/{len(cases)} already saved", flush=True)
            continue
        result = run_case(case, index, policy, args, model_dir, ops)
        print(f"{args.worker}: {index}/{len(cases)} N={result['n']} "
              f"raw={result['raw_best']} A={result['a_best']} B={result['b_best']} "
              f"time={result['total_seconds']:.1f}s", flush=True)


def parent(args, cases, output):
    for name in BACKUPS:
        command = [sys.executable, "-B", str(Path(__file__).resolve()),
                   "--worker", name, "--output-dir", str(output), "--resume",
                   "--limit", str(args.limit), "--device", args.device,
                   "--seed", str(args.seed), "--max-starts", str(args.max_starts),
                   "--samples", str(args.samples), "--max-2opt", str(args.max_2opt),
                   "--optb-checks", str(args.optb_checks)]
        subprocess.run(command, check=True)
    from report import write_comparison
    write_comparison(output, cases)
    print(f"Completed {len(cases)} cases for each model. Results: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=130)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-starts", type=int, default=32)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-2opt", type=int, default=32)
    parser.add_argument("--optb-checks", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", choices=tuple(BACKUPS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_starts < 1 or args.samples < 0 or args.max_2opt < 1 or args.optb_checks < 1:
        parser.error("budgets must be positive; --samples may be zero")
    cases = read_cases(args.limit)
    output = prepare_output(args, cases)
    print(f"Output: {output}", flush=True)
    if args.worker:
        worker(args, cases, output)
    else:
        parent(args, cases, output)


if __name__ == "__main__":
    main()
