"""Independent 138-case complex test; certified distance only, never F-beta."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from gb_2_maco_POMO.runner import prepare_tsp
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, equal_plateau, tour_length
from .evaluate25 import load_selected_checkpoint
from .rollout import generate_routes
from .train import PROJECT, write_csv, write_json

DATA = PROJECT / "benachmark_STSP_Complex_Multiopt_Test"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--selector", choices=("best_length", "best_multi"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output directory must be new")
    checkpoint = args.train_dir / "checkpoints" / f"{args.selector}.pt"
    model, state = load_selected_checkpoint(checkpoint)
    torch.set_num_threads(1)
    with (DATA / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        cases = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    if len(cases) != 138 or any(row["complete_optimal_set_known"] != "no"
                                     for row in cases):
        raise RuntimeError("complex test manifest differs from the verified 138-case set")
    output.mkdir(parents=True)
    (output / "raw_routes").mkdir()
    (output / "best_routes").mkdir()
    results = []
    for index, row in enumerate(cases, 1):
        case_started = perf_counter()
        path = DATA / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest().lower() != row["sha256"].lower():
            raise RuntimeError(f"complex input changed: {path}")
        proof = json.loads((DATA / row["proof_file"]).read_text(encoding="utf-8"))
        if (proof["instance"] != row["name"] or
                int(proof["known_exact_optimal_length"]) != int(row["exact_optimal_length"])):
            raise RuntimeError(f"complex proof differs from manifest: {row['name']}")
        problem = prepare_tsp(path)
        n = len(problem.cities)
        raw = generate_routes(model, problem, samples=8, seed=42 + index,
                              device=torch.device("cpu"), starts=list(range(min(n, 32))))
        (output / "raw_routes" / f"{row['name']}.json").write_text(
            json.dumps(raw, separators=(",", ":")), encoding="utf-8")
        matrix = problem.distance_matrix
        lengths = [tour_length(route, matrix) for route in raw]
        unique = {}
        for route in sorted(raw, key=lambda r: tour_length(r, matrix)):
            unique.setdefault(canonical(route), route)
            if len(unique) == 32:
                break
        refined = [two_opt(route, matrix) for route in unique.values()]
        best = min([*lengths, *(x.length for x in refined)])
        a = {canonical(route): route for route, length in zip(raw, lengths)
             if length == best}
        a.update({canonical(x.route): x.route for x in refined if x.length == best})
        b = equal_plateau(list(a.values()), matrix, 50_000)
        reference = int(row["exact_optimal_length"])
        if min(lengths) < reference or best < reference or b.best_length < reference:
            raise RuntimeError(f"route shorter than certified optimum: {row['name']}")
        (output / "best_routes" / f"{row['name']}.json").write_text(
            json.dumps({"A": [list(x) for x in sorted(a)],
                        "B": [list(x) for x in sorted(b.routes)]},
                       separators=(",", ":")), encoding="utf-8")
        results.append({"instance": row["name"], "city_count": n,
                        "certified_length": reference, "raw_best": min(lengths),
                        "two_opt_best": best, "optb_best": b.best_length,
                        "raw_gap_pct": 100 * (min(lengths) / reference - 1),
                        "two_opt_gap_pct": 100 * (best / reference - 1),
                        "distance_hit": int(b.best_length == reference),
                        "a_certified_length_routes": len(a) if best == reference else 0,
                        "b_certified_length_routes": len(b.routes)
                        if b.best_length == reference else 0,
                        "elapsed_seconds": perf_counter() - case_started})
        print(f"{row['name']}: {len(results)}/138", flush=True)
    write_csv(output / "per_instance.csv", results)
    summary = {"group": state["group"], "selector": args.selector,
               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "cases": len(results), "distance_hits": sum(x["distance_hit"] for x in results),
               "mean_raw_gap_pct": sum(x["raw_gap_pct"] for x in results) / len(results),
               "mean_two_opt_gap_pct": sum(x["two_opt_gap_pct"] for x in results) / len(results),
               "certified_length_routes_A": sum(x["a_certified_length_routes"] for x in results),
               "certified_length_routes_B": sum(x["b_certified_length_routes"] for x in results),
               "mean_elapsed_seconds": sum(x["elapsed_seconds"] for x in results) / len(results),
               "f_beta_available": False, "complete_optimal_sets_known": False,
               "budget": "32 starts; 1 argmax + 8 samples; first 32 unique raw cycles by length; full 2-opt; Opt-B 50000 checks"}
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
