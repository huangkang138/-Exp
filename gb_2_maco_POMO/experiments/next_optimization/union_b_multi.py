"""Merge saved S0/S2/S3/S4 Opt-B tours with saved dual-reward B_multi tours.

This command only reads existing route records. It never runs inference or search.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from gb_maco_POMO.experiments.compare import fbeta, load_ground_truth
from gb_2_maco_POMO.experiments.hierarchical.original25 import BENCHMARKS, DATA
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, tour_length
from gb_2_maco_POMO.runner import prepare_tsp

PROJECT = Path(__file__).resolve().parents[3]
PRIOR = PROJECT / "gb_2_maco_POMO/results/next_optimization/solution_union"
DUAL = (PROJECT / "gb_2_maco_POMO/results/dual_reward/"
        "run_seed_42_multiopt_500_10epoch/experiment/test")
DEFAULT_OUTPUT = PROJECT / "gb_2_maco_POMO/results/next_optimization/solution_union_plus_B_multi"


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _routes(raw: list[list[int]], city_count: int, name: str) -> set[tuple[int, ...]]:
    if any(len(route) != city_count or set(route) != set(range(city_count)) for route in raw):
        raise RuntimeError(f"{name}: invalid saved route")
    result = {canonical(route) for route in raw}
    if len(result) != len(raw):
        raise RuntimeError(f"{name}: duplicate canonical route in saved file")
    return result


def run(output: Path = DEFAULT_OUTPUT) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"new output directory required: {output}")
    prior_rows = {r["instance"]: r for r in _read_csv(PRIOR / "per_instance.csv")
                  if r["combination"] == "ALL"}
    dual_rows = {r["instance"]: r for r in _read_csv(DUAL / "per_instance.csv")
                 if r["mode"] == "B_multi"}
    prior_summary = {r["combination"]: r for r in _read_csv(PRIOR / "summary.csv")}
    dual_summary = json.loads((DUAL / "summary.json").read_text(encoding="utf-8"))["B_multi"]
    if (len(prior_rows) != 25 or len(dual_rows) != 25 or
            prior_summary["ALL"]["cases"] != "25" or
            dual_summary["selected_epoch"] != 6):
        raise RuntimeError("source records are incomplete or B_multi selection changed")
    rows, merged = [], {}
    for index, (stem, city_count, _) in enumerate(BENCHMARKS, 1):
        name = f"MSTSP{index}"
        truth = load_ground_truth(DATA / f"{stem}.solution", city_count)
        distance = prepare_tsp(DATA / f"{stem}.tsp").distance_matrix
        old_json = json.loads((PRIOR / "merged_routes" / f"{name}.json").read_text(encoding="utf-8"))["ALL"]
        new_json = json.loads((DUAL / "B_multi/best_routes" / f"{name}.json").read_text(encoding="utf-8"))["B"]
        old = _routes(old_json["routes"], city_count, f"{name} previous ALL")
        added_source = _routes(new_json["routes"], city_count, f"{name} B_multi")
        old_record, new_record = prior_rows[name], dual_rows[name]
        if (old != truth.intersection(old) or
                len(old) != int(old_record["returned_tours"]) or
                len(old & truth) != int(old_record["true_optimal_tours"]) or
                len(added_source) != int(new_record["b_routes"]) or
                len(added_source & truth) != int(new_record["b_hits"]) or
                any(tour_length(route, distance) != int(new_json["best_length"])
                    for route in added_source) or
                int(new_json["best_length"]) != int(new_record["b_best"]) or
                abs(fbeta(old, truth) - float(old_record["fbeta"])) > 1e-12 or
                abs(fbeta(added_source, truth) - float(new_record["b_fbeta"])) > 1e-12):
            raise RuntimeError(f"{name}: source routes differ from saved evaluation")
        both = old | added_source
        hits = both & truth
        false = both - truth
        score = fbeta(both, truth)
        row = {"instance": name, "truth_tours": len(truth),
               "prior_all_routes": len(old), "b_multi_routes": len(added_source),
               "merged_routes": len(both), "merged_true_optimal_routes": len(hits),
               "merged_nonoptimal_routes": len(false),
               "new_true_optimal_vs_prior": len((added_source - old) & truth),
               "new_nonoptimal_vs_prior": len((added_source - old) - truth),
               "precision": len(hits) / len(both), "recall": len(hits) / len(truth),
               "fbeta": score, "prior_all_fbeta": float(old_record["fbeta"]),
               "delta_fbeta": score - float(old_record["fbeta"])}
        rows.append(row)
        merged[name] = {"routes": sorted(both), "true_optimal_routes": sorted(hits),
                        "nonoptimal_routes": sorted(false)}
    if (sum(r["prior_all_routes"] for r in rows) != int(prior_summary["ALL"]["returned_tours"]) or
            sum(r["b_multi_routes"] for r in rows) != int(dual_summary["opt_b_hits"]) +
            sum(r["b_multi_routes"] - int(dual_rows[r["instance"]]["b_hits"]) for r in rows)):
        raise RuntimeError("source aggregate counts disagree")
    summary = {"cases": 25,
               "prior_all_mean_fbeta": sum(r["prior_all_fbeta"] for r in rows) / 25,
               "merged_mean_fbeta": sum(r["fbeta"] for r in rows) / 25,
               "delta_mean_fbeta": sum(r["delta_fbeta"] for r in rows) / 25,
               "prior_all_routes": sum(r["prior_all_routes"] for r in rows),
               "b_multi_routes": sum(r["b_multi_routes"] for r in rows),
               "merged_routes": sum(r["merged_routes"] for r in rows),
               "merged_true_optimal_routes": sum(r["merged_true_optimal_routes"] for r in rows),
               "merged_nonoptimal_routes": sum(r["merged_nonoptimal_routes"] for r in rows),
               "new_true_optimal_vs_prior": sum(r["new_true_optimal_vs_prior"] for r in rows),
               "new_nonoptimal_vs_prior": sum(r["new_nonoptimal_vs_prior"] for r in rows),
               "mean_precision": sum(r["precision"] for r in rows) / 25,
               "mean_recall": sum(r["recall"] for r in rows) / 25}
    if abs(summary["prior_all_mean_fbeta"] - float(prior_summary["ALL"]["mean_fbeta"])) > 1e-12:
        raise RuntimeError("previous four-group mean Fbeta differs")
    output.mkdir(parents=True)
    routes_dir = output / "merged_routes"
    routes_dir.mkdir()
    for name, value in merged.items():
        _write_json(routes_dir / f"{name}.json", value)
    with (output / "per_instance.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(output / "summary.json", summary)
    _write_json(output / "config.json", {
        "previous_union": str(PRIOR), "previous_per_instance_sha256": _hash(PRIOR / "per_instance.csv"),
        "b_multi_routes": str(DUAL / "B_multi/best_routes"),
        "b_multi_per_instance_sha256": _hash(DUAL / "per_instance.csv"),
        "b_multi_checkpoint_sha256": dual_summary["checkpoint_sha256"],
        "seed": 42, "beta_squared": 0.3,
        "method": "canonical union of saved final Opt-B routes; use .solution only to score the union",
        "budget_note": "combines five saved searches; not a same-budget model comparison"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
