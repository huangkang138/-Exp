"""Summarize every distinct saved 2-opt endpoint for the two target instances."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from core import canonical, compare, edges, order_signature, spatial_labels


def run(project, output, number):
    from gb_2_maco_POMO.experiments.hierarchical.original25 import BENCHMARKS, DATA
    from gb_2_maco_POMO.runner import prepare_tsp
    from gb_maco_POMO.experiments.compare import load_ground_truth
    from gb_2_maco_POMO.experiments.shared.opt_b_search import tour_length

    name = f"MSTSP{number}"
    stem, n, _ = BENCHMARKS[number-1]
    problem = prepare_tsp(DATA / f"{stem}.tsp")
    spatial = spatial_labels(problem.cities)[0]
    balls = problem.city_to_ball
    truth = sorted(load_ground_truth(DATA / f"{stem}.solution", n))
    truth_edges = [(route, edges(route)) for route in truth]
    optimum = 10442 if number == 22 else 9521
    source = project / "gb_2_maco_POMO/results/multi_solution_diagnosis_20261017_seeds42_46/raw_to_2opt.csv"
    unique = {}
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["instance"] != name or int(row["two_opt_length"]) <= optimum:
                continue
            route = json.loads(row["two_opt_key"])
            key = canonical(route)
            item = unique.setdefault(key, {"length": int(row["two_opt_length"]), "sources": Counter()})
            if item["length"] != int(row["two_opt_length"]):
                raise RuntimeError("canonical route has inconsistent saved lengths")
            item["sources"][(row["model"], row["seed"])] += 1
    target = output / f"{name}_all_saved_2opt_routes.csv"
    columns = ["route", "length", "gap", "source_occurrences", "source_models_seeds",
               "closest_truth_edge_distance", "all_truth_edge_distance_max",
               "ball_order_same", "spatial_order_same", "ball_crossings", "spatial_crossings",
               "remove_A", "remove_B", "remove_C", "add_A", "add_B", "add_C",
               "wrong_edges", "interface_error_ratio", "normal_interface_ratio"]
    bins = {"gap_1_to_3": [], "gap_4_to_10": [], "gap_over_10": []}
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for key, item in sorted(unique.items(), key=lambda x: (x[1]["length"], x[0])):
            route = list(key)
            if tour_length(route, problem.distance_matrix) != item["length"]:
                raise RuntimeError("saved 2-opt route length differs from benchmark matrix")
            route_edges = edges(route)
            distances = [(len(route_edges - e), truth_route) for truth_route, e in truth_edges]
            closest_distance, closest = min(distances, key=lambda x: (x[0], x[1]))
            counts, *_ = compare(route, list(closest), balls)
            gap = item["length"] - optimum
            row = {"route": json.dumps(key), "length": item["length"], "gap": gap,
                   "source_occurrences": sum(item["sources"].values()),
                   "source_models_seeds": json.dumps([f"{model}/{seed}:{count}" for (model,seed),count in sorted(item["sources"].items())]),
                   "closest_truth_edge_distance": closest_distance,
                   "all_truth_edge_distance_max": max(distance for distance,_ in distances),
                   "ball_order_same": int(order_signature(route, balls) == order_signature(closest, balls)),
                   "spatial_order_same": int(order_signature(route, spatial) == order_signature(closest, spatial)),
                   "ball_crossings": sum(balls[route[i]] != balls[route[(i+1)%n]] for i in range(n)),
                   "spatial_crossings": sum(spatial[route[i]] != spatial[route[(i+1)%n]] for i in range(n)),
                   **{k: counts[k] for k in ("remove_A","remove_B","remove_C","add_A","add_B","add_C",
                                               "wrong_edges","interface_error_ratio","normal_interface_ratio")}}
            writer.writerow(row)
            bins["gap_1_to_3" if gap <= 3 else "gap_4_to_10" if gap <= 10 else "gap_over_10"].append(row)
    summary = {}
    for label, rows in bins.items():
        total_wrong = sum(r["wrong_edges"] for r in rows)
        interface = sum(r[k] for r in rows for k in ("remove_A","remove_B","add_A","add_B"))
        summary[label] = {"unique_routes": len(rows),
                          "ball_order_same": sum(r["ball_order_same"] for r in rows),
                          "spatial_order_same": sum(r["spatial_order_same"] for r in rows),
                          "interface_error_ratio_weighted": interface/total_wrong if total_wrong else None,
                          "normal_interface_ratio_mean": sum(r["normal_interface_ratio"] for r in rows)/len(rows) if rows else None}
    print(name, len(unique), summary)
    return {"source": str(source), "unique_failed_2opt_routes": len(unique), "gap_bins": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    sys.path.insert(0, str(project))
    summary = {f"MSTSP{n}": run(project, output, n) for n in (22,25)}
    (output / "all_saved_2opt_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
