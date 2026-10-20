"""Read saved routes only; write reproducible MSTSP22/25 diagnosis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from core import (canonical, compare, edge_classes, edges, order_signature,
                  segment_comparison, spatial_labels)
from plot import draw


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def record_routes(results, name, optimum, upper):
    """Inventory every saved best 2-opt/Opt-B route in the near-optimal range."""
    found = {}
    source_files = []
    for path in results.rglob(f"best_routes/{name}.json"):
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            continue
        if not isinstance(saved, dict):
            continue
        for stage in ("A", "B"):
            group = saved.get(stage)
            if not isinstance(group, dict) or not optimum < int(group.get("best_length", 0)) <= upper:
                continue
            source_files.append(str(path))
            for route in group.get("routes", []):
                key = canonical(route)
                item = found.setdefault(key, {"route": list(key), "sources": []})
                item["sources"].append(f"{path.relative_to(results)}#{stage}")
    return found, sorted(set(source_files))


def paired_routes(results, name, target, found):
    path = results / "multi_solution_diagnosis_20261017_seeds42_46/raw_to_2opt.csv"
    pairs = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["instance"] != name or int(row["two_opt_length"]) != target:
                continue
            final = json.loads(row["two_opt_key"])
            key = canonical(final)
            item = found.setdefault(key, {"route": list(key), "sources": []})
            item["sources"].append(f"{path.relative_to(results)}#model={row['model']},seed={row['seed']},raw_index={row['raw_index']}")
            pairs.append({"route_key": json.dumps(key), "model": row["model"], "seed": row["seed"],
                          "raw_index": row["raw_index"], "raw_length": int(row["raw_length"]),
                          "two_opt_length": target, "raw_route": json.loads(row["raw_key"]),
                          "two_opt_route": final})
    return pairs, path


def diagnose_case(number, project, output):
    from gb_2_maco_POMO.experiments.hierarchical.original25 import BENCHMARKS, DATA
    from gb_2_maco_POMO.runner import prepare_tsp
    from gb_maco_POMO.experiments.compare import load_ground_truth
    from gb_2_maco_POMO.experiments.shared.opt_b_search import tour_length

    name = f"MSTSP{number}"
    stem, n, _ = BENCHMARKS[number - 1]
    tsp, solution = DATA / f"{stem}.tsp", DATA / f"{stem}.solution"
    problem = prepare_tsp(tsp)
    truth = sorted(load_ground_truth(solution, n))
    optimum = min(tour_length(route, problem.distance_matrix) for route in truth)
    assert (number, optimum) in ((22, 10442), (25, 9521))
    upper = 10445 if number == 22 else 9524
    spatial, spatial_groups, threshold = spatial_labels(problem.cities)
    balls = problem.city_to_ball
    results = project / "gb_2_maco_POMO/results"
    found, files = record_routes(results, name, optimum, upper)
    pairs, pair_path = paired_routes(results, name, upper, found)
    inventory, differences, segment_rows, pair_rows = [], [], [], []
    records = []
    for key, item in sorted(found.items(), key=lambda x: (tour_length(x[0], problem.distance_matrix), x[0])):
        route = item["route"]
        length = tour_length(route, problem.distance_matrix)
        if not optimum < length <= upper:
            raise RuntimeError(f"{name}: saved route length mismatch")
        ranked = sorted(((len(edges(route) - edges(other)), other) for other in truth),
                        key=lambda x: (x[0], x[1]))
        nearest_distance, nearest = ranked[0]
        counts, removed, added, failed_classes, optimal_classes = compare(route, nearest, balls)
        statuses, failed_segments, optimal_segments = segment_comparison(route, list(nearest), balls)
        identifier = f"{name}_{length}_{len(records)+1:02d}"
        ball_match = order_signature(route, balls) == order_signature(nearest, balls)
        spatial_match = order_signature(route, spatial) == order_signature(nearest, spatial)
        status_text = statuses if isinstance(statuses, str) else json.dumps(statuses, ensure_ascii=False, sort_keys=True)
        row = {"route_id": identifier, "instance": name, "length": length,
               "gap": length - optimum, "closest_truth_index": truth.index(nearest),
               "closest_truth_edge_distance": nearest_distance,
               "all_truth_edge_distance_min": ranked[0][0],
               "all_truth_edge_distance_max": ranked[-1][0],
               "ball_order_same": int(ball_match), "spatial_order_same": int(spatial_match),
               "ball_crossings": sum(balls[route[i]] != balls[route[(i+1)%n]] for i in range(n)),
               "spatial_crossings": sum(spatial[route[i]] != spatial[route[(i+1)%n]] for i in range(n)),
               **counts, "segment_status": status_text,
               "source_count": len(set(item["sources"])),
               "sources": json.dumps(sorted(set(item["sources"])), ensure_ascii=False),
               "route": json.dumps(route), "closest_truth_route": json.dumps(nearest)}
        inventory.append(row)
        records.append((row, route, list(nearest)))
        for side, selected, classes in (("remove", removed, failed_classes), ("add", added, optimal_classes)):
            for a, b in sorted(selected):
                differences.append({"route_id": identifier, "side": side, "class": classes[a,b],
                                    "city_a": a, "city_b": b, "ball_a": balls[a], "ball_b": balls[b],
                                    "spatial_a": spatial[a], "spatial_b": spatial[b],
                                    "length": problem.distance_matrix[a][b]})
        for side, pieces in (("failed", failed_segments), ("closest_optimal", optimal_segments)):
            for index, part in enumerate(pieces):
                segment_rows.append({"route_id": identifier, "side": side, "segment_index": index,
                                     **{k: json.dumps(v) if isinstance(v, list) else v for k, v in part.items()}})
    by_key = {canonical(json.loads(row["route"])): row for row in inventory}
    for pair in pairs:
        final = canonical(pair["two_opt_route"])
        row = by_key[final]
        optimal = json.loads(row["closest_truth_route"])
        raw_classes = edge_classes(pair["raw_route"], balls)
        final_classes = edge_classes(pair["two_opt_route"], balls)
        truth_edges = edges(optimal)
        raw_wrong = set(raw_classes) - truth_edges
        final_wrong = set(final_classes) - truth_edges
        pair_rows.append({"route_id": row["route_id"], "model": pair["model"], "seed": pair["seed"],
                          "raw_index": pair["raw_index"], "raw_length": pair["raw_length"],
                          "two_opt_length": pair["two_opt_length"],
                          "raw_wrong_edges": len(raw_wrong), "removed_wrong_edges": len(raw_wrong - final_wrong),
                          "retained_wrong_edges": len(raw_wrong & final_wrong),
                          "new_wrong_edges": len(final_wrong - raw_wrong),
                          **{f"retained_{kind}": sum(raw_classes[e] == kind for e in raw_wrong & final_wrong)
                             for kind in "ABC"}})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / f"{name}_routes.csv", inventory, list(inventory[0]))
    write_csv(output / f"{name}_edge_differences.csv", differences, list(differences[0]))
    write_csv(output / f"{name}_segments.csv", segment_rows, list(segment_rows[0]))
    write_csv(output / f"{name}_raw_2opt_pairs.csv", pair_rows,
              list(pair_rows[0]) if pair_rows else ["route_id", "model", "seed", "raw_index", "raw_length",
                                                   "two_opt_length", "raw_wrong_edges", "removed_wrong_edges",
                                                   "retained_wrong_edges", "new_wrong_edges", "retained_A", "retained_B", "retained_C"])
    # Prefer a paired 10445 route and Joint diversity's saved 9524 2-opt result.
    candidates = [r for r in records if r[0]["length"] == upper]
    if number == 22:
        preferred = {row["route_id"] for row in pair_rows}
        candidates.sort(key=lambda r: (r[0]["route_id"] not in preferred, r[0]["closest_truth_edge_distance"]))
    else:
        candidates.sort(key=lambda r: (not any("joint_ball_city" in s and s.endswith("#A")
                                               for s in json.loads(r[0]["sources"])),
                                       r[0]["closest_truth_edge_distance"]))
    chosen = candidates[0]
    draw(output / f"{name}_route_comparison.png", problem, spatial, chosen[1], chosen[2],
         chosen[0]["route_id"], optimum, chosen[0]["length"])
    return {"instance": name, "tsp": str(tsp), "solution": str(solution),
            "optimum": optimum, "truth_routes": len(truth), "city_count": n,
            "ball_count": len(problem.granular_balls), "spatial_cluster_count": len(spatial_groups),
            "spatial_cluster_sizes": list(map(len, spatial_groups)), "spatial_mst_threshold": threshold,
            "source_files": files, "paired_source": str(pair_path),
            "saved_near_optimal_unique_routes": len(records), "raw_2opt_pairs": len(pair_rows),
            "typical_route_id": chosen[0]["route_id"], "routes": inventory}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    sys.path.insert(0, str(project))
    if output.exists():
        raise FileExistsError(output)
    data = [diagnose_case(n, project, output) for n in (22, 25)]
    (output / "source_manifest.json").write_text(
        json.dumps([{k:v for k,v in item.items() if k != "routes"} for item in data],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    for item in data:
        rows = item["routes"]
        print(item["instance"], "unique", len(rows), "lengths", dict(Counter(r["length"] for r in rows)),
              "typical", item["typical_route_id"])
        for length in sorted(set(r["length"] for r in rows)):
            subset = [r for r in rows if r["length"] == length]
            print(length, "count", len(subset), "ball_same", sum(r["ball_order_same"] for r in subset),
                  "spatial_same", sum(r["spatial_order_same"] for r in subset),
                  "mean_interface_error", round(sum(r["interface_error_ratio"] for r in subset)/len(subset),3),
                  "mean_baseline", round(sum(r["normal_interface_ratio"] for r in subset)/len(subset),3))


if __name__ == "__main__":
    main()
