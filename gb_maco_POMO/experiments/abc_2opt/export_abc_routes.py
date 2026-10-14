"""Replay saved raw routes and persist every distinct best A/B/C cycle."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from plateau_perturb_search import canonical, equal_plateau, perturb_and_descend, tour_length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project.resolve()))
    from gb_maco_POMO.runner import prepare_tsp
    from gb_maco_POMO.experiments.compare import BENCHMARKS
    from gb_maco_POMO.experiments.diagnose_2opt import _select_candidates
    from gb_maco_POMO.experiments.local_search_2opt import two_opt

    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((args.result_dir / "per_instance.csv").open(
        encoding="utf-8-sig", newline="")))
    output = args.result_dir / "best_routes"
    output.mkdir(exist_ok=True)
    for row in rows:
        name = row["instance"]
        index = int(name[5:])
        stem, city_count, _ = BENCHMARKS[index - 1]
        distance = prepare_tsp(args.benchmark_dir / f"{stem}.tsp").distance_matrix
        routes = json.loads((args.result_dir / "raw_routes" / f"{name}.json").read_text(
            encoding="utf-8"))
        if len(routes) != int(row["raw_routes"]):
            raise RuntimeError(f"{name}: saved raw route count changed")
        lengths = [tour_length(route, distance) for route in routes]
        candidates = _select_candidates(routes, lengths, city_count, len(routes),
                                        all_generated=True)
        refined = [two_opt(route, distance) for route in candidates]
        best = min([*lengths, *(item.length for item in refined)])
        a_routes = [route for route, length in zip(routes, lengths) if length == best]
        a_routes.extend(item.route for item in refined if item.length == best)
        starts = {canonical(route): route for route in a_routes}
        b = equal_plateau(list(starts.values()), distance,
                          summary["extra_pair_check_ceiling_per_instance"])
        c = perturb_and_descend(list(starts.values()), distance,
                                summary["extra_pair_check_ceiling_per_instance"],
                                summary["seed"] + index)
        for group, pool, length, checks in (
            ("baseline", starts, best, sum(item.pair_checks for item in refined)),
            ("b", b.routes, b.best_length, b.checks),
            ("c", c.routes, c.best_length, c.checks),
        ):
            if (len(pool) != int(row[f"{group}_routes"])
                    or length != int(row[f"{group}_best"])
                    or checks != int(row["a_checks" if group == "baseline" else f"{group}_checks"])):
                raise RuntimeError(f"{name}: {group} differs from recorded run")
            if any(tour_length(route, distance) != length for route in pool.values()):
                raise RuntimeError(f"{name}: invalid route length")
        payload = {
            "instance": name,
            "route_format": "zero-based city indices; closed cycle, last city returns to first",
            "groups": {
                group: {"best_length": length,
                        "routes": [list(key) for key in sorted(pool)]}
                for group, pool, length in (("A", starts, best),
                                            ("B", b.routes, b.best_length),
                                            ("C", c.routes, c.best_length))
            },
        }
        (output / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8")
        print(f"{name}: A/B/C routes {len(starts)}/{len(b.routes)}/{len(c.routes)}", flush=True)


if __name__ == "__main__":
    main()
