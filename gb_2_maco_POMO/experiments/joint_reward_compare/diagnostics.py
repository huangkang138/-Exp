"""Compare raw greedy interfaces and sampled structure diversity without truth labels."""

from __future__ import annotations

import json
from pathlib import Path

from gb_2_maco_POMO.experiments.dual_reward.train import save_csv
from gb_2_maco_POMO.experiments.hierarchical.original25 import BENCHMARKS, DATA
from gb_2_maco_POMO.experiments.joint_ball_city.evaluate import BASELINE
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical
from gb_2_maco_POMO.runner import prepare_tsp

from .reward import structure


def _load(path, n):
    routes = json.loads(path.read_text(encoding="utf-8"))
    if len(routes) != n * 9 or any(len(route) != n or set(route) != set(range(n)) for route in routes):
        raise RuntimeError(f"invalid saved raw route set: {path}")
    return routes


def _unique_structures(routes, mapping):
    unique = {canonical(route) for route in routes}
    pairs = [structure(key, mapping) for key in unique]
    return len(unique), len({cross for cross, _ in pairs}), len({within for _, within in pairs})


def run(root: Path, j1: Path, j2: Path):
    rows = []
    for index, (stem, n, _) in enumerate(BENCHMARKS, 1):
        name = f"MSTSP{index}"
        problem = prepare_tsp(DATA / f"{stem}.tsp")
        mapping = problem.city_to_ball
        sources = {"A0": BASELINE / "A0/raw_routes"}
        for group, folder in (("J1", j1), ("J2", j2)):
            for kind in ("shortest", "diversity"):
                sources[f"{group}_{kind}"] = folder / "test" / f"Joint_{kind}" / "raw_routes"
        routes = {label: _load(path / f"{name}.json", n) for label, path in sources.items()}
        base_cross = [structure(canonical(route), mapping)[0] for route in routes["A0"][:n]]
        record = {"instance": name, "city_count": n}
        for label, values in routes.items():
            unique, cross, within = _unique_structures(values, mapping)
            greedy_cross = [structure(canonical(route), mapping)[0] for route in values[:n]]
            record.update({f"{label}_raw_unique_cycles": unique,
                           f"{label}_raw_cross_structures": cross,
                           f"{label}_raw_within_structures": within,
                           f"{label}_greedy_cross_edge_set_changed_vs_A0": sum(a != b for a,b in zip(base_cross, greedy_cross))})
        rows.append(record)
    save_csv(root / "raw_structure_diagnostics.csv", rows)
    return rows
