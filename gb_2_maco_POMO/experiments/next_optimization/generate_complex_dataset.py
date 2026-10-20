"""Prepare one gated 300-original + 200-complex train / 100+50 validation split.

New cases carry two distinct equal-length *heuristic best* tours. No exact
optimality claim is made: the RL reward uses route length, not these labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

from gb_2_maco_POMO.experiments.next_optimization.dataset_diagnosis import (
    ORIGINAL, OUTPUT as DIAGNOSIS, _spatial_clusters,
)
from gb_2_maco_POMO.experiments.shared.data import load_manifest
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, equal_plateau
from gb_2_maco_POMO.tsp import City, init_distance_matrix, read_cities
from gb_2_maco_POMO.experiments.hierarchical.original25 import BENCHMARKS, DATA

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT / "gb_2_maco_POMO/results/next_optimization/complex_training/dataset"


def _signature(points):
    """Translation, 90-degree rotation, reflection and uniform-scale guard."""
    forms = []
    for sx, sy, swap in ((sx, sy, swap) for sx in (-1, 1) for sy in (-1, 1)
                         for swap in (False, True)):
        transformed = [(sx * (y if swap else x), sy * (x if swap else y)) for x, y in points]
        min_x = min(x for x, _ in transformed)
        min_y = min(y for _, y in transformed)
        shifted = [(x - min_x, y - min_y) for x, y in transformed]
        scale = max(max(max(x, y) for x, y in shifted), 1.0)
        forms.append(tuple(sorted((round(x / scale, 8), round(y / scale, 8))
                                  for x, y in shifted)))
    return min(forms)


def _coordinates(n, k, rng):
    """Independent 4–6 cluster geometries with deliberately unequal group sizes."""
    sizes = [max(5, n // k + rng.randint(-3, 3)) for _ in range(k)]
    while sum(sizes) < n:
        sizes[rng.randrange(k)] += 1
    while sum(sizes) > n:
        j = max(range(k), key=lambda i: sizes[i])
        if sizes[j] <= 5:
            raise RuntimeError("invalid cluster allocation")
        sizes[j] -= 1
    if len(set(sizes)) == 1:
        sizes[0] += 1
        sizes[1] -= 1
    layout = rng.choice(("ring", "two_rows", "asymmetric"))
    angle = rng.uniform(0, math.tau)
    centers = []
    for j in range(k):
        if layout == "ring":
            theta = angle + math.tau * j / k
            centers.append((500 + 320 * math.cos(theta), 500 + 320 * math.sin(theta)))
        elif layout == "two_rows":
            cols = math.ceil(k / 2)
            centers.append((130 + 740 * (j % cols) / max(cols - 1, 1),
                            230 + 540 * (j // cols)))
        else:
            theta = angle + math.tau * j / k
            radius = 250 + 75 * (j % 2)
            centers.append((500 + radius * math.cos(theta),
                            500 + radius * math.sin(theta)))
    points = []
    for (cx, cy), count in zip(centers, sizes):
        sigma_x, sigma_y = rng.uniform(12, 27), rng.uniform(12, 27)
        local_angle = rng.uniform(0, math.tau)
        co, si = math.cos(local_angle), math.sin(local_angle)
        for _ in range(count):
            u, v = rng.gauss(0, sigma_x), rng.gauss(0, sigma_y)
            points.append((round(cx + u * co - v * si),
                           round(cy + u * si + v * co)))
    if len(points) != n or len(set(points)) != n:
        return None
    return points


def _two_equal_best(points):
    """Verify only the best length found by the unchanged heuristic search."""
    cities = [City(i, x, y) for i, (x, y) in enumerate(points)]
    matrix = init_distance_matrix(cities)
    n = len(points)
    starts = (0, n // 4, n // 2, 3 * n // 4)
    seeds = []
    for start in starts:
        route, remaining = [start], set(range(n)) - {start}
        while remaining:
            nxt = min(remaining, key=lambda j: (matrix[route[-1]][j], j))
            route.append(nxt)
            remaining.remove(nxt)
        seeds.append(two_opt(route, matrix))
    best = min(item.length for item in seeds)
    best_routes = [item.route for item in seeds if item.length == best]
    plateau = equal_plateau(best_routes, matrix, 50_000)
    keys = sorted(plateau.routes)
    return best, keys[:2] if len(keys) >= 2 else []


def _source_signatures():
    signatures = set()
    train, validation = load_manifest({"root": str(ORIGINAL), "train_dir": "train",
                                       "validation_dir": "validation"})
    for partition in (train, validation):
        for item in partition:
            signatures.add(_signature([(c.x, c.y) for c in read_cities(item.path)]))
    for stem, _, _ in BENCHMARKS:
        signatures.add(_signature([(c.x, c.y) for c in read_cities(DATA / f"{stem}.tsp")]))
    return signatures


def _write_manifest(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "split", "relative_path",
                                                    "city_count", "distribution", "sha256"))
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path = DEFAULT_OUTPUT):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"new dataset directory required: {output}")
    summary = {r["split"]: r for r in csv.DictReader(
        (DIAGNOSIS / "distribution_summary.csv").open(encoding="utf-8-sig", newline=""))}
    # Fail closed if the diagnosis no longer shows the observed gap.
    if (int(summary["train"]["clear_4to6_cluster_unequal"]) >= 50 and
            int(summary["validation"]["clear_4to6_cluster_unequal"]) >= 10 and
            int(summary["train"]["matched_with_known_multiple_optima"]) >= 20):
        raise RuntimeError("original data already covers the configured target; no new data needed")
    train, validation = load_manifest({"root": str(ORIGINAL),
                                       "train_dir": "train", "validation_dir": "validation"})
    rng = random.Random(42)
    chosen = rng.sample(train, 300)
    chosen_signatures = {_signature([(c.x, c.y) for c in read_cities(item.path)])
                         for item in chosen}
    validation_signatures = {_signature([(c.x, c.y) for c in read_cities(item.path)])
                             for item in validation}
    if (len(chosen_signatures) != len(chosen) or
            len(validation_signatures) != len(validation) or
            chosen_signatures & validation_signatures):
        raise RuntimeError("original sampled train/validation contain duplicate or equivalent cases")
    signatures = _source_signatures()
    output.mkdir(parents=True)
    (output / "INCOMPLETE").write_text("dataset generation in progress\n", encoding="utf-8")
    rows, labels = [], []
    for split, partition in (("train", chosen), ("validation", validation)):
        for item in partition:
            relative = Path(split) / "original" / item.path.name
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item.path, destination)
            rows.append({"id": item.path.stem, "split": split,
                "relative_path": relative.as_posix(), "city_count": item.city_count,
                "distribution": item.distribution,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    for split, count in (("train", 200), ("validation", 50)):
        accepted = attempts = 0
        while accepted < count:
            attempts += 1
            if attempts > count * 100:
                raise RuntimeError(f"{split}: could not certify two equal heuristic-best routes; partial dataset marked INCOMPLETE")
            n = rng.choice((50, 55, 60, 65, 66, 70))
            k = rng.randint(4, 6)
            points = _coordinates(n, k, rng)
            if points is None:
                continue
            signature = _signature(points)
            if signature in signatures:
                continue
            cities = [City(i, x, y) for i, (x, y) in enumerate(points)]
            found_k, sizes, noise, _, _, _, ratio = _spatial_clusters(cities)
            if (not 4 <= found_k <= 6 or len(set(sizes)) < 2 or noise > 2 or
                    ratio is None or ratio < 3):
                continue
            best, tours = _two_equal_best(points)
            if len(tours) < 2 or tours[0] == tours[1]:
                continue
            accepted += 1
            signatures.add(signature)
            identity = f"complex_{split}_{accepted:03d}"
            relative = Path(split) / "complex" / f"{identity}.tsp"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("".join(f"{x} {y}\n" for x, y in points), encoding="utf-8")
            rows.append({"id": identity, "split": split,
                "relative_path": relative.as_posix(), "city_count": n,
                "distribution": "complex_multicluster",
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
            labels.append({"id": identity, "split": split, "city_count": n,
                "spatial_clusters": found_k, "spatial_cluster_sizes": sizes,
                "between_within_ratio": ratio, "heuristic_best_length": best,
                "two_equal_heuristic_best_routes": [list(x) for x in tours],
                "exact_optimality_proved": False,
                "note": "equal best within nearest-neighbor + ordinary 2-opt + 50000-check equal plateau; not certified global optimum"})
    _write_manifest(output / "manifest.csv", rows)
    (output / "heuristic_equal_routes.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    check_train, check_validation = load_manifest({"root": str(output),
        "train_dir": "train", "validation_dir": "validation"})
    if (len(check_train), len(check_validation), len(labels)) != (500, 150, 250):
        raise RuntimeError("mixed dataset counts mismatch")
    (output / "config.json").write_text(json.dumps({
        "seed": 42, "original_train_sample": 300, "new_complex_train": 200,
        "original_validation": 100, "new_complex_validation": 50,
        "test_coordinates_used_only_for_duplicate_guard": True,
        "test_optimal_answers_used": False,
        "optimality_claim": "none; only two distinct equal heuristic-best tours confirmed per new case",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "INCOMPLETE").unlink()
    (output / "READY").write_text("500 train / 150 validation\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
