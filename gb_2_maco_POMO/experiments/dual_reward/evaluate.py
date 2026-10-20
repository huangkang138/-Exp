"""Fixed seed-42 S0 protocol for the baseline and both selected checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from gb_maco_POMO.experiments.compare import BENCHMARKS, load_ground_truth
from gb_2_maco_POMO.experiments.hierarchical import original25
from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import (
    generate_learned_routes, load_checkpoint,
)
from gb_2_maco_POMO.experiments.region_supervised_pomo.config import INITIAL, INITIAL_SHA256
from gb_2_maco_POMO.experiments.region_supervised_pomo.data import _project_components
from gb_2_maco_POMO.runner import prepare_tsp

from .settings import Settings, load as load_settings
from .train import save_csv, save_json

EXPECTED_A0 = {"distance_hits": 25, "two_opt_hits": 283,
               "opt_b_hits": 669, "regions_hit": 105}


def _regions(path: Path, truth, regions):
    saved = json.loads(path.read_text(encoding="utf-8"))
    returned = {tuple(route) for route in saved["B"]["routes"]} & truth
    return sum(bool(returned & region) for region in regions)


def run(root: Path, config: Settings | None = None) -> dict:
    config = config or load_settings()
    test = root / "test"
    if test.exists():
        raise FileExistsError(f"new test directory required: {test}")
    test.mkdir(parents=True)
    torch.set_num_threads(1)
    original25.SEED = config.seed
    if hashlib.sha256(INITIAL.read_bytes()).hexdigest() != INITIAL_SHA256:
        raise RuntimeError("A0 checkpoint changed")
    sources = {"A0": config.initial_checkpoint,
               **{f"{group}_{kind}": root / group / "checkpoints" / f"best_{kind}.pt"
                  for group in ("A", "B") for kind in ("length", "multi")}}
    training = {group: json.loads((root / group / "train_result.json").read_text(encoding="utf-8"))
                for group in ("A", "B")}
    paper = original25._csv(original25.PAPER)
    rows = []
    for label, path in sources.items():
        if label != "A0":
            group, kind = label.split("_")
            selection = json.loads((root / group / "selection.json").read_text(encoding="utf-8"))
            if hashlib.sha256(path.read_bytes()).hexdigest() != selection[f"best_{kind}"]["checkpoint_sha256"]:
                raise RuntimeError(f"{label}: selected checkpoint hash differs from training record")
        model, _ = load_checkpoint(path)
        model.cpu().eval()
        for folder in ("raw_routes", "best_routes", "cases"):
            (test / label / folder).mkdir(parents=True, exist_ok=True)
        for index, (stem, n, _) in enumerate(BENCHMARKS, 1):
            name = f"MSTSP{index}"
            row = original25.run_case(model, index, label, 0.0, test,
                                      float(paper[name]["paper_fbeta"]),
                                      route_generator=generate_learned_routes)
            problem = prepare_tsp(original25.DATA / f"{stem}.tsp")
            truth = load_ground_truth(original25.DATA / f"{stem}.solution", n)
            tours = tuple(sorted(truth))
            components = [set(tours[j] for j in indices)
                          for indices in _project_components(tours, problem.distance_matrix)]
            row["regions_hit"] = _regions(test / label / "best_routes" / f"{name}.json",
                                          truth, components)
            row["known_regions"] = len(components)
            save_json(test / label / "cases" / f"{name}.json", row)
            rows.append(row)
            print(f"{label} {name}: 2-opt={row['a_hits']} Opt-B={row['b_hits']} "
                  f"regions={row['regions_hit']}", flush=True)
    save_csv(test / "per_instance.csv", rows)
    summary = {}
    for label, path in sources.items():
        part = [row for row in rows if row["mode"] == label]
        summary[label] = {
            "checkpoint": str(path),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "validation_selected": True,
            "selected_epoch": 0 if label == "A0" else training[label[0]][f"best_{label.split('_')[1]}_epoch"],
            "distance_hits": sum(row["b_distance_hit"] for row in part),
            "raw_distance_hits": sum(row["raw_best"] == row["reference_length"] for row in part),
            "two_opt_distance_hits": sum(row["a_distance_hit"] for row in part),
            "two_opt_hits": sum(row["a_hits"] for row in part),
            "opt_b_hits": sum(row["b_hits"] for row in part),
            "regions_hit": sum(row["regions_hit"] for row in part),
            "mean_raw_gap_pct": sum(row["raw_gap_pct"] for row in part) / 25,
            "mean_two_opt_gap_pct": sum(row["a_gap_pct"] for row in part) / 25,
            "mean_opt_b_gap_pct": sum(row["b_gap_pct"] for row in part) / 25,
            "mean_two_opt_fbeta": sum(row["a_fbeta"] for row in part) / 25,
            "mean_opt_b_fbeta": sum(row["b_fbeta"] for row in part) / 25,
        }
    if any(summary["A0"][key] != value for key, value in EXPECTED_A0.items()) or abs(
            summary["A0"]["mean_opt_b_fbeta"] - 0.9011984576780031) > 1e-9:
        raise RuntimeError("A0 historical protocol baseline does not reproduce")
    save_json(test / "summary.json", summary)
    paired = []
    for i in range(1, 26):
        name = f"MSTSP{i}"
        cases = {row["mode"]: row for row in rows if row["instance"] == name}
        record = {"instance": name, "reference_length": cases["A0"]["reference_length"],
                  "truth_count": cases["A0"]["truth_count"],
                  "known_regions": cases["A0"]["known_regions"],
                  "paper_fbeta": cases["A0"]["paper_fbeta"]}
        for label in sources:
            case = cases[label]
            for key in ("raw_best", "raw_gap_pct", "a_best", "a_gap_pct",
                        "a_routes", "a_hits", "a_precision", "a_recall", "a_fbeta",
                        "b_best", "b_gap_pct", "b_routes", "b_hits",
                        "b_precision", "b_recall", "b_fbeta", "regions_hit"):
                record[f"{label}_{key}"] = case[key]
        paired.append(record)
    save_csv(test / "paired_per_instance.csv", paired)
    return summary
