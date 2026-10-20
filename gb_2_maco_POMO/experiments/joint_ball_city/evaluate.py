"""Formal seed-42 MSTSP1-25 evaluation of both validation-selected checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch

from gb_maco_POMO.experiments.compare import BENCHMARKS, load_ground_truth
from gb_2_maco_POMO.experiments.dual_reward.train import save_csv, save_json
from gb_2_maco_POMO.experiments.hierarchical import original25
from gb_2_maco_POMO.experiments.region_supervised_pomo.config import INITIAL_SHA256
from gb_2_maco_POMO.experiments.region_supervised_pomo.data import _project_components
from gb_2_maco_POMO.runner import prepare_tsp

from .inference import generate_routes, load_checkpoint

BASELINE = (Path(__file__).resolve().parents[3] /
            "gb_2_maco_POMO/results/dual_reward/"
            "run_seed_42_multiopt_500_10epoch/experiment/test")
LABELS = ("A0", "Joint_shortest", "Joint_diversity")


def _regions(path: Path, truth, regions) -> int:
    saved = json.loads(path.read_text(encoding="utf-8"))
    returned = {tuple(route) for route in saved["B"]["routes"]} & truth
    return sum(bool(returned & region) for region in regions)


def _baseline_rows(config):
    summary = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))["A0"]
    if (summary["checkpoint_sha256"] != INITIAL_SHA256 or
            summary["distance_hits"] != 25 or summary["two_opt_hits"] != 283 or
            summary["opt_b_hits"] != 669 or summary["regions_hit"] != 105 or
            abs(summary["mean_opt_b_fbeta"] - 0.9011984576780031) > 1e-10):
        raise RuntimeError("historical A0 formal-test baseline changed")
    with (BASELINE / "per_instance.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["mode"] == "A0"]
    if len(rows) != 25 or any(int(r["seed"]) != config.seed + index or
                              int(r["samples_per_start"]) != config.validation_samples_per_start
                              for index, r in enumerate(rows, 1)):
        raise RuntimeError("historical A0 per-instance protocol differs")
    return rows, summary


def run(root: Path, config) -> dict:
    output = root / "test"
    if output.exists():
        raise FileExistsError(f"test output exists: {output}")
    output.mkdir()
    torch.set_num_threads(1)
    original25.SEED = config.seed
    original25.SAMPLES = config.validation_samples_per_start
    original25.OPT_B_CHECKS = config.validation_optb_checks
    baseline, baseline_summary = _baseline_rows(config)
    rows = list(baseline)
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    paper = original25._csv(original25.PAPER)
    for kind in ("shortest", "diversity"):
        label = f"Joint_{kind}"
        path = root / "checkpoints" / f"best_{kind}.pt"
        chosen = selection[f"best_{kind}"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != chosen["checkpoint_sha256"]:
            raise RuntimeError(f"{label}: checkpoint differs from independent validation")
        model, _ = load_checkpoint(path)
        # Historical A0 formal testing used CPU. Keep the evaluation device
        # identical; CUDA cost is measured separately for all three models.
        model.cpu().eval()
        for folder in ("raw_routes", "best_routes", "cases"):
            (output / label / folder).mkdir(parents=True, exist_ok=True)
        for index, (stem, n, _) in enumerate(BENCHMARKS, 1):
            name = f"MSTSP{index}"
            row = original25.run_case(model, index, label, 0.0, output,
                                      float(paper[name]["paper_fbeta"]),
                                      route_generator=generate_routes)
            prepared = prepare_tsp(original25.DATA / f"{stem}.tsp")
            truth = load_ground_truth(original25.DATA / f"{stem}.solution", n)
            tours = tuple(sorted(truth))
            regions = [set(tours[j] for j in indices)
                       for indices in _project_components(tours, prepared.distance_matrix)]
            row["regions_hit"] = _regions(output / label / "best_routes" / f"{name}.json",
                                          truth, regions)
            row["known_regions"] = len(regions)
            save_json(output / label / "cases" / f"{name}.json", row)
            rows.append(row)
            print(f"{label} {name}: distance={row['b_best']} "
                  f"true tours={row['b_hits']} regions={row['regions_hit']}", flush=True)
        del model
    # The reused A0 rows and fresh Joint rows use identical field names.
    fields = list(rows[-1])
    if any(set(row) != set(fields) for row in rows):
        raise RuntimeError("A0 and Joint formal-test record schemas differ")
    save_csv(output / "per_instance.csv", rows)
    summary = {"A0": baseline_summary}
    for kind in ("shortest", "diversity"):
        label = f"Joint_{kind}"
        part = [r for r in rows if r["mode"] == label]
        path = root / "checkpoints" / f"best_{kind}.pt"
        summary[label] = {"checkpoint": str(path),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selected_epoch": selection[f"best_{kind}"]["epoch"],
            "distance_hits": sum(int(r["b_distance_hit"]) for r in part),
            "raw_distance_hits": sum(int(r["raw_best"]) == int(r["reference_length"]) for r in part),
            "two_opt_distance_hits": sum(int(r["a_distance_hit"]) for r in part),
            "two_opt_hits": sum(int(r["a_hits"]) for r in part),
            "opt_b_hits": sum(int(r["b_hits"]) for r in part),
            "regions_hit": sum(int(r["regions_hit"]) for r in part),
            "mean_raw_gap_pct": sum(float(r["raw_gap_pct"]) for r in part) / 25,
            "mean_two_opt_gap_pct": sum(float(r["a_gap_pct"]) for r in part) / 25,
            "mean_opt_b_gap_pct": sum(float(r["b_gap_pct"]) for r in part) / 25,
            "mean_two_opt_fbeta": sum(float(r["a_fbeta"]) for r in part) / 25,
            "mean_opt_b_fbeta": sum(float(r["b_fbeta"]) for r in part) / 25}
    save_json(output / "summary.json", summary)
    paired = []
    per_case = {(r["instance"], r["mode"]): r for r in rows}
    keys = ("raw_best", "raw_gap_pct", "a_best", "a_gap_pct", "a_routes",
            "a_hits", "a_precision", "a_recall", "a_fbeta", "b_best",
            "b_gap_pct", "b_routes", "b_hits", "b_precision", "b_recall",
            "b_fbeta", "regions_hit")
    for index in range(1, 26):
        name = f"MSTSP{index}"
        a0 = per_case[name, "A0"]
        record = {"instance": name, "reference_length": a0["reference_length"],
                  "truth_count": a0["truth_count"], "known_regions": a0["known_regions"],
                  "paper_fbeta": a0["paper_fbeta"]}
        for label in LABELS:
            row = per_case[name, label]
            record.update({f"{label}_{key}": row[key] for key in keys})
        paired.append(record)
    save_csv(output / "paired_per_instance.csv", paired)
    return summary
