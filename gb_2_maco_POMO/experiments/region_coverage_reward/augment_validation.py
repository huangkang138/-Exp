"""Recover raw-route mean validation metrics without changing model selection."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import torch

from gb_2_maco_POMO.experiments.learnable_ball_bias.rollout import generate_routes
from gb_2_maco_POMO.experiments.shared.data import load_manifest
from gb_2_maco_POMO.experiments.shared.opt_b_search import tour_length
from gb_2_maco_POMO.runner import prepare_tsp

from .evaluate25 import load_model, write_csv
from .train import DATA_ROOT, GROUPS, PROJECT


ROOT = PROJECT / "gb_2_maco_POMO/results/region_coverage_reward"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    torch.set_num_threads(1)
    _, validation = load_manifest({"root": str(DATA_ROOT), "train_dir": "train",
                                   "validation_dir": "validation"})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for group in GROUPS:
        for seed in (42, 43, 44):
            directory = ROOT / group / f"train_seed_{seed}"
            history = json.loads((directory / "validation_history.json").read_text(encoding="utf-8"))
            for entry in history:
                epoch = entry["epoch"]
                output = directory / f"validation_augmented_epoch_{epoch:03d}.csv"
                if output.is_file():
                    continue
                rows = read_csv(directory / f"validation_epoch_{epoch:03d}.csv")
                if len(rows) != len(validation):
                    raise RuntimeError("incomplete validation rows")
                checkpoint = directory / "checkpoints" / f"epoch_{epoch:03d}.pt"
                model = load_model(checkpoint, group).to(device)
                augmented = []
                for index, (item, row) in enumerate(zip(validation, rows), 1):
                    if row["instance"] != item.path.name:
                        raise RuntimeError("validation case order changed")
                    problem = prepare_tsp(item.path)
                    routes = generate_routes(model, problem, samples=8, seed=seed + index,
                                             device=device)
                    lengths = [tour_length(route, problem.distance_matrix) for route in routes]
                    matched = len(lengths) == item.city_count * 9 and min(lengths) == int(row["raw_best"])
                    reference = int(row["reference_length"])
                    augmented.append({**row, "raw_replay_match": int(matched),
                                      "raw_mean_length": statistics.mean(lengths) if matched else "",
                                      "raw_mean_gap_pct": (100 * (statistics.mean(lengths) / reference - 1)
                                                           if matched else "")})
                write_csv(output, augmented)
                matched_rows = [row for row in augmented if row["raw_replay_match"]]
                metrics = {"epoch": epoch, "seed": seed, "group": group,
                           "raw_replay_matched_cases": len(matched_rows),
                           "raw_replay_best_matched_all_cases": len(matched_rows) == 100,
                           "limitation": "original validation raw tours were not saved; matching best length does not prove route-by-route identity",
                           "mean_raw_route_length": (statistics.mean(r["raw_mean_length"] for r in matched_rows)
                                                     if len(matched_rows) == 100 else None),
                           "mean_raw_route_gap_pct": (statistics.mean(r["raw_mean_gap_pct"] for r in matched_rows)
                                                      if len(matched_rows) == 100 else None),
                           "mean_best_2opt_length": statistics.mean(int(r["two_opt_best"]) for r in augmented),
                           "mean_best_2opt_gap_pct": statistics.mean(100 * (int(r["two_opt_best"])
                                                                      / int(r["reference_length"]) - 1)
                                                                     for r in augmented)}
                (directory / f"validation_augmented_epoch_{epoch:03d}.json").write_text(
                    json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(group, seed, epoch, f"raw replay matched {len(matched_rows)}/100", flush=True)


if __name__ == "__main__":
    main()
