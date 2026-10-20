"""Official 25-case evaluation; call only after the user finishes training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gb_maco_POMO.experiments.compare import BENCHMARKS, load_ground_truth
from gb_2_maco_POMO.experiments.hierarchical import original25
from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import generate_learned_routes
from gb_2_maco_POMO.experiments.region_supervised_pomo.data import _project_components
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical
from gb_2_maco_POMO.runner import prepare_tsp

from .common import load_selected, sha256
from .config import (INFERENCE_SEEDS, INITIAL_SHA256, OPT_B_CHECKS, RESULTS,
                     TRAIN_SEEDS)
from .train import save_csv, save_json


def run_test(group, train_seed, inference_seed):
    if group not in ("A", "B") or train_seed not in TRAIN_SEEDS or inference_seed not in INFERENCE_SEEDS:
        raise ValueError("unknown model or seed")
    if original25.SAMPLES != 8 or original25.OPT_B_CHECKS != OPT_B_CHECKS:
        raise RuntimeError("historical inference or Opt-B budget changed")
    torch.set_num_threads(1)
    train_dir = RESULTS / f"train_seed_{train_seed}" / group
    result = json.loads((train_dir / "train_result.json").read_text(encoding="utf-8"))
    selection = json.loads((train_dir / "selection.json").read_text(encoding="utf-8"))
    if (result["completed_epochs"] != 3 or result["total_optimizer_updates"] != 33
            or result["selected_epoch"] != selection["selected_epoch"]
            or selection["test_set_used_for_selection"] is not False):
        raise RuntimeError("training incomplete or test-set selection detected")
    checkpoint = Path(selection["checkpoint"])
    digest = sha256(checkpoint)
    if digest != result["selected_checkpoint_sha256"] or result["initial_sha256"] != INITIAL_SHA256:
        raise RuntimeError("selected checkpoint changed")
    output = RESULTS / "tests" / group / f"train_seed_{train_seed}" / f"infer_seed_{inference_seed}"
    config = {"group": group, "train_seed": train_seed,
              "inference_seed": inference_seed, "checkpoint": str(checkpoint),
              "checkpoint_sha256": digest, "initial_sha256": INITIAL_SHA256,
              "device": "cpu", "starts": "all cities", "argmax_rounds": 1,
              "sampling_rounds": 8, "two_opt": "all unique raw routes",
              "optb_pair_checks": OPT_B_CHECKS, "source": "original25.run_case",
              "beta_squared": 0.3}
    if output.exists():
        if json.loads((output / "run_config.json").read_text(encoding="utf-8")) != config:
            raise RuntimeError(f"existing test protocol differs: {output}")
    else:
        output.mkdir(parents=True)
        save_json(output / "run_config.json", config)
    for name in ("raw_routes", "best_routes"):
        (output / group / name).mkdir(parents=True, exist_ok=True)
    policy = load_selected(checkpoint, group, torch.device("cpu"))
    policy.eval()
    original25.SEED = inference_seed
    paper = original25._csv(original25.PAPER)
    rows = []
    for index, (stem, city_count, _) in enumerate(BENCHMARKS, 1):
        name = f"MSTSP{index}"
        case_path = output / f"case_{index:02d}.json"
        if case_path.is_file():
            rows.append(json.loads(case_path.read_text(encoding="utf-8")))
            continue
        row = original25.run_case(
            policy, index, group, 0.0, output, float(paper[name]["paper_fbeta"]),
            route_generator=generate_learned_routes)
        problem = prepare_tsp(original25.DATA / f"{stem}.tsp")
        truth = load_ground_truth(original25.DATA / f"{stem}.solution", city_count)
        truth_tours = tuple(sorted(truth))
        components = _project_components(truth_tours, problem.distance_matrix)
        best = json.loads((output / group / "best_routes" / f"{name}.json").read_text(encoding="utf-8"))
        a = {tuple(t) for t in best["A"]["routes"]} & truth
        b = {tuple(t) for t in best["B"]["routes"]} & truth
        raw = json.loads((output / group / "raw_routes" / f"{name}.json").read_text(encoding="utf-8"))
        raw_unique = {canonical(route) for route in raw}
        raw_true = raw_unique & truth
        region_sets = [{truth_tours[i] for i in comp} for comp in components]
        item = {"model": group, "train_seed": train_seed,
                "inference_seed": inference_seed, **row,
                "raw_unique": len(raw_unique), "raw_true_optimal": len(raw_true),
                "raw_duplicate_rate": 1 - len(raw_unique) / len(raw),
                "raw_regions_hit": sum(bool(raw_true & region) for region in region_sets),
                "a_regions_hit": sum(bool(a & region) for region in region_sets),
                "b_regions_hit": sum(bool(b & region) for region in region_sets),
                "truth_regions": len(region_sets)}
        save_json(case_path, item)
        rows.append(item)
        print(f"{group} train={train_seed} infer={inference_seed} {name} "
              f"distance_hit={row['b_distance_hit']} true_tours={row['b_hits']}", flush=True)
    save_csv(output / "per_instance.csv", rows)
    save_json(output / "summary.json", {
        "group": group, "train_seed": train_seed, "inference_seed": inference_seed,
        "checkpoint_sha256": digest, "cases": len(rows),
        "raw_distance_hits": sum(r["raw_gap_pct"] == 0 for r in rows),
        "two_opt_distance_hits": sum(r["a_distance_hit"] for r in rows),
        "optb_distance_hits": sum(r["b_distance_hit"] for r in rows),
        "raw_true_optimal": sum(r["raw_true_optimal"] for r in rows),
        "two_opt_true_optimal": sum(r["a_hits"] for r in rows),
        "optb_true_optimal": sum(r["b_hits"] for r in rows),
        "regions_hit": sum(r["b_regions_hit"] for r in rows),
        "mean_fbeta": sum(r["b_fbeta"] for r in rows) / len(rows),
        "inference_seconds": sum(r["inference_seconds"] for r in rows),
        "two_opt_seconds": sum(r["two_opt_seconds"] for r in rows),
        "optb_seconds": sum(r["optb_seconds"] for r in rows)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seed", type=int, choices=TRAIN_SEEDS)
    parser.add_argument("--inference-seed", type=int, choices=INFERENCE_SEEDS)
    args = parser.parse_args()
    for seed in (args.train_seed,) if args.train_seed is not None else TRAIN_SEEDS:
        for inference in ((args.inference_seed,) if args.inference_seed is not None
                          else INFERENCE_SEEDS):
            for group in ("A", "B"):
                run_test(group, seed, inference)


if __name__ == "__main__":
    main()
