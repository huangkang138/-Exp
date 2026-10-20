"""Original MSTSP1-25 inference and unchanged 2-opt/Opt-B postprocessing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from gb_maco_POMO.experiments.compare import BENCHMARKS, load_ground_truth
from gb_2_maco_POMO.experiments.hierarchical import original25
from gb_2_maco_POMO.experiments.hierarchical.selected_rollout import load_policy
from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import (
    generate_learned_routes, load_checkpoint)
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, tour_length
from gb_2_maco_POMO.runner import prepare_tsp

from .config import INITIAL, INITIAL_SHA256, INFERENCE_SEEDS, RESULTS, TRAIN_SEEDS
from .data import _project_components, sha256
from .train import save_csv, save_json


def checkpoint_for(mode, train_seed):
    if mode == "fixed_soft":
        checkpoint = original25.CHECKPOINT
        expected_sha = original25.EXPECTED_CHECKPOINT
    elif mode == "A0":
        checkpoint = INITIAL
        expected_sha = INITIAL_SHA256
    else:
        if train_seed not in TRAIN_SEEDS:
            raise ValueError("trained group requires train seed 42/43/44")
        directory = RESULTS / f"train_seed_{train_seed}" / mode
        result = json.loads((directory / "train_result.json").read_text(encoding="utf-8"))
        if result["completed_epochs"] < 3 and not result["stopped_early"]:
            raise RuntimeError(f"training incomplete: {directory}")
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        if selection["test_set_used_for_selection"] is not False:
            raise RuntimeError("checkpoint selection used the final test set")
        checkpoint = Path(selection["checkpoint"])
        expected_sha = sha256(checkpoint)
        if result["selected_epoch"] != selection["selected_epoch"]:
            raise RuntimeError("selection result mismatch")
        if (selection["selected_epoch"] == 0) != (checkpoint.resolve() == INITIAL.resolve()):
            raise RuntimeError("epoch 0 must use the common initial checkpoint")
    if sha256(checkpoint) != expected_sha:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
    return checkpoint, expected_sha


def test_model(mode, train_seed, inference_seed):
    if mode not in ("fixed_soft", "A0", "A", "B", "C") or inference_seed not in INFERENCE_SEEDS:
        raise ValueError("unknown test mode or inference seed")
    torch.set_num_threads(1)
    checkpoint, digest = checkpoint_for(mode, train_seed)
    output = RESULTS / "tests" / mode / (
        f"train_seed_{train_seed}" if train_seed is not None else "initial") / f"infer_seed_{inference_seed}"
    config = {"mode": mode, "train_seed": train_seed, "inference_seed": inference_seed,
              "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
              "starts": "all cities", "argmax_rounds": 1, "sampling_rounds": 8,
              "two_opt": "all unique raw routes", "optb_pair_checks": 50_000,
              "source": "original25.run_case", "beta_squared": 0.3}
    if output.exists():
        saved = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        if saved != config:
            raise RuntimeError(f"test output protocol mismatch: {output}")
    else:
        output.mkdir(parents=True)
        save_json(output / "run_config.json", config)
    if mode == "fixed_soft":
        policy, _ = load_policy(checkpoint, "cpu")
    else:
        policy, state = load_checkpoint(checkpoint)
        if (mode in ("A", "B", "C") and checkpoint.resolve() != INITIAL.resolve()
                and (state.get("group"), state.get("seed")) != (mode, train_seed)):
            raise RuntimeError("trained checkpoint identity mismatch")
    (output / mode / "raw_routes").mkdir(parents=True, exist_ok=True)
    (output / mode / "best_routes").mkdir(parents=True, exist_ok=True)
    original25.SEED = inference_seed
    paper = original25._csv(original25.PAPER)
    cases = []
    for index, (stem, n, _) in enumerate(BENCHMARKS, 1):
        name = f"MSTSP{index}"
        per_case = output / f"case_{index:02d}.json"
        if per_case.is_file():
            cases.append(json.loads(per_case.read_text(encoding="utf-8")))
            continue
        row = original25.run_case(
            policy, index, mode, 0.0, output, float(paper[name]["paper_fbeta"]),
            route_generator=None if mode == "fixed_soft" else generate_learned_routes)
        problem = prepare_tsp(original25.DATA / f"{stem}.tsp")
        truth = load_ground_truth(original25.DATA / f"{stem}.solution", n)
        truth_tours = tuple(sorted(truth))
        components = _project_components(truth_tours, problem.distance_matrix)
        best = json.loads((output / mode / "best_routes" / f"{name}.json").read_text(encoding="utf-8"))
        a = {tuple(t) for t in best["A"]["routes"]} & truth
        b = {tuple(t) for t in best["B"]["routes"]} & truth
        raw = json.loads((output / mode / "raw_routes" / f"{name}.json").read_text(encoding="utf-8"))
        unique_raw = {canonical(route) for route in raw}
        raw_true = unique_raw & truth
        region_sets = [{truth_tours[i] for i in region} for region in components]
        item = {"model": mode, "train_seed": train_seed if train_seed is not None else "",
                "inference_seed": inference_seed, **row,
                "raw_unique": len(unique_raw), "raw_true_optimal": len(raw_true),
                "raw_duplicate_rate": 1 - len(unique_raw) / len(raw),
                "raw_regions_hit": sum(bool(raw_true & region) for region in region_sets),
                "a_regions_hit": sum(bool(a & region) for region in region_sets),
                "b_regions_hit": sum(bool(b & region) for region in region_sets),
                "truth_regions": len(region_sets),
                "b_new_true_optimal": len(b - a)}
        per_case.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
        cases.append(item)
        print(f"{mode} train={train_seed} infer={inference_seed} {name} "
              f"distance_hit={row['b_distance_hit']} true_tours={row['b_hits']}", flush=True)
    save_csv(output / "per_instance.csv", cases)
    summary = {"mode": mode, "train_seed": train_seed, "inference_seed": inference_seed,
               "checkpoint_sha256": digest, "cases": len(cases),
               "raw_distance_hits": sum(r["raw_gap_pct"] == 0 for r in cases),
               "two_opt_distance_hits": sum(r["a_distance_hit"] for r in cases),
               "optb_distance_hits": sum(r["b_distance_hit"] for r in cases),
               "raw_true_optimal": sum(r["raw_true_optimal"] for r in cases),
               "two_opt_true_optimal": sum(r["a_hits"] for r in cases),
               "optb_true_optimal": sum(r["b_hits"] for r in cases),
               "truth_regions_hit": sum(r["b_regions_hit"] for r in cases),
               "truth_regions_total": sum(r["truth_regions"] for r in cases),
               "mean_fbeta": sum(r["b_fbeta"] for r in cases) / len(cases),
               "mean_raw_duplicate_rate": sum(r["raw_duplicate_rate"] for r in cases) / len(cases),
               "inference_seconds": sum(r["inference_seconds"] for r in cases),
               "two_opt_seconds": sum(r["two_opt_seconds"] for r in cases),
               "optb_seconds": sum(r["optb_seconds"] for r in cases)}
    if mode == "fixed_soft" and inference_seed == 42:
        if (summary["optb_distance_hits"] != 25 or summary["optb_true_optimal"] != 663
                or abs(summary["mean_fbeta"] - 0.8968434047253819) > 1e-12):
            raise RuntimeError("fixed Soft historical seed-42 result mismatch")
    if mode == "A0" and inference_seed == 42:
        if (summary["optb_distance_hits"] != 25 or summary["optb_true_optimal"] != 669
                or abs(summary["mean_fbeta"] - 0.901198457678003) > 1e-12):
            raise RuntimeError("initial ball policy historical seed-42 result mismatch")
    save_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixed_soft", "A0", "A", "B", "C"), required=True)
    parser.add_argument("--train-seed", type=int, choices=TRAIN_SEEDS)
    parser.add_argument("--inference-seed", type=int, choices=INFERENCE_SEEDS, required=True)
    args = parser.parse_args()
    if (args.mode in ("A", "B", "C")) != (args.train_seed is not None):
        parser.error("train seed required only for A/B/C")
    test_model(args.mode, args.train_seed, args.inference_seed)


if __name__ == "__main__":
    main()
