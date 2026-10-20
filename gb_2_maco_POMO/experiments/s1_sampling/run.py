"""Run S1 against the existing, immutable A0/S0 test records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gb_maco_POMO.experiments.compare import BENCHMARKS, load_ground_truth
from gb_2_maco_POMO.experiments.hierarchical import original25
from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import load_checkpoint
from gb_2_maco_POMO.experiments.region_supervised_pomo.config import (
    INFERENCE_SEEDS, INITIAL, INITIAL_SHA256, RESULTS,
)
from gb_2_maco_POMO.experiments.region_supervised_pomo.data import _project_components, sha256
from gb_2_maco_POMO.experiments.region_supervised_pomo.train import save_csv, save_json
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical
from gb_2_maco_POMO.runner import prepare_tsp

from .routes import generate_s1_routes

OUTPUT = RESULTS.parent / "s1_sampling"


def _s0_folder(seed: int) -> Path:
    folder = RESULTS / "tests" / "A0" / "initial" / f"infer_seed_{seed}"
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    expected = {"mode": "A0", "inference_seed": seed,
                "checkpoint_sha256": INITIAL_SHA256, "starts": "all cities",
                "argmax_rounds": 1, "sampling_rounds": 8,
                "two_opt": "all unique raw routes", "optb_pair_checks": 50_000,
                "source": "original25.run_case", "beta_squared": 0.3}
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"S0 protocol differs: {folder}")
    if Path(config["checkpoint"]).resolve() != INITIAL.resolve():
        raise RuntimeError("S0 used a different checkpoint")
    if len(list(folder.glob("case_*.json"))) != 25 or not (folder / "summary.json").is_file():
        raise RuntimeError(f"S0 seed {seed} is incomplete")
    return folder


def run_seed(seed: int) -> None:
    if seed not in INFERENCE_SEEDS or sha256(INITIAL) != INITIAL_SHA256:
        raise RuntimeError("seed or checkpoint changed")
    s0 = _s0_folder(seed)
    output = OUTPUT / f"infer_seed_{seed}"
    config = {"experiment": "S1 ball argmax + city sampling", "inference_seed": seed,
              "checkpoint": str(INITIAL), "checkpoint_sha256": INITIAL_SHA256,
              "s0_source": str(s0), "starts": "all cities", "argmax_rounds": 1,
              "sampling_rounds": 8, "sampling_ball": "argmax each step",
              "sampling_city": "categorical within chosen ball",
              "two_opt": "all unique raw routes", "optb_pair_checks": 50_000,
              "source": "original25.run_case", "beta_squared": 0.3, "device": "cpu"}
    if output.exists():
        if json.loads((output / "run_config.json").read_text(encoding="utf-8")) != config:
            raise RuntimeError(f"existing S1 protocol differs: {output}")
    else:
        output.mkdir(parents=True)
        save_json(output / "run_config.json", config)
    for name in ("raw_routes", "best_routes"):
        (output / "S1" / name).mkdir(parents=True, exist_ok=True)
    policy, _ = load_checkpoint(INITIAL)
    torch.set_num_threads(1)
    original25.SEED = seed
    paper = original25._csv(original25.PAPER)
    cases = []
    for index, (stem, n, _) in enumerate(BENCHMARKS, 1):
        name = f"MSTSP{index}"
        case_path = output / f"case_{index:02d}.json"
        if case_path.exists():
            cases.append(json.loads(case_path.read_text(encoding="utf-8")))
            continue
        row = original25.run_case(policy, index, "S1", 0.0, output,
                                  float(paper[name]["paper_fbeta"]),
                                  route_generator=generate_s1_routes)
        raw = json.loads((output / "S1" / "raw_routes" / f"{name}.json").read_text(encoding="utf-8"))
        old_raw = json.loads((s0 / "A0" / "raw_routes" / f"{name}.json").read_text(encoding="utf-8"))
        if raw[:n] != old_raw[:n]:
            raise RuntimeError(f"S1 changed the argmax routes: {name}, seed {seed}")
        old_row = json.loads((s0 / f"case_{index:02d}.json").read_text(encoding="utf-8"))
        if (row["reference_length"], row["truth_count"], row["raw_routes"]) != (
                old_row["reference_length"], old_row["truth_count"], old_row["raw_routes"]):
            raise RuntimeError(f"S0/S1 case protocol differs: {name}, seed {seed}")
        problem = prepare_tsp(original25.DATA / f"{stem}.tsp")
        truth = load_ground_truth(original25.DATA / f"{stem}.solution", n)
        truth_tours = tuple(sorted(truth))
        components = _project_components(truth_tours, problem.distance_matrix)
        best = json.loads((output / "S1" / "best_routes" / f"{name}.json").read_text(encoding="utf-8"))
        a = {tuple(route) for route in best["A"]["routes"]} & truth
        b = {tuple(route) for route in best["B"]["routes"]} & truth
        raw_unique = {canonical(route) for route in raw}
        raw_true = raw_unique & truth
        region_sets = [{truth_tours[i] for i in component} for component in components]
        item = {"model": "S1", "inference_seed": seed, **row,
                "raw_unique": len(raw_unique), "raw_true_optimal": len(raw_true),
                "raw_duplicate_rate": 1 - len(raw_unique) / len(raw),
                "raw_regions_hit": sum(bool(raw_true & region) for region in region_sets),
                "a_regions_hit": sum(bool(a & region) for region in region_sets),
                "b_regions_hit": sum(bool(b & region) for region in region_sets),
                "truth_regions": len(region_sets),
                "b_new_true_optimal": len(b - a), "argmax_matches_s0": True}
        save_json(case_path, item)
        cases.append(item)
        print(f"S1 seed={seed} {name} distance_hit={row['b_distance_hit']} "
              f"true_tours={row['b_hits']}", flush=True)
    save_csv(output / "per_instance.csv", cases)
    save_json(output / "summary.json", {
        "seed": seed, "cases": len(cases), "checkpoint_sha256": INITIAL_SHA256,
        "distance_hits": sum(r["b_distance_hit"] for r in cases),
        "true_optimal_tours": sum(r["b_hits"] for r in cases),
        "regions_hit": sum(r["b_regions_hit"] for r in cases),
        "mean_fbeta": sum(r["b_fbeta"] for r in cases) / len(cases),
        "argmax_matches_s0": all(r["argmax_matches_s0"] for r in cases),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=INFERENCE_SEEDS)
    args = parser.parse_args()
    for seed in (args.seed,) if args.seed is not None else INFERENCE_SEEDS:
        run_seed(seed)


if __name__ == "__main__":
    main()
