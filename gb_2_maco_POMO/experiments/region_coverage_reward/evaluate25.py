"""Final-only evaluation through the project's unchanged original25.run_case."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from gb_maco_POMO.experiments.compare import BENCHMARKS, load_ground_truth
from gb_2_maco_POMO.experiments.hierarchical import original25
from gb_2_maco_POMO.experiments.hierarchical.selected_rollout import load_policy
from gb_2_maco_POMO.experiments.learnable_ball_bias.model import BallBiasPolicy
from gb_2_maco_POMO.experiments.learnable_ball_bias.rollout import generate_routes
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, tour_length
from gb_2_maco_POMO.runner import prepare_tsp

from .train import INITIAL_SHA


def truth_regions(truth, distance):
    """Connected components of the listed true tours under one equal 2-opt swap."""
    remaining = set(truth)
    regions = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        region = {first}
        stack = [first]
        while stack:
            route = stack.pop()
            n = len(route)
            for i in range(n - 2):
                a, b = route[i], route[i + 1]
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    c, d = route[j], route[(j + 1) % n]
                    if distance[a][c] + distance[b][d] != distance[a][b] + distance[c][d]:
                        continue
                    neighbor = canonical(route[:i + 1] + tuple(reversed(route[i + 1:j + 1])) + route[j + 1:])
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        region.add(neighbor)
                        stack.append(neighbor)
        regions.append(region)
    return regions


def load_model(checkpoint, mode):
    if mode == "fixed_soft":
        policy, _ = load_policy(checkpoint, "cpu")
        return policy
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    arch = state["architecture"]
    use_bias = bool(state.get("use_bias", False))
    model = BallBiasPolicy(int(arch["embedding_dim"]), int(arch["num_heads"]),
                           int(arch["encoder_layers"]), use_bias=use_bias,
                           bias_bound=float(state.get("bias_bound", 2.0)))
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"no rows: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(mode, train_seed, inference_seed, checkpoint, output, *, resume=False):
    if mode not in {"fixed_soft", "short", "G0", "Q", "G1", "G2", "G3"}:
        raise ValueError(mode)
    torch.set_num_threads(1)
    checkpoint = checkpoint.resolve()
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if mode == "fixed_soft" and checkpoint_sha != original25.EXPECTED_CHECKPOINT:
        raise RuntimeError("fixed Soft checkpoint differs from historical baseline")
    if mode == "short" and checkpoint_sha != INITIAL_SHA:
        raise RuntimeError("short-training checkpoint differs from common initialization")
    if mode in {"G0", "Q", "G1", "G2", "G3"}:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["experiment_group"] != mode or state["seed"] != train_seed:
            raise RuntimeError("selected checkpoint identity mismatch")
    config = {"mode": mode, "train_seed": train_seed, "inference_seed": inference_seed,
              "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
              "starts": "all", "argmax_rounds": 1, "sampling_rounds": 8,
              "two_opt": "all unique original routes", "optb_pair_checks": 50000,
              "beta_squared": 0.3,
              "source": "original25.run_case"}
    if output.exists():
        if not resume or json.loads((output / "run_config.json").read_text(encoding="utf-8")) != config:
            raise RuntimeError("existing final test requires identical --resume")
    else:
        output.mkdir(parents=True)
        (output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    model = load_model(checkpoint, mode)
    label = mode
    for folder in ("raw_routes", "best_routes"):
        (output / label / folder).mkdir(parents=True)
    original25.SEED = inference_seed
    paper = original25._csv(original25.PAPER)

    def route_generator(policy, prepared, starts, samples, seed):
        if starts != list(range(len(prepared.cities))) or samples != 8:
            raise RuntimeError("original25 inference protocol changed")
        routes = generate_routes(policy, prepared, samples=samples, seed=seed,
                                 device=torch.device("cpu"))
        return routes, None

    cases = []
    region_rows = []
    for index, (stem, n, _) in enumerate(BENCHMARKS, 1):
        name = f"MSTSP{index}"
        case_file = output / f"case_{index:02d}.json"
        if case_file.is_file():
            saved = json.loads(case_file.read_text(encoding="utf-8"))
            cases.append(saved["case"])
            region_rows.extend(saved["regions"])
            continue
        row = original25.run_case(model, index, label, 0.0, output,
                                  float(paper[name]["paper_fbeta"]),
                                  route_generator=None if mode == "fixed_soft" else route_generator)
        problem = prepare_tsp(original25.DATA / f"{stem}.tsp")
        truth = load_ground_truth(original25.DATA / f"{stem}.solution", n)
        regions = truth_regions(truth, problem.distance_matrix)
        membership = {key: region_id for region_id, region in enumerate(regions)
                      for key in region}
        best_file = json.loads((output / label / "best_routes" / f"{name}.json").read_text(encoding="utf-8"))
        a = {tuple(key) for key in best_file["A"]["routes"]} & truth
        b = {tuple(key) for key in best_file["B"]["routes"]} & truth
        raw = json.loads((output / label / "raw_routes" / f"{name}.json").read_text(encoding="utf-8"))
        raw_lengths = [tour_length(route, problem.distance_matrix) for route in raw]
        unique_raw = {}
        for route in raw:
            unique_raw.setdefault(canonical(route), route)
        entries = Counter()
        for route in unique_raw.values():
            refined = two_opt(route, problem.distance_matrix)
            key = canonical(refined.route)
            if key in membership:
                entries[membership[key]] += 1
        case_regions = []
        for region_id, region in enumerate(regions):
            case_regions.append({"instance": name, "region_id": region_id,
                                "truth_routes_in_region": len(region),
                                "two_opt_entry_count": entries[region_id],
                                "repeated_two_opt_entries": max(0, entries[region_id] - 1),
                                "a_true_routes": len(a & region),
                                "b_true_routes": len(b & region),
                                "b_new_true_routes": len((b - a) & region)})
        case_row = {"model": mode, "train_seed": train_seed if train_seed is not None else "",
                      "inference_seed": inference_seed, **row,
                      "raw_duplicate_rate": 1 - len(unique_raw) / len(raw),
                      "raw_mean_length": sum(raw_lengths) / len(raw_lengths),
                      "raw_mean_gap_pct": 100 * (sum(raw_lengths) / len(raw_lengths)
                                                 / row["reference_length"] - 1),
                      "truth_regions_listed": len(regions),
                      "a_regions_hit": len({membership[key] for key in a}),
                      "b_regions_hit": len({membership[key] for key in b}),
                      "region_coverage_rate": len({membership[key] for key in b}) / len(regions),
                      "two_opt_optimal_entries": sum(entries.values())}
        cases.append(case_row)
        region_rows.extend(case_regions)
        case_file.write_text(json.dumps({"case": case_row, "regions": case_regions},
                                        ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{mode} train={train_seed} infer={inference_seed} {name}: "
              f"hits={row['b_distance_hit']} tours={row['b_hits']} "
              f"regions={cases[-1]['b_regions_hit']}/{len(regions)}", flush=True)
    write_csv(output / "per_instance.csv", cases)
    write_csv(output / "region_coverage.csv", region_rows)
    summary = {"model": mode, "train_seed": train_seed, "inference_seed": inference_seed,
               "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
               "distance_hits": sum(r["b_distance_hit"] for r in cases),
               "true_optimal_routes": sum(r["b_hits"] for r in cases),
               "mean_fbeta": sum(r["b_fbeta"] for r in cases) / len(cases),
               "truth_regions_hit": sum(r["b_regions_hit"] for r in cases),
               "truth_regions_total": sum(r["truth_regions_listed"] for r in cases),
               "mean_raw_duplicate_rate": sum(r["raw_duplicate_rate"] for r in cases) / len(cases),
               "inference_seconds": sum(r["inference_seconds"] for r in cases),
               "two_opt_seconds": sum(r["two_opt_seconds"] for r in cases),
               "optb_seconds": sum(r["optb_seconds"] for r in cases)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--train-seed", type=int)
    parser.add_argument("--inference-seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    evaluate(args.mode, args.train_seed, args.inference_seed, args.checkpoint,
             args.output_dir.resolve(), resume=args.resume)


if __name__ == "__main__":
    main()
