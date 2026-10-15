"""Evaluate the new-data Bias model without changing checkpoint selection or search rules.

Run after training with --checkpoint pointing to its best.pt. Each invocation
creates a fresh output directory under results/learnable_bias_new_dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch

from ..attention import POMOPolicy
from ..pomo import search_routes
from ..runner import prepare_tsp
from .compare import BENCHMARKS, load_ground_truth, load_policy, read_optimum
from .compare_learnable_bias import BENCHMARK_DIR, PROJECT, canonical, equal_plateau, fbeta
from .data import load_manifest
from .diagnose_2opt import _select_candidates
from .local_search_2opt import two_opt
from .train_loop import evaluate_validation


ROOT = PROJECT / "gb_maco_POMO/results/learnable_bias_new_dataset"
NEW_DATA = PROJECT / "benachmark_Train_MATSP_Multiopt_500"
FIXED = PROJECT / "gb_maco_POMO/results/checkpoints/run_20261013_230742_828b99/best.pt"
OLD_LEARNED = (PROJECT / "gb_maco_POMO/results/learnable_bias/checkpoints"
               / "run_20261014_223323_7de9e6/best.pt")
OLD_SUMMARY = (PROJECT / "gb_maco_POMO/results/learnable_bias/comparison"
               / "run_20261014_233707/summary.json")
FIXED_SHA256 = "dbdfc5cb2faef0d7cbecb7c5c7cc77db16049e5d10e5d584a26e288f1bd88410"
FIELDS = ("model", "instance", "seed", "reference_length", "a_best", "a_hits",
          "b_best", "b_hits", "b_fbeta", "b_checks", "raw_routes", "unique_candidates")


def _output(kind: str) -> Path:
    path = ROOT / kind / f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _new_policy(checkpoint: Path) -> tuple[POMOPolicy, dict[str, object]]:
    policy, state = load_policy(checkpoint, torch.device("cpu"))
    data = state["config"]["data"]
    if (policy.bias_mode != "learnable"
            or Path(data.get("train_root", "")).resolve() != NEW_DATA.resolve()
            or Path(data["root"]).resolve() != (PROJECT / "benachmark_Train_MATSP_500").resolve()):
        raise ValueError("Checkpoint is not a new-data learnable Bias run")
    return policy, state


def extra_validation(checkpoint: Path) -> Path:
    policy, state = _new_policy(checkpoint)
    _, instances = load_manifest({"root": str(NEW_DATA), "train_dir": "train",
                                  "validation_dir": "validation"})
    torch.set_num_threads(1)
    mean, groups = evaluate_validation(policy, instances, 64, 8, torch.device("cpu"),
                                       int(state["best_epoch"]))
    output = _output("extra_validation")
    with (output / "by_group.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=groups[0].keys())
        writer.writeheader()
        writer.writerows(groups)
    (output / "summary.json").write_text(json.dumps({
        "checkpoint": str(checkpoint.resolve()), "best_epoch": state["best_epoch"],
        "instances": len(instances), "mean_best_length": mean,
        "purpose": "additional evaluation only; not used for checkpoint selection",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def benchmark(checkpoint: Path, seeds: list[int]) -> Path:
    if not seeds or any(seed == 42 for seed in seeds):
        raise ValueError("Use compare_learnable_bias for seed 42; this command is for 123/2026")
    old = json.loads(OLD_SUMMARY.read_text(encoding="utf-8"))
    if (hashlib.sha256(FIXED.read_bytes()).hexdigest() != FIXED_SHA256
            or hashlib.sha256(OLD_LEARNED.read_bytes()).hexdigest()
            != old["learnable_checkpoint_sha256"]):
        raise ValueError("Historical checkpoint identity differs from saved comparison")
    models = {"fixed": FIXED, "old_learned": OLD_LEARNED, "new_learned": checkpoint}
    policies = {name: load_policy(path, torch.device("cpu"))[0]
                for name, path in models.items() if name != "new_learned"}
    policies["new_learned"], _ = _new_policy(checkpoint)
    torch.set_num_threads(1)
    output = _output("seed_stability")
    rows = []
    for seed in seeds:
        for index, (stem, count, _) in enumerate(BENCHMARKS, start=1):
            problem = prepare_tsp(BENCHMARK_DIR / f"{stem}.tsp")
            reference = read_optimum(BENCHMARK_DIR / f"{stem}.solution")
            truth = load_ground_truth(BENCHMARK_DIR / f"{stem}.solution", count)
            for name, policy in policies.items():
                search = search_routes(policy, problem, inference_mode="mixed",
                                       samples_per_start=8, seed=seed + index)
                candidates = _select_candidates(search.routes, search.lengths, count,
                                                len(search.routes), all_generated=True)
                refined = [two_opt(route, problem.distance_matrix) for route in candidates]
                best = min([*search.lengths, *(item.length for item in refined)])
                if best < reference:
                    raise RuntimeError(f"MSTSP{index}: route shorter than reference")
                starts = [route for route, length in zip(search.routes, search.lengths)
                          if length == best]
                starts.extend(item.route for item in refined if item.length == best)
                a_pool = {canonical(route): route for route in starts}
                b = equal_plateau(list(a_pool.values()), problem.distance_matrix, 50_000)
                a_hits = len(set(a_pool) & truth)
                b_hits = len(set(b.routes) & truth)
                rows.append(dict(zip(FIELDS, (name, f"MSTSP{index}", seed, reference,
                                              best, a_hits, b.best_length, b_hits,
                                              fbeta(set(b.routes), truth), b.checks,
                                              len(search.routes), len(candidates)))))
                print(f"seed={seed} MSTSP{index} {name}: {best}, Opt-B={b_hits}", flush=True)
    with (output / "per_instance.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for seed in seeds:
        for name in models:
            subset = [row for row in rows if row["seed"] == seed and row["model"] == name]
            summary.append({"seed": seed, "model": name,
                            "distance_hits": sum(r["a_best"] == r["reference_length"] for r in subset),
                            "mean_gap_pct": sum(100 * (r["a_best"] / r["reference_length"] - 1)
                                                for r in subset) / 25,
                            "a_hits": sum(r["a_hits"] for r in subset),
                            "b_hits": sum(r["b_hits"] for r in subset),
                            "mean_fbeta": sum(r["b_fbeta"] for r in subset) / 25})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (output / "sources.json").write_text(json.dumps({
        "checkpoints": {name: str(path.resolve()) for name, path in models.items()},
        "device": "cpu", "seeds": seeds, "samples_per_start": 8,
        "optb_extra_checks": 50_000,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("validation", "stability"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 2026])
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("checkpoint does not exist")
    if args.mode == "validation":
        output = extra_validation(args.checkpoint)
    else:
        if not OLD_SUMMARY.is_file():
            parser.error("historical old-learned result is missing")
        output = benchmark(args.checkpoint, args.seeds)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
