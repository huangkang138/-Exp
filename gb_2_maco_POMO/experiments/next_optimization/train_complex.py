"""Continue A0 for at most one epoch on the gated mixed dataset.

The entry point estimates CPU runtime first and stops before any optimizer
step if the estimate exceeds 30 minutes. Formal MSTSP testing runs only when
the independently validated epoch-1 checkpoint beats epoch 0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from time import perf_counter

import torch

from gb_maco_POMO.experiments.compare import BENCHMARKS, load_ground_truth
from gb_2_maco_POMO.experiments.hierarchical import original25
from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import (
    generate_learned_routes, load_checkpoint,
)
from gb_2_maco_POMO.experiments.learnable_ball.learnable_rollout import rollout_batch
from gb_2_maco_POMO.experiments.learnable_ball.train_learnable import _loss, _validate
from gb_2_maco_POMO.experiments.region_supervised_pomo.config import INITIAL, INITIAL_SHA256
from gb_2_maco_POMO.experiments.region_supervised_pomo.data import _project_components
from gb_2_maco_POMO.experiments.shared.data import batches_by_city_count, load_manifest
from gb_2_maco_POMO.runner import prepare_tsp

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT / "gb_2_maco_POMO/results/next_optimization/complex_training/dataset"
DEFAULT_OUTPUT = PROJECT / "gb_2_maco_POMO/results/next_optimization/complex_training/run_seed_42"
HISTORY = PROJECT / "gb_2_maco_POMO/results/learnable_ball_test25_20261015"
CPU_LIMIT_SECONDS = 1800


def _save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _estimate_cpu(model, train, validation):
    """A no-update sample backward pass; scaled conservatively to one epoch."""
    torch.set_num_threads(1)
    model.cpu().train()
    batch = max(batches_by_city_count(train, 8), key=lambda items: items[0].city_count)
    start = perf_counter()
    outcome = rollout_batch(model, batch, torch.device("cpu"), sample=True)
    _loss(outcome).backward()
    sample_seconds = perf_counter() - start
    model.zero_grad(set_to_none=True)
    sample_n = batch[0].city_count
    # Decoder computation grows with city count; square scaling is deliberately
    # conservative and includes 2x reserve for optimizer/validation/I/O.
    weighted_microbatches = sum((part[0].city_count / sample_n) ** 2
                                for part in batches_by_city_count(train, 8))
    validation_equivalent = sum((part[0].city_count / sample_n) ** 2
                                for part in batches_by_city_count(validation, 8)) * 0.5
    estimate = 2 * sample_seconds * (weighted_microbatches + validation_equivalent)
    return {"sample_n": sample_n, "sample_instances": len(batch),
            "sample_seconds": sample_seconds, "estimated_cpu_seconds": estimate,
            "limit_seconds": CPU_LIMIT_SECONDS,
            "method": "one no-update backward pass; squared-city weighting; 2x reserve"}


def _snapshot(model, optimizer, architecture, epoch, validation_score, dataset_sha):
    return {"model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "step": epoch, "train_instances_seen": 0 if epoch == 0 else 500,
            "validation_mean_best_length": validation_score,
            "architecture": architecture,
            "initialized_from": str(INITIAL), "initial_checkpoint_sha256": INITIAL_SHA256,
            "dataset_manifest_sha256": dataset_sha}


def _test(policy, output, selected_path):
    """Historical S0 route protocol and postprocessing, only after validation win."""
    original25.SEED = 42
    paper = original25._csv(original25.PAPER)
    per_case = []
    for index, (stem, n, _) in enumerate(BENCHMARKS, 1):
        name = f"MSTSP{index}"
        for folder in ("raw_routes", "best_routes", "cases"):
            (output / "test" / "new" / folder).mkdir(parents=True, exist_ok=True)
        row = original25.run_case(policy, index, "new", 0.0, output / "test",
                                  float(paper[name]["paper_fbeta"]),
                                  route_generator=generate_learned_routes)
        baseline = json.loads((HISTORY / "learned_ball/cases" / f"{name}.json").read_text(encoding="utf-8"))
        if row["reference_length"] != baseline["reference_length"] or row["truth_count"] != baseline["truth_count"]:
            raise RuntimeError(f"{name}: historical target differs")
        problem = prepare_tsp(original25.DATA / f"{stem}.tsp")
        truth = load_ground_truth(original25.DATA / f"{stem}.solution", n)
        tours = tuple(sorted(truth))
        regions = [set(tours[j] for j in group)
                   for group in _project_components(tours, problem.distance_matrix)]
        def count_regions(path):
            saved = json.loads(path.read_text(encoding="utf-8"))
            returned = {tuple(route) for route in saved["B"]["routes"]} & truth
            return sum(bool(returned & region) for region in regions)
        current_region_count = count_regions(output / "test/new/best_routes" / f"{name}.json")
        baseline_region_count = count_regions(HISTORY / "learned_ball/best_routes" / f"{name}.json")
        record = {"instance": name, "reference_length": row["reference_length"],
                  "truth_count": row["truth_count"], "paper_fbeta": row["paper_fbeta"]}
        for label, item, area_count in (("a0", baseline, baseline_region_count),
                                        ("new", row, current_region_count)):
            for key in ("raw_best", "raw_gap_pct", "a_best", "a_gap_pct",
                        "a_hits", "b_best", "b_gap_pct", "b_distance_hit",
                        "b_hits", "b_precision", "b_recall", "b_fbeta"):
                record[f"{label}_{key}"] = item[key]
            record[f"{label}_regions_hit"] = area_count
        per_case.append(record)
        _save_json(output / "test/new/cases" / f"{name}.json", row)
        print(f"{name}: A0 {baseline['b_hits']} / new {row['b_hits']} true tours", flush=True)
    _save_csv(output / "test/per_instance.csv", per_case)
    summary = {"checkpoint": str(selected_path),
               "checkpoint_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
               "seed": 42, "cases": 25}
    for label in ("a0", "new"):
        summary[label] = {
            "distance_hits": sum(int(r[f"{label}_b_distance_hit"]) for r in per_case),
            "true_optimal_tours": sum(int(r[f"{label}_b_hits"]) for r in per_case),
            "regions_hit": sum(int(r[f"{label}_regions_hit"]) for r in per_case),
            "mean_fbeta": sum(float(r[f"{label}_b_fbeta"]) for r in per_case) / 25}
    _save_json(output / "test/summary.json", summary)
    lines = ["# 复杂数据续训：A0 与新模型正式测试", "",
        "同一 seed=42、全部起点、每起点 1 Argmax+8 Sampling、完整普通 2-opt、Opt-B 每例最多 50,000 检查。"
        "新模型只训练一组混合数据；若改善，不能独立证明改善全部由数据分布变化引起。", "",
        "| 指标 | A0 | 新模型 |", "|---|---:|---:|"]
    for title, key in (("最优距离命中", "distance_hits"),
                       ("Opt-B 真最优回路", "true_optimal_tours"),
                       ("最优区域覆盖", "regions_hit"), ("平均 Fβ", "mean_fbeta")):
        lines.append(f"| {title} | {summary['a0'][key]} | {summary['new'][key]} |")
    lines += ["", "## 逐例", "",
              "| 算例 | 参考距离 | A0 / 新模型原始距离 | A0 / 新模型 2-opt 距离 | A0 / 新模型 Opt-B 距离 | A0 / 新模型回路 | A0 / 新模型区域 | A0 / 新模型 Fβ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in per_case:
        lines.append(f"| {r['instance']} | {r['reference_length']} | "
                     f"{r['a0_raw_best']}/{r['new_raw_best']} | "
                     f"{r['a0_a_best']}/{r['new_a_best']} | "
                     f"{r['a0_b_best']}/{r['new_b_best']} | "
                     f"{r['a0_b_hits']}/{r['new_b_hits']} | "
                     f"{r['a0_regions_hit']}/{r['new_regions_hit']} | "
                     f"{r['a0_b_fbeta']:.4f}/{r['new_b_fbeta']:.4f} |")
    lines += ["", "完整 Gap、Precision、Recall、Fβ 字段见 `per_instance.csv`。", ""]
    (output / "test/analysis.md").write_text("\n".join(lines), encoding="utf-8")


def run(dataset_root: Path, output: Path, device_name: str):
    dataset_root, output = dataset_root.resolve(), output.resolve()
    if not (dataset_root / "READY").is_file() or (dataset_root / "INCOMPLETE").exists():
        raise RuntimeError("mixed dataset is incomplete")
    if output.exists():
        raise FileExistsError(f"new output directory required: {output}")
    if hashlib.sha256(INITIAL.read_bytes()).hexdigest() != INITIAL_SHA256:
        raise RuntimeError("A0 checkpoint changed")
    train, validation = load_manifest({"root": str(dataset_root),
                                       "train_dir": "train", "validation_dir": "validation"})
    if len(train) != 500 or len(validation) != 150:
        raise RuntimeError("mixed data split must be 500/150")
    output.mkdir(parents=True)
    manifest_sha = hashlib.sha256((dataset_root / "manifest.csv").read_bytes()).hexdigest()
    _save_json(output / "config.json", {"dataset_root": str(dataset_root),
        "dataset_manifest_sha256": manifest_sha, "A0": str(INITIAL),
        "A0_sha256": INITIAL_SHA256, "seed": 42, "epochs_max": 1,
        "batch_size": 64, "micro_batch_size": 8,
        "optimizer": "Adam resumed from A0", "learning_rate": 1e-4,
        "reward": "negative closed-tour length", "validation": "150 independent cases; greedy best distance",
        "device": device_name})
    random.seed(42)
    torch.manual_seed(42)
    policy, state = load_checkpoint(INITIAL)
    architecture = dict(state["architecture"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    estimate = _estimate_cpu(policy, train, validation)
    _save_json(output / "cpu_estimate.json", estimate)
    if estimate["estimated_cpu_seconds"] > CPU_LIMIT_SECONDS:
        _save_json(output / "stopped.json", {"reason": "estimated CPU training exceeds 30 minutes",
                                              **estimate})
        print("CPU estimate exceeds 30 minutes; no training started", flush=True)
        return
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(device_name)
    policy.to(device)
    for values in optimizer.state.values():
        for name, value in values.items():
            if torch.is_tensor(value):
                values[name] = value.to(device)
    torch.manual_seed(42)  # The CPU estimate must not change training sampling.
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    else:
        torch.set_num_threads(1)
    # Epoch 0 is independently scored on the new fixed validation split.
    score0, details0 = _validate(policy, validation, device, 8)
    (output / "checkpoints").mkdir()
    torch.save(_snapshot(policy, optimizer, architecture, 0, score0, manifest_sha),
               output / "checkpoints/epoch_0.pt")
    history = [{"epoch": 0, "validation_mean_best_length": score0,
                "train_instances": 0, "validation_groups": details0}]
    policy.train()
    batches = batches_by_city_count(train, 64, random.Random(42))
    start = perf_counter()
    for batch in batches:
        optimizer.zero_grad()
        for offset in range(0, len(batch), 8):
            micro = batch[offset:offset + 8]
            outcome = rollout_batch(policy, micro, device, sample=True)
            (_loss(outcome) * (len(micro) / len(batch))).backward()
        if any(not torch.isfinite(p.grad).all().item() for p in policy.parameters()
               if p.grad is not None):
            raise RuntimeError("nonfinite gradient; training stopped")
        optimizer.step()
    score1, details1 = _validate(policy, validation, device, 8)
    epoch1 = output / "checkpoints/epoch_1.pt"
    torch.save(_snapshot(policy, optimizer, architecture, 1, score1, manifest_sha), epoch1)
    history.append({"epoch": 1, "validation_mean_best_length": score1,
                    "train_instances": 500, "validation_groups": details1})
    _save_json(output / "validation_history.json", history)
    selected = epoch1 if score1 < score0 else output / "checkpoints/epoch_0.pt"
    _save_json(output / "train_result.json", {"training_seconds": perf_counter() - start,
        "epoch_0_score": score0, "epoch_1_score": score1,
        "selected_epoch": 1 if score1 < score0 else 0,
        "selected_checkpoint": str(selected),
        "selected_checkpoint_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "formal_test_executed": False})
    if score1 >= score0:
        print("Epoch 1 did not beat epoch 0 validation; formal test skipped", flush=True)
        return
    # Formal test is CPU to match A0 historical seed/protocol.
    policy.cpu().eval()
    _test(policy, output, selected)
    _save_json(output / "train_result.json", {"training_seconds": perf_counter() - start,
        "epoch_0_score": score0, "epoch_1_score": score1,
        "selected_epoch": 1, "selected_checkpoint": str(selected),
        "selected_checkpoint_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "formal_test_executed": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run(args.dataset_root, args.output_dir, args.device)


if __name__ == "__main__":
    main()
