"""Resume completed stages without rerunning training or final cases."""

from __future__ import annotations

import argparse
import csv
import json

import torch

from .config import GROUPS, INFERENCE_SEEDS, LAMBDA, RESULTS, TRAIN_SEEDS
from .data import load_experiment_data
from .report import write_report
from .test25 import test_model
from .train import (PREFLIGHT_VERSION, dataset_signature, ensure_a0,
                    finalize_group, preflight, run_group, save_json)


def ready_training():
    missing = []
    for seed in TRAIN_SEEDS:
        for group in GROUPS:
            path = RESULTS / f"train_seed_{seed}" / group / "train_result.json"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise RuntimeError(f"training incomplete, first missing result: {missing[0]}")


def run_all(device_name, phase):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if device.type == "cpu":
        torch.set_num_threads(1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if phase in ("all", "train"):
        partitions, labels, _ = load_experiment_data(
            audit_path=RESULTS / "region_rule_audit.csv")
        for seed in TRAIN_SEEDS:
            marker = RESULTS / f"preflight_seed_{seed}_{device_name}.json"
            if marker.is_file():
                stats = json.loads(marker.read_text(encoding="utf-8"))
                if stats["dataset_signature"] != dataset_signature():
                    raise RuntimeError(f"preflight dataset/checkpoint changed: {marker}")
            if (not marker.is_file() or stats.get("probe_version") != PREFLIGHT_VERSION
                    or stats.get("scale_gate", {}).get("lambda") != LAMBDA):
                stats = preflight(seed, device_name)
                save_json(marker, stats)
            if not stats["passed"]:
                raise RuntimeError(f"gradient scale gate failed for seed {seed}: "
                                   f"{stats['scale_gate']}; training paused")
            ensure_a0(seed, partitions, labels, device)
            completed = [RESULTS / f"train_seed_{seed}" / group / "train_result.json"
                         for group in GROUPS]
            cohort_result = RESULTS / f"train_seed_{seed}" / "cohort_result.json"
            if cohort_result.is_file():
                cohort = json.loads(cohort_result.read_text(encoding="utf-8"))
                for group in GROUPS:
                    if not (RESULTS / f"train_seed_{seed}" / group / "train_result.json").is_file():
                        finalize_group(group, seed, cohort["completed_epochs"],
                                       cohort["stopped_early"])
                continue
            if all(path.is_file() for path in completed):
                print(f"skip completed training seed={seed}", flush=True)
                continue
            if any(path.is_file() for path in completed):
                raise RuntimeError(f"partially finalized matched cohort: seed={seed}")
            for epoch in range(1, 4):
                for group in GROUPS:
                    directory = RESULTS / f"train_seed_{seed}" / group
                    run_group(group, seed, partitions, labels, device,
                              resume=directory.exists(), stop_after=epoch)
                history = [json.loads((RESULTS / f"train_seed_{seed}" / group /
                                      "validation_history.json").read_text(encoding="utf-8"))
                           for group in GROUPS]
                if any(len(records) != epoch + 1 for records in history):
                    raise RuntimeError(f"unequal completed training epochs: seed={seed}")
                no_improvement = []
                for group in GROUPS:
                    path = RESULTS / f"train_seed_{seed}" / group / "training_history.csv"
                    with path.open(encoding="utf-8-sig", newline="") as stream:
                        training_rows = list(csv.DictReader(stream))
                    no_improvement.append(int(training_rows[-1]["early_wait"]) >= 2)
                if all(no_improvement) or epoch == 3:
                    stopped_early = epoch < 3 and all(no_improvement)
                    save_json(cohort_result, {"completed_epochs": epoch,
                                             "stopped_early": stopped_early,
                                             "optimizer_updates_equal": True})
                    for group in GROUPS:
                        finalize_group(group, seed, epoch, stopped_early)
                    break
    if phase in ("all", "test"):
        ready_training()
        for mode in ("fixed_soft", "A0"):
            for infer_seed in INFERENCE_SEEDS:
                test_model(mode, None, infer_seed)
        for seed in TRAIN_SEEDS:
            for group in GROUPS:
                for infer_seed in INFERENCE_SEEDS:
                    test_model(group, seed, infer_seed)
    if phase in ("all", "report"):
        write_report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "train", "test", "report"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_all(args.device, args.phase)


if __name__ == "__main__":
    main()
