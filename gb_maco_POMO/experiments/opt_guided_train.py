"""Five-epoch A/B/C training from the verified fixed Soft checkpoint.

The original POMO loss stays exact. Auxiliary feedback is detached full 2-opt
on eight preselected starts, with gradients on their original route logP.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch

from ..runner import prepare_tsp
from .batch_rollout import rollout_batch
from .compare import load_policy
from .data import batches_by_city_count, load_manifest
from .opt_guided_diagnose import DATA, PROJECT, SOURCE, SOURCE_SHA
from .opt_guided_feedback import (
    auxiliary_losses, heuristic_reference, select_indices, selected_feedback,
)
from .opt_guided_validation import validate


ROOT = PROJECT / "gb_maco_POMO/results/opt_guided_pomo"
OLD_DATA = PROJECT / "benachmark_Train_MATSP_500"
CONFIG = {
    "source_checkpoint": str(SOURCE), "source_sha256": SOURCE_SHA,
    "train_root": str(DATA), "checkpoint_validation_root": str(OLD_DATA),
    "epochs": 5, "batch_size": 64, "micro_batch_size": 8,
    "learning_rate": 1e-5, "seed": 42, "selected_starts": 8,
    "quality_tolerance": 0.005, "optimizer": "Adam(default betas/eps/weight_decay)",
    "reference": "deterministic NN(start=0) plus original complete 2-opt",
    "validation": "all N argmax starts for best.pt; eight fixed argmax starts for 2-opt metrics",
}
EPOCH_FIELDS = (
    "epoch", "train_mean_raw_length", "train_mean_best_raw_length",
    "train_mean_selected_two_opt_length", "validation_mean_best_raw_length",
    "validation_mean_best_two_opt_length", "validation_mean_unique_good_cycles",
    "main_loss", "quality_loss", "coverage_loss", "main_grad_norm",
    "quality_grad_norm", "coverage_grad_norm", "pass_rate",
    "mean_unique_two_opt_cycles", "marginal_nonzero_rate", "training_seconds",
    "two_opt_seconds", "validation_seconds", "peak_gpu_mb", "peak_cpu_rss_mb",
)
BATCH_FIELDS = ("epoch", "batch", "instances", "main_loss", "quality_loss",
                "coverage_loss", "main_grad_norm", "quality_grad_norm",
                "coverage_grad_norm", "two_opt_seconds", "batch_seconds")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _norm(grads: tuple[torch.Tensor | None, ...]) -> float:
    return math.sqrt(sum(float(g.detach().square().sum().item()) for g in grads if g is not None))


def _grads(loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...]
           ) -> tuple[torch.Tensor | None, ...]:
    return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)


def _cosine(left: tuple[torch.Tensor | None, ...], right: tuple[torch.Tensor | None, ...]
            ) -> float:
    dot = sum(float((a.detach() * b.detach()).sum().item())
              for a, b in zip(left, right) if a is not None and b is not None)
    denominator = _norm(left) * _norm(right)
    return dot / denominator if denominator else 0.0


def _reference_table(instances) -> dict[Path, int]:
    return {item.path: heuristic_reference(prepare_tsp(item.path).distance_matrix)
            for item in instances}


def _main_loss(outcome) -> torch.Tensor:
    assert outcome.route_log_probs is not None
    rewards = -outcome.lengths
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    return -(advantages.detach() * outcome.route_log_probs).mean()


def calibrate(policy, train, references, device) -> dict[str, float]:
    """Measure raw gradient directions, then fix coefficients for all runs."""
    policy.train()
    parameters = tuple(policy.parameters())
    rng = random.Random(CONFIG["seed"] + 1)
    sample = [item for item in train if item.city_count == 30][:8]
    if len(sample) != 8:
        raise RuntimeError("Need eight 30-city pilot instances")
    selected = select_indices(sample, rng)
    torch.manual_seed(CONFIG["seed"])
    outcome = rollout_batch(policy, sample, device, sample_actions=True)
    feedback = selected_feedback(outcome, sample, selected, references,
                                 CONFIG["quality_tolerance"])
    main = _main_loss(outcome)
    quality, coverage = auxiliary_losses(outcome.route_log_probs, feedback)
    gradients = tuple(_grads(loss, parameters) for loss in (main, quality, coverage))
    norms = [_norm(group) for group in gradients]
    if not all(math.isfinite(value) for value in norms) or min(norms) <= 0:
        raise RuntimeError(f"Pilot gradients inactive or invalid: {norms}")
    cos_quality = _cosine(gradients[0], gradients[1])
    cos_coverage = _cosine(gradients[0], gradients[2])
    if min(cos_quality, cos_coverage) < -0.9:
        raise RuntimeError(f"Pilot auxiliary gradient strongly opposes main: "
                           f"{cos_quality}, {cos_coverage}")
    # Numeric coefficients can exceed one because quality is divided by L_ref;
    # effective initial gradient ratios are each 5% of the main gradient.
    alpha = 0.05 * norms[0] / norms[1]
    lam = 0.05 * norms[0] / norms[2]
    return {"main_grad_norm": norms[0], "quality_grad_norm": norms[1],
            "coverage_grad_norm": norms[2], "main_quality_cosine": cos_quality,
            "main_coverage_cosine": cos_coverage, "alpha": alpha, "lambda": lam,
            "initial_weighted_quality_ratio": alpha * norms[1] / norms[0],
            "initial_weighted_coverage_ratio": lam * norms[2] / norms[0]}


def _save(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _cpu_rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1_048_576
    except ImportError:
        return None


def checkpoint_config(group: str, run_root: Path, source: dict[str, object],
                      calibration: dict[str, float], device: torch.device
                      ) -> dict[str, dict[str, object]]:
    """Keep the project's nested checkpoint config for standard load_policy()."""
    configured = copy.deepcopy(source["config"])
    configured["training"].update({
        "epochs": CONFIG["epochs"], "batch_size": CONFIG["batch_size"],
        "micro_batch_size": CONFIG["micro_batch_size"],
        "learning_rate": CONFIG["learning_rate"], "seed": CONFIG["seed"],
    })
    configured["model"]["bias_mode"] = "fixed"
    configured["data"].update({"root": str(OLD_DATA), "train_root": str(DATA)})
    configured["runtime"]["device"] = str(device)
    configured["output"].update({"results_root": str(run_root),
                                         "checkpoints_root": str(run_root / group)})
    configured["opt_guided"] = {
        "group": group, "selected_starts": CONFIG["selected_starts"],
        "quality_tolerance": CONFIG["quality_tolerance"],
        "alpha": calibration["alpha"] if group != "A" else 0.0,
        "lambda": calibration["lambda"] if group == "C" else 0.0,
        "reference": CONFIG["reference"],
    }
    return configured


def train_group(group: str, run_root: Path, train, validation, references,
                calibration: dict[str, float], device: torch.device) -> dict[str, object]:
    policy, checkpoint = load_policy(SOURCE, device)
    effective_checkpoint_config = checkpoint_config(group, run_root, checkpoint,
                                                    calibration, device)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=CONFIG["learning_rate"])
    parameters = tuple(policy.parameters())
    shuffle_rng = random.Random(CONFIG["seed"])
    selection_rng = random.Random(CONFIG["seed"] + 1)
    torch.manual_seed(CONFIG["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(CONFIG["seed"])
        torch.cuda.reset_peak_memory_stats()
    directory = run_root / group
    directory.mkdir()
    (directory / "effective_config.json").write_text(json.dumps({
        **CONFIG, "group": group, "alpha": calibration["alpha"] if group != "A" else 0,
        "lambda": calibration["lambda"] if group == "C" else 0,
        "initial_epoch": checkpoint["epoch"], "device": str(device),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    epochs: list[dict[str, object]] = []
    batches_log: list[dict[str, object]] = []
    best = float("inf")
    best_epoch = 0
    for epoch in range(1, CONFIG["epochs"] + 1):
        policy.train()
        start_epoch = time.perf_counter()
        sums = {name: 0.0 for name in ("raw", "raw_best", "opt", "main", "quality",
                                      "coverage", "main_norm", "quality_norm", "coverage_norm",
                                      "passed", "unique", "nonzero", "two_opt")}
        selected_total = 0
        batch_count = 0
        for batch_index, batch in enumerate(batches_by_city_count(train, CONFIG["batch_size"],
                                                                  shuffle_rng), start=1):
            start_batch = time.perf_counter()
            optimizer.zero_grad()
            batch_losses = {"main": 0.0, "quality": 0.0, "coverage": 0.0,
                            "main_norm": 0.0, "quality_norm": 0.0, "coverage_norm": 0.0,
                            "two_opt": 0.0}
            for start in range(0, len(batch), CONFIG["micro_batch_size"]):
                micro = batch[start:start + CONFIG["micro_batch_size"]]
                # Draw before the policy rollout. Python RNG is independent of torch sampling.
                indices = select_indices(micro, selection_rng) if group != "A" else None
                outcome = rollout_batch(policy, micro, device, sample_actions=True)
                main = _main_loss(outcome)
                quality = main * 0
                coverage = main * 0
                if indices is not None:
                    feedback = selected_feedback(outcome, micro, indices, references,
                                                 CONFIG["quality_tolerance"])
                    quality, coverage = auxiliary_losses(outcome.route_log_probs, feedback)
                    for item in feedback:
                        sums["opt"] += sum(item.refined_lengths)
                        sums["passed"] += item.pass_count
                        sums["unique"] += item.unique_count
                        sums["nonzero"] += sum(item.coverage_rewards)
                        sums["two_opt"] += item.two_opt_seconds
                        batch_losses["two_opt"] += item.two_opt_seconds
                        selected_total += len(item.selected_indices)
                ratio = len(micro) / len(batch)
                components = (main, quality, coverage)
                for name, component in zip(("main", "quality", "coverage"), components):
                    batch_losses[name] += float(component.detach().item()) * ratio
                    sums[name] += float(component.detach().item()) * len(micro)
                # Measure each actual component before the combined backward.
                for name, component in zip(("main_norm", "quality_norm", "coverage_norm"),
                                           components):
                    if name == "coverage_norm" and group != "C":
                        continue
                    if name == "quality_norm" and group == "A":
                        continue
                    norm = _norm(_grads(component, parameters))
                    batch_losses[name] += norm * ratio
                    sums[name] += norm * len(micro)
                combined = main
                if group in ("B", "C"):
                    combined = combined + calibration["alpha"] * quality
                if group == "C":
                    combined = combined + calibration["lambda"] * coverage
                if not torch.isfinite(combined).item():
                    raise RuntimeError("Nonfinite loss")
                (combined * ratio).backward()
                sums["raw"] += outcome.lengths.detach().sum().item() / micro[0].city_count
                sums["raw_best"] += outcome.lengths.detach().min(dim=1).values.sum().item()
            optimizer.step()
            batch_count += 1
            batches_log.append({"epoch": epoch, "batch": batch_index, "instances": len(batch),
                                "main_loss": batch_losses["main"],
                                "quality_loss": batch_losses["quality"],
                                "coverage_loss": batch_losses["coverage"],
                                "main_grad_norm": batch_losses["main_norm"],
                                "quality_grad_norm": batch_losses["quality_norm"],
                                "coverage_grad_norm": batch_losses["coverage_norm"],
                                "two_opt_seconds": batch_losses["two_opt"],
                                "batch_seconds": time.perf_counter() - start_batch})
            print(f"{group} epoch {epoch}: {batch_index} batches", flush=True)
        training_seconds = time.perf_counter() - start_epoch
        start_validation = time.perf_counter()
        validation_metrics, mappings = validate(policy, validation, device, references,
                                                CONFIG["quality_tolerance"], epoch)
        validation_seconds = time.perf_counter() - start_validation
        count = len(train)
        row = {
            "epoch": epoch, "train_mean_raw_length": sums["raw"] / count,
            "train_mean_best_raw_length": sums["raw_best"] / count,
            "train_mean_selected_two_opt_length": sums["opt"] / selected_total
            if selected_total else None,
            **validation_metrics, "main_loss": sums["main"] / count,
            "quality_loss": sums["quality"] / count,
            "coverage_loss": sums["coverage"] / count,
            "main_grad_norm": sums["main_norm"] / count,
            "quality_grad_norm": sums["quality_norm"] / count,
            "coverage_grad_norm": sums["coverage_norm"] / count,
            "pass_rate": sums["passed"] / selected_total if selected_total else None,
            "mean_unique_two_opt_cycles": sums["unique"] / count if selected_total else None,
            "marginal_nonzero_rate": sums["nonzero"] / selected_total
            if selected_total else None,
            "training_seconds": training_seconds, "two_opt_seconds": sums["two_opt"],
            "validation_seconds": validation_seconds,
            "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1_048_576
            if device.type == "cuda" else 0,
            "peak_cpu_rss_mb": _cpu_rss_mb(),
        }
        epochs.append(row)
        _write_csv(directory / "epoch_metrics.csv", EPOCH_FIELDS, epochs)
        _write_csv(directory / "batch_metrics.csv", BATCH_FIELDS, batches_log)
        _write_csv(directory / f"validation_route_mapping_epoch_{epoch:02d}.csv",
                   ("epoch", "instance", "pomo_start", "reference", "raw_length",
                    "two_opt_length", "good", "canonical_two_opt_cycle"), mappings)
        state = {
            "group": group, "run_id": run_root.name, "epoch": epoch,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": effective_checkpoint_config,
            "source_checkpoint_sha256": SOURCE_SHA,
            "python_shuffle_rng_state": shuffle_rng.getstate(),
            "python_selection_rng_state": selection_rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "best_validation": min(best, validation_metrics["validation_mean_best_raw_length"]),
            "best_epoch": epoch if validation_metrics["validation_mean_best_raw_length"] < best
            else best_epoch,
            "epoch_metrics": epochs,
        }
        _save(directory / "latest.pt", state)
        if validation_metrics["validation_mean_best_raw_length"] < best:
            best = validation_metrics["validation_mean_best_raw_length"]
            best_epoch = epoch
            _save(directory / "best.pt", state)
        print(f"{group} Epoch {epoch}: raw={row['train_mean_best_raw_length']:.2f}, "
              f"validation={row['validation_mean_best_raw_length']:.2f}, "
              f"opt={row['validation_mean_best_two_opt_length']:.2f}, "
              f"seconds={training_seconds:.1f}+{validation_seconds:.1f}", flush=True)
    return {"group": group, "best_epoch": best_epoch, "best_validation": best,
            "final": epochs[-1], "result_dir": str(directory)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--diagnosis", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Source checkpoint SHA256 mismatch")
    diagnosis = json.loads((args.diagnosis / "summary.json").read_text(encoding="utf-8"))
    if (diagnosis["checkpoint_sha256"] != SOURCE_SHA
            or diagnosis["quality_variable_instances"] < 25
            or diagnosis["marginal_nonzero_rate_0_5"] < 0.05):
        raise RuntimeError("Offline feasibility gate did not pass")
    train, validation = load_manifest({"root": str(OLD_DATA), "train_root": str(DATA),
                                       "train_dir": "train", "validation_dir": "validation"})
    if len(train) != 500 or len(validation) != 100:
        raise ValueError("Expected new 500 train / old 100 validation")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for paired A/B/C training")
    references = _reference_table(train + validation)
    run_root = args.output_root / f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    run_root.mkdir(parents=True, exist_ok=False)
    test = subprocess.run(
        [sys.executable, "-B", "-m", "unittest",
         "gb_maco_POMO.experiments.test_opt_guided_feedback", "-v"],
        cwd=PROJECT, text=True, capture_output=True, check=False,
    )
    (run_root / "correctness_tests.txt").write_text(test.stdout + test.stderr,
                                                     encoding="utf-8")
    if test.returncode:
        raise RuntimeError("Correctness tests failed; no A/B/C training started")
    (run_root / "offline_diagnosis.json").write_text(
        json.dumps({"source": str(args.diagnosis.resolve()), "summary": diagnosis},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_root / "references.csv").write_text(
        "instance,reference_length\n" + "".join(f"{path},{length}\n"
                                                for path, length in references.items()),
        encoding="utf-8")
    policy, _ = load_policy(SOURCE, device)
    calibration = calibrate(policy, train, references, device)
    (run_root / "calibration.json").write_text(json.dumps(calibration, indent=2)
                                               + "\n", encoding="utf-8")
    print(f"Calibration: {calibration}", flush=True)
    results = []
    for group in ("A", "B", "C"):
        results.append(train_group(group, run_root, train, validation, references,
                                   calibration, device))
    (run_root / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2)
                                           + "\n", encoding="utf-8")
    print(f"Finished: {run_root}", flush=True)


if __name__ == "__main__":
    main()
