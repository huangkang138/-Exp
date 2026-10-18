"""Formal 35-epoch joint ball/city POMO training with explicit data roots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from time import perf_counter

import torch

from gb_2_maco_POMO.experiments.shared.data import batches_by_city_count, load_manifest
from gb_2_maco_POMO.experiments.learnable_ball.learnable_policy import LearnableBallPolicy
from gb_2_maco_POMO.experiments.learnable_ball.learnable_rollout import rollout_batch
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.learnable_ball.train_learnable import (
    DATA, EXPECTED_FIXED_SHA, FIXED, _grad_norm, _loss, _probe,
)
from gb_2_maco_POMO.runner import prepare_tsp

EPOCHS = 35
BATCH_SIZE = 64
MICRO_BATCH_SIZE = 8
LR = 1e-4
SEED = 42
VALIDATION_INTERVAL = 4
EXPECTED_UPDATES = 385


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _reference_lengths(instances, output: Path) -> dict[str, int]:
    """Fixed nearest-neighbor + ordinary 2-opt references; these are not optima."""
    rows = []
    for item in instances:
        matrix = prepare_tsp(item.path).distance_matrix
        n = item.city_count
        route, remaining = [0], set(range(1, n))
        while remaining:
            city = min(remaining, key=lambda j: (matrix[route[-1]][j], j))
            route.append(city)
            remaining.remove(city)
        rows.append({"instance": item.path.name, "reference_length": two_opt(route, matrix).length})
    _csv(output / "validation_reference.csv", rows)
    return {row["instance"]: row["reference_length"] for row in rows}


@torch.no_grad()
def _validate(model, instances, device, reference):
    model.eval()
    rows = []
    for batch in batches_by_city_count(instances, MICRO_BATCH_SIZE):
        best = rollout_batch(model, batch, device, sample=False).lengths.min(dim=1).values.cpu().tolist()
        rows.extend({"instance": item.path.name, "best_length": length,
                     "reference_length": reference[item.path.name],
                     "reference_gap_pct": 100 * (length / reference[item.path.name] - 1)}
                    for item, length in zip(batch, best))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, default=DATA)
    parser.add_argument("--validation-root", type=Path, default=DATA)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    train_root = args.train_root.resolve()
    validation_root = args.validation_root.resolve()
    if output.exists():
        parser.error("output directory must be new")
    if not torch.cuda.is_available():
        parser.error("formal historical configuration requires CUDA")
    started = perf_counter()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    train, validation = load_manifest({"root": str(validation_root),
                                       "train_root": str(train_root), "train_dir": "train",
                                       "validation_dir": "validation"})
    if (len(train) != 500 or len(validation) != 100
            or {i.path for i in train} & {i.path for i in validation}
            or {hashlib.sha256(i.path.read_bytes()).hexdigest() for i in train}
            & {hashlib.sha256(i.path.read_bytes()).hexdigest() for i in validation}):
        raise RuntimeError("expected disjoint historical 500/100 split")
    batches_per_epoch = len(batches_by_city_count(train, BATCH_SIZE))
    if batches_per_epoch * EPOCHS != EXPECTED_UPDATES:
        raise RuntimeError(f"expected 385 updates, found {batches_per_epoch * EPOCHS}")
    fixed_sha = hashlib.sha256(FIXED.read_bytes()).hexdigest()
    if fixed_sha != EXPECTED_FIXED_SHA:
        raise RuntimeError("fixed Soft checkpoint changed")
    fixed = torch.load(FIXED, map_location="cpu", weights_only=True)
    architecture = {key: int(fixed["config"]["model"][key])
                    for key in ("embedding_dim", "num_heads", "encoder_layers")}
    model = LearnableBallPolicy(
        architecture["embedding_dim"], architecture["num_heads"],
        architecture["encoder_layers"],
    ).cuda()
    incompatible = model.load_state_dict(fixed["model_state_dict"], strict=False)
    expected_missing = {f"ball_head.{layer}.{suffix}" for layer in (0, 2)
                        for suffix in ("weight", "bias")}
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(f"fixed checkpoint incompatible: {incompatible}")
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    _json(output / "train_config.json", {
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "micro_batch_size": MICRO_BATCH_SIZE,
        "optimizer": "Adam", "learning_rate": LR, "seed": SEED,
        "updates_per_epoch": batches_per_epoch, "expected_updates": EXPECTED_UPDATES,
        "validation_interval_epochs": VALIDATION_INTERVAL,
        "historical_validation_batch_size_config": 64,
        "actual_validation_inference_chunk_size": MICRO_BATCH_SIZE,
        "device": "cuda", "train_instances": len(train), "validation_instances": len(validation),
        "reward": "negative complete closed-tour distance",
        "loss": "POMO advantage times joint ball/city log probability",
        "checkpoint_selection": "minimum validation greedy POMO mean best length",
        "validation_gap_definition": "relative to frozen nearest-neighbor plus ordinary 2-opt; not optimality gap",
        "train_dataset": str(train_root), "validation_dataset": str(validation_root),
        "train_manifest_sha256": hashlib.sha256((train_root / "manifest.csv").read_bytes()).hexdigest(),
        "validation_manifest_sha256": hashlib.sha256((validation_root / "manifest.csv").read_bytes()).hexdigest(),
        "initialized_from": str(FIXED), "fixed_checkpoint_sha256": fixed_sha,
        "start_mode": "restart from fixed Soft; short run lacks RNG state and used a different validation schedule",
        "formal_sources": ["fixed Soft run_20261013_230742_828b99",
                           "learnable Bias run_20261014_223323_7de9e6"],
    })
    reference = _reference_lengths(validation, output)
    probe = _probe(model, train, torch.device("cuda"), LR)
    _json(output / "probe.json", probe)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng = random.Random(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    initial_head = [p.detach().clone() for p in model.ball_head.parameters()]
    history, validation_history = [], []
    best_validation, best_epoch, best_step = float("inf"), 0, 0
    max_ball_grad, step = 0.0, 0
    for epoch in range(1, EPOCHS + 1):
        epoch_started = perf_counter()
        reward_sum = best_sum = loss_sum = 0.0
        seen = 0
        for batch in batches_by_city_count(train, BATCH_SIZE, rng):
            model.train()
            optimizer.zero_grad()
            for offset in range(0, len(batch), MICRO_BATCH_SIZE):
                micro = batch[offset:offset + MICRO_BATCH_SIZE]
                outcome = rollout_batch(model, micro, torch.device("cuda"), sample=True)
                loss = _loss(outcome)
                (loss * (len(micro) / len(batch))).backward()
                count = len(micro)
                reward_sum -= float(outcome.lengths.detach().mean().item()) * count
                best_sum += float(outcome.lengths.detach().min(dim=1).values.mean().item()) * count
                loss_sum += float(loss.detach().item()) * count
                seen += count
            ball_grad = _grad_norm(model.ball_head.parameters())
            if not all(torch.isfinite(p.grad).all().item()
                       for p in model.parameters() if p.grad is not None):
                raise RuntimeError(f"invalid RL gradient at epoch {epoch}, update {step+1}")
            max_ball_grad = max(max_ball_grad, ball_grad)
            optimizer.step()
            step += 1
        if seen != 500 or step != epoch * batches_per_epoch:
            raise RuntimeError("unexpected training count or update count")
        validation_mean = validation_gap = None
        if epoch % VALIDATION_INTERVAL == 0 or epoch == EPOCHS:
            val_rows = _validate(model, validation, torch.device("cuda"), reference)
            validation_mean = sum(r["best_length"] for r in val_rows) / len(val_rows)
            validation_gap = sum(r["reference_gap_pct"] for r in val_rows) / len(val_rows)
            _csv(output / f"validation_epoch_{epoch:03d}.csv", val_rows)
            validation_history.append({"epoch": epoch, "step": step,
                                       "mean_best_length": validation_mean,
                                       "mean_reference_gap_pct": validation_gap})
        # Latest includes optimizer and random state so an interrupted run is recoverable.
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch, "step": step, "train_instances_seen": epoch * seen,
                      "validation_mean_best_length": validation_mean,
                      "architecture": architecture, "fixed_checkpoint_sha256": fixed_sha,
                      "python_rng_state": rng.getstate(),
                      "torch_rng_state": torch.get_rng_state(),
                      "cuda_rng_state": torch.cuda.get_rng_state_all()}
        torch.save(checkpoint, output / "checkpoints" / "latest.pt")
        if epoch % 20 == 0:
            torch.save(checkpoint, output / "checkpoints" / f"epoch_{epoch:03d}.pt")
        if validation_mean is not None and validation_mean < best_validation:
            best_validation, best_epoch, best_step = validation_mean, epoch, step
            torch.save(checkpoint, output / "checkpoints" / "best.pt")
        row = {"epoch": epoch, "updates": step, "train_mean_reward": reward_sum / seen,
               "train_mean_length": -reward_sum / seen,
               "train_mean_best_length": best_sum / seen, "mean_loss": loss_sum / seen,
               "validation_mean_best_length": validation_mean,
               "validation_reference_gap_pct": validation_gap,
               "best_epoch_so_far": best_epoch, "seconds": perf_counter() - epoch_started}
        history.append(row)
        _csv(output / "training_history.csv", history)
        _csv(output / "validation_history.csv", validation_history) if validation_history else None
        head_delta = sum((p.detach() - old).abs().sum().item()
                         for p, old in zip(model.ball_head.parameters(), initial_head))
        _json(output / "progress.json", {"completed_epochs": epoch, "updates": step,
                                         "best_epoch": best_epoch, "best_step": best_step,
                                         "best_validation_mean_best_length": best_validation,
                                         "ball_parameter_delta_l1": head_delta,
                                         "elapsed_seconds": perf_counter() - started})
        print(f"Epoch {epoch}/{EPOCHS} updates={step} reward={row['train_mean_reward']:.2f} "
              f"val={validation_mean} ref_gap={validation_gap} best={best_epoch} "
              f"seconds={row['seconds']:.1f}", flush=True)
    if step != EXPECTED_UPDATES or max_ball_grad <= 0 or head_delta <= 0:
        raise RuntimeError("formal training or ball-head update incomplete")
    _json(output / "train_result.json", {
        "completed_epochs": EPOCHS, "train_instances": len(train),
        "validation_instances": len(validation), "total_optimizer_updates": step,
        "best_epoch": best_epoch, "best_step": best_step,
        "best_validation_mean_best_length": best_validation,
        "max_ball_grad_norm": max_ball_grad, "ball_parameter_delta_l1": head_delta,
        "best_checkpoint_sha256": hashlib.sha256((output / "checkpoints" / "best.pt").read_bytes()).hexdigest(),
        "training_seconds": perf_counter() - started,
    })
    print(f"Training complete: {output}", flush=True)


if __name__ == "__main__":
    main()
