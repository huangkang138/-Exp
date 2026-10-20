"""Matched A/B POMO training; the user starts this command explicitly."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from time import perf_counter

import torch

from gb_2_maco_POMO.attention import build_city_features
from gb_2_maco_POMO.experiments.learnable_ball.learnable_rollout import rollout_batch
from gb_2_maco_POMO.experiments.learnable_ball.train_learnable import _grad_norm, _loss
from gb_2_maco_POMO.experiments.shared.data import batches_by_city_count
from gb_2_maco_POMO.runner import prepare_tsp

from .common import initial_model, load_data, sha256
from .config import (BATCH_SIZE, DATA, EPOCHS, INFERENCE_SEEDS, INITIAL,
                     INITIAL_SHA256, LEARNING_RATE, MICRO_BATCH_SIZE, RESULTS,
                     TRAIN_MANIFEST_SHA256, TRAIN_SEEDS, VALIDATION_MANIFEST_SHA256)
from .validation import VALIDATION_SEED, choose, evaluate


def save_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validation(model, validation, device):
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        return evaluate(model, validation, device)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@torch.inference_mode()
def _attention_probe(model, item, device):
    if not hasattr(model, "ball_attention_alpha"):
        return {"alpha": None, "weight_change_by_current_city": None}
    problem = prepare_tsp(item.path)
    features = build_city_features(problem.cities, problem.granular_balls,
                                   problem.city_to_ball).to(device)
    _, encoded = model.encode(features)
    n = item.city_count
    mapping = torch.tensor(problem.city_to_ball, dtype=torch.long, device=device)[None, :]
    available = torch.ones((1, 1, n), dtype=torch.bool, device=device)
    available[:, :, :3] = False
    first = torch.zeros((1, 1), dtype=torch.long, device=device)
    _, _, counts, w1 = model.ball_summary(encoded, first, torch.ones_like(first),
                                           available, mapping)
    _, _, _, w2 = model.ball_summary(encoded, first, torch.full_like(first, 2),
                                     available, mapping)
    return {"alpha": float(model.ball_attention_alpha.item()),
            "weight_change_by_current_city": float(
                ((w1 - w2).abs() * (counts >= 2)[..., None]).max().item())}


def _snapshot(path, model, optimizer, group, seed, epoch, architecture,
              rng, records, history, device):
    torch.save({"group": group, "train_seed": seed, "epoch": epoch,
                "initial_sha256": INITIAL_SHA256,
                "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
                "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
                "architecture": {"embedding_dim": architecture[0],
                                 "num_heads": architecture[1],
                                 "encoder_layers": architecture[2]},
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "data_rng_state": rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "validation_records": records, "training_history": history}, path)


def run_group(group, seed, train, validation, device):
    if group not in ("A", "B") or seed not in TRAIN_SEEDS:
        raise ValueError("invalid group or seed")
    marker = RESULTS / f"preflight_{device.type}.json"
    checked = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
    if (not checked.get("passed") or checked.get("checkpoint_sha256") != INITIAL_SHA256
            or checked.get("max_ball_logit_delta") != 0
            or checked.get("max_joint_probability_delta") != 0):
        raise RuntimeError(f"epoch-0 preflight must pass before training: {marker}")
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model, architecture = initial_model(group, device)
    # Model construction consumes RNG; reset to match the POMO draws exactly.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = random.Random(seed)
    output = RESULTS / f"train_seed_{seed}" / group
    config = {"group": group, "train_seed": seed, "initial_checkpoint": str(INITIAL),
              "initial_sha256": INITIAL_SHA256, "train_dataset": str(DATA),
              "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
              "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
              "train_cases": 500, "validation_cases": 100, "epochs": EPOCHS,
              "updates_per_epoch": 11, "batch_size": BATCH_SIZE,
              "micro_batch_size": MICRO_BATCH_SIZE, "optimizer": "Adam",
              "learning_rate": LEARNING_RATE, "device": str(device),
              "reward": "negative closed-tour length, per-instance POMO baseline",
              "validation": "all-city 1 argmax + 8 sampling, ordinary 2-opt on every unique raw cycle",
              "validation_seed": VALIDATION_SEED,
              "selection": "lowest 100-case mean 2-opt relative reference gap, then raw gap, then earliest epoch",
              "reference": "certified optimum when available; otherwise nearest-neighbor plus 2-opt heuristic upper bound",
              "test_set_used_for_selection": False}
    if (output / "train_result.json").is_file():
        print(f"skip completed {group} seed={seed}", flush=True)
        return
    if output.exists():
        if json.loads((output / "config.json").read_text(encoding="utf-8")) != config:
            raise RuntimeError(f"existing training configuration differs: {output}")
        state = torch.load(output / "checkpoints/last.pt", map_location="cpu", weights_only=True)
        if (state["group"], state["train_seed"], state["initial_sha256"],
                state["train_manifest_sha256"], state["validation_manifest_sha256"]) != (
                group, seed, INITIAL_SHA256, TRAIN_MANIFEST_SHA256, VALIDATION_MANIFEST_SHA256):
            raise RuntimeError("resume checkpoint identity differs")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        rng.setstate(state["data_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        records, history = state["validation_records"], state["training_history"]
        first_epoch = state["epoch"] + 1
    else:
        (output / "checkpoints").mkdir(parents=True)
        save_json(output / "config.json", config)
        model.eval()
        validation_started = perf_counter()
        val_rows, metrics = _validation(model, validation, device)
        save_json(output / "epoch0_validation_cost.json", {
            "seconds": perf_counter() - validation_started})
        save_csv(output / "validation_epoch_000.csv", val_rows)
        epoch0 = output / "checkpoints/epoch_000.pt"
        records = [{"epoch": 0, "checkpoint": str(epoch0), "metrics": metrics,
                    "checkpoint_selection_used_test": False}]
        history = []
        _snapshot(epoch0, model, optimizer, group, seed, 0, architecture,
                  rng, records, history, device)
        _snapshot(output / "checkpoints/last.pt", model, optimizer, group, seed, 0,
                  architecture, rng, records, history, device)
        save_json(output / "validation_history.json", records)
        first_epoch = 1
    probe_item = next(item for item in train if item.city_count >= 50
                      and len(prepare_tsp(item.path).granular_balls) > 1)
    for epoch in range(first_epoch, EPOCHS + 1):
        started = perf_counter()
        batches = batches_by_city_count(train, BATCH_SIZE, rng)
        if len(batches) != 11:
            raise RuntimeError("optimizer update count changed")
        step_rows = []
        for step, batch in enumerate(batches, 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            lengths = []
            for offset in range(0, len(batch), MICRO_BATCH_SIZE):
                micro = batch[offset:offset + MICRO_BATCH_SIZE]
                outcome = rollout_batch(model, micro, device, sample=True)
                loss = _loss(outcome)
                (loss * len(micro) / len(batch)).backward()
                losses.append((float(loss.detach()), len(micro)))
                lengths.extend(outcome.lengths.detach().min(dim=1).values.cpu().tolist())
            if not all(torch.isfinite(p.grad).all().item() for p in model.parameters()
                       if p.grad is not None):
                raise RuntimeError(f"nonfinite gradient: {group} seed={seed} epoch={epoch} step={step}")
            alpha_grad = (float(model.ball_attention_alpha.grad.item())
                          if group == "B" and model.ball_attention_alpha.grad is not None else None)
            row = {"epoch": epoch, "step": step, "city_count": batch[0].city_count,
                   "instances": len(batch), "mean_pomo_loss": sum(v*n for v,n in losses)/len(batch),
                   "mean_best_raw_length": sum(lengths)/len(lengths),
                   "ball_grad_norm": _grad_norm(model.ball_head.parameters()),
                   "decoder_grad_norm": _grad_norm(model.decoder.parameters()),
                   "alpha_grad": alpha_grad}
            optimizer.step()
            row["alpha_after_update"] = (float(model.ball_attention_alpha.item())
                                         if group == "B" else None)
            step_rows.append(row)
            print(f"{group} seed={seed} epoch={epoch} step={step}/11 "
                  f"loss={row['mean_pomo_loss']:.3f} alpha={row['alpha_after_update']}", flush=True)
        train_seconds = perf_counter() - started
        save_csv(output / f"training_steps_epoch_{epoch:03d}.csv", step_rows)
        model.eval()
        attention = _attention_probe(model, probe_item, device)
        validation_started = perf_counter()
        val_rows, metrics = _validation(model, validation, device)
        validation_seconds = perf_counter() - validation_started
        save_csv(output / f"validation_epoch_{epoch:03d}.csv", val_rows)
        snapshot = output / "checkpoints" / f"epoch_{epoch:03d}.pt"
        records.append({"epoch": epoch, "checkpoint": str(snapshot), "metrics": metrics,
                        "checkpoint_selection_used_test": False})
        selected = choose(records)
        history.append({"epoch": epoch, "updates": 11, "instances": 500,
                        "train_seconds_including_validation": perf_counter() - started,
                        "train_seconds": train_seconds,
                        "validation_seconds": validation_seconds,
                        "mean_pomo_loss": sum(r["mean_pomo_loss"] for r in step_rows)/11,
                        "mean_best_raw_length": sum(r["mean_best_raw_length"] for r in step_rows)/11,
                        "max_abs_alpha_grad": max((abs(r["alpha_grad"]) for r in step_rows
                                                   if r["alpha_grad"] is not None), default=None),
                        **attention, "selected_epoch_so_far": selected["epoch"]})
        save_csv(output / "training_history.csv", history)
        save_json(output / "validation_history.json", records)
        save_json(output / "selection.json", {"selected_epoch": selected["epoch"],
                  "checkpoint": selected["checkpoint"], "metrics": selected["metrics"],
                  "test_set_used_for_selection": False})
        _snapshot(snapshot, model, optimizer, group, seed, epoch, architecture,
                  rng, records, history, device)
        _snapshot(output / "checkpoints/last.pt", model, optimizer, group, seed,
                  epoch, architecture, rng, records, history, device)
        print(f"{group} seed={seed} epoch={epoch}/3 selected={selected['epoch']} "
              f"val_2opt_reference_gap={metrics['mean_two_opt_relative_reference_pct']:.4f}",
              flush=True)
    selected = choose(records)
    if group == "B" and not any(x["max_abs_alpha_grad"] and x["max_abs_alpha_grad"] > 0
                                for x in history):
        raise RuntimeError("dynamic attention alpha never received a gradient")
    save_json(output / "train_result.json", {
        "group": group, "train_seed": seed, "completed_epochs": EPOCHS,
        "total_optimizer_updates": EPOCHS * 11, "selected_epoch": selected["epoch"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": sha256(Path(selected["checkpoint"])),
        "initial_sha256": INITIAL_SHA256,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_seconds": sum(x["train_seconds"] for x in history),
        "validation_seconds": (json.loads((output / "epoch0_validation_cost.json").read_text(
            encoding="utf-8"))["seconds"] + sum(x["validation_seconds"] for x in history)),
        "training_and_validation_seconds": (json.loads((output / "epoch0_validation_cost.json")
            .read_text(encoding="utf-8"))["seconds"] +
            sum(x["train_seconds_including_validation"] for x in history))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=TRAIN_SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    train, validation = load_data()
    for seed in (args.seed,) if args.seed is not None else TRAIN_SEEDS:
        for group in ("A", "B"):
            run_group(group, seed, train, validation, device)


if __name__ == "__main__":
    main()
