"""Ten-epoch Joint Head-only continuation and independent dual checkpoint selection."""

from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path
from time import perf_counter

import torch

from gb_2_maco_POMO.experiments.dual_reward.selection import select_checkpoints
from gb_2_maco_POMO.experiments.dual_reward.train import _validation_optima, save_csv, save_json
from gb_2_maco_POMO.experiments.learnable_ball.train_learnable import _loss
from gb_2_maco_POMO.experiments.shared.data import batches_by_city_count, load_manifest
from gb_2_maco_POMO.experiments.region_supervised_pomo.config import INITIAL_SHA256

from .preflight import check, initialize
from .rollout import rollout_batch
from .settings import Settings
from .validation import evaluate


def _snapshot(model, optimizer, source, config: Settings, epoch: int) -> dict:
    return {"model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": dict(source["architecture"]),
            "epoch": epoch, "train_instances_seen": epoch * 500,
            "initialized_from": str(config.initial_checkpoint),
            "initial_checkpoint_sha256": INITIAL_SHA256,
            "trainable_module": "joint_head",
            "frozen_modules": ["encoder", "ball_head", "decoder"]}


def _validate_and_save(model, validation, optima, config, output, epoch, optimizer, source):
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    try:
        rows, selection_metrics = evaluate(model, validation, optima, config)
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
    path = output / "checkpoints" / f"epoch_{epoch}.pt"
    torch.save(_snapshot(model, optimizer, source, config, epoch), path)
    save_csv(output / f"validation_epoch_{epoch}.csv", rows)
    record = {"epoch": epoch, "checkpoint": str(path),
              "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "metrics": selection_metrics}
    flat = {"epoch": epoch, **selection_metrics,
            "mean_raw_length": sum(r["raw_best_length"] for r in rows) / 100,
            "mean_two_opt_length": sum(r["two_opt_best_length"] for r in rows) / 100,
            "mean_optb_length": sum(r["optb_best_length"] for r in rows) / 100,
            "mean_raw_unique_routes": sum(r["raw_unique_routes"] for r in rows) / 100,
            "mean_optb_routes": sum(r["optb_routes"] for r in rows) / 100,
            "ball_argmax_changes": sum(r["ball_argmax_changes"] for r in rows),
            "city_argmax_changes": sum(r["city_argmax_changes"] for r in rows),
            "ball_probability_l1_mean": sum(r["ball_probability_l1_mean"] for r in rows) / 100,
            "city_probability_l1_mean": sum(r["city_probability_l1_mean"] for r in rows) / 100,
            "joint_probability_l1_mean": sum(r["joint_probability_l1_mean"] for r in rows) / 100,
            "score_mean": sum(r["score_mean"] for r in rows) / 100,
            "score_std_mean": sum(r["score_std"] for r in rows) / 100,
            "score_min": min(r["score_min"] for r in rows),
            "score_max": max(r["score_max"] for r in rows),
            "checkpoint_sha256": record["checkpoint_sha256"]}
    print(f"epoch {epoch}: certified hits={selection_metrics['certified_hits']}/56 "
          f"gap={selection_metrics['mean_certified_gap_pct']:.4f}% "
          f"multi={selection_metrics['mean_certified_optb_routes']:.4f}", flush=True)
    return record, flat


def run(output: Path, config: Settings) -> dict:
    output = output.resolve()
    if output.exists() and {p.name for p in output.iterdir()} != {"run_status.json"}:
        raise FileExistsError(f"new output directory required: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Joint training refuses CPU fallback")
    device = torch.device("cuda")
    if hashlib.sha256(config.initial_checkpoint.read_bytes()).hexdigest() != INITIAL_SHA256:
        raise RuntimeError("A0 checkpoint changed")
    train, validation = load_manifest({"root": str(config.data_root),
                                       "train_dir": "train", "validation_dir": "validation"})
    if (len(train), len(validation)) != (500, 100) or {x.path for x in train} & {
            x.path for x in validation}:
        raise RuntimeError("Multiopt 500/100 split changed")
    optima = _validation_optima(config.data_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    shutil.copyfile(config.config_path, output / "config.toml")
    save_json(output / "config.json", {
        "data_root": str(config.data_root), "manifest_sha256": hashlib.sha256(
            (config.data_root / "manifest.csv").read_bytes()).hexdigest(),
        "initial_checkpoint": str(config.initial_checkpoint),
        "initial_checkpoint_sha256": INITIAL_SHA256,
        "seed": config.seed, "epochs": config.epochs,
        "validation_interval": config.validation_interval,
        "batch_size": config.batch_size, "micro_batch_size": config.micro_batch_size,
        "learning_rate": config.learning_rate, "device": config.device,
        "reward": "negative closed-tour length; original POMO baseline",
        "trainable": "joint_head only", "certified_validation_cases": len(optima)})
    save_json(output / "preflight.json", check(config, train, device))
    print("Joint preflight passed; training starts", flush=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    _, model, source = initialize(config.initial_checkpoint)
    model.to(device).eval()  # Frozen backbone keeps the A0 evaluation behavior.
    optimizer = torch.optim.Adam(model.joint_head.parameters(), lr=config.learning_rate)
    original = {name: p.detach().clone() for name, p in model.named_parameters()
                if not name.startswith("joint_head.")}
    joint_start = {name: p.detach().clone() for name, p in model.joint_head.named_parameters()}
    records, flat_history, training_rows = [], [], []
    record, flat = _validate_and_save(model, validation, optima, config, output, 0, optimizer, source)
    records.append(record)
    flat_history.append(flat)
    save_csv(output / "validation_history.csv", flat_history)
    rng = random.Random(config.seed)
    started = perf_counter()
    step_number = 0
    for epoch in range(1, config.epochs + 1):
        seen = 0
        batches = batches_by_city_count(train, config.batch_size, rng)
        for step, batch in enumerate(batches, 1):
            step_number += 1
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            for offset in range(0, len(batch), config.micro_batch_size):
                micro = batch[offset:offset + config.micro_batch_size]
                outcome = rollout_batch(model, micro, device, sample=True)
                loss = _loss(outcome)
                weight = len(micro) / len(batch)
                (loss * weight).backward()
                weighted_loss += loss.item() * weight
            if any(p.grad is None or not torch.isfinite(p.grad).all().item()
                   for p in model.joint_head[-1].parameters()):
                raise RuntimeError(f"epoch {epoch} step {step}: invalid output-layer gradient")
            grad_norm = sum(float(p.grad.detach().square().sum()) for p in
                            model.joint_head.parameters() if p.grad is not None) ** .5
            optimizer.step()
            seen += len(batch)
            training_rows.append({"epoch": epoch, "step": step_number,
                                  "instances_seen": (epoch - 1) * 500 + seen,
                                  "city_count": batch[0].city_count,
                                  "loss": weighted_loss, "joint_gradient_norm": grad_norm})
            print(f"epoch {epoch}/{config.epochs} step {step}/{len(batches)}: {seen}/500", flush=True)
        if seen != 500 or any(not torch.equal(original[name], p) for name, p in
                              model.named_parameters() if name in original):
            raise RuntimeError(f"epoch {epoch}: incomplete data or frozen A0 changed")
        save_csv(output / "training_history.csv", training_rows)
        if epoch % config.validation_interval == 0:
            record, flat = _validate_and_save(model, validation, optima, config,
                                              output, epoch, optimizer, source)
            records.append(record)
            flat_history.append(flat)
            save_csv(output / "validation_history.csv", flat_history)
    if not any(not torch.equal(joint_start[name], p) for name, p in
               model.joint_head.named_parameters()):
        raise RuntimeError("Joint Head did not change after training")
    shortest, diversity = select_checkpoints(records,
        quality_gap_tolerance_pct=config.multi_quality_gap_tolerance_pct,
        allowed_hit_loss=config.multi_allowed_certified_hit_loss)
    for alias, selected in (("best_shortest", shortest), ("best_diversity", diversity)):
        shutil.copyfile(selected["checkpoint"], output / "checkpoints" / f"{alias}.pt")
    selection = {"best_shortest": shortest, "best_diversity": diversity,
                 "quality_gap_tolerance_pct": config.multi_quality_gap_tolerance_pct,
                 "allowed_certified_hit_loss": config.multi_allowed_certified_hit_loss,
                 "definition": "56 certified cases; distance gap first, eligible certified Opt-B route mean second"}
    save_json(output / "selection.json", selection)
    result = {"epochs": config.epochs, "validation_epochs": [r["epoch"] for r in records],
              "best_shortest_epoch": shortest["epoch"],
              "best_diversity_epoch": diversity["epoch"],
              "training_seconds": perf_counter() - started,
              "original_parameters_frozen": True,
              "joint_parameter_delta_l1": sum((p - joint_start[name]).abs().sum().item()
                                                for name, p in model.joint_head.named_parameters())}
    save_json(output / "train_result.json", result)
    return result
