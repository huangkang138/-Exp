"""Train one A/B/C group; choose two checkpoints using independent validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tomllib
from pathlib import Path

import torch

from gb_2_maco_POMO.experiments.shared.data import batches_by_city_count, load_manifest
from .model import BallBiasPolicy
from .reward import diversity_signal, policy_loss
from .rollout import rollout_batch
from .validation import evaluate, select_checkpoints

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("config.toml")
INITIAL_SHA = "efffddda5fe28f678dd835b7d238ffe4e89d740f40045dd761db1e4a46bfd4bc"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path):
    with path.open("rb") as stream:
        cfg = tomllib.load(stream)
    positive = ("epochs", "batch_size", "validation_interval",
                "validation_optb_checks", "diversity_routes_per_instance")
    if any(int(cfg[key]) < 1 for key in positive) or cfg["diversity_routes_per_instance"] < 2:
        raise ValueError("epoch, batch, validation and diversity budgets must be positive")
    if cfg["validation_samples_per_start"] < 0 or cfg["learning_rate"] <= 0:
        raise ValueError("invalid validation samples or learning rate")
    if not 0 <= cfg["diversity_lambda"] <= 0.05 or cfg["diversity_tolerance_pct"] < 0:
        raise ValueError("diversity lambda must be 0..0.05 and tolerance nonnegative")
    if cfg["multi_quality_gap_tolerance_pct"] < 0 or cfg["multi_allowed_certified_hit_loss"] < 0:
        raise ValueError("invalid multi-objective quality gate")
    return cfg


def initialize(group, cfg, device):
    source = (PROJECT / cfg["initial_checkpoint"]).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != INITIAL_SHA:
        raise RuntimeError("original two-level checkpoint differs from the verified source")
    state = torch.load(source, map_location="cpu", weights_only=True)
    arch = state["architecture"]
    model = BallBiasPolicy(int(arch["embedding_dim"]), int(arch["num_heads"]),
                           int(arch["encoder_layers"]), use_bias=group != "A",
                           bias_bound=cfg["ball_bias_bound"])
    incompatible = model.load_state_dict(state["model_state_dict"], strict=False)
    expected = ({f"bias_head.{layer}.{suffix}" for layer in (0, 2)
                 for suffix in ("weight", "bias")} if group != "A" else set())
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError(f"initial checkpoint incompatible: {incompatible}")
    return model.to(device), arch, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("A", "B", "C"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output directory must be new; historical results are never overwritten")
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("configured CUDA device is unavailable")
    if device.type == "cpu":
        torch.set_num_threads(1)
    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    data_root = (PROJECT / cfg["data_root"]).resolve()
    train, validation = load_manifest({"root": str(data_root), "train_dir": "train",
                                       "validation_dir": "validation"})
    if len(train) != 500 or len(validation) != 100 or \
            {x.path for x in train} & {x.path for x in validation}:
        raise RuntimeError("expected independent 500/100 split")
    if {hashlib.sha256(x.path.read_bytes()).digest() for x in train} & \
            {hashlib.sha256(x.path.read_bytes()).digest() for x in validation}:
        raise RuntimeError("identical training/validation instance contents")
    model, arch, source_sha = initialize(args.group, cfg, device)
    # Creating the optional head consumes RNG; reset so A/B/C begin sampling
    # from the same stream and B/C differ only in the configured reward.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    output.mkdir(parents=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    effective = dict(cfg, group=args.group, data_root_resolved=str(data_root),
                     initial_checkpoint_sha256=source_sha,
                     bias_parameters=model.bias_parameter_count,
                     train_instances=len(train), validation_instances=len(validation),
                     validation_reference="certified lengths where proof exists; heuristic elsewhere")
    write_json(output / "effective_config.json", effective)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    batch_rng = random.Random(seed)
    diversity_rng = random.Random(seed + 100_003)
    history, records = [], []
    max_grad = {"ball": 0.0, "bias": 0.0, "city": 0.0}
    update = 0
    for epoch in range(1, cfg["epochs"] + 1):
        train_length, seen, assessed, eligible = 0.0, 0, 0, 0
        for batch in batches_by_city_count(train, cfg["batch_size"], batch_rng):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outcome = rollout_batch(model, batch, device, sample=True)
            if args.group == "C":
                diversity, selected, detail = diversity_signal(
                    outcome, batch, routes_per_instance=cfg["diversity_routes_per_instance"],
                    tolerance_pct=cfg["diversity_tolerance_pct"], rng=diversity_rng)
                assessed += sum(x["evaluated"] for x in detail)
                eligible += sum(x["eligible"] for x in detail)
            else:
                diversity = selected = None
            loss = policy_loss(outcome, diversity_weight=cfg["diversity_lambda"]
                               if args.group == "C" else 0,
                               diversity=diversity, assessed=selected)
            loss.backward()
            for label, params in (("ball", model.ball_head.parameters()),
                                  ("bias", model.bias_head.parameters() if model.use_bias else ()),
                                  ("city", model.decoder.parameters())):
                norm = sum(float(p.grad.detach().square().sum()) for p in params
                           if p.grad is not None) ** 0.5
                max_grad[label] = max(max_grad[label], norm)
            if any(not torch.isfinite(p.grad).all().item() for p in model.parameters()
                   if p.grad is not None):
                raise RuntimeError(f"nonfinite gradient at epoch {epoch}, update {update + 1}")
            optimizer.step()
            update += 1
            seen += len(batch)
            train_length += float(outcome.lengths.detach().mean()) * len(batch)
        if seen != len(train):
            raise RuntimeError("training epoch did not cover every instance")
        row = {"epoch": epoch, "updates": update, "mean_raw_length": train_length / seen,
               "diversity_routes_evaluated": assessed, "diversity_quality_eligible": eligible}
        history.append(row)
        write_csv(output / "training_history.csv", history)
        if epoch % cfg["validation_interval"] and epoch != cfg["epochs"]:
            print(f"{args.group} epoch {epoch}/{cfg['epochs']} updates={update}", flush=True)
            continue
        val_rows, metrics = evaluate(
            model, validation, data_root, device,
            samples=cfg["validation_samples_per_start"],
            optb_checks=cfg["validation_optb_checks"], seed=seed)
        write_csv(output / f"validation_epoch_{epoch:03d}.csv", val_rows)
        snapshot = checkpoints / f"epoch_{epoch:03d}.pt"
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "architecture": arch, "group": args.group, "use_bias": model.use_bias,
                    "bias_bound": model.bias_bound, "epoch": epoch, "step": update,
                    "config": effective, "validation_metrics": metrics,
                    "python_rng_state": batch_rng.getstate(),
                    "diversity_rng_state": diversity_rng.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                   snapshot)
        records.append({"epoch": epoch, "updates": update, "checkpoint": str(snapshot),
                        "metrics": metrics})
        write_json(output / "validation_history.json", records)
        print(f"{args.group} epoch {epoch}/{cfg['epochs']} validation: {metrics}", flush=True)
    if max_grad["ball"] <= 0 or max_grad["city"] <= 0 or \
            (args.group != "A" and max_grad["bias"] <= 0):
        raise RuntimeError(f"missing effective RL gradient: {max_grad}")
    best_length, best_multi = select_checkpoints(
        records, quality_gap_tolerance_pct=cfg["multi_quality_gap_tolerance_pct"],
        allowed_hit_loss=cfg["multi_allowed_certified_hit_loss"])
    shutil.copy2(best_length["checkpoint"], checkpoints / "best_length.pt")
    shutil.copy2(best_multi["checkpoint"], checkpoints / "best_multi.pt")
    write_json(output / "selection.json", {
        "best_length": best_length, "best_multi": best_multi,
        "quality_gate_pct": cfg["multi_quality_gap_tolerance_pct"],
        "allowed_certified_hit_loss": cfg["multi_allowed_certified_hit_loss"],
        "f_beta_available": False, "test_set_used_for_selection": False,
        "note": ("no certified validation lengths; best_multi equals best_length; "
                 "selection uses 2-opt gap relative to a fixed nearest-neighbor plus 2-opt reference"
                 if best_length["metrics"]["certified_cases"] == 0 else
                 "certified optimum count is incomplete; selection uses distinct certified-length tours, not recall")})
    write_json(output / "train_result.json", {"group": args.group, "epochs": cfg["epochs"],
               "updates": update, "max_gradient_norm": max_grad,
               "best_length_epoch": best_length["epoch"], "best_multi_epoch": best_multi["epoch"]})


if __name__ == "__main__":
    main()
