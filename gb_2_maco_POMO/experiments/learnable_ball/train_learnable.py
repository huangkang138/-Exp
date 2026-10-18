"""One short RL epoch for the learnable ball policy; validation selects best.pt."""

from __future__ import annotations

import argparse
import copy
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
from gb_2_maco_POMO.runner import prepare_tsp

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "benachmark_Train_MATSP_500"
FIXED = PROJECT / "gb_maco_POMO/results/checkpoints/run_20261013_230742_828b99/best.pt"
EXPECTED_FIXED_SHA = "dbdfc5cb2faef0d7cbecb7c5c7cc77db16049e5d10e5d584a26e288f1bd88410"


def _save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _grad_norm(parameters) -> float:
    return sum(float(p.grad.detach().square().sum().item()) for p in parameters
               if p.grad is not None) ** 0.5


def _loss(outcome) -> torch.Tensor:
    if outcome.joint_log_probs is None:
        raise RuntimeError("sampled rollout must carry joint ball/city logP")
    rewards = -outcome.lengths  # unchanged closed-tour reward
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    result = -(advantages.detach() * outcome.joint_log_probs).mean()
    if not torch.isfinite(result).item():
        raise RuntimeError("joint policy-gradient loss is not finite")
    return result


def _probe(model, train_instances, device, learning_rate):
    """Use a multi-ball case so the ball choice is nontrivial."""
    item = next((x for x in train_instances if x.city_count >= 50
                 and len(prepare_tsp(x.path).granular_balls) > 1), None)
    if item is None:
        raise RuntimeError("no multi-ball training case for gradient probe")
    trial = copy.deepcopy(model)
    trial.train()
    optimizer = torch.optim.Adam(trial.parameters(), lr=learning_rate)
    torch.manual_seed(12345)
    outcome = rollout_batch(trial, [item], device, sample=True)
    loss = _loss(outcome)
    loss.backward()
    ball_grad = _grad_norm(trial.ball_head.parameters())
    city_grad = _grad_norm(trial.decoder.parameters())
    if ball_grad <= 0 or city_grad <= 0:
        raise RuntimeError(f"missing RL gradient: ball={ball_grad}, city={city_grad}")
    before_ball = [p.detach().clone() for p in trial.ball_head.parameters()]
    before_city = [p.detach().clone() for p in trial.decoder.parameters()]
    optimizer.step()
    ball_delta = sum((p.detach() - old).abs().sum().item()
                     for p, old in zip(trial.ball_head.parameters(), before_ball))
    city_delta = sum((p.detach() - old).abs().sum().item()
                     for p, old in zip(trial.decoder.parameters(), before_city))
    if ball_delta <= 0 or city_delta <= 0:
        raise RuntimeError("ball or city parameters did not update")
    return {"case": item.path.name, "city_count": item.city_count,
            "ball_count": len(prepare_tsp(item.path).granular_balls),
            "route_count": outcome.routes.shape[1], "loss": float(loss.item()),
            "ball_grad_norm": ball_grad, "city_grad_norm": city_grad,
            "ball_parameter_delta_l1": ball_delta, "city_parameter_delta_l1": city_delta,
            "routes_legal": True}


@torch.no_grad()
def _validate(model, instances, device, micro_batch_size):
    model.eval()
    total = 0.0
    groups: dict[tuple[str, int], list[float]] = {}
    for batch in batches_by_city_count(instances, micro_batch_size):
        output = rollout_batch(model, batch, device, sample=False)
        best = output.lengths.min(dim=1).values.cpu().tolist()
        for item, length in zip(batch, best):
            total += length
            groups.setdefault((item.distribution, item.city_count), []).append(length)
    details = [{"distribution": distribution, "city_count": city_count,
                "instance_count": len(values), "mean_best_length": sum(values) / len(values)}
               for (distribution, city_count), values in sorted(groups.items())]
    return total / len(instances), details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--micro-batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.micro_batch_size < 1 or args.output_dir.exists():
        parser.error("micro batch must be positive and output directory must be new")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("selected Python environment has no CUDA")
    started = perf_counter()
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    seed, batch_size, learning_rate = 42, 64, 1e-4
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train, validation = load_manifest({"root": str(DATA), "train_dir": "train",
                                       "validation_dir": "validation"})
    if len(train) != 500 or len(validation) != 100:
        raise RuntimeError("expected 500 disjoint training and 100 validation instances")
    if {x.path for x in train} & {x.path for x in validation}:
        raise RuntimeError("training and validation overlap")
    fixed_sha = hashlib.sha256(FIXED.read_bytes()).hexdigest()
    if fixed_sha != EXPECTED_FIXED_SHA:
        raise RuntimeError("fixed Soft checkpoint changed")
    state = torch.load(FIXED, map_location="cpu", weights_only=True)
    config = state["config"]["model"]
    model = LearnableBallPolicy(int(config["embedding_dim"]), int(config["num_heads"]),
                                int(config["encoder_layers"])).to(device)
    incompatible = model.load_state_dict(state["model_state_dict"], strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != {
        f"ball_head.{layer}.{suffix}" for layer in (0, 2) for suffix in ("weight", "bias")
    }:
        raise RuntimeError(f"fixed weights are incompatible: {incompatible}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    _save_json(output / "train_config.json", {
        "epochs": 1, "batch_size": batch_size, "micro_batch_size": args.micro_batch_size,
        "optimizer": "Adam", "learning_rate": learning_rate, "seed": seed,
        "device": args.device, "reward": "negative complete closed-tour distance",
        "loss": "negative mean of POMO advantage times sum(logP(ball)+logP(city|ball))",
        "validation": "100 disjoint cases, greedy POMO best length; midpoint/end",
        "train_instances": len(train), "validation_instances": len(validation),
        "dataset": str(DATA),
        "train_manifest_sha256": hashlib.sha256((DATA / "train_manifest.csv").read_bytes()).hexdigest(),
        "validation_manifest_sha256": hashlib.sha256((DATA / "validation_manifest.csv").read_bytes()).hexdigest(),
        "initialized_from": str(FIXED), "fixed_checkpoint_sha256": fixed_sha,
    })
    probe = _probe(model, train, device, learning_rate)
    _save_json(output / "probe.json", probe)
    print(f"Probe passed: ball grad={probe['ball_grad_norm']:.3f}, "
          f"city grad={probe['city_grad_norm']:.3f}", flush=True)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batches = batches_by_city_count(train, batch_size, rng)
    processed = 0
    best_validation = float("inf")
    best_step = 0
    history = []
    validation_rows = []
    max_ball_grad = 0.0
    initial_head = [p.detach().clone() for p in model.ball_head.parameters()]
    for step, batch in enumerate(batches, start=1):
        step_started = perf_counter()
        model.train()
        optimizer.zero_grad()
        losses = []
        best_lengths = []
        for offset in range(0, len(batch), args.micro_batch_size):
            micro = batch[offset:offset + args.micro_batch_size]
            outcome = rollout_batch(model, micro, device, sample=True)
            loss = _loss(outcome)
            (loss * (len(micro) / len(batch))).backward()
            losses.append((float(loss.detach().item()), len(micro)))
            best_lengths.extend(outcome.lengths.detach().min(dim=1).values.cpu().tolist())
        ball_grad = _grad_norm(model.ball_head.parameters())
        city_grad = _grad_norm(model.decoder.parameters())
        if not all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None):
            raise RuntimeError(f"nonfinite gradient at step {step}")
        max_ball_grad = max(max_ball_grad, ball_grad)
        optimizer.step()
        processed += len(batch)
        row = {"step": step, "processed": processed, "city_count": batch[0].city_count,
               "instances": len(batch), "mean_loss": sum(v * n for v, n in losses) / len(batch),
               "mean_best_length": sum(best_lengths) / len(best_lengths),
               "ball_grad_norm": ball_grad, "city_grad_norm": city_grad,
               "seconds": perf_counter() - step_started}
        history.append(row)
        print(f"Train {step}/{len(batches)}: {processed}/500 N={batch[0].city_count} "
              f"ball_grad={ball_grad:.3f} seconds={row['seconds']:.1f}", flush=True)
        if (processed >= 250 and not validation_rows) or step == len(batches):
            validation_started = perf_counter()
            score, details = _validate(model, validation, device, args.micro_batch_size)
            checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": 1, "step": step, "train_instances_seen": processed,
                          "validation_mean_best_length": score,
                          "architecture": {"embedding_dim": int(config["embedding_dim"]),
                                           "num_heads": int(config["num_heads"]),
                                           "encoder_layers": int(config["encoder_layers"])},
                          "fixed_checkpoint_sha256": fixed_sha}
            snapshot = output / "checkpoints" / f"step_{step:03d}.pt"
            torch.save(checkpoint, snapshot)
            if score < best_validation:
                best_validation, best_step = score, step
                torch.save(checkpoint, output / "checkpoints" / "best.pt")
            validation_rows.append({"step": step, "processed": processed,
                                    "mean_best_length": score,
                                    "seconds": perf_counter() - validation_started,
                                    "groups": details})
            print(f"Validation step {step}: mean best={score:.3f}, "
                  f"best step={best_step}", flush=True)
    head_delta = sum((p.detach() - old).abs().sum().item()
                     for p, old in zip(model.ball_head.parameters(), initial_head))
    if max_ball_grad <= 0 or head_delta <= 0:
        raise RuntimeError("ball head did not learn during the short RL epoch")
    with (output / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    _save_json(output / "validation_history.json", validation_rows)
    _save_json(output / "train_result.json", {
        "completed_epochs": 1, "train_instances": processed,
        "validation_instances": len(validation), "best_step": best_step,
        "best_validation_mean_best_length": best_validation,
        "max_ball_grad_norm": max_ball_grad,
        "ball_parameter_delta_l1": head_delta,
        "best_checkpoint_sha256": hashlib.sha256((output / "checkpoints" / "best.pt").read_bytes()).hexdigest(),
        "training_seconds": perf_counter() - started,
    })
    print(f"Training complete: {output}", flush=True)


if __name__ == "__main__":
    main()
