"""Matched POMO training with optional tour or region-balanced teacher forcing."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch

from gb_2_maco_POMO.attention import build_city_features
from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import load_checkpoint
from gb_2_maco_POMO.experiments.learnable_ball.learnable_rollout import rollout_batch
from gb_2_maco_POMO.experiments.shared.data import batches_by_city_count
from gb_2_maco_POMO.runner import prepare_tsp

from .config import (BATCH_SIZE, EPOCHS, GROUPS, INITIAL, INITIAL_SHA256,
                     LAMBDA, LEARNING_RATE, MICRO_BATCH_SIZE, ORIGINAL, PACKAGE,
                     RESULTS, TRAIN_SEEDS)
from .data import load_experiment_data, sha256
from .supervision import draw_route, teacher_forcing_loss
from .validation import choose_checkpoint, evaluate

PREFLIGHT_VERSION = 3


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset_signature():
    return {"initial_sha256": sha256(INITIAL),
            "original_manifest_sha256": sha256(ORIGINAL / "manifest.csv"),
            "label_manifest_sha256": sha256(PACKAGE / "manifest.csv")}


def seeded_model(seed, device):
    if sha256(INITIAL) != INITIAL_SHA256:
        raise RuntimeError("initial 1-epoch/11-step checkpoint hash changed")
    model, state = load_checkpoint(INITIAL)
    if state.get("epoch") != 1 or state.get("step") != 11:
        raise RuntimeError("initial checkpoint is not the 1-epoch, 11-update model")
    model.to(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return model


def pomo_loss(outcome):
    if outcome.joint_log_probs is None:
        raise RuntimeError("sampled POMO rollout has no joint log probability")
    rewards = -outcome.lengths
    advantages = rewards - rewards.mean(1, keepdim=True)
    loss = -(advantages.detach() * outcome.joint_log_probs).mean()
    if not torch.isfinite(loss).item():
        raise RuntimeError("nonfinite POMO loss")
    return loss


def grad_norm(gradients):
    return sum(float(g.detach().square().sum().item()) for g in gradients) ** 0.5


def grad_cosine(left, right):
    left_norm, right_norm = grad_norm(left), grad_norm(right)
    if not left_norm or not right_norm:
        return None
    dot = sum(float((a.detach() * b.detach()).sum().item())
              for a, b in zip(left, right))
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _grads(loss, params):
    values = torch.autograd.grad(loss, params, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, values)]


def batch_gradients(model, batch, labels, group, teacher_rng, device):
    """Return separate exact batch gradients before combining at fixed lambda."""
    params = list(model.parameters())
    rl_grad = [torch.zeros_like(p) for p in params]
    sup_grad = [torch.zeros_like(p) for p in params]
    sampled = {}
    regions, tours = Counter(), Counter()
    if group != "A":
        for item in batch:
            case = labels.get(item.path)
            if case is None:
                continue
            route, region, tour = draw_route(case, group, teacher_rng)
            sampled[item.path] = (case, route)
            regions[f"{case.label_id}:{region}"] += 1
            tours[f"{case.label_id}:{tour}"] += 1
    labeled_count = len(sampled)
    rl_loss_sum = sup_loss_sum = raw_sum = reward_sum = 0.0
    route_count = 0
    for offset in range(0, len(batch), MICRO_BATCH_SIZE):
        micro = batch[offset:offset + MICRO_BATCH_SIZE]
        outcome = rollout_batch(model, micro, device, sample=True)
        rl_loss = pomo_loss(outcome)
        weight = len(micro) / len(batch)
        for target, part in zip(rl_grad, _grads(rl_loss * weight, params)):
            target.add_(part)
        rl_loss_sum += float(rl_loss.detach()) * weight
        raw_sum += float(outcome.lengths.detach().sum())
        reward_sum -= float(outcome.lengths.detach().sum())
        route_count += outcome.lengths.numel()
        selected = [sampled[item.path] for item in micro if item.path in sampled]
        if selected:
            sup_loss = teacher_forcing_loss(model, selected, device)
            supervision_weight = len(selected) / labeled_count
            for target, part in zip(sup_grad, _grads(sup_loss * supervision_weight, params)):
                target.add_(part)
            sup_loss_sum += float(sup_loss.detach()) * supervision_weight
    rl_norm = grad_norm(rl_grad)
    sup_norm = grad_norm(sup_grad)
    ball_ids = {id(p) for p in model.ball_head.parameters()}
    decoder_ids = {id(p) for p in model.decoder.parameters()}
    ball_pairs = [(a, b) for p, a, b in zip(params, rl_grad, sup_grad)
                  if id(p) in ball_ids]
    decoder_pairs = [(a, b) for p, a, b in zip(params, rl_grad, sup_grad)
                     if id(p) in decoder_ids]
    for parameter, rl_part, sup_part in zip(params, rl_grad, sup_grad):
        parameter.grad = rl_part + LAMBDA * sup_part
        if not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError("nonfinite combined gradient")
    if rl_norm <= 0 or (group != "A" and labeled_count and sup_norm <= 0):
        raise RuntimeError(f"vanishing gradient: POMO={rl_norm}, supervised={sup_norm}")
    diagnostics = {"pomo_loss": rl_loss_sum, "supervised_loss": sup_loss_sum,
                   "pomo_grad_norm": rl_norm, "supervised_grad_norm": sup_norm,
                   "weighted_grad_ratio": LAMBDA * sup_norm / rl_norm,
                   "grad_cosine": grad_cosine(rl_grad, sup_grad),
                   "ball_grad_cosine": grad_cosine(*zip(*ball_pairs)),
                   "decoder_grad_cosine": grad_cosine(*zip(*decoder_pairs)),
                   "cancellation_ratio": (grad_norm([p.grad for p in params]) /
                                          (rl_norm + LAMBDA * sup_norm)),
                   "combined_grad_norm": grad_norm([p.grad for p in params]),
                   "mean_raw_length": raw_sum / route_count,
                   "mean_pomo_reward": reward_sum / route_count,
                   "supervised_cases": labeled_count,
                   "ball_grad_norm": grad_norm([p.grad for p in model.ball_head.parameters()]),
                   "decoder_grad_norm": grad_norm([p.grad for p in model.decoder.parameters()])}
    return diagnostics, regions, tours


@torch.inference_mode()
def policy_probe(model, item, device):
    problem = prepare_tsp(item.path)
    features = build_city_features(problem.cities, problem.granular_balls,
                                   problem.city_to_ball).to(device)
    _, encoded = model.encode(features)
    n = item.city_count
    mapping = torch.tensor(problem.city_to_ball, dtype=torch.long, device=device)[None, :]
    first = torch.zeros((1, 1), dtype=torch.long, device=device)
    available = torch.ones((1, 1, n), dtype=torch.bool, device=device)
    available[:, :, 0] = False
    ball_logits, city_logits = model.action_logits(encoded, first, first, available, mapping)
    ball = torch.distributions.Categorical(logits=ball_logits)
    target = ball_logits.argmax(-1)
    city_mask = available & (mapping[:, None, :] == target[..., None])
    city = torch.distributions.Categorical(logits=city_logits.masked_fill(~city_mask, -torch.inf))
    return {"ball_entropy": float(ball.entropy().item()),
            "city_entropy_given_top_ball": float(city.entropy().item()),
            "first_step_top_ball": int(target.item())}


def _validate_without_advancing_rng(model, instances, labels, seed, device):
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        return evaluate(model, instances, labels, seed)
    finally:
        torch.set_rng_state(state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def preflight(seed=42, device_name=None):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(1)
    partitions, labels, _ = load_experiment_data()
    model = seeded_model(seed, device)
    by_size = {}
    for size in (10, 50, 60, 66):
        selected = [item for item in partitions["train"] if item.city_count == size
                    and item.path in labels["train"]][:4]
        if selected:
            by_size[size] = selected
    diagnostics = {}
    for size, batch in by_size.items():
        for group in ("B", "C"):
            torch.manual_seed(seed + size)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + size)
            model.zero_grad(set_to_none=True)
            stats, _, _ = batch_gradients(model, batch, labels["train"], group,
                                          random.Random(seed + size), device)
            diagnostics[f"{size}_{group}"] = stats
    # The supervised objective must reach both policy heads on a multi-ball case.
    multi = next((x for x in partitions["train"] if x.path in labels["train"]
                  and len(prepare_tsp(x.path).granular_balls) > 1), None)
    if multi is None:
        raise RuntimeError("no labeled multi-ball case for teacher-forcing gradient check")
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed + 999)
    stats, _, _ = batch_gradients(model, [multi], labels["train"], "C",
                                  random.Random(seed + 999), device)
    diagnostics["multi_ball"] = {"case": multi.path.name, **stats}
    route, _, _ = draw_route(labels["train"][multi.path], "C", random.Random(seed))
    sup = teacher_forcing_loss(model, [(labels["train"][multi.path], route)], device)
    grads = torch.autograd.grad(sup, tuple(model.ball_head.parameters()) +
                               tuple(model.decoder.parameters()), allow_unused=True)
    ball_count = len(tuple(model.ball_head.parameters()))
    ball_norm = grad_norm([g for g in grads[:ball_count] if g is not None])
    city_norm = grad_norm([g for g in grads[ball_count:] if g is not None])
    diagnostics["teacher_head_gradients"] = {"ball": ball_norm, "city": city_norm}
    if ball_norm <= 0 or city_norm <= 0:
        raise RuntimeError("teacher forcing does not update both ball and city heads")
    ratios = [value["weighted_grad_ratio"] for key, value in diagnostics.items()
              if key != "teacher_head_gradients"]
    diagnostics["scale_gate"] = {"lambda": LAMBDA,
                                  "min_weighted_ratio": min(ratios),
                                  "max_weighted_ratio": max(ratios),
                                  "acceptable_range": [0.01, 100]}
    diagnostics["passed"] = all(0.01 <= ratio <= 100 for ratio in ratios)
    diagnostics["gradient_conflict"] = {
        "opposing_probes": [key for key, value in diagnostics.items()
                            if isinstance(value, dict) and value.get("grad_cosine") is not None
                            and value["grad_cosine"] < 0],
        "severe_cancellation_probes": [key for key, value in diagnostics.items()
                                        if isinstance(value, dict)
                                        and value.get("cancellation_ratio") is not None
                                        and value["cancellation_ratio"] < 0.25]}
    diagnostics["device"] = str(device)
    diagnostics["seed"] = seed
    diagnostics["probe_version"] = PREFLIGHT_VERSION
    diagnostics["dataset_signature"] = dataset_signature()
    return diagnostics


def run_group(group, seed, partitions, labels, device, *, resume, stop_after=EPOCHS):
    output = RESULTS / f"train_seed_{seed}" / group
    signature = dataset_signature()
    if signature["initial_sha256"] != INITIAL_SHA256:
        raise RuntimeError("initial checkpoint hash changed")
    model = seeded_model(seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    data_rng = random.Random(seed)
    teacher_rng = random.Random(seed + 100_000)
    records, history, region_history, tour_history = [], [], Counter(), Counter()
    if resume:
        saved = torch.load(output / "checkpoints/last.pt", map_location="cpu", weights_only=True)
        if (saved["group"], saved["seed"], saved["signature"]) != (group, seed, signature):
            raise RuntimeError("resume identity or dataset mismatch")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        data_rng.setstate(saved["data_rng_state"])
        teacher_rng.setstate(saved["teacher_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        records, history = saved["validation_records"], saved["training_history"]
        region_history, tour_history = Counter(saved["region_counts"]), Counter(saved["tour_counts"])
        first_epoch = saved["epoch"] + 1
    else:
        if output.exists():
            raise FileExistsError(f"existing run requires --resume: {output}")
        (output / "checkpoints").mkdir(parents=True)
        save_json(output / "config.json", {
            "group": group, "train_seed": seed, "dataset_signature": signature,
            "train_cases": 580, "validation_cases": 120, "labeled_train": 362,
            "labeled_validation": 76, "batch_size": BATCH_SIZE,
            "micro_batch_size": MICRO_BATCH_SIZE, "epochs_max": EPOCHS,
            "learning_rate": LEARNING_RATE, "optimizer": "Adam",
            "lambda": LAMBDA if group != "A" else 0,
            "pomo": "negative closed-tour length with per-instance POMO baseline",
            "teacher_sampling": ("none" if group == "A" else
                                 "uniform known tour" if group == "B" else
                                 "uniform known region then uniform tour"),
            "selection": "epoch0 quality gate: certified hits >= baseline-1; mean gap <= baseline+0.5 percentage points; then full-region coverage, known-region coverage, distance hits, gap, earliest epoch",
            "training_test_solutions_used": False, "device": str(device)})
        baseline = RESULTS / f"train_seed_{seed}" / "A0"
        if not (baseline / "validation_epoch_000.csv").is_file():
            raise RuntimeError(f"A0 baseline validation missing: {baseline}")
        baseline_metrics = json.loads((baseline / "validation_history.json").read_text(encoding="utf-8"))[0]["metrics"]
        save_csv(output / "validation_epoch_000.csv", list(csv.DictReader(
            (baseline / "validation_epoch_000.csv").open(encoding="utf-8-sig", newline=""))))
        records = [{"epoch": 0, "checkpoint": str(INITIAL), "metrics": baseline_metrics}]
        first_epoch = 1
        _save_checkpoint(output / "checkpoints/last.pt", group, seed, 0, signature,
                         model, optimizer, data_rng, teacher_rng, records, history,
                         region_history, tour_history, device)
    if first_epoch > stop_after or (output / "train_result.json").is_file():
        return
    probe_item = next(item for item in partitions["train"] if item.city_count >= 50)
    for epoch in range(first_epoch, stop_after + 1):
        started = time.perf_counter()
        steps = []
        epoch_regions, epoch_tours = Counter(), Counter()
        batches = batches_by_city_count(partitions["train"], BATCH_SIZE, data_rng)
        for step, batch in enumerate(batches, 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            stats, region_counts, tour_counts = batch_gradients(
                model, batch, labels["train"], group, teacher_rng, device)
            if group != "A" and stats["supervised_cases"] and not (
                0.01 <= stats["weighted_grad_ratio"] <= 100
            ):
                raise RuntimeError("supervised/POMO gradient scale gate failed at "
                                   f"{group} seed={seed} epoch={epoch} step={step}: "
                                   f"ratio={stats['weighted_grad_ratio']}")
            optimizer.step()
            epoch_regions.update(region_counts)
            epoch_tours.update(tour_counts)
            steps.append({"epoch": epoch, "step": step, "city_count": batch[0].city_count,
                          "instances": len(batch), **stats})
            print(f"{group} seed={seed} epoch={epoch} step={step}/{len(batches)} "
                  f"POMO={stats['pomo_loss']:.3f} SL={stats['supervised_loss']:.3f} "
                  f"ratio={stats['weighted_grad_ratio']:.4g}", flush=True)
        region_history.update(epoch_regions)
        tour_history.update(epoch_tours)
        save_csv(output / f"training_steps_epoch_{epoch:03d}.csv", steps)
        model.eval()
        policy_metrics = policy_probe(model, probe_item, device)
        validation_started = time.perf_counter()
        val_rows, metrics = _validate_without_advancing_rng(
            model, partitions["validation"], labels["validation"], seed, device)
        validation_seconds = time.perf_counter() - validation_started
        save_csv(output / f"validation_epoch_{epoch:03d}.csv", val_rows)
        snapshot = output / "checkpoints" / f"epoch_{epoch:03d}.pt"
        records.append({"epoch": epoch, "checkpoint": str(snapshot), "metrics": metrics,
                        "validation_seconds": validation_seconds})
        chosen = choose_checkpoint(records)
        previous = choose_checkpoint(records[:-1])
        patience = 0 if chosen["epoch"] != previous["epoch"] else (
            history[-1]["early_wait"] + 1 if history else 1)
        row = {"epoch": epoch, "steps": len(steps), "train_seconds": validation_started - started,
               "validation_seconds": validation_seconds,
               "mean_pomo_loss": sum(x["pomo_loss"] for x in steps) / len(steps),
               "mean_supervised_loss": sum(x["supervised_loss"] for x in steps) / len(steps),
               "mean_pomo_grad_norm": sum(x["pomo_grad_norm"] for x in steps) / len(steps),
               "mean_supervised_grad_norm": sum(x["supervised_grad_norm"] for x in steps) / len(steps),
               "mean_raw_length": sum(x["mean_raw_length"] for x in steps) / len(steps),
               "teacher_regions_seen": len(epoch_regions), "teacher_tours_seen": len(epoch_tours),
               "ball_entropy": policy_metrics["ball_entropy"],
               "city_entropy": policy_metrics["city_entropy_given_top_ball"],
               "selected_epoch": chosen["epoch"], "early_wait": patience}
        history.append(row)
        save_csv(output / "training_history.csv", history)
        save_json(output / "validation_history.json", records)
        save_json(output / "selection.json", {"selected_epoch": chosen["epoch"],
                                                  "checkpoint": chosen["checkpoint"],
                                                  "metrics": chosen["metrics"],
                                                  "test_set_used_for_selection": False})
        save_json(output / "teacher_sampling_counts.json", {
            "regions": dict(region_history), "tours": dict(tour_history),
            "note": "B/C distribution is over provided labels; partial labels are not full global region truth."})
        _save_checkpoint(snapshot, group, seed, epoch, signature, model, optimizer,
                         data_rng, teacher_rng, records, history, region_history,
                         tour_history, device)
        _save_checkpoint(output / "checkpoints/last.pt", group, seed, epoch, signature,
                         model, optimizer, data_rng, teacher_rng, records, history,
                         region_history, tour_history, device)
        print(f"{group} seed={seed} epoch={epoch} validation={metrics['certified_hits']}/76 "
              f"selected={chosen['epoch']} seconds={time.perf_counter()-started:.1f}", flush=True)


def finalize_group(group, seed, completed_epoch, stopped_early):
    output = RESULTS / f"train_seed_{seed}" / group
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    history = list(csv.DictReader((output / "training_history.csv").open(
        encoding="utf-8-sig", newline="")))
    if len(history) != completed_epoch or int(history[-1]["epoch"]) != completed_epoch:
        raise RuntimeError(f"training history incomplete: {output}")
    save_json(output / "train_result.json", {
        "group": group, "seed": seed, "completed_epochs": completed_epoch,
        "stopped_early": stopped_early, "selected_epoch": selection["selected_epoch"],
        "selected_checkpoint": selection["checkpoint"],
        "early_stop_rule": "cohort stops after all A/B/C have two consecutive unimproved epochs; equal optimizer updates"})


def _save_checkpoint(path, group, seed, epoch, signature, model, optimizer,
                     data_rng, teacher_rng, records, history, regions, tours, device):
    torch.save({"group": group, "seed": seed, "epoch": epoch,
                "signature": signature, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "architecture": {"embedding_dim": model.decoder.embedding_dim,
                                 "num_heads": model.decoder.cross_attention.num_heads,
                                 "encoder_layers": len(model.encoder.layers)},
                "data_rng_state": data_rng.getstate(),
                "teacher_rng_state": teacher_rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "validation_records": records, "training_history": history,
                "region_counts": dict(regions), "tour_counts": dict(tours)}, path)


def ensure_a0(seed, partitions, labels, device):
    output = RESULTS / f"train_seed_{seed}" / "A0"
    if (output / "validation_history.json").is_file():
        return
    if output.exists():
        raise RuntimeError(f"incomplete A0 validation: {output}")
    output.mkdir(parents=True)
    model = seeded_model(seed, device)
    probe_item = next(item for item in partitions["train"] if item.city_count >= 50)
    save_json(output / "policy_distribution_epoch_000.json",
              policy_probe(model, probe_item, device))
    rows, metrics = _validate_without_advancing_rng(
        model, partitions["validation"], labels["validation"], seed, device)
    save_csv(output / "validation_epoch_000.csv", rows)
    save_json(output / "validation_history.json", [{"epoch": 0,
              "checkpoint": str(INITIAL), "metrics": metrics}])
    save_json(output / "config.json", {"seed": seed, "checkpoint": str(INITIAL),
                                           "checkpoint_sha256": INITIAL_SHA256,
                                           "dataset_signature": dataset_signature()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=TRAIN_SEEDS, required=True)
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        stats = preflight(args.seed, args.device)
        save_json(RESULTS / f"preflight_seed_{args.seed}_{args.device}.json", stats)
        if not stats["passed"]:
            raise RuntimeError(f"gradient scale gate failed: {stats['scale_gate']}")
        print("preflight passed", flush=True)
        return
    if args.group is None:
        parser.error("--group required for training")
    marker = RESULTS / f"preflight_seed_{args.seed}_{args.device}.json"
    checked = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
    if (not checked.get("passed") or checked.get("probe_version") != PREFLIGHT_VERSION
            or checked.get("dataset_signature") != dataset_signature()
            or checked.get("scale_gate", {}).get("lambda") != LAMBDA):
        raise RuntimeError(f"run --preflight for this seed before training: {marker}")
    partitions, labels, _ = load_experiment_data(
        audit_path=RESULTS / "region_rule_audit.csv")
    ensure_a0(args.seed, partitions, labels, device)
    run_group(args.group, args.seed, partitions, labels, device, resume=args.resume)
    finalize_group(args.group, args.seed, EPOCHS, False)


if __name__ == "__main__":
    main()
