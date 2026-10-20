"""Matched A/B continuation from A0 on the independent Multiopt 500/100 split."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from time import perf_counter

import torch

from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import generate_learned_routes, load_checkpoint
from gb_2_maco_POMO.experiments.learnable_ball_bias.validation import heuristic_reference
from gb_2_maco_POMO.experiments.learnable_ball.train_learnable import _loss as original_loss
from gb_2_maco_POMO.experiments.shared.data import batches_by_city_count, load_manifest
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, equal_plateau, tour_length
from gb_2_maco_POMO.experiments.region_supervised_pomo.config import INITIAL, INITIAL_SHA256
from gb_2_maco_POMO.runner import prepare_tsp

from .checks import check
from .reward import losses
from .rollout import rollout_batch
from .selection import metrics, select_checkpoints
from .settings import PROJECT, Settings, load as load_settings


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _norm(parameters) -> float:
    return sum(float(p.grad.detach().square().sum()) for p in parameters
               if p.grad is not None) ** 0.5


def _check_step_gradients(model, group: str, step: int) -> tuple[float, float]:
    """A single-ball batch may legitimately have zero Ball Head gradient."""
    if any(not torch.isfinite(p.grad).all().item() for p in model.parameters()
           if p.grad is not None):
        raise RuntimeError(f"{group} step {step}: nonfinite gradient")
    return _norm(model.ball_head.parameters()), _norm(model.decoder.parameters())


def _validation_optima(data_root: Path | None = None) -> dict[str, int]:
    """Proofs are validation diagnostics only, never rewards or selectors."""
    data_root = data_root or load_settings().data_root
    certified = {}
    with (data_root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "validation" or row["multiopt_certified"] != "1":
                continue
            proof = json.loads((data_root / row["proof_file"]).read_text(encoding="utf-8"))
            certificate = proof["certificate"]
            length = certificate.get("exact_optimal_length",
                                     certificate.get("proven_optimal_length"))
            manifest_length = row["exact_optimal_length"] or row["proven_optimal_length"]
            if (length is None or not manifest_length or int(length) != int(manifest_length) or
                    proof["tsp_path"] != row["relative_path"] or
                    proof["split"] != "validation"):
                raise RuntimeError(f"validation certificate disagrees with manifest: {row['id']}")
            name = Path(row["relative_path"]).name
            if name in certified:
                raise RuntimeError(f"duplicate certified validation file: {name}")
            certified[name] = int(length)
    if len(certified) != 56:
        raise RuntimeError(f"expected 56 certified validation cases, got {len(certified)}")
    return certified


@torch.no_grad()
def validate(policy, instances, device, label: str, optima: dict[str, int], config: Settings):
    """Greedy diagnostics plus equal-budget sampled/2-opt/Opt-B selection data."""
    policy.eval()
    rows = []
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    for batch in batches_by_city_count(instances, config.micro_batch_size):
        outcome = rollout_batch(policy, batch, device, sample=False)
        for i, item in enumerate(batch):
            routes = outcome.routes[i].cpu().tolist()
            lengths = outcome.lengths[i].cpu().tolist()
            best = min(lengths)
            best_indices = [j for j, length in enumerate(lengths) if length == best]
            best_index = best_indices[0]
            matrix = prepare_tsp(item.path).distance_matrix
            unique = {canonical(route): route for route in routes}
            refined = [two_opt(route, matrix) for route in unique.values()]
            post_best = min(result.length for result in refined)
            optimum = optima.get(item.path.name)
            if optimum is not None and (best < optimum or post_best < optimum):
                raise RuntimeError(f"{item.path.name}: found distance below certified optimum")
            sampled, _ = generate_learned_routes(policy, prepare_tsp(item.path),
                list(range(item.city_count)), config.validation_samples_per_start,
                config.seed + len(rows) + 1)
            sampled_lengths = [tour_length(route, matrix) for route in sampled]
            # Keep the final-test order: shortest raw route, then original index.
            unique_sampled = {}
            for j in sorted(range(len(sampled)), key=lambda j: (sampled_lengths[j], j)):
                route = sampled[j]
                unique_sampled.setdefault(canonical(route), route)
            sampled_refined = [two_opt(route, matrix) for route in unique_sampled.values()]
            sampled_best = min([*sampled_lengths, *(r.length for r in sampled_refined)])
            plateau = {canonical(route): route for route, length in zip(sampled, sampled_lengths)
                       if length == sampled_best}
            plateau.update({canonical(r.route): r.route for r in sampled_refined
                            if r.length == sampled_best})
            optb = equal_plateau(list(plateau.values()), matrix, config.validation_optb_checks)
            heuristic = heuristic_reference(matrix)
            if optimum is not None and (sampled_best < optimum or optb.best_length < optimum):
                raise RuntimeError(f"{item.path.name}: sampled route below certified optimum")
            rows.append({"instance": item.path.name, "distribution": item.distribution,
                "city_count": item.city_count, "group": label,
                "raw_mean_length": sum(lengths) / len(lengths),
                "raw_best_length": best, "two_opt_best_length": post_best,
                "inter_length_at_raw_best": outcome.inter_lengths[i, best_index].item(),
                "intra_length_at_raw_best": outcome.intra_lengths[i, best_index].item(),
                "crossings_at_raw_best": int(outcome.crossings[i, best_index].item()),
                "distinct_raw_best_routes": len({canonical(routes[j]) for j in best_indices}),
                "distinct_two_opt_best_routes": len({canonical(r.route) for r in refined
                                                     if r.length == post_best}),
                "certified_optimal_length": optimum if optimum is not None else "",
                "raw_optimal_gap_pct": 100 * (best / optimum - 1) if optimum else "",
                "two_opt_optimal_gap_pct": 100 * (post_best / optimum - 1) if optimum else "",
                "raw_optimal_hit": int(best == optimum) if optimum else "",
                "two_opt_optimal_hit": int(post_best == optimum) if optimum else "",
                "sampled_raw_best_length": min(sampled_lengths),
                "sampled_two_opt_best_length": sampled_best,
                "sampled_optb_best_length": optb.best_length,
                "sampled_optb_routes": len(optb.routes),
                "sampled_certified_optb_routes": len(optb.routes) if optimum is not None
                    and optb.best_length == optimum else 0,
                "sampled_two_opt_optimal_hit": int(sampled_best == optimum) if optimum else "",
                "sampled_two_opt_optimal_gap_pct": 100 * (sampled_best / optimum - 1) if optimum else "",
                "sampled_heuristic_relative_raw_pct": 100 * (min(sampled_lengths) / heuristic - 1),
                "sampled_heuristic_relative_2opt_pct": 100 * (sampled_best / heuristic - 1)})
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    return rows


def _optimizer(model, state, learning_rate: float):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    device = next(model.parameters()).device
    for values in optimizer.state.values():
        for name, value in values.items():
            if torch.is_tensor(value):
                values[name] = value.to(device)
    return optimizer


def _snapshot(model, optimizer, source_state, epoch, score, group):
    return {"model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": dict(source_state["architecture"]),
            "epoch": epoch, "step": epoch, "train_instances_seen": epoch * 500,
            "validation_mean_best_length": score,
            "initialized_from": str(INITIAL),
            "initial_checkpoint_sha256": INITIAL_SHA256,
            "reward_group": group, "encoder_frozen": True}


def run(output: Path, config: Settings | None = None) -> dict:
    config = config or load_settings()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"new output directory required: {output}")
    if hashlib.sha256(config.initial_checkpoint.read_bytes()).hexdigest() != INITIAL_SHA256:
        raise RuntimeError("A0 checkpoint hash differs from the fixed baseline")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(config.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    train, validation = load_manifest({"root": str(config.data_root), "train_dir": "train",
                                       "validation_dir": "validation"})
    if (len(train), len(validation)) != (500, 100) or {x.path for x in train} & {x.path for x in validation}:
        raise RuntimeError("Multiopt 500/100 train-validation split changed")
    optima = _validation_optima(config.data_root)
    output.mkdir(parents=True)
    shutil.copyfile(config.config_path, output / "config.toml")
    save_json(output / "config.json", {
        "data": str(config.data_root), "train": 500, "validation": 100,
        "train_manifest_sha256": hashlib.sha256((config.data_root / "manifest.csv").read_bytes()).hexdigest(),
        "initial": str(INITIAL), "initial_sha256": INITIAL_SHA256,
        "seed": config.seed, "groups": {"A": 0, "B": config.auxiliary_lambda},
        "epochs": config.epochs, "batch_size": config.batch_size,
        "micro_batch_size": config.micro_batch_size,
        "optimizer": "A0 Adam state resumed", "learning_rate": config.learning_rate,
        "encoder": "frozen for both groups", "device": config.device,
        "validation_selection": "certified gap and sampled Opt-B route diversity; see selection.json",
        "validation_certified_cases": len(optima),
        "validation_optimal_gap": "certified 56 cases only; other 44 unknown",
        "certificates_used_for": "validation checkpoint selection only; never training rewards"})
    probe_model, _ = load_checkpoint(config.initial_checkpoint)
    probe_model.to(device)
    preflight = check(probe_model, train, device)
    save_json(output / "preflight.json", preflight)
    print("Dual-reward preflight passed", flush=True)
    results = {}
    for group in ("A", "B"):
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        model, state = load_checkpoint(config.initial_checkpoint)
        model.to(device)
        model.encoder.requires_grad_(False)
        optimizer = _optimizer(model, state, config.learning_rate)
        folder = output / group
        (folder / "checkpoints").mkdir(parents=True)
        initial_ball = [p.detach().clone() for p in model.ball_head.parameters()]
        initial_city = [p.detach().clone() for p in model.decoder.parameters()]
        initial_encoder = [p.detach().clone() for p in model.encoder.parameters()]
        started = perf_counter()
        train_rows, records = [], []

        def checkpoint(epoch: int):
            rows = validate(model, validation, device, f"{group}{epoch}", optima, config)
            score = sum(r["raw_best_length"] for r in rows) / len(rows)
            path = folder / "checkpoints" / f"epoch_{epoch}.pt"
            torch.save(_snapshot(model, optimizer, state, epoch, score, group), path)
            save_csv(folder / f"validation_epoch_{epoch}.csv", rows)
            record = {"epoch": epoch, "checkpoint": str(path),
                      "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "metrics": metrics(rows), "greedy_mean_best_length": score}
            records.append(record)
            save_json(folder / "validation_history.json", records)
            print(f"{group} epoch {epoch}: validation complete", flush=True)

        checkpoint(0)
        rng = random.Random(config.seed)
        global_step = 0
        for epoch in range(1, config.epochs + 1):
            model.train()
            seen = 0
            batches = batches_by_city_count(train, config.batch_size, rng)
            for step, batch in enumerate(batches, 1):
                global_step += 1
                optimizer.zero_grad(set_to_none=True)
                accumulated = {"ball_loss": 0.0, "city_loss": 0.0,
                               "total_adv_std": 0.0, "inter_adv_std": 0.0,
                               "intra_adv_std": 0.0, "inter_length": 0.0,
                               "intra_length": 0.0, "crossings": 0.0}
                for offset in range(0, len(batch), config.micro_batch_size):
                    micro = batch[offset:offset + config.micro_batch_size]
                    outcome = rollout_batch(model, micro, device, sample=True)
                    if group == "A":
                        total_loss = original_loss(outcome)
                        _, _, stats = losses(outcome, 0.0)
                    else:
                        ball_loss, city_loss, stats = losses(outcome, config.auxiliary_lambda)
                        total_loss = ball_loss + city_loss
                    weight = len(micro) / len(batch)
                    (total_loss * weight).backward()
                    for key in ("ball_loss", "city_loss", "total_adv_std",
                                "inter_adv_std", "intra_adv_std"):
                        accumulated[key] += stats[key] * weight
                    for key, value in (("inter_length", outcome.inter_lengths.mean()),
                                       ("intra_length", outcome.intra_lengths.mean()),
                                       ("crossings", outcome.crossings.float().mean())):
                        accumulated[key] += value.item() * weight
                ball_grad, city_grad = _check_step_gradients(model, group, global_step)
                optimizer.step()
                seen += len(batch)
                train_rows.append({"epoch": epoch, "step": global_step,
                    "instances_seen": (epoch - 1) * 500 + seen,
                    "city_count": batch[0].city_count, "group": group,
                    "ball_grad_norm": ball_grad, "decoder_grad_norm": city_grad,
                    **accumulated})
                print(f"{group} epoch {epoch} step {step}/{len(batches)}: {seen}/500", flush=True)
            if seen != 500 or any(not torch.equal(a, b) for a, b in
                                  zip(initial_encoder, model.encoder.parameters())):
                raise RuntimeError(f"{group} epoch {epoch}: incomplete epoch or encoder changed")
            if epoch % config.validation_interval == 0:
                checkpoint(epoch)

        ball_delta = sum((p.detach() - old).abs().sum().item()
                         for p, old in zip(model.ball_head.parameters(), initial_ball))
        city_delta = sum((p.detach() - old).abs().sum().item()
                         for p, old in zip(model.decoder.parameters(), initial_city))
        if ball_delta <= 0 or city_delta <= 0:
            raise RuntimeError(f"{group}: trainable head did not change")
        best_length, best_multi = select_checkpoints(records,
            quality_gap_tolerance_pct=config.multi_quality_gap_tolerance_pct,
            allowed_hit_loss=config.multi_allowed_certified_hit_loss)
        for alias, selected in (("best_length", best_length), ("best_multi", best_multi)):
            shutil.copyfile(selected["checkpoint"], folder / "checkpoints" / f"{alias}.pt")
        selection = {"best_length": best_length, "best_multi": best_multi,
                     "quality_gap_tolerance_pct": config.multi_quality_gap_tolerance_pct,
                     "allowed_certified_hit_loss": config.multi_allowed_certified_hit_loss}
        save_json(folder / "selection.json", selection)
        save_csv(folder / "training_steps.csv", train_rows)
        result = {"group": group, "epochs": config.epochs,
                  "validated_epochs": [r["epoch"] for r in records],
                  "best_length_epoch": best_length["epoch"],
                  "best_multi_epoch": best_multi["epoch"],
                  "checkpoint_sha256": {str(r["epoch"]): r["checkpoint_sha256"] for r in records},
                  "ball_parameter_delta_l1": ball_delta,
                  "decoder_parameter_delta_l1": city_delta,
                  "training_seconds": perf_counter() - started}
        save_json(folder / "train_result.json", result)
        results[group] = result
    save_json(output / "train_summary.json", results)
    return results
