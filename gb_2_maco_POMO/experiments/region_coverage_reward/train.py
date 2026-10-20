"""Train five controlled reward groups from the same two-level checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import torch

from gb_2_maco_POMO.experiments.learnable_ball_bias.model import BallBiasPolicy
from gb_2_maco_POMO.experiments.learnable_ball_bias.reward import policy_loss
from gb_2_maco_POMO.experiments.learnable_ball_bias.rollout import rollout_batch
from gb_2_maco_POMO.experiments.learnable_ball_bias.rollout import generate_routes
from gb_2_maco_POMO.runner import prepare_tsp
from gb_2_maco_POMO.experiments.shared.data import batches_by_city_count, load_manifest

from .references import build_references, reference_digest
from .reward import compute_advantages, coverage_loss
from .validation import choose_checkpoint, evaluate, validation_references


PROJECT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT / "benachmark_Train_MATSP_Multiopt_500"
INITIAL = PROJECT / "gb_2_maco_POMO/results/learnable_ball_20261015/checkpoints/best.pt"
INITIAL_SHA = "efffddda5fe28f678dd835b7d238ffe4e89d740f40045dd761db1e4a46bfd4bc"
GROUPS = ("G0", "Q", "G1", "G2", "G3")


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def initialized_model(group, seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    source = torch.load(INITIAL, map_location="cpu", weights_only=True)
    arch = source["architecture"]
    bias = group in ("G2", "G3")
    model = BallBiasPolicy(int(arch["embedding_dim"]), int(arch["num_heads"]),
                           int(arch["encoder_layers"]), use_bias=bias, bias_bound=2.0)
    incompatible = model.load_state_dict(source["model_state_dict"], strict=False)
    expected = ({f"bias_head.{layer}.{suffix}" for layer in (0, 2)
                 for suffix in ("weight", "bias")} if bias else set())
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint/architecture mismatch: {incompatible}")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return model.to(device), arch


def checkpoint(path, model, optimizer, group, seed, epoch, update, arch, batch_rng,
               references, records, history, early_wait):
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "architecture": arch,
                "group": "B" if model.use_bias else "A", "experiment_group": group,
                "use_bias": model.use_bias, "bias_bound": model.bias_bound,
                "epoch": epoch, "step": update, "seed": seed,
                "batch_rng_state": batch_rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (torch.cuda.get_rng_state_all()
                                   if next(model.parameters()).device.type == "cuda" else None),
                "references": references, "validation_records": records,
                "training_history": history, "early_wait": early_wait}, path)


def run(group, seed, output, *, stop_after=5, resume=False, baseline_dir=None):
    if group not in GROUPS or seed not in (42, 43, 44) or not 1 <= stop_after <= 5:
        raise ValueError("invalid group, seed or epoch limit")
    if hashlib.sha256(INITIAL.read_bytes()).hexdigest() != INITIAL_SHA:
        raise RuntimeError("initial short-training checkpoint changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    train, validation = load_manifest({"root": str(DATA_ROOT), "train_dir": "train",
                                       "validation_dir": "validation"})
    if len(train) != 500 or len(validation) != 100:
        raise RuntimeError("dataset size mismatch")
    expected_references = build_references(DATA_ROOT, train, validation)
    validation_refs = validation_references(DATA_ROOT, validation)
    ref_sha = reference_digest(expected_references)
    model, arch = initialized_model(group, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch_rng = random.Random(seed)
    if resume:
        state = torch.load(output / "checkpoints/last.pt", map_location="cpu", weights_only=True)
        if state["experiment_group"] != group or state["seed"] != seed:
            raise RuntimeError("resume group or seed mismatch")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        batch_rng.setstate(state["batch_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        references = state["references"]
        if set(references) != set(expected_references) or any(
            references[k]["certified"] != expected_references[k]["certified"] for k in references
        ):
            raise RuntimeError("resume training reference mismatch")
        records, history = state["validation_records"], state["training_history"]
        update, first_epoch, early_wait = state["step"], state["epoch"] + 1, state["early_wait"]
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        (output / "checkpoints").mkdir()
        references = expected_references
        records, history, update, first_epoch, early_wait = [], [], 0, 1, 0
        config = {"group": group, "seed": seed, "data_root": str(DATA_ROOT),
                  "initial_checkpoint": str(INITIAL), "initial_sha256": INITIAL_SHA,
                  "train_instances": 500, "validation_instances": 100,
                  "certified_training_instances": sum(r["certified"] for r in references.values()),
                  "training_references_sha256": ref_sha,
                  "max_epochs": 5, "early_stop_no_improve_epochs": 2,
                  "batch_size": 8, "learning_rate": 1e-4, "optimizer": "Adam",
                  "device": str(device), "validation_samples_per_start": 8,
                  "validation_optb_checks_per_case": 50000,
                  "quality_gate_gap_tolerance_pct": 0.5, "quality_gate_allowed_certified_hit_loss": 1,
                  "reward_two_opt_routes_per_instance": 16,
                  "reward_optb_entrances_per_instance": 8,
                  "reward_optb_checks_per_instance": 8000,
                  "coverage_region_weight": 0.7, "coverage_solution_weight": 0.3,
                  "coverage_advantage_weight": 0.5,
                  "uncertified_high_quality_tolerance_pct": 0.5,
                  "source_config": "learnable_ball_bias/config.toml: batch=8, lr=1e-4, cuda"}
        if baseline_dir is not None:
            config["epoch0_reused_from"] = str(baseline_dir.resolve())
        save_json(output / "effective_config.json", config)
        model.eval()
        before = time.perf_counter()
        if baseline_dir is None:
            val_rows, metrics = evaluate(model, validation, validation_refs, device, seed=seed)
            write_csv(output / "validation_epoch_000.csv", val_rows)
        else:
            baseline_dir = baseline_dir.resolve()
            source_config = json.loads((baseline_dir / "effective_config.json").read_text(encoding="utf-8"))
            if (source_config["group"] != "G0" or source_config["seed"] != seed
                    or source_config["initial_sha256"] != INITIAL_SHA
                    or source_config["validation_samples_per_start"] != 8
                    or source_config["validation_optb_checks_per_case"] != 50000):
                raise RuntimeError("epoch-0 baseline identity or budget mismatch")
            # Bias last layer is initialized to zero; verify actual sampled
            # routes coincide before reusing the identical initial evaluation.
            reference_model, _ = initialized_model("G0", seed, device)
            for item in (validation[0], validation[len(validation) // 2], validation[-1]):
                problem = prepare_tsp(item.path)
                a = generate_routes(reference_model, problem, samples=8, seed=seed + 1, device=device)
                b = generate_routes(model, problem, samples=8, seed=seed + 1, device=device)
                if a != b:
                    raise RuntimeError("initial route mismatch; cannot reuse baseline")
            source_rows = baseline_dir / "validation_epoch_000.csv"
            with source_rows.open(encoding="utf-8-sig", newline="") as stream:
                if len(list(csv.DictReader(stream))) != 100:
                    raise RuntimeError("incomplete epoch-0 baseline")
            shutil.copyfile(source_rows, output / "validation_epoch_000.csv")
            source_records = json.loads((baseline_dir / "validation_history.json").read_text(encoding="utf-8"))
            metrics = source_records[0]["metrics"]
            source_state = torch.load(baseline_dir / "checkpoints/epoch_000.pt",
                                      map_location="cpu", weights_only=True)
            torch.set_rng_state(source_state["torch_rng_state"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(source_state["cuda_rng_state"])
        records.append({"epoch": 0, "step": 0, "validation_seconds": time.perf_counter() - before,
                        "metrics": metrics,
                        "checkpoint": str(output / "checkpoints/epoch_000.pt")})
        checkpoint(output / "checkpoints/epoch_000.pt", model, optimizer, group, seed,
                   0, 0, arch, batch_rng, references, records, history, 0)
        save_json(output / "validation_history.json", records)
        print(f"{group} seed={seed} epoch=0 validation {metrics}; seconds={time.perf_counter()-before:.1f}", flush=True)
    if first_epoch > stop_after or early_wait >= 2:
        return
    caches = {"two_opt": {}, "opt_b": {}}
    baseline = records[0]["metrics"]
    for epoch in range(first_epoch, stop_after + 1):
        started = time.perf_counter()
        pending = {}
        totals = {"selected": 0, "eligible": 0, "entrances": 0, "regions": 0,
                  "nonzero_coverage_advantages": 0, "optb_checks": 0,
                  "two_opt_checks": 0, "actual_optb_checks": 0,
                  "actual_two_opt_checks": 0, "optb_cache_hits": 0,
                  "two_opt_cache_hits": 0, "lower_reference_candidates": 0}
        reward_seconds = 0.0
        raw_length_sum, raw_length_count, loss_sum, batch_count = 0.0, 0, 0.0, 0
        for batch in batches_by_city_count(train, 8, batch_rng):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outcome = rollout_batch(model, batch, device, sample=True)
            raw_length_sum += float(outcome.lengths.detach().sum().item())
            raw_length_count += outcome.lengths.numel()
            if group in ("G0", "G2"):
                loss = policy_loss(outcome)
            else:
                before = time.perf_counter()
                advantage, diag, lower = compute_advantages(
                    outcome, batch, references, group, caches)
                reward_seconds += time.perf_counter() - before
                for key, value in vars(diag).items():
                    totals[key] += value
                for key, value in lower.items():
                    pending[key] = min(pending.get(key, references[key]["length"]), value)
                loss = coverage_loss(outcome, advantage)
            loss_sum += float(loss.detach().item())
            batch_count += 1
            loss.backward()
            if any(not torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None):
                raise RuntimeError(f"nonfinite gradient: {group} seed={seed} epoch={epoch}")
            optimizer.step()
            update += 1
        for key, length in pending.items():
            references[key]["length"] = length
            references[key]["source"] = f"training epoch {epoch} 2-opt; available epoch {epoch+1}"
        row = {"epoch": epoch, "step": update, "train_seconds": time.perf_counter() - started,
               "reward_seconds": reward_seconds,
               "mean_raw_route_length": raw_length_sum / raw_length_count,
               "mean_batch_loss": loss_sum / batch_count, **totals,
               "next_epoch_reference_updates": len(pending)}
        history.append(row)
        write_csv(output / "training_history.csv", history)
        model.eval()
        before = time.perf_counter()
        val_rows, metrics = evaluate(model, validation, validation_refs, device, seed=seed)
        validation_seconds = time.perf_counter() - before
        write_csv(output / f"validation_epoch_{epoch:03d}.csv", val_rows)
        snapshot = output / "checkpoints" / f"epoch_{epoch:03d}.pt"
        records.append({"epoch": epoch, "step": update,
                        "validation_seconds": validation_seconds, "metrics": metrics,
                        "checkpoint": str(snapshot)})
        chosen = choose_checkpoint(records, baseline=baseline)
        previous = choose_checkpoint(records[:-1], baseline=baseline)
        early_wait = 0 if chosen["epoch"] != previous["epoch"] else early_wait + 1
        checkpoint(snapshot, model, optimizer, group, seed, epoch, update, arch,
                   batch_rng, references, records, history, early_wait)
        checkpoint(output / "checkpoints/last.pt", model, optimizer, group, seed, epoch,
                   update, arch, batch_rng, references, records, history, early_wait)
        save_json(output / "validation_history.json", records)
        save_json(output / "selection.json", {
            "selected_epoch": chosen["epoch"], "checkpoint": chosen["checkpoint"],
            "epoch0_metrics": baseline, "selected_metrics": chosen["metrics"],
            "quality_gate_gap_tolerance_pct": 0.5,
            "quality_gate_allowed_certified_hit_loss": 1,
            "test_set_used_for_selection": False})
        save_json(output / "train_result.json", {
            "group": group, "seed": seed, "completed_epochs": epoch, "updates": update,
            "early_wait": early_wait, "stopped_early": early_wait >= 2,
            "selected_epoch": chosen["epoch"], "validation_seconds_last": validation_seconds,
            "reference_count": len(references)})
        print(f"{group} seed={seed} epoch={epoch} updates={update} reward={totals} "
              f"validation={metrics} selected={chosen['epoch']} "
              f"train_s={row['train_seconds']:.1f} val_s={validation_seconds:.1f}", flush=True)
        if early_wait >= 2:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args()
    run(args.group, args.seed, args.output_dir.resolve(),
        stop_after=args.stop_after, resume=args.resume, baseline_dir=args.baseline_dir)


if __name__ == "__main__":
    main()
