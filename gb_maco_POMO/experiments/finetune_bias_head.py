"""Fine-tune only the learnable Bias Head from the fixed Soft-POMO best checkpoint.

Check setup: python -m gb_maco_POMO.experiments.finetune_bias_head --check
Train:       python -m gb_maco_POMO.experiments.finetune_bias_head
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch

from ..attention import POMOPolicy, build_city_features
from ..pomo import search_routes
from ..runner import prepare_tsp
from .compare import BENCHMARKS, load_policy
from .compare_learnable_bias import BENCHMARK_DIR
from .config import load_config
from .data import load_manifest
from .train_loop import (
    _save_checkpoint, _train_epoch, _write_csv, _write_effective_config,
    evaluate_validation, select_device, GROUP_FIELDS,
)


DEFAULT_CONFIG = Path(__file__).with_name("finetune_bias_head_config.toml")
EXPECTED_SOURCE_SHA256 = "dbdfc5cb2faef0d7cbecb7c5c7cc77db16049e5d10e5d584a26e288f1bd88410"
HISTORY_FIELDS = (
    "epoch", "train_mean_length", "train_mean_best_length", "validation_mean_best_length",
    "mean_loss", "bias_head_grad_norm_mean", "same_bias_mean", "same_bias_min",
    "same_bias_max", "same_bias_std", "adjacent_bias_mean", "adjacent_bias_min",
    "adjacent_bias_max", "adjacent_bias_std", "fallback_count", "fallback_rate",
    "average_candidate_size", "attention_unchanged", "seconds",
)


def _source_path(config: dict[str, dict[str, object]], config_path: Path) -> Path:
    value = Path(str(config["finetune"]["source_checkpoint"]))
    return (config_path.parent / value).resolve() if not value.is_absolute() else value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_frozen_policy(source: Path) -> tuple[POMOPolicy, POMOPolicy, dict[str, torch.Tensor], dict]:
    """Strictly load fixed weights, then permit only the four new Head tensors to be missing."""
    if _sha256(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("source checkpoint SHA256 differs from the original fixed Soft-POMO best.pt")
    fixed, state = load_policy(source, torch.device("cpu"))  # strict=True inside load_policy
    if (fixed.candidate_mode, fixed.bias_mode, fixed.same_ball_bias,
            fixed.adjacent_ball_bias) != ("soft", "fixed", 2.0, 1.0):
        raise ValueError("source checkpoint must use Soft candidates and fixed +2/+1 Bias")
    model = state["config"]["model"]
    learned = POMOPolicy(
        embedding_dim=int(model["embedding_dim"]), head_num=int(model["num_heads"]),
        encoder_layer_num=int(model["encoder_layers"]), candidate_mode="soft",
        same_ball_bias=2.0, adjacent_ball_bias=1.0, bias_mode="learnable",
    )
    transfer = learned.load_state_dict(fixed.state_dict(), strict=False)
    expected_missing = {f"decoder.bias_head.{suffix}" for suffix in
                        ("0.weight", "0.bias", "2.weight", "2.bias")}
    if set(transfer.missing_keys) != expected_missing or transfer.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint transfer result: {transfer}")
    original = {key: value.detach().cpu().clone() for key, value in fixed.state_dict().items()}
    _assert_frozen_unchanged(learned, original)
    for name, parameter in learned.named_parameters():
        parameter.requires_grad_(name.startswith("decoder.bias_head."))
    if {name for name, p in learned.named_parameters() if p.requires_grad} != expected_missing:
        raise RuntimeError("trainable parameter set is not exactly the Bias Head")
    return fixed, learned, original, state


def _assert_frozen_unchanged(policy: POMOPolicy, original: dict[str, torch.Tensor]) -> None:
    for name, reference in original.items():
        if not torch.equal(policy.state_dict()[name].detach().cpu(), reference):
            raise RuntimeError(f"frozen source parameter changed: {name}")


@torch.no_grad()
def _verify_initial_equivalence(fixed: POMOPolicy, learned: POMOPolicy) -> None:
    """Compare pre-mask logits, masked probabilities, and paired mixed rollout routes."""
    fixed.eval()
    learned.eval()
    # Smallest existing multi-ball benchmark exercises same, adjacent and other-ball Bias.
    problem = prepare_tsp(BENCHMARK_DIR / f"{BENCHMARKS[14][0]}.tsp")
    n = len(problem.cities)
    if len(problem.granular_balls) < 3:
        raise RuntimeError("initialization check requires a multi-ball problem")
    features = build_city_features(problem.cities, problem.granular_balls, problem.city_to_ball)
    _, encoded_fixed = fixed.encode(features)
    _, encoded_learned = learned.encode(features)
    if not torch.equal(encoded_fixed, encoded_learned):
        raise RuntimeError("Encoder outputs changed during checkpoint transfer")
    starts = torch.arange(n).unsqueeze(0)
    mask = torch.ones((1, n, n), dtype=torch.bool)
    mask[0, torch.arange(n), torch.arange(n)] = False
    same = torch.zeros_like(mask)
    adjacent = torch.zeros_like(mask)
    fixed_bias = torch.zeros((1, n, n))
    for current in range(n):
        current_ball = problem.city_to_ball[current]
        for candidate in range(n):
            ball = problem.city_to_ball[candidate]
            same[0, current, candidate] = ball == current_ball
            adjacent[0, current, candidate] = (
                ball != current_ball and ball in problem.ball_adjacency[current_ball]
            )
            fixed_bias[0, current, candidate] = (
                2.0 if ball == current_ball else
                1.0 if ball in problem.ball_adjacency[current_ball] else 0.0
            )
    old = fixed(encoded_fixed, starts, starts, mask, fixed_bias)
    new = learned(encoded_learned, starts, starts, mask, None, same, adjacent)
    if not torch.equal(learned.decoder.last_bias_values[..., 0],
                       torch.full((1, n), 2.0)) or not torch.equal(
                           learned.decoder.last_bias_values[..., 1], torch.full((1, n), 1.0)):
        raise RuntimeError("Bias Head does not initialize to exact +2/+1")
    if not torch.equal(fixed.decoder.last_logits, learned.decoder.last_logits):
        raise RuntimeError("initial fixed and learnable logits differ")
    if not torch.equal(old, new) or not torch.all(new[~mask] == 0):
        raise RuntimeError("initial probability or visited-city Mask differs")
    old_routes = search_routes(fixed, problem, inference_mode="mixed", samples_per_start=1, seed=42).routes
    new_routes = search_routes(learned, problem, inference_mode="mixed", samples_per_start=1, seed=42).routes
    if old_routes != new_routes:
        raise RuntimeError("paired initial mixed rollout routes differ")


def _config(config_path: Path) -> dict[str, dict[str, object]]:
    config = load_config(config_path, {})
    if "finetune" not in config or "source_checkpoint" not in config["finetune"]:
        raise ValueError("config requires [finetune] source_checkpoint")
    model = config["model"]
    if (model["candidate_mode"], model["bias_mode"], model["same_ball_bias"],
            model["adjacent_ball_bias"]) != ("soft", "learnable", 2.0, 1.0):
        raise ValueError("fine-tune config requires learnable Soft Bias initialized at +2/+1")
    return config


def run(config_path: Path, *, check_only: bool = False) -> dict[str, object] | None:
    config_path = config_path.resolve()
    config = _config(config_path)
    source = _source_path(config, config_path)
    train_instances, validation_instances = load_manifest(config["data"])
    if len(train_instances) != 500 or len(validation_instances) != 100:
        raise ValueError("expected exactly 500 training and 100 validation instances")
    torch.set_num_threads(1)  # Also keeps CPU preflight deterministic and fast.
    fixed, policy, original, source_state = _load_frozen_policy(source)
    source_model = source_state["config"]["model"]
    for key in ("embedding_dim", "encoder_layers", "num_heads"):
        if int(config["model"][key]) != int(source_model[key]):
            raise ValueError(f"fine-tune model {key} differs from source checkpoint")
    _verify_initial_equivalence(fixed, policy)
    if check_only:
        print(f"Check OK: 500/100 cases; source epoch {source_state['best_epoch']}; "
              f"logits, probabilities and paired routes match; Bias Head only is trainable; "
              f"CUDA available={torch.cuda.is_available()}")
        return None
    device = select_device(str(config["runtime"]["device"]))
    if device.type != "cuda":
        raise ValueError("this experiment requires CUDA")
    seed = int(config["training"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    policy.to(device)
    optimizer = torch.optim.Adam(policy.decoder.bias_head.parameters(),
                                 lr=float(config["training"]["learning_rate"]))
    run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    results = Path(str(config["output"]["results_root"])) / run_id
    checkpoints = Path(str(config["output"]["checkpoints_root"])) / run_id
    results.mkdir(parents=True, exist_ok=False)
    checkpoints.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, results / "source_config.toml")
    _write_effective_config(results / "effective_config.toml", config)
    (results / "source.json").write_text(json.dumps({
        "checkpoint": str(source), "sha256": _sha256(source),
        "source_best_epoch": source_state["best_epoch"],
        "frozen_keys": list(original), "trainable_keys": [
            name for name, p in policy.named_parameters() if p.requires_grad],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    history: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    def save(epoch: int, best_val: float, best_epoch: int, improved: bool) -> None:
        state = {
            "run_id": run_id, "epoch": epoch, "config": config,
            "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
            "model_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "python_rng_state": rng.getstate(), "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "best_validation": best_val, "best_epoch": best_epoch,
            "history": history, "validation_groups": group_rows,
        }
        _save_checkpoint(checkpoints / "latest.pt", state)
        if improved:
            _save_checkpoint(checkpoints / "best.pt", state)
        if epoch == 0 or epoch % int(config["checkpoint"]["save_interval"]) == 0:
            _save_checkpoint(checkpoints / f"epoch_{epoch:04d}.pt", state)

    initial_val, initial_groups = evaluate_validation(
        policy, validation_instances, int(config["validation"]["batch_size"]),
        int(config["training"]["micro_batch_size"]), device, 0,
    )
    group_rows.extend(initial_groups)
    initial_row = {"epoch": 0, "validation_mean_best_length": initial_val,
                   "attention_unchanged": True}
    history.append(initial_row)
    save(0, initial_val, 0, True)
    _write_csv(results / "training_history.csv", HISTORY_FIELDS, history)
    _write_csv(results / "validation_by_group.csv", GROUP_FIELDS, group_rows)
    best_val, best_epoch = initial_val, 0
    print(f"New {run_id}: source fixed epoch={source_state['best_epoch']}, "
          f"epoch 0 validation={initial_val:.2f}, train=500, validation=100, device={device}", flush=True)
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        stats = _train_epoch(
            policy, optimizer, train_instances, int(config["training"]["batch_size"]),
            int(config["training"]["micro_batch_size"]), device, rng, epoch,
        )
        _assert_frozen_unchanged(policy, original)
        val_mean = None
        improved = False
        if epoch % int(config["validation"]["interval"]) == 0 or epoch == int(config["training"]["epochs"]):
            val_mean, rows = evaluate_validation(
                policy, validation_instances, int(config["validation"]["batch_size"]),
                int(config["training"]["micro_batch_size"]), device, epoch,
            )
            group_rows.extend(rows)
            if val_mean < best_val:
                best_val, best_epoch, improved = val_mean, epoch, True
        row = {"epoch": epoch, **stats, "validation_mean_best_length": val_mean,
               "attention_unchanged": True}
        history.append(row)
        _write_csv(results / "training_history.csv", HISTORY_FIELDS, history)
        _write_csv(results / "validation_by_group.csv", GROUP_FIELDS, group_rows)
        save(epoch, best_val, best_epoch, improved)
        print(f"Epoch {epoch}: train={stats['train_mean_best_length']:.2f}, "
              f"validation={val_mean if val_mean is not None else '-'}, best={best_val:.2f}, "
              f"loss={stats['mean_loss']:.4f}, grad={stats['bias_head_grad_norm_mean']:.5f}, "
              f"same={stats['same_bias_mean']:.3f}"
              f"[{stats['same_bias_min']:.3f},{stats['same_bias_max']:.3f}]"
              f"±{stats['same_bias_std']:.3f}, "
              f"adjacent={stats['adjacent_bias_mean']:.3f}"
              f"[{stats['adjacent_bias_min']:.3f},{stats['adjacent_bias_max']:.3f}]"
              f"±{stats['adjacent_bias_std']:.3f}, "
              f"attention_unchanged=True, seconds={stats['seconds']:.1f}", flush=True)
    result = {"run_id": run_id, "source_checkpoint": str(source),
              "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
              "train_instances": 500, "validation_instances": 100,
              "completed_epochs": history[-1]["epoch"], "best_epoch": best_epoch,
              "best_validation_mean_best_length": best_val, "device": str(device),
              "attention_unchanged": True, "results_dir": str(results),
              "checkpoints_dir": str(checkpoints)}
    (results / "final_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Finished: {results}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="check data, transfer and paired routes without training")
    args = parser.parse_args()
    run(args.config, check_only=args.check)


if __name__ == "__main__":
    main()
