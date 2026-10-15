"""多实例训练循环：梯度累积、固定验证、独立实验目录与断点续训。"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch

from ..attention import POMOPolicy
from .batch_rollout import rollout_batch
from .data import DatasetInstance, batches_by_city_count, load_manifest


HISTORY_FIELDS = (
    "epoch", "train_mean_length", "train_mean_best_length", "validation_mean_best_length",
    "mean_loss", "fallback_count", "fallback_rate", "average_candidate_size", "seconds",
    "same_bias_mean", "same_bias_min", "same_bias_max", "same_bias_std",
    "adjacent_bias_mean", "adjacent_bias_min", "adjacent_bias_max", "adjacent_bias_std",
    "bias_head_grad_norm_mean",
)
GROUP_FIELDS = ("epoch", "distribution", "city_count", "instance_count", "mean_best_length")


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 检测不到 GPU")
    return torch.device(name)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    """由 checkpoint 中的完整记录重写 CSV；续训时不会重复追加旧 epoch。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_effective_config(path: Path, config: dict[str, dict[str, object]]) -> None:
    """写成可直接阅读的 TOML；它记录环境变量和 CLI 覆盖后的实际值。"""
    lines: list[str] = []
    for section, values in config.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_checkpoint(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate_validation(
    policy: POMOPolicy, instances: list[DatasetInstance], batch_size: int,
    micro_batch_size: int, device: torch.device, epoch: int,
) -> tuple[float, list[dict[str, object]]]:
    """对固定验证集每个实例取 N 个起点中的最短路线；完全不更新参数。"""
    policy.eval()
    groups: dict[tuple[str, int], list[float]] = {}
    for batch in batches_by_city_count(instances, batch_size):
        for start in range(0, len(batch), micro_batch_size):
            micro = batch[start:start + micro_batch_size]
            best = rollout_batch(policy, micro, device, sample_actions=False).lengths.min(dim=1).values
            for item, length in zip(micro, best.tolist()):
                groups.setdefault((item.distribution, item.city_count), []).append(length)
    rows: list[dict[str, object]] = [
        {"epoch": epoch, "distribution": distribution, "city_count": city_count,
         "instance_count": len(values), "mean_best_length": sum(values) / len(values)}
        for (distribution, city_count), values in sorted(groups.items())
    ]
    total = sum(sum(values) for values in groups.values())
    return total / len(instances), rows


def _train_epoch(
    policy: POMOPolicy, optimizer: torch.optim.Optimizer,
    instances: list[DatasetInstance], batch_size: int, micro_batch_size: int,
    device: torch.device, rng: random.Random, epoch: int,
) -> dict[str, float | int]:
    """一个有效 batch 更新一次；多个 micro batch 的梯度按实例数加权累积。"""
    policy.train()
    batches = batches_by_city_count(instances, batch_size, rng)
    route_total = best_total = loss_total = 0.0
    fallback_total = candidate_total = decision_total = instance_total = 0
    bias_sum = {kind: 0.0 for kind in ("same", "adjacent")}
    bias_square_sum = {kind: 0.0 for kind in bias_sum}
    bias_min = {kind: float("inf") for kind in bias_sum}
    bias_max = {kind: float("-inf") for kind in bias_sum}
    bias_count = 0.0
    head_grad_norm_total = 0.0
    head_grad_norm_count = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(batches, start=1):
        optimizer.zero_grad()
        batch_loss = 0.0
        for start in range(0, len(batch), micro_batch_size):
            micro = batch[start:start + micro_batch_size]
            outcome = rollout_batch(policy, micro, device, sample_actions=True)
            assert outcome.route_log_probs is not None
            rewards = -outcome.lengths
            advantages = rewards - rewards.mean(dim=1, keepdim=True)
            # micro 的 mean 已除以自身实例数；再按 micro/batch 比例还原整批 mean。
            loss = -(advantages.detach() * outcome.route_log_probs).mean()
            weighted_loss = loss * (len(micro) / len(batch))
            if not torch.isfinite(weighted_loss).item():
                raise RuntimeError("Policy Gradient loss 出现 NaN 或 Inf")
            weighted_loss.backward()
            batch_loss += weighted_loss.detach().item()
            route_total += outcome.lengths.detach().sum().item() / micro[0].city_count
            best_total += outcome.lengths.detach().min(dim=1).values.sum().item()
            fallback_total += outcome.fallback_count
            candidate_total += outcome.candidate_total
            decision_total += outcome.decision_count
            if outcome.bias_stats is not None:
                count = outcome.bias_stats["count"]
                bias_count += count
                for kind in bias_sum:
                    bias_sum[kind] += outcome.bias_stats[f"{kind}_mean"] * count
                    bias_square_sum[kind] += outcome.bias_stats[f"{kind}_square_mean"] * count
                    bias_min[kind] = min(bias_min[kind], outcome.bias_stats[f"{kind}_min"])
                    bias_max[kind] = max(bias_max[kind], outcome.bias_stats[f"{kind}_max"])
        if policy.decoder.bias_head is not None:
            gradients = [p.grad.detach() for p in policy.decoder.bias_head.parameters()
                         if p.requires_grad and p.grad is not None]
            if not gradients:
                raise RuntimeError("Bias Head 没有梯度")
            norm = math.sqrt(sum(float(g.square().sum().item()) for g in gradients))
            if not math.isfinite(norm):
                raise RuntimeError("Bias Head 梯度范数出现 NaN 或 Inf")
            head_grad_norm_total += norm
            head_grad_norm_count += 1
        optimizer.step()
        loss_total += batch_loss * len(batch)
        instance_total += len(batch)
        if batch_index % 10 == 0 or batch_index == len(batches):
            print(f"epoch {epoch}: {batch_index}/{len(batches)} batches", flush=True)
    bias_columns = {}
    if bias_count:
        for kind in bias_sum:
            bias_columns[f"{kind}_bias_mean"] = bias_sum[kind] / bias_count
            bias_columns[f"{kind}_bias_std"] = math.sqrt(max(
                0.0, bias_square_sum[kind] / bias_count - bias_columns[f"{kind}_bias_mean"] ** 2
            ))
            bias_columns[f"{kind}_bias_min"] = bias_min[kind]
            bias_columns[f"{kind}_bias_max"] = bias_max[kind]
        bias_columns["bias_head_grad_norm_mean"] = head_grad_norm_total / head_grad_norm_count
    return {
        "train_mean_length": route_total / instance_total,
        "train_mean_best_length": best_total / instance_total,
        "mean_loss": loss_total / instance_total,
        "fallback_count": fallback_total,
        "fallback_rate": fallback_total / decision_total,
        "average_candidate_size": candidate_total / decision_total,
        "seconds": time.perf_counter() - started,
        **bias_columns,
    }


def _check_resume_config(current: dict[str, dict[str, object]], saved: dict[str, dict[str, object]]) -> None:
    """允许延长总轮数和调整 checkpoint 保存间隔；训练语义必须一致。"""
    for section in ("model", "data", "output", "validation"):
        if current[section] != saved[section]:
            raise ValueError(f"续训时 [{section}] 与原运行配置不同，请恢复原配置")
    for key in ("batch_size", "micro_batch_size", "learning_rate", "seed"):
        if current["training"][key] != saved["training"][key]:
            raise ValueError(f"续训时 [training] {key} 与原运行配置不同")


def run_training(config: dict[str, dict[str, object]], source_config: Path) -> dict[str, object]:
    """主流程；新运行创建唯一目录，续训恢复完整状态到原目录。"""
    train_instances, validation_instances = load_manifest(config["data"])
    device = select_device(str(config["runtime"]["device"]))
    if device.type == "cpu":
        torch.set_num_threads(1)
    resume = str(config["checkpoint"]["resume"])
    rng = random.Random()
    if not resume:
        seed = int(config["training"]["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        rng.seed(seed)
    model_cfg = config["model"]
    policy = POMOPolicy(
        embedding_dim=int(model_cfg["embedding_dim"]),
        head_num=int(model_cfg["num_heads"]),
        encoder_layer_num=int(model_cfg["encoder_layers"]),
        candidate_mode=str(model_cfg["candidate_mode"]),
        same_ball_bias=float(model_cfg["same_ball_bias"]),
        adjacent_ball_bias=float(model_cfg["adjacent_ball_bias"]),
        bias_mode=str(model_cfg.get("bias_mode", "fixed")),
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config["training"]["learning_rate"]))
    results_root = Path(str(config["output"]["results_root"]))
    checkpoints_root = Path(str(config["output"]["checkpoints_root"]))

    if resume:
        resume_path = Path(resume)
        if resume_path.name != "latest.pt" or not resume_path.is_file():
            raise ValueError("请用本项目生成的 latest.pt 续训")
        # 只读取用户指定的本地训练 checkpoint；其中包含 Python RNG 状态。
        state = torch.load(resume_path, map_location=device, weights_only=False)
        _check_resume_config(config, state["config"])
        run_id = state["run_id"]
        result_dir = results_root / run_id
        checkpoint_dir = checkpoints_root / run_id
        if resume_path.resolve() != (checkpoint_dir / "latest.pt").resolve() or not result_dir.is_dir():
            raise ValueError("续训文件与原 run 目录不匹配")
        policy.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        rng.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda":
            # map_location=device 会把 checkpoint 里的 RNG ByteTensor 也搬到 GPU；
            # PyTorch 的 set_rng_state_all 要求这些状态保存在 CPU ByteTensor 中。
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng_states"]])
        start_epoch = int(state["epoch"]) + 1
        if start_epoch > int(config["training"]["epochs"]):
            raise ValueError("目标 epochs 必须大于 checkpoint 已完成的 epoch")
        best_val = float(state["best_validation"])
        best_epoch = int(state["best_epoch"])
        history = state["history"]
        group_rows = state["validation_groups"]
        # 保留首次运行的实际配置；续训时另存本次目标轮数和覆盖参数。
        _write_effective_config(result_dir / f"resume_config_epoch_{start_epoch:04d}.toml", config)
        print(f"Resume {run_id}: epoch {start_epoch}/{config['training']['epochs']}", flush=True)
    else:
        run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        result_dir = results_root / run_id
        checkpoint_dir = checkpoints_root / run_id
        result_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source_config, result_dir / "source_config.toml")
        _write_effective_config(result_dir / "effective_config.toml", config)
        start_epoch, best_val, best_epoch = 1, float("inf"), 0
        history: list[dict[str, object]] = []
        group_rows: list[dict[str, object]] = []
        print(f"New {run_id}: train={len(train_instances)}, validation={len(validation_instances)}, device={device}", flush=True)

    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        stats = _train_epoch(
            policy, optimizer, train_instances,
            int(config["training"]["batch_size"]),
            int(config["training"]["micro_batch_size"]), device, rng, epoch,
        )
        val_mean: float | None = None
        improved = False
        if epoch % int(config["validation"]["interval"]) == 0 or epoch == int(config["training"]["epochs"]):
            val_mean, rows = evaluate_validation(
                policy, validation_instances, int(config["validation"]["batch_size"]),
                int(config["training"]["micro_batch_size"]), device, epoch,
            )
            group_rows.extend(rows)
            if val_mean < best_val:
                best_val, best_epoch, improved = val_mean, epoch, True

        row: dict[str, object] = {"epoch": epoch, **stats, "validation_mean_best_length": val_mean}
        history.append(row)
        _write_csv(result_dir / "training_history.csv", HISTORY_FIELDS, history)
        _write_csv(result_dir / "validation_by_group.csv", GROUP_FIELDS, group_rows)
        checkpoint_state: dict[str, object] = {
            "run_id": run_id, "epoch": epoch, "config": config,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "python_rng_state": rng.getstate(), "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "best_validation": best_val, "best_epoch": best_epoch,
            "history": history, "validation_groups": group_rows,
        }
        _save_checkpoint(checkpoint_dir / "latest.pt", checkpoint_state)
        if improved:
            _save_checkpoint(checkpoint_dir / "best.pt", checkpoint_state)
        if epoch % int(config["checkpoint"]["save_interval"]) == 0:
            _save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pt", checkpoint_state)
        print(
            f"Epoch {epoch}: train={stats['train_mean_best_length']:.2f}, "
            f"validation={val_mean if val_mean is not None else '-'}, best={best_val:.2f}, "
            f"seconds={stats['seconds']:.1f}, loss={stats['mean_loss']:.4f}"
            + (f", same={stats['same_bias_mean']:.3f}"
               f"[{stats['same_bias_min']:.3f},{stats['same_bias_max']:.3f}]"
               f", adjacent={stats['adjacent_bias_mean']:.3f}"
               f"[{stats['adjacent_bias_min']:.3f},{stats['adjacent_bias_max']:.3f}]"
               if 'same_bias_mean' in stats else ''), flush=True,
        )

    result: dict[str, object] = {
        "run_id": run_id, "train_instances": len(train_instances),
        "validation_instances": len(validation_instances), "device": str(device),
        "completed_epochs": history[-1]["epoch"],
        "best_epoch": best_epoch, "best_validation_mean_best_length": best_val,
        "results_dir": str(result_dir), "checkpoints_dir": str(checkpoint_dir),
    }
    (result_dir / "final_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Finished: {result_dir}", flush=True)
    return result
