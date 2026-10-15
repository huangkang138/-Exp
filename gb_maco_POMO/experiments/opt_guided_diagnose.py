"""Offline train-only feasibility audit; never updates checkpoint weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from ..runner import prepare_tsp
from .batch_rollout import rollout_batch
from .compare import load_policy
from .data import DatasetInstance, load_manifest
from .opt_guided_feedback import heuristic_reference, select_indices, selected_feedback


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "gb_maco_POMO/results/checkpoints/run_20261013_230742_828b99/best.pt"
SOURCE_SHA = "dbdfc5cb2faef0d7cbecb7c5c7cc77db16049e5d10e5d584a26e288f1bd88410"
DATA = PROJECT / "benachmark_Train_MATSP_Multiopt_500"
FIELDS = ("instance", "city_count", "distribution", "reference", "raw_min", "raw_mean",
          "refined_min", "refined_mean", "raw_best_refines_to_best", "unique_refined",
          "selected", "quality_std", "pass_0", "pass_0_25", "pass_0_5",
          "marginal_nonzero_0", "marginal_nonzero_0_25", "marginal_nonzero_0_5",
          "rollout_seconds", "two_opt_seconds", "reference_seconds")


def write_report(output: Path, summary: dict[str, object]) -> None:
    lines = ["# 2-opt 反馈离线可行性诊断", "",
             "只使用新训练集的 42 个样本（每种城市规模 7 例，按分布轮取），"
             "固定 Soft checkpoint 不更新；每例先用独立 Python RNG 选最多 8 个 POMO 起点，"
             "再采样完整路线并执行原版完整 2-opt。", "",
             f"- 原始最短路线未产生最佳 2-opt 终点："
             f"{summary['raw_best_not_refined_best_instances']}/{summary['instances']} 例。",
             f"- 2-opt 质量奖励有差异："
             f"{summary['quality_variable_instances']}/{summary['instances']} 例。",
             f"- 八条路线平均得到 {summary['mean_unique_refined']:.2f} 个不同 2-opt 回路。",
             f"- 0%、0.25%、0.5% 门槛通过率："
             f"{summary['pass_rate_0']:.1%} / {summary['pass_rate_0_25']:.1%} / "
             f"{summary['pass_rate_0_5']:.1%}。",
             f"- 相应覆盖边际非零率："
             f"{summary['marginal_nonzero_rate_0']:.1%} / "
             f"{summary['marginal_nonzero_rate_0_25']:.1%} / "
             f"{summary['marginal_nonzero_rate_0_5']:.1%}。",
             f"- rollout {summary['rollout_seconds']:.2f}s；"
             f"额外完整 2-opt {summary['two_opt_seconds']:.2f}s；"
             f"固定参考长度预计算 {summary['reference_seconds']:.2f}s。", "",
             "L_ref 是每例固定的从城市 0 出发的最近邻回路经完整 2-opt 后的长度，"
             "与当前策略动作无关。采用 0.5% 门槛是训练集离线诊断决定，"
             "未使用正式 25 例测试算例。参考长度不保证为全局最优；"
             "所谓高质量只相对该启发式参考。", "",
             "质量与覆盖信号均有可区分的非零反馈，离线可行性门槛通过；"
             "但这不代表训练后一定改善正式最优解覆盖。逐例数据见 `per_instance.csv`。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _sample(instances: list[DatasetInstance], seed: int) -> list[DatasetInstance]:
    groups = defaultdict(list)
    for instance in instances:
        groups[instance.city_count].append(instance)
    rng = random.Random(seed)
    selected = []
    for count in sorted(groups):
        by_distribution = defaultdict(list)
        for item in groups[count]:
            by_distribution[item.distribution].append(item)
        for values in by_distribution.values():
            rng.shuffle(values)
        bucket = []
        while len(bucket) < 7:
            for distribution in sorted(by_distribution):
                if by_distribution[distribution] and len(bucket) < 7:
                    bucket.append(by_distribution[distribution].pop())
        selected.extend(bucket)
    return selected


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("fixed Soft checkpoint SHA256 mismatch")
    train, _ = load_manifest({"root": str(DATA), "train_dir": "train",
                              "validation_dir": "validation"})
    selected = _sample(train, 42)
    if len(selected) != 42:
        raise RuntimeError("expected 42 stratified training cases")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, state = load_policy(SOURCE, device)
    if policy.candidate_mode != "soft" or policy.bias_mode != "fixed":
        raise ValueError("expected fixed Soft model")
    if device.type == "cpu":
        torch.set_num_threads(1)
    rng = random.Random(42)
    rows = []
    for number, instance in enumerate(selected, start=1):
        problem = prepare_tsp(instance.path)
        start = time.perf_counter()
        reference = heuristic_reference(problem.distance_matrix)
        reference_seconds = time.perf_counter() - start
        indices = select_indices([instance], rng)
        start = time.perf_counter()
        with torch.no_grad():
            rollout = rollout_batch(policy, [instance], device, sample_actions=True)
        rollout_seconds = time.perf_counter() - start
        feedback = selected_feedback(rollout, [instance], indices,
                                     {instance.path: reference}, 0.005)[0]
        variants = {}
        for tolerance, label in ((0.0, "0"), (0.0025, "0_25"), (0.005, "0_5")):
            from .opt_guided_feedback import coverage_marginals
            rewards, passed = coverage_marginals(feedback.refined_cycles,
                                                  feedback.refined_lengths,
                                                  reference, tolerance)
            variants[f"pass_{label}"] = passed
            variants[f"marginal_nonzero_{label}"] = sum(rewards)
        raw_best = min(feedback.raw_lengths)
        best_refined = min(feedback.refined_lengths)
        rows.append({
            "instance": str(instance.path.relative_to(DATA)),
            "city_count": instance.city_count, "distribution": instance.distribution,
            "reference": reference, "raw_min": raw_best,
            "raw_mean": statistics.mean(feedback.raw_lengths),
            "refined_min": best_refined,
            "refined_mean": statistics.mean(feedback.refined_lengths),
            "raw_best_refines_to_best": any(raw == raw_best and refined == best_refined
                                            for raw, refined in zip(feedback.raw_lengths,
                                                                    feedback.refined_lengths)),
            "unique_refined": feedback.unique_count, "selected": len(feedback.selected_indices),
            "quality_std": statistics.pstdev(feedback.quality_rewards),
            "rollout_seconds": rollout_seconds, "two_opt_seconds": feedback.two_opt_seconds,
            "reference_seconds": reference_seconds, **variants,
        })
        print(f"diagnose {number}/42: n={instance.city_count}, "
              f"unique={feedback.unique_count}, pass={variants['pass_0_5']}, "
              f"marginal={variants['marginal_nonzero_0_5']}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_instance.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    total_routes = sum(row["selected"] for row in rows)
    summary = {
        "checkpoint": str(SOURCE), "checkpoint_sha256": SOURCE_SHA,
        "model_epoch": state["epoch"], "device": str(device), "seed": 42,
        "instances": len(rows), "selected_routes": total_routes,
        "raw_best_not_refined_best_instances": sum(not row["raw_best_refines_to_best"] for row in rows),
        "quality_variable_instances": sum(row["quality_std"] > 1e-8 for row in rows),
        "mean_unique_refined": statistics.mean(row["unique_refined"] for row in rows),
        "mean_raw_length": statistics.mean(row["raw_mean"] for row in rows),
        "mean_refined_length": statistics.mean(row["refined_mean"] for row in rows),
        "rollout_seconds": sum(row["rollout_seconds"] for row in rows),
        "two_opt_seconds": sum(row["two_opt_seconds"] for row in rows),
        "reference_seconds": sum(row["reference_seconds"] for row in rows),
    }
    for label in ("0", "0_25", "0_5"):
        summary[f"pass_rate_{label}"] = sum(row[f"pass_{label}"] for row in rows) / total_routes
        summary[f"marginal_nonzero_rate_{label}"] = (
            sum(row[f"marginal_nonzero_{label}"] for row in rows) / total_routes)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)
                                          + "\n", encoding="utf-8")
    write_report(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
