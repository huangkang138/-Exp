"""Compare a trained learnable Bias model with the saved fixed-Bias CPU baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "gb_maco_POMO/results/learnable_bias"
EXPERIMENT = PROJECT / "最优结果的实验/目前最优"
sys.path.insert(0, str(EXPERIMENT))
from opt_b_search import canonical, equal_plateau

from gb_maco_POMO.pomo import search_routes
from gb_maco_POMO.runner import prepare_tsp
from gb_maco_POMO.experiments.compare import (
    BENCHMARKS, fbeta, load_ground_truth, load_policy, read_optimum,
)
from gb_maco_POMO.experiments.diagnose_2opt import _select_candidates
from gb_maco_POMO.experiments.local_search_2opt import two_opt


BASELINE = EXPERIMENT / "runs/run_20261014_220649_876728/per_instance.csv"
BASELINE_SUMMARY = BASELINE.with_name("summary.json")
BENCHMARK_DIR = PROJECT.parent / "GB-MACO_PPSN2026-main/PPSN_GB-MACO/benchmark_MSTSP"
FIELDS = (
    "instance", "city_count", "reference_length", "raw_routes", "unique_candidates",
    "mixed_best", "a_best", "a_routes", "a_hits", "a_fbeta", "a_checks",
    "b_best", "b_routes", "b_hits", "b_fbeta", "b_checks",
    "same_bias_mean", "same_bias_min", "same_bias_max", "same_bias_std",
    "adjacent_bias_mean", "adjacent_bias_min", "adjacent_bias_max", "adjacent_bias_std",
    "fixed_a_best", "fixed_a_hits", "fixed_a_fbeta", "fixed_b_best",
    "fixed_b_hits", "fixed_b_fbeta", "b_hits_delta", "b_fbeta_delta",
)


def _baseline() -> dict[str, dict[str, str]]:
    with BASELINE.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    if len(rows) != 25 or summary["checkpoint_sha256"] != (
        "dbdfc5cb2faef0d7cbecb7c5c7cc77db16049e5d10e5d584a26e288f1bd88410"
    ) or summary["device"] != "cpu" or summary["seed"] != 42 or (
        summary["samples_per_start"] != 8 or summary["extra_pair_check_ceiling_per_instance"] != 50_000
    ):
        raise ValueError("saved fixed-Bias baseline does not match the formal CPU protocol")
    return {row["instance"]: row for row in rows}


def _checkpoint() -> Path:
    candidates = sorted(RESULTS.glob("training/run_*/final_result.json"), reverse=True)
    for result in candidates:
        checkpoint = RESULTS / "checkpoints" / result.parent.name / "best.pt"
        if checkpoint.is_file():
            return checkpoint
    raise FileNotFoundError("learnable Bias best.pt not found; train with the dedicated config first")


def _metrics(routes: dict[tuple[int, ...], list[int]], truth: set[tuple[int, ...]]) -> dict[str, float | int]:
    keys = set(routes)
    return {"routes": len(keys), "hits": len(keys & truth), "fbeta": fbeta(keys, truth)}


def run(checkpoint: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    fixed = _baseline()
    torch.set_num_threads(1)
    policy, state = load_policy(checkpoint, torch.device("cpu"))
    if policy.candidate_mode != "soft" or policy.bias_mode != "learnable":
        raise ValueError("checkpoint is not a learnable soft-Bias model")
    output.mkdir(parents=True)
    (output / "raw_routes").mkdir()
    rows: list[dict[str, object]] = []
    for index, (stem, city_count, _) in enumerate(BENCHMARKS, start=1):
        name = f"MSTSP{index}"
        problem = prepare_tsp(BENCHMARK_DIR / f"{stem}.tsp")
        search = search_routes(policy, problem, inference_mode="mixed", samples_per_start=8, seed=42 + index)
        routes, lengths = search.routes, search.lengths
        (output / "raw_routes" / f"{name}.json").write_text(
            json.dumps(routes, separators=(",", ":")), encoding="utf-8"
        )
        candidates = _select_candidates(routes, lengths, city_count, len(routes), all_generated=True)
        refined = [two_opt(route, problem.distance_matrix) for route in candidates]
        best = min([*lengths, *(item.length for item in refined)])
        best_routes = [route for route, length in zip(routes, lengths) if length == best]
        best_routes.extend(item.route for item in refined if item.length == best)
        a_pool = {canonical(route): route for route in best_routes}
        b = equal_plateau(list(a_pool.values()), problem.distance_matrix, 50_000)
        reference = read_optimum(BENCHMARK_DIR / f"{stem}.solution")
        truth = load_ground_truth(BENCHMARK_DIR / f"{stem}.solution", city_count)
        if best < reference or b.best_length < reference or b.checks > 50_000:
            raise RuntimeError(f"{name}: invalid distance or Opt-B budget")
        am, bm = _metrics(a_pool, truth), _metrics(b.routes, truth)
        old = fixed[name]
        bias = search.bias_values
        if bias is None:
            raise RuntimeError("learnable Bias values were not recorded")
        bias_stats = {
            f"{kind}_bias_{stat}": float(getattr(bias[..., axis], stat)())
            for axis, kind in enumerate(("same", "adjacent"))
            for stat in ("mean", "min", "max", "std")
        }
        row = {
            "instance": name, "city_count": city_count, "reference_length": reference,
            "raw_routes": len(routes), "unique_candidates": len(candidates),
            "mixed_best": min(lengths), "a_best": best, "a_routes": am["routes"],
            "a_hits": am["hits"], "a_fbeta": am["fbeta"],
            "a_checks": sum(item.pair_checks for item in refined),
            "b_best": b.best_length, "b_routes": bm["routes"], "b_hits": bm["hits"],
            "b_fbeta": bm["fbeta"], "b_checks": b.checks, **bias_stats,
            "fixed_a_best": int(old["a_best"]), "fixed_a_hits": int(old["a_hits"]),
            "fixed_a_fbeta": float(old["a_fbeta"]),
            "fixed_b_best": int(old["b_best"]), "fixed_b_hits": int(old["b_hits"]),
            "fixed_b_fbeta": float(old["b_fbeta"]),
            "b_hits_delta": bm["hits"] - int(old["b_hits"]),
            "b_fbeta_delta": bm["fbeta"] - float(old["b_fbeta"]),
        }
        rows.append(row)
        print(f"{name}: distance {best}/{b.best_length}; Opt-B hits {old['b_hits']} -> {bm['hits']}", flush=True)
    with (output / "per_instance.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "fixed_baseline": str(BASELINE), "fixed_baseline_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        "learnable_checkpoint": str(checkpoint), "learnable_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "learnable_best_epoch": state["best_epoch"], "device": "cpu", "seed": 42,
        "inference": "per start 1 argmax + 8 sampling", "local_search": "full 2-opt; Opt-B 50000 additional checks/case",
        "cases": len(rows),
        "fixed_distance_hits": sum(r["fixed_a_best"] == r["reference_length"] for r in rows),
        "learnable_distance_hits": sum(r["a_best"] == r["reference_length"] for r in rows),
        "fixed_mean_gap_pct": sum(100 * (r["fixed_a_best"] / r["reference_length"] - 1) for r in rows) / len(rows),
        "learnable_mean_gap_pct": sum(100 * (r["a_best"] / r["reference_length"] - 1) for r in rows) / len(rows),
        "fixed_a_hits": sum(r["fixed_a_hits"] for r in rows),
        "learnable_a_hits": sum(r["a_hits"] for r in rows),
        "fixed_b_hits": sum(r["fixed_b_hits"] for r in rows),
        "learnable_b_hits": sum(r["b_hits"] for r in rows),
        "fixed_mean_fbeta": sum(r["fixed_b_fbeta"] for r in rows) / len(rows),
        "learnable_mean_fbeta": sum(r["b_fbeta"] for r in rows) / len(rows),
        "better_b_hits_cases": sum(r["b_hits_delta"] > 0 for r in rows),
        "worse_b_hits_cases": sum(r["b_hits_delta"] < 0 for r in rows),
        "better_b_fbeta_cases": sum(r["b_fbeta_delta"] > 1e-12 for r in rows),
        "worse_b_fbeta_cases": sum(r["b_fbeta_delta"] < -1e-12 for r in rows),
        "same_bias_mean": sum(r["same_bias_mean"] for r in rows) / len(rows),
        "adjacent_bias_mean": sum(r["adjacent_bias_mean"] for r in rows) / len(rows),
        "same_bias_mean_std": sum(r["same_bias_std"] for r in rows) / len(rows),
        "adjacent_bias_mean_std": sum(r["adjacent_bias_std"] for r in rows) / len(rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Learnable Bias 与固定 Bias 对照", "", "同一 CPU 条件：seed=42；每起点 1 Argmax + 8 Sampling；完整普通 2-opt；每例 Opt-B 最多 50,000 次额外换边检查。", "",
             "| 指标 | 固定 Bias | 可学习 Bias |", "|---|---:|---:|",
             f"| 最短距离命中 | {summary['fixed_distance_hits']}/25 | {summary['learnable_distance_hits']}/25 |",
             f"| 平均 Gap | {summary['fixed_mean_gap_pct']:.4f}% | {summary['learnable_mean_gap_pct']:.4f}% |",
             f"| 普通 2-opt 真最优解 | {summary['fixed_a_hits']} | {summary['learnable_a_hits']} |",
             f"| Opt-B 真最优解 | {summary['fixed_b_hits']} | {summary['learnable_b_hits']} |",
             f"| 平均 Fβ | {summary['fixed_mean_fbeta']:.6f} | {summary['learnable_mean_fbeta']:.6f} |", "",
             f"Opt-B 命中逐例改善/退步：{summary['better_b_hits_cases']}/{summary['worse_b_hits_cases']}；"
             f"Fβ 逐例改善/退步：{summary['better_b_fbeta_cases']}/{summary['worse_b_fbeta_cases']}。", "",
             "| 算例 | 距离 固定/可学 | Opt-B 命中 固定/可学 (差值) | Fβ 固定/可学 (差值) | 同球 Bias 均值 [最小,最大] | 邻球 Bias 均值 [最小,最大] |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['instance']} | {r['fixed_a_best']}/{r['a_best']} | {r['fixed_b_hits']}/{r['b_hits']} ({r['b_hits_delta']:+d}) | "
                     f"{r['fixed_b_fbeta']:.4f}/{r['b_fbeta']:.4f} ({r['b_fbeta_delta']:+.4f}) | "
                     f"{r['same_bias_mean']:.3f} [{r['same_bias_min']:.3f},{r['same_bias_max']:.3f}] | "
                     f"{r['adjacent_bias_mean']:.3f} [{r['adjacent_bias_min']:.3f},{r['adjacent_bias_max']:.3f}] |")
    lines += ["", "## 判断", ""]
    if summary["learnable_mean_fbeta"] > summary["fixed_mean_fbeta"] and summary["learnable_b_hits"] >= summary["fixed_b_hits"]:
        lines.append("可学习 Bias 在平均 Fβ 上提高，且 Opt-B 真最优解数量未下降。")
    elif summary["learnable_mean_fbeta"] < summary["fixed_mean_fbeta"] and summary["learnable_b_hits"] <= summary["fixed_b_hits"]:
        lines.append("可学习 Bias 在平均 Fβ 上下降，且 Opt-B 真最优解数量未增加。")
    else:
        lines.append("平均 Fβ 与 Opt-B 真最优解数量的变化方向不同，需要按目标指标判断。")
    lines.append(f"贪心 POMO 的 Bias 输出：同球平均 {summary['same_bias_mean']:.4f}，"
                 f"逐例平均标准差 {summary['same_bias_mean_std']:.4f}；邻球平均 "
                 f"{summary['adjacent_bias_mean']:.4f}，逐例平均标准差 "
                 f"{summary['adjacent_bias_mean_std']:.4f}。")
    if summary["same_bias_mean_std"] < 1e-4 and summary["adjacent_bias_mean_std"] < 1e-4:
        lines.append("两项 Bias 几乎不随起点/步骤变化，状态差异很弱。")
    else:
        lines.append("Bias 随起点/步骤变化；各例变化范围和标准差见逐例表及 CSV。")
    lines += ["", "Bias 统计来自每例贪心 POMO rollout 的逐起点、逐步骤输出。"
              "现有 Query 仅由起点与当前城市的编码表示构成，不直接编码完整已访问路径。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint or _checkpoint()
    output = args.output_dir or RESULTS / "comparison" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run(checkpoint, output)


if __name__ == "__main__":
    main()
