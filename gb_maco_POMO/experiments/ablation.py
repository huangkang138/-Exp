"""五组最终验证；.solution 仅在路线生成结束后用于评价。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import torch

from ..pomo import search_routes
from ..runner import prepare_tsp
from .compare import (BASELINE_CSV, BENCHMARK_ROOT, PAPER_CSV, POMO_ROOT, BENCHMARKS,
                      load_baseline, load_paper, load_policy, load_ground_truth,
                      fbeta, diversity_indicator)
from .train_loop import select_device


GROUPS = {
    "A": ("hard", "argmax"), "B": ("soft", "argmax"),
    "C": ("hard", "sampling"), "D": ("soft", "sampling"),
    "E": ("none", "sampling"),
}
FIELDS = (
    "group", "instance", "city_count", "best_length", "optimal_length", "gap_pct",
    "hit_optimum", "fbeta", "di", "generated_routes", "unique_routes",
    "duplicate_rate", "high_quality_routes", "high_quality_rate", "seconds",
    "starts", "samples_per_start", "decoder_decisions", "checkpoint_sha256",
)


def run_group(
    group: str, checkpoint: Path, output: Path, instances: set[str] | None,
    device: torch.device, samples_per_start: int, seed: int, quality_threshold: float,
    benchmark_root: Path, baseline_csv: Path, paper_csv: Path,
) -> list[dict[str, object]]:
    expected_mode, inference_mode = GROUPS[group]
    policy, state = load_policy(checkpoint, device)
    actual_mode = str(state["config"]["model"].get("candidate_mode", "hard"))
    if actual_mode != expected_mode:
        raise ValueError(f"{group} 需要 {expected_mode} 模型，实际是 {actual_mode}")
    # 先确认对照材料齐全；此处不读取任何 .solution。
    baseline, paper = load_baseline(baseline_csv), load_paper(paper_csv)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows: list[dict[str, object]] = []
    for index, (stem, count, _budget) in enumerate(BENCHMARKS, start=1):
        name = f"MSTSP{index}"
        if instances is not None and name not in instances:
            continue
        source = benchmark_root / f"{stem}.tsp"
        if not source.is_file() or name not in baseline or name not in paper:
            raise FileNotFoundError(f"缺少算例或对照材料：{name}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        prepared = prepare_tsp(source)
        if len(prepared.cities) != count:
            raise ValueError(f"{name} 城市数量不匹配")
        search = search_routes(
            policy, prepared, inference_mode=inference_mode,
            samples_per_start=samples_per_start, seed=seed + index,
            quality_threshold=quality_threshold,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        # 搜索已经结束，才读取真值；真值不参与动作选择或路线筛选。
        truth = load_ground_truth(benchmark_root / f"{stem}.solution", count)
        optimum = min(int(line.split()[0]) for line in
                      (benchmark_root / f"{stem}.solution").read_text(encoding="utf-8").splitlines()
                      if line.strip())
        if search.best_length < optimum:
            raise ValueError(f"{name} 路线短于参考最优，检查距离定义")
        from ..pomo import canonical_cycle
        returned = {canonical_cycle(route) for route in search.best_routes}
        f_score = fbeta(returned, truth)
        di_score = diversity_indicator(returned, truth)
        if f_score is None or di_score is None:
            raise ValueError(f"{name} 缺少可用真值")
        rows.append({
            "group": group, "instance": name, "city_count": count,
            "best_length": search.best_length, "optimal_length": optimum,
            "gap_pct": 100 * (search.best_length / optimum - 1),
            "hit_optimum": int(search.best_length == optimum),
            "fbeta": f_score, "di": di_score,
            "generated_routes": search.generated_count,
            "unique_routes": search.unique_count,
            "duplicate_rate": search.duplicate_rate,
            "high_quality_routes": search.high_quality_count,
            "high_quality_rate": search.high_quality_rate,
            "seconds": search.elapsed_seconds, "starts": count,
            "samples_per_start": samples_per_start if inference_mode == "sampling" else 1,
            "decoder_decisions": search.generated_count * (count - 1),
            "checkpoint_sha256": checkpoint_hash,
        })
        print(f"{group} {name}: length={search.best_length}, unique={search.unique_count}/"
              f"{search.generated_count}, seconds={search.elapsed_seconds:.2f}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_instance.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "group": group, "candidate_mode": expected_mode, "inference_mode": inference_mode,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": state["epoch"], "run_id": state["run_id"],
        "seed": seed, "samples_per_start": samples_per_start if inference_mode == "sampling" else 1,
        "quality_threshold": quality_threshold, "device": str(device),
        "instances": len(rows), "mean_gap_pct": sum(float(r["gap_pct"]) for r in rows) / len(rows),
        "optimal_hits": sum(int(r["hit_optimum"]) for r in rows),
        "mean_fbeta": sum(float(r["fbeta"]) for r in rows) / len(rows),
        "total_seconds": sum(float(r["seconds"]) for r in rows),
        "total_generated_routes": sum(int(r["generated_routes"]) for r in rows),
        "total_unique_routes": sum(int(r["unique_routes"]) for r in rows),
        "total_high_quality_routes": sum(int(r["high_quality_routes"]) for r in rows),
        "total_decoder_decisions": sum(int(r["decoder_decisions"]) for r in rows),
        "baseline_csv": str(baseline_csv.resolve()), "paper_csv": str(paper_csv.resolve()),
    }
    summary["overall_duplicate_rate"] = 1 - summary["total_unique_routes"] / summary["total_generated_routes"]
    summary["overall_high_quality_rate"] = summary["total_high_quality_routes"] / summary["total_unique_routes"]
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "config.json").write_text(json.dumps({
        "group": group, "instances": sorted(instances) if instances else "all",
        "benchmark_root": str(benchmark_root.resolve()), "checkpoint": str(checkpoint.resolve()),
        "seed": seed, "samples_per_start": samples_per_start,
        "quality_threshold": quality_threshold,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "run.log").write_text(
        "\n".join(f"{row['instance']}: best={row['best_length']}, "
                  f"unique={row['unique_routes']}/{row['generated_routes']}, "
                  f"seconds={float(row['seconds']):.4f}" for row in rows) + "\n",
        encoding="utf-8",
    )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="POMO 五组候选机制与搜索方式对照")
    parser.add_argument("--group", choices=(*GROUPS, "all"), default="all")
    parser.add_argument("--hard-checkpoint", type=Path)
    parser.add_argument("--soft-checkpoint", type=Path)
    parser.add_argument("--none-checkpoint", type=Path)
    parser.add_argument("--instances", nargs="+")
    parser.add_argument("--samples-per-start", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-threshold", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--benchmark-dir", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--baseline-csv", type=Path, default=BASELINE_CSV)
    parser.add_argument("--paper-csv", type=Path, default=PAPER_CSV)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    groups = list(GROUPS) if args.group == "all" else [args.group]
    checkpoints = {"hard": args.hard_checkpoint, "soft": args.soft_checkpoint,
                   "none": args.none_checkpoint}
    if any(checkpoints[GROUPS[group][0]] is None for group in groups):
        parser.error("所选组缺少对应模式的 checkpoint")
    for group in groups:
        checkpoint = checkpoints[GROUPS[group][0]]
        if not checkpoint.is_file():
            parser.error(f"{group} 的 checkpoint 不存在：{checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        actual = str(state["config"]["model"].get("candidate_mode", "hard"))
        if actual != GROUPS[group][0]:
            parser.error(f"{group} 的模型机制应为 {GROUPS[group][0]}，实际为 {actual}")
    if args.samples_per_start < 1 or not math.isfinite(args.quality_threshold) or args.quality_threshold < 0:
        parser.error("采样次数或高质量阈值无效")
    names = {f"MSTSP{i}" for i in range(1, len(BENCHMARKS) + 1)}
    selected = set(args.instances) if args.instances else None
    if selected is not None and (not selected or not selected <= names):
        parser.error("--instances 含未知算例")
    device = select_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    root = args.output_root or POMO_ROOT / "results" / "ablation" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if root.exists():
        parser.error(f"结果目录已存在，避免覆盖：{root}")
    # 先核验全部输入，避免运行到一半才发现某个论文算例缺失。
    for index, (stem, _count, _budget) in enumerate(BENCHMARKS, start=1):
        if selected is not None and f"MSTSP{index}" not in selected:
            continue
        for suffix in (".tsp", ".solution"):
            if not (args.benchmark_dir / f"{stem}{suffix}").is_file():
                parser.error(f"缺少验证输入：{args.benchmark_dir / (stem + suffix)}")
    summaries: dict[str, dict[str, object]] = {}
    for group in groups:
        mode = GROUPS[group][0]
        run_group(group, checkpoints[mode], root / group, selected, device,
                  args.samples_per_start, args.seed, args.quality_threshold,
                  args.benchmark_dir, args.baseline_csv, args.paper_csv)
        summaries[group] = json.loads((root / group / "summary.json").read_text(encoding="utf-8"))
    pairs = (("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E"))
    comparisons = {
        f"{left}_to_{right}": {
            metric + "_delta": float(summaries[right][metric]) - float(summaries[left][metric])
            for metric in ("mean_gap_pct", "mean_fbeta", "total_seconds",
                           "total_generated_routes", "total_unique_routes",
                           "overall_duplicate_rate", "overall_high_quality_rate")
        }
        for left, right in pairs if left in summaries and right in summaries
    }
    (root / "comparisons.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Results: {root}")


if __name__ == "__main__":
    main()
