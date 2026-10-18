"""Paired fixed-Soft / hierarchical comparison on selected certified cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import isfinite
from datetime import datetime
from pathlib import Path
from time import perf_counter

from gb_2_maco_POMO.runner import prepare_tsp
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, equal_plateau, tour_length
from gb_2_maco_POMO.experiments.hierarchical.selected_rollout import (
    generate_routes, load_policy, selected_starts,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "benachmark_STSP_Complex_Multiopt_Test"
CHECKPOINT = ROOT / "gb_maco_POMO" / "results" / "checkpoints" / "run_20261013_230742_828b99" / "best.pt"
DEFAULT_CASES = (
    "mstsp_n080_rectangle_01",
    "mstsp_n128_compact_01",
    "mstsp_n160_rectangle_01",
)
DEFAULT_LAMBDAS = (0.0, 0.5, 1.0, 2.0)


def choose_cases(names: list[str]) -> list[tuple[int, dict[str, str]]]:
    with (DATA / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        all_cases = list(csv.DictReader(stream))
    selected = []
    for name in names:
        found = [(i + 1, case) for i, case in enumerate(all_cases[:90]) if case["name"] == name]
        if len(found) != 1:
            raise ValueError(f"case is absent from first 90: {name}")
        selected.append(found[0])
    return selected


def run_one(policy, case, index, args, mode, penalty):
    prepared = prepare_tsp(DATA / case["file"])
    n = len(prepared.cities)
    if n != int(case["n"]):
        raise ValueError("manifest city count mismatch")
    starts = selected_starts(n, args.max_starts)
    seed = args.seed + index  # Original manifest index, independent of subset order.
    started = perf_counter()
    routes, scale = generate_routes(policy, prepared, starts, args.samples, seed,
                                    mode=mode, penalty=penalty)
    inference = perf_counter() - started
    if any(len(route) != n or set(route) != set(range(n)) for route in routes):
        raise RuntimeError("route has missing/repeated cities")
    distance = prepared.distance_matrix
    raw = {canonical(route): route for route in routes}
    lengths = {key: tour_length(route, distance) for key, route in raw.items()}
    raw_best = min(lengths.values())
    ordered = sorted(raw, key=lambda key: (lengths[key], key))
    chosen = [raw[key] for key in ordered[:args.max_2opt]]
    post_started = perf_counter()
    refined = [two_opt(route, distance) for route in chosen]
    two_opt_seconds = perf_counter() - post_started
    a_best = min([raw_best, *(item.length for item in refined)])
    a_routes = {key: route for key, route in raw.items() if lengths[key] == a_best}
    a_routes.update({canonical(item.route): item.route
                     for item in refined if item.length == a_best})
    b_started = perf_counter()
    b = equal_plateau(list(a_routes.values()), distance, args.optb_checks)
    optb_seconds = perf_counter() - b_started
    optimum = int(case["exact_optimal_length"])
    if min(raw_best, a_best, b.best_length) < optimum:
        raise RuntimeError("route shorter than certified optimum")
    if any(tour_length(route, distance) != b.best_length for route in b.routes.values()):
        raise RuntimeError("Opt-B route length mismatch")
    result = {
        "case": case["name"], "n": n, "mode": mode if mode == "fixed_soft" else f"hierarchical_{penalty:g}",
        "lambda": "" if mode == "fixed_soft" else penalty,
        "seed": seed, "starts": len(starts), "samples_per_start": args.samples,
        "raw_routes": len(routes), "raw_unique": len(raw),
        "two_opt_candidates": len(chosen), "optb_checks": b.checks,
        "distance_scale": scale, "optimum": optimum,
        "raw_best": raw_best, "raw_gap_pct": 100 * (raw_best / optimum - 1),
        "two_opt_best": a_best, "two_opt_gap_pct": 100 * (a_best / optimum - 1),
        "two_opt_hit": int(a_best == optimum),
        "optb_best": b.best_length, "optb_gap_pct": 100 * (b.best_length / optimum - 1),
        "optb_hit": int(b.best_length == optimum),
        "optb_distinct_proven_optima": len(b.routes) if b.best_length == optimum else 0,
        "inference_seconds": inference, "two_opt_seconds": two_opt_seconds,
        "optb_seconds": optb_seconds, "postprocess_seconds": two_opt_seconds + optb_seconds,
    }
    return result


def write_report(output: Path, rows: list[dict], args, digest: str) -> None:
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest,
        "dataset": str(DATA), "manifest_sha256": hashlib.sha256((DATA / "manifest.csv").read_bytes()).hexdigest(),
        "cases": args.cases, "seed": args.seed, "device": args.device,
        "strategy": args.strategy, "lambdas": args.lambdas,
        "max_starts": args.max_starts, "samples": args.samples,
        "max_2opt": args.max_2opt, "optb_checks": args.optb_checks,
        "distance_scale": "median positive nearest-neighbor edge; minimum 1",
        "recall_note": "Full optimal-tour set is unknown; recall/F-beta cannot be computed.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 粒球两级决策小样本对照", "", "同一 checkpoint、起点、种子和后处理预算；Gap 相对认证最优距离。",
             "λ=0 的概率等于无偏置城市 Softmax，但逐级 Argmax 路线可以不同。", "",
             "| 算例 | 模式 | 原始距离 / Gap | 2-opt 距离 / Gap | 2-opt 命中 | Opt-B 最优回路数 | 推理秒 | 后处理秒 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['mode']} | {row['raw_best']} / {row['raw_gap_pct']:.2f}% | "
                     f"{row['two_opt_best']} / {row['two_opt_gap_pct']:.2f}% | {row['two_opt_hit']} | "
                     f"{row['optb_distinct_proven_optima']} | {row['inference_seconds']:.2f} | "
                     f"{row['postprocess_seconds']:.2f} |")
    lines += ["", "## 汇总（本次选定算例）", "",
              "| 模式 | 原始平均 Gap | 2-opt 平均 Gap | 2-opt 命中率 | Opt-B 命中率 | Opt-B 全局最优回路总数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for mode in dict.fromkeys(row["mode"] for row in rows):
        group = [r for r in rows if r["mode"] == mode]
        count = len(group)
        lines.append(f"| {mode} | {sum(r['raw_gap_pct'] for r in group)/count:.2f}% | "
                     f"{sum(r['two_opt_gap_pct'] for r in group)/count:.2f}% | "
                     f"{sum(r['two_opt_hit'] for r in group)}/{count} | "
                     f"{sum(r['optb_hit'] for r in group)}/{count} | "
                     f"{sum(r['optb_distinct_proven_optima'] for r in group)} |")
    lines += ["", "验证集未给出完整最优回路集合，因此不计算召回率或 Fβ。"
              " `results.csv` 另含逐例实际起点数、候选数、Opt-B 检查数与各阶段耗时。", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--strategy", choices=("all", "fixed_soft", "hierarchical"), default="all")
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-starts", type=int, default=32)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-2opt", type=int, default=32)
    parser.add_argument("--optb-checks", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if (min(args.max_starts, args.max_2opt, args.optb_checks) < 1 or args.samples < 0
            or any(not isfinite(value) or value < 0 for value in args.lambdas)
            or len(set(args.lambdas)) != len(args.lambdas)):
        parser.error("invalid search budget")
    args.checkpoint = args.checkpoint.resolve()
    cases = choose_cases(args.cases)
    output = args.output_dir or Path(__file__).resolve().parents[2] / "results" / f"hierarchical_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    policy, digest = load_policy(args.checkpoint, args.device)
    modes = []
    if args.strategy in ("all", "fixed_soft"):
        modes.append(("fixed_soft", 0.0))
    if args.strategy in ("all", "hierarchical"):
        modes.extend(("hierarchical", value) for value in args.lambdas)
    rows = []
    for index, case in cases:
        for mode, penalty in modes:
            label = "fixed_soft" if mode == "fixed_soft" else f"hierarchical_{penalty:g}"
            row = run_one(policy, case, index, args, mode, penalty)
            rows.append(row)
            (output / f"{row['case']}_{label}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{row['case']} {row['mode']}: {row['raw_best']} → {row['two_opt_best']} → {row['optb_best']}", flush=True)
    write_report(output, rows, args, digest)
    print(output, flush=True)


if __name__ == "__main__":
    main()
