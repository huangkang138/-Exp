"""Reproducible A/B/C post-search comparison for the saved Soft POMO model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

from plateau_perturb_search import canonical, equal_plateau, perturb_and_descend


FIELDS = (
    "instance", "city_count", "raw_routes", "unique_candidates", "raw_sha256",
    "reference_length", "truth_count", "baseline_best", "baseline_routes",
    "baseline_hits", "baseline_precision", "baseline_recall", "baseline_fbeta",
    "b_best", "b_routes", "b_hits", "b_precision", "b_recall", "b_fbeta",
    "b_new_routes", "b_checks", "b_seconds", "b_expanded_routes",
    "c_best", "c_routes", "c_hits", "c_precision", "c_recall", "c_fbeta",
    "c_new_routes", "c_checks", "c_seconds", "c_attempts", "c_completed_descents",
    "a_checks", "a_seconds", "prepare_seconds", "pomo_seconds", "case_seconds",
)


def metrics(routes, truth, score_fn):
    returned = {canonical(route) for route in routes}
    hits = len(returned & truth)
    return {
        "routes": len(returned), "hits": hits,
        "precision": hits / len(returned), "recall": hits / len(truth),
        "fbeta": score_fn(returned, truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-checks", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-start", type=int, default=8)
    parser.add_argument("--instances", nargs="+", type=int)
    args = parser.parse_args()
    if args.extra_checks < 1 or args.samples_per_start < 1 or args.output_dir.exists():
        parser.error("invalid budget/sampling, or output directory already exists")
    indices = sorted(set(args.instances or range(1, 26)))
    if not indices or indices[0] < 1 or indices[-1] > 25:
        parser.error("instances must be in 1..25")
    sys.path.insert(0, str(args.project.resolve()))
    from gb_maco_POMO.pomo import canonical_cycle, search_routes
    from gb_maco_POMO.runner import prepare_tsp
    from gb_maco_POMO.experiments.compare import (
        BENCHMARKS, fbeta, load_ground_truth, load_policy, read_optimum,
    )
    from gb_maco_POMO.experiments.diagnose_2opt import _select_candidates
    from gb_maco_POMO.experiments.local_search_2opt import two_opt

    torch.set_num_threads(1)
    policy, state = load_policy(args.checkpoint, torch.device("cpu"))
    if policy.candidate_mode != "soft":
        parser.error("checkpoint must be Soft POMO")
    baseline = {row["instance"]: row for row in csv.DictReader(
        args.baseline_csv.open(encoding="utf-8-sig", newline=""))}
    for index in indices:
        stem, _, _ = BENCHMARKS[index - 1]
        if not all((args.benchmark_dir / f"{stem}{suffix}").is_file()
                   for suffix in (".tsp", ".solution")):
            parser.error(f"missing benchmark input: {stem}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "raw_routes").mkdir()
    rows = []
    batch_started = perf_counter()
    for index in indices:
        name = f"MSTSP{index}"
        stem, city_count, _ = BENCHMARKS[index - 1]
        case_started = perf_counter()
        prepared = prepare_tsp(args.benchmark_dir / f"{stem}.tsp")
        prepare_seconds = perf_counter() - case_started
        if len(prepared.cities) != city_count:
            raise RuntimeError(f"{name}: city count mismatch")
        # One sampling call per instance. All groups share these exact routes.
        search = search_routes(policy, prepared, inference_mode="mixed",
                               samples_per_start=args.samples_per_start,
                               seed=args.seed + index)
        raw_payload = json.dumps(search.routes, separators=(",", ":")).encode("utf-8")
        raw_hash = hashlib.sha256(raw_payload).hexdigest()
        (args.output_dir / "raw_routes" / f"{name}.json").write_bytes(raw_payload)
        candidates = _select_candidates(search.routes, search.lengths, city_count,
                                        len(search.routes), all_generated=True)
        if len(candidates) != len({canonical_cycle(route) for route in search.routes}):
            raise RuntimeError(f"{name}: candidate deduplication mismatch")
        a_started = perf_counter()
        refined = [two_opt(route, prepared.distance_matrix) for route in candidates]
        a_seconds = perf_counter() - a_started
        best = min([*search.lengths, *(item.length for item in refined)])
        best_routes = [route for route, length in zip(search.routes, search.lengths)
                       if length == best]
        best_routes.extend(item.route for item in refined if item.length == best)
        start_pool = {canonical(route): route for route in best_routes}
        if not start_pool:
            raise RuntimeError(f"{name}: no best route")

        b_started = perf_counter()
        b = equal_plateau(list(start_pool.values()), prepared.distance_matrix,
                          args.extra_checks)
        b_seconds = perf_counter() - b_started
        c_started = perf_counter()
        c = perturb_and_descend(list(start_pool.values()), prepared.distance_matrix,
                                args.extra_checks, args.seed + index)
        c_seconds = perf_counter() - c_started
        if b.checks > args.extra_checks or c.checks > args.extra_checks:
            raise RuntimeError(f"{name}: extra budget exceeded")
        if any(canonical(route) != key for key, route in b.routes.items()) or any(
                canonical(route) != key for key, route in c.routes.items()):
            raise RuntimeError(f"{name}: invalid route key")

        # The reference set is loaded only after A, B and C have finished.
        solution = args.benchmark_dir / f"{stem}.solution"
        reference = read_optimum(solution)
        truth = load_ground_truth(solution, city_count)
        if min(best, b.best_length, c.best_length) < reference:
            raise RuntimeError(f"{name}: search beat reference optimum")
        am = metrics(start_pool.values(), truth, fbeta)
        bm = metrics(b.routes.values(), truth, fbeta)
        cm = metrics(c.routes.values(), truth, fbeta)
        saved = baseline[name]
        for key, value in (
            ("two_opt_best", best), ("selected_routes", len(candidates)),
            ("two_opt_optimal_hits", am["hits"]),
            ("two_opt_best_route_count", am["routes"]),
        ):
            if int(saved[key]) != value:
                raise RuntimeError(f"{name}: A disagrees with saved baseline ({key})")
        if abs(float(saved["two_opt_fbeta"]) - am["fbeta"]) > 1e-12:
            raise RuntimeError(f"{name}: A Fbeta disagrees with saved baseline")
        row = {
            "instance": name, "city_count": city_count,
            "raw_routes": len(search.routes), "unique_candidates": len(candidates),
            "raw_sha256": raw_hash, "reference_length": reference,
            "truth_count": len(truth), "baseline_best": best,
            **{f"baseline_{key}": value for key, value in am.items()},
            "b_best": b.best_length,
            **{f"b_{key}": value for key, value in bm.items()},
            "b_new_routes": len(set(b.routes) - set(start_pool)),
            "b_checks": b.checks, "b_seconds": b_seconds,
            "b_expanded_routes": b.attempts,
            "c_best": c.best_length,
            **{f"c_{key}": value for key, value in cm.items()},
            "c_new_routes": len(set(c.routes) - set(start_pool)),
            "c_checks": c.checks, "c_seconds": c_seconds,
            "c_attempts": c.attempts,
            "c_completed_descents": c.completed_descents,
            "a_checks": sum(item.pair_checks for item in refined),
            "a_seconds": a_seconds, "prepare_seconds": prepare_seconds,
            "pomo_seconds": search.elapsed_seconds,
            "case_seconds": perf_counter() - case_started,
        }
        rows.append(row)
        print(f"{name}: A={am['routes']} B={bm['routes']} C={cm['routes']} "
              f"checks={b.checks}/{c.checks}", flush=True)
    with (args.output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch": state.get("epoch"), "benchmark_dir": str(args.benchmark_dir.resolve()),
        "device": "cpu", "seed": args.seed,
        "samples_per_start": args.samples_per_start,
        "extra_pair_check_ceiling_per_instance": args.extra_checks,
        "instances": len(rows), "batch_seconds": perf_counter() - batch_started,
        "groups": {group: {
            "optimal_distance_cases": sum(r[f"{group}_best"] == r["reference_length"] for r in rows),
            "mean_fbeta": sum(r[f"{group}_fbeta"] for r in rows) / len(rows),
            "total_optimal_hits": sum(r[f"{group}_hits"] for r in rows),
            "distance_checks": sum(r["a_checks" if group == "baseline" else f"{group}_checks"] for r in rows),
            "seconds": sum(r["a_seconds" if group == "baseline" else f"{group}_seconds"] for r in rows),
        } for group in ("baseline", "b", "c")},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Soft POMO + 2-opt：A/B/C 对照", "",
        "同一检查点和种子，每例仅生成一次原始路线并保存于 `raw_routes/`。",
        "A：所有去重原始路线做普通完整 2-opt。",
        "B：从 A 的最佳去重回路出发，广度优先探索等长 2-opt 邻居。",
        "C：从 A 的最佳去重回路出发，固定种子的双桥扰动后做完整 2-opt；未完成的下降不收录。",
        f"B、C 各自的额外预算上限为每例 {args.extra_checks:,} 次换边距离检查。",
        "`.solution` 只在三组搜索结束后用于评价；所有最佳回路按旋转、反向等价去重。",
        "耗时为本次实测；A 的逐例数值与已保存基线核对。", "",
        "| 组别 | 达参考最优距离 | 平均 Fβ | 真最优命中总数 | 距离检查 | 搜索耗时 s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, label in (("baseline", "A 普通 2-opt"), ("b", "B 等长交换"),
                         ("c", "C 双桥扰动")):
        item = summary["groups"][group]
        lines.append(f"| {label} | {item['optimal_distance_cases']}/{len(rows)} | "
                     f"{item['mean_fbeta']:.4f} | {item['total_optimal_hits']} | "
                     f"{item['distance_checks']:,} | {item['seconds']:.2f} |")
    lines.extend(("", "| 算例 | 参考最优 | A/B/C 距离 | A/B/C 最短去重 | A/B/C 真最优命中 | A/B/C Fβ | B/C 实际检查 | B/C 耗时 s |",
                  "|---|---:|---|---|---|---|---|---|"))
    for r in rows:
        triad = lambda suffix: "/".join(str(r[f"{prefix}_{suffix}"]) for prefix in ("baseline", "b", "c"))
        scores = "/".join(f"{r[p + '_fbeta']:.4f}" for p in ("baseline", "b", "c"))
        lines.append(f"| {r['instance']} | {r['reference_length']} | {triad('best')} | "
                     f"{triad('routes')} | {triad('hits')} | "
                     f"{scores} | "
                     f"{r['b_checks']}/{r['c_checks']} | "
                     f"{r['b_seconds']:.3f}/{r['c_seconds']:.3f} |")
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
