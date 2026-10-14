"""同一推理预算比较旧/新 checkpoint：Mixed POMO -> 完整 2-opt -> Opt-B。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-start", type=int, default=8)
    parser.add_argument("--extra-checks", type=int, default=50_000)
    args = parser.parse_args()
    if args.output_dir.exists() or args.samples_per_start < 1 or args.extra_checks < 1:
        parser.error("输出目录须不存在，采样次数与检查预算须为正数")
    checkpoints: dict[str, Path] = {}
    for item in args.checkpoint:
        label, separator, path = item.partition("=")
        if not separator or not label.isidentifier() or label in checkpoints:
            parser.error("--checkpoint 格式为唯一的 LABEL=PATH，LABEL 须是标识符")
        checkpoints[label] = Path(path).resolve()
        if not checkpoints[label].is_file():
            parser.error(f"checkpoint 不存在：{checkpoints[label]}")
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(Path(__file__).with_name("abc_2opt")))
    from gb_maco_POMO.pomo import search_routes
    from gb_maco_POMO.runner import prepare_tsp
    from gb_maco_POMO.experiments.compare import (
        BENCHMARKS, fbeta, load_ground_truth, load_policy, read_optimum,
    )
    from gb_maco_POMO.experiments.diagnose_2opt import _select_candidates
    from gb_maco_POMO.experiments.local_search_2opt import two_opt
    from plateau_perturb_search import canonical, equal_plateau

    for stem, _, _ in BENCHMARKS:
        for suffix in (".tsp", ".solution"):
            if not (args.benchmark_dir / f"{stem}{suffix}").is_file():
                parser.error(f"缺少 benchmark：{stem}{suffix}")
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for label, checkpoint in checkpoints.items():
        policy, state = load_policy(checkpoint, torch.device("cpu"))
        if policy.candidate_mode != "soft":
            raise ValueError(f"{label} 不是 Soft POMO checkpoint")
        model_dir = args.output_dir / label
        model_dir.mkdir()
        (model_dir / "raw_routes").mkdir()
        (model_dir / "best_routes").mkdir()
        for index, (stem, city_count, _) in enumerate(BENCHMARKS, start=1):
            name = f"MSTSP{index}"
            problem = prepare_tsp(args.benchmark_dir / f"{stem}.tsp")
            if len(problem.cities) != city_count:
                raise ValueError(f"{name} 城市数与 benchmark 定义不符")
            # 两个模型都用相同推理预算与种子；各模型仅采样一次，不重采样挑结果。
            raw = search_routes(policy, problem, inference_mode="mixed",
                                samples_per_start=args.samples_per_start,
                                seed=args.seed + index)
            payload = json.dumps(raw.routes, separators=(",", ":")).encode("utf-8")
            (model_dir / "raw_routes" / f"{name}.json").write_bytes(payload)
            candidates = _select_candidates(raw.routes, raw.lengths, city_count,
                                            len(raw.routes), all_generated=True)
            started = perf_counter()
            refined = [two_opt(route, problem.distance_matrix) for route in candidates]
            a_seconds = perf_counter() - started
            best = min([*raw.lengths, *(item.length for item in refined)])
            best_routes = [route for route, length in zip(raw.routes, raw.lengths) if length == best]
            best_routes.extend(item.route for item in refined if item.length == best)
            a_pool = {canonical(route): route for route in best_routes}
            started = perf_counter()
            b = equal_plateau(list(a_pool.values()), problem.distance_matrix, args.extra_checks)
            b_seconds = perf_counter() - started
            if b.checks > args.extra_checks:
                raise RuntimeError(f"{name}: Opt-B 超出检查预算")
            # 搜索全部结束后才读取 .solution，确保它只用于评估。
            solution = args.benchmark_dir / f"{stem}.solution"
            reference = read_optimum(solution)
            truth = load_ground_truth(solution, city_count)
            if min(best, b.best_length) < reference:
                raise RuntimeError(f"{name}: 搜索距离低于参考最优值")
            (model_dir / "best_routes" / f"{name}.json").write_text(
                json.dumps({"A": [list(x) for x in sorted(a_pool)],
                            "B": [list(x) for x in sorted(b.routes)]}, ensure_ascii=False),
                encoding="utf-8",
            )
            row: dict[str, object] = {
                "model": label, "instance": name, "reference_length": reference,
                "raw_sha256": hashlib.sha256(payload).hexdigest(),
                "raw_count": len(raw.routes), "a_best": best,
                "a_gap_pct": 100 * (best - reference) / reference,
                "a_routes": len(a_pool),
                "a_hits": len(set(a_pool) & truth), "a_fbeta": fbeta(set(a_pool), truth),
                "a_checks": sum(item.pair_checks for item in refined), "a_seconds": a_seconds,
                "b_best": b.best_length,
                "b_gap_pct": 100 * (b.best_length - reference) / reference,
                "b_routes": len(b.routes),
                "b_hits": len(set(b.routes) & truth), "b_fbeta": fbeta(set(b.routes), truth),
                "b_checks": b.checks, "b_seconds": b_seconds,
            }
            rows.append(row)
            print(f"{label} {name}: A={row['a_routes']} B={row['b_routes']}", flush=True)
        (model_dir / "checkpoint.json").write_text(json.dumps({
            "path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "epoch": state.get("epoch"), "device": "cpu", "seed": args.seed,
            "samples_per_start": args.samples_per_start, "extra_checks": args.extra_checks,
        }, indent=2), encoding="utf-8")
    fields = tuple(rows[0])
    with (args.output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for label in checkpoints:
        own = [row for row in rows if row["model"] == label]
        summary.append({"model": label, "cases": len(own),
                        "optimal_distance_cases": sum(r["b_best"] == r["reference_length"] for r in own),
                        "a_mean_gap_pct": sum(float(r["a_gap_pct"]) for r in own) / len(own),
                        "b_mean_gap_pct": sum(float(r["b_gap_pct"]) for r in own) / len(own),
                        "a_mean_fbeta": sum(float(r["a_fbeta"]) for r in own) / len(own),
                        "b_mean_fbeta": sum(float(r["b_fbeta"]) for r in own) / len(own),
                        "a_hits": sum(int(r["a_hits"]) for r in own),
                        "b_hits": sum(int(r["b_hits"]) for r in own),
                        "b_checks": sum(int(r["b_checks"]) for r in own),
                        "b_seconds": sum(float(r["b_seconds"]) for r in own)})
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"结果已保存：{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
