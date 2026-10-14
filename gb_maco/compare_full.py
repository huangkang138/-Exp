"""Run one full-budget GB-MACO search per MSTSP instance and write comparable diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

from .aco import GBMACO, SearchConfig
from .granular_ball import Point2D, gbc
from .runner import BENCHMARKS
from .tsp import city_to_ball_mapping, greed_route, init_distance_matrix, read_cities

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "reproduction" / "scripts"))
from calculate_metrics import canonical_cycle, fbeta, load_ground_truth  # noqa: E402

FIELDS = (
    "instance", "city_count", "optimal_length", "best_length", "gap_pct",
    "generated_routes", "best_solution_count", "optimal_route_hits", "precision",
    "recall", "fbeta", "fallback_count", "truth_solution_count", "unique_routes",
    "candidate_routes", "paper_fbeta", "preprocess_seconds", "search_seconds",
    "total_seconds", "max_fes", "ant_count", "alpha", "beta", "rho", "seed",
)


def _cycle_bytes(route: list[int]) -> bytes:
    """Compact exact key for these benchmarks (city IDs are 0..65)."""
    values = tuple(route)
    start = values.index(min(values))
    forward = values[start:] + values[:start]
    reverse = (forward[0],) + tuple(reversed(forward[1:]))
    return bytes(min(forward, reverse))


class MeasuredGBMACO(GBMACO):
    """Observe complete ant routes without changing the search decisions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_routes = 0
        self.generated_cycles: set[bytes] = set()

    def _build_ants(self, assignments):
        ants = super()._build_ants(assignments)
        self.generated_routes += len(ants)
        self.generated_cycles.update(_cycle_bytes(ant.route.city_path) for ant in ants)
        return ants


def _run_case(index: int, benchmark_dir: str, seed_start: int, paper_fbeta: float) -> dict[str, object]:
    stem, expected_count, max_fes = BENCHMARKS[index]
    source = Path(benchmark_dir) / f"{stem}.tsp"
    started = perf_counter()
    cities = read_cities(source)
    if len(cities) != expected_count:
        raise ValueError(f"{source}: expected {expected_count} cities, got {len(cities)}")
    distance = init_distance_matrix(cities)
    balls = gbc([Point2D(city.x, city.y, city.number) for city in cities])
    mapping = city_to_ball_mapping(cities, balls)
    initial = greed_route(cities, distance, 0, mapping, balls)
    preprocess_seconds = perf_counter() - started

    seed = seed_start + index + 1
    config = SearchConfig(max_fes=max_fes, seed=seed)
    search = MeasuredGBMACO(cities, distance, mapping, balls, initial, config)
    search_started = perf_counter()
    routes = search.solve()
    search_seconds = perf_counter() - search_started
    if not routes:
        raise RuntimeError(f"{stem}: GB-MACO returned no routes")

    # Read the reference solution only after the search has finished.
    truth_path = source.with_suffix(".solution")
    truth = load_ground_truth(truth_path, expected_count)
    optimum = min(int(line.split()[0]) for line in truth_path.read_text(encoding="utf-8").splitlines() if line.strip())
    best = routes[0].length
    if best < optimum:
        raise ValueError(f"{stem}: route shorter than reference optimum")
    returned = {canonical_cycle(route.city_path) for route in routes if route.length == best}
    hits = len(returned & truth)
    precision = hits / len(returned)
    recall = hits / len(truth)
    score = fbeta(returned, truth)
    if score is None:
        raise ValueError(f"{stem}: no reference tours")
    return {
        "instance": f"MSTSP{index + 1}", "city_count": expected_count,
        "optimal_length": optimum, "best_length": best,
        "gap_pct": 100 * (best / optimum - 1),
        "generated_routes": search.generated_routes,
        "best_solution_count": len(returned), "optimal_route_hits": hits,
        "precision": precision, "recall": recall, "fbeta": score,
        "fallback_count": search.zero_weight_fallback_count,
        "truth_solution_count": len(truth), "unique_routes": len(search.generated_cycles),
        "candidate_routes": len(routes), "paper_fbeta": paper_fbeta,
        "preprocess_seconds": preprocess_seconds, "search_seconds": search_seconds,
        "total_seconds": perf_counter() - started,
        "max_fes": max_fes, "ant_count": config.ant_count,
        "alpha": config.alpha, "beta": config.beta, "rho": config.rho,
        "seed": seed,
    }


def _write_tables(rows: list[dict[str, object]], output_dir: Path) -> None:
    rows.sort(key=lambda row: int(str(row["instance"])[5:]))
    with (output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    headers = ("实例", "参考最优距离", "蚁群最佳距离", "Gap %", "生成路线", "最佳去重",
               "命中", "Precision", "Recall", "Fβ", "Fallback", "真最优总数",
               "全部去重", "时间(s)", "论文 Fβ")
    keys = ("instance", "optimal_length", "best_length", "gap_pct", "generated_routes",
            "best_solution_count", "optimal_route_hits", "precision", "recall", "fbeta",
            "fallback_count", "truth_solution_count", "unique_routes", "total_seconds",
            "paper_fbeta")
    decimals = {"gap_pct", "precision", "recall", "fbeta", "total_seconds", "paper_fbeta"}
    lines = ["# GB-MACO 25 例单次完整预算结果", "",
             "参数：ant_count=150，alpha=1，beta=1.5，rho=0.3；按原代码使用各算例 FES 预算。", "",
             "生成路线与全部去重统计构造完成的蚂蚁路线；最佳去重统计最终候选队列中达到最短距离的不同回路。",
             "Fallback 统计轮盘赌权重全为零时选择首个未访问城市的次数，与 POMO 的局部候选回退含义不同。",
             "时间为该算例从读取坐标至指标计算结束的墙钟时间；多实例并行运行，时间包含并行资源竞争。", "",
             "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = [f"{float(row[key]):.4f}" if key in decimals else str(row[key]) for key in keys]
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run each MSTSP once with full GB-MACO parameters")
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=42)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configuration = {"benchmark_dir": str(args.benchmark_dir.resolve()), "workers": args.workers,
                     "seed_start": args.seed_start, "runs_per_instance": 1,
                     "max_fes": [item[2] for item in BENCHMARKS], "ant_count": 150,
                     "alpha": 1.0, "beta": 1.5, "rho": 0.3}
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("Existing output directory has a different configuration")
    else:
        config_path.write_text(json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paper = {row["instance"]: float(row["paper_fbeta"]) for row in
             csv.DictReader((PROJECT_ROOT / "reproduction/config/paper_results.csv").open(encoding="utf-8-sig"))}
    rows = []
    pending = []
    for index, (stem, count, _) in enumerate(BENCHMARKS):
        name = f"MSTSP{index + 1}"
        for suffix in (".tsp", ".solution"):
            if not (args.benchmark_dir / f"{stem}{suffix}").is_file():
                raise FileNotFoundError(args.benchmark_dir / f"{stem}{suffix}")
        case_file = output / f"{name}.json"
        if case_file.exists():
            rows.append(json.loads(case_file.read_text(encoding="utf-8")))
        else:
            pending.append((index, str(args.benchmark_dir.resolve()), args.seed_start, paper[name]))
    _write_tables(rows, output)
    started = perf_counter()
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_case, *task): task[0] for task in pending}
        for future in as_completed(futures):
            index = futures[future]
            name = f"MSTSP{index + 1}"
            try:
                row = future.result()
                (output / f"{name}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                rows.append(row)
                _write_tables(rows, output)
                print(f"{name}: best={row['best_length']}, Fβ={row['fbeta']:.4f}, seconds={row['total_seconds']:.1f}", flush=True)
            except Exception as exc:
                failures.append({"instance": name, "error": str(exc)})
                print(f"{name}: FAILED {exc}", flush=True)
    summary = {"completed": len(rows), "failed": failures, "batch_seconds": perf_counter() - started,
               "mean_gap_pct": sum(float(row["gap_pct"]) for row in rows) / len(rows) if rows else None,
               "mean_fbeta": sum(float(row["fbeta"]) for row in rows) / len(rows) if rows else None}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output}", flush=True)
    return 1 if failures or len(rows) != len(BENCHMARKS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
