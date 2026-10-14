"""Compare Argmax, Sampling, mixed search and a fixed-budget 2-opt refinement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from ..pomo import canonical_cycle, search_routes
from ..runner import prepare_tsp
from .compare import BENCHMARKS, fbeta, load_ground_truth, load_policy, read_optimum
from .local_search_2opt import two_opt
from .train_loop import select_device

FIELDS = (
    "instance", "city_count", "seed", "optimal_length", "argmax_best", "sampling_best",
    "mixed_best", "two_opt_best", "argmax_gap_pct", "sampling_gap_pct", "mixed_gap_pct",
    "two_opt_gap_pct", "mixed_fbeta", "two_opt_fbeta", "mixed_precision", "mixed_recall",
    "two_opt_precision", "two_opt_recall", "mixed_optimal_hits", "two_opt_optimal_hits",
    "truth_solution_count", "mixed_best_route_count", "two_opt_best_route_count",
    "generated_routes", "selected_routes", "post_opt_candidate_routes",
    "post_opt_unique_routes", "two_opt_moves",
    "two_opt_pair_checks", "prepare_seconds", "search_seconds", "two_opt_seconds",
    "total_seconds", "paper_fbeta",
)


def _metrics(routes: list[list[int]], truth: set[tuple[int, ...]]) -> tuple[float, float, float, int, int]:
    returned = {canonical_cycle(route) for route in routes}
    hits = len(returned & truth)
    return hits / len(returned), hits / len(truth), fbeta(returned, truth), hits, len(returned)


def _select_candidates(
    routes: list[list[int]], lengths: list[int], city_count: int, limit: int,
    all_generated: bool = False,
) -> list[list[int]]:
    """Select unique tours under the chosen 2-opt candidate policy."""
    if all_generated:
        selected = []
        seen = set()
        for index in sorted(range(len(routes)), key=lambda i: (lengths[i], i)):
            key = canonical_cycle(routes[index])
            if key not in seen:
                selected.append(routes[index])
                seen.add(key)
                if len(selected) == limit:
                    break
        return selected

    # Historical policy: reserve one best Argmax tour, then use unique samples.
    argmax_index = min(range(city_count), key=lambda i: lengths[i])
    selected = [routes[argmax_index]]
    if limit == 1:
        return selected
    seen = {canonical_cycle(selected[0])}
    sample_indices = sorted(range(city_count, len(routes)), key=lambda i: (lengths[i], i))
    for index in sample_indices:
        key = canonical_cycle(routes[index])
        if key not in seen:
            selected.append(routes[index])
            seen.add(key)
            if len(selected) == limit:
                break
    return selected


def _run_case(
    index: int, benchmark_dir: Path, policy: torch.nn.Module, seed: int,
    samples_per_start: int, top_k: int, paper_fbeta: float,
    all_generated: bool = False,
) -> dict[str, object]:
    stem, city_count, _ = BENCHMARKS[index - 1]
    name = f"MSTSP{index}"
    started = perf_counter()
    prepared = prepare_tsp(benchmark_dir / f"{stem}.tsp")
    if len(prepared.cities) != city_count:
        raise ValueError(f"{name}: unexpected city count")
    prepare_seconds = perf_counter() - started
    search = search_routes(policy, prepared, inference_mode="mixed",
                           samples_per_start=samples_per_start, seed=seed + index)
    if len(search.routes) != city_count * (1 + samples_per_start):
        raise RuntimeError(f"{name}: candidate count does not match mixed search")
    argmax_best = min(search.lengths[:city_count])
    sampling_best = min(search.lengths[city_count:])
    mixed_best = min(argmax_best, sampling_best)
    if mixed_best != search.best_length:
        raise RuntimeError(f"{name}: mixed best does not match its candidates")
    selected = _select_candidates(search.routes, search.lengths, city_count, top_k,
                                  all_generated=all_generated)
    local_started = perf_counter()
    refined = [two_opt(route, prepared.distance_matrix) for route in selected]
    two_opt_seconds = perf_counter() - local_started
    post_opt_routes = search.routes + [item.route for item in refined]
    post_opt_unique_routes = len({canonical_cycle(route) for route in post_opt_routes})
    best = min([mixed_best, *(item.length for item in refined)])
    best_routes = [route for route, length in zip(search.routes, search.lengths) if length == best]
    best_routes.extend(item.route for item in refined if item.length == best)

    # Ground truth is loaded only after candidate generation and local search.
    solution = benchmark_dir / f"{stem}.solution"
    optimum = read_optimum(solution)
    truth = load_ground_truth(solution, city_count)
    if best < optimum:
        raise RuntimeError(f"{name}: refined route is shorter than reference optimum")
    mixed_precision, mixed_recall, mixed_fbeta, mixed_hits, mixed_best_count = _metrics(search.best_routes, truth)
    precision, recall, score, hits, best_count = _metrics(best_routes, truth)
    gap = lambda length: 100 * (length / optimum - 1)
    return {
        "instance": name, "city_count": city_count, "seed": seed + index,
        "optimal_length": optimum, "argmax_best": argmax_best,
        "sampling_best": sampling_best, "mixed_best": mixed_best, "two_opt_best": best,
        "argmax_gap_pct": gap(argmax_best), "sampling_gap_pct": gap(sampling_best),
        "mixed_gap_pct": gap(mixed_best), "two_opt_gap_pct": gap(best),
        "mixed_fbeta": mixed_fbeta, "two_opt_fbeta": score,
        "mixed_precision": mixed_precision, "mixed_recall": mixed_recall,
        "two_opt_precision": precision, "two_opt_recall": recall,
        "mixed_optimal_hits": mixed_hits, "two_opt_optimal_hits": hits,
        "truth_solution_count": len(truth), "mixed_best_route_count": mixed_best_count,
        "two_opt_best_route_count": best_count, "generated_routes": len(search.routes),
        "selected_routes": len(selected), "two_opt_moves": sum(item.moves for item in refined),
        "post_opt_candidate_routes": len(post_opt_routes),
        "post_opt_unique_routes": post_opt_unique_routes,
        "two_opt_pair_checks": sum(item.pair_checks for item in refined),
        "prepare_seconds": prepare_seconds, "search_seconds": search.elapsed_seconds,
        "two_opt_seconds": two_opt_seconds, "total_seconds": perf_counter() - started,
        "paper_fbeta": paper_fbeta,
    }


def _write_report(rows: list[dict[str, object]], summary: dict[str, object], path: Path) -> None:
    reference_color = "#38BDF8"
    refined_color = "#34D399"

    def colored(value: object, color: str) -> str:
        return f'<span style="color:{color}">{value}</span>'

    def cell(value: object, *, color: str | None = None, header: bool = False) -> str:
        tag = "th" if header else "td"
        style = f' style="color:{color}"' if color else ""
        return f"<{tag}{style}>{value}</{tag}>"

    lines = [
        "# Soft POMO + 2-opt 对照", "",
        f"同一 Soft 检查点、{summary['device'].upper()}、每起点 1 条 Argmax + "
        f"{summary['samples_per_start']} 条 Sampling；每例最多优化 "
        f"{summary['top_k']} 条去重路线（实际条数见下表）。",
        ("候选来自全部 Argmax 和 Sampling，按距离排序并按回路去重；"
         if summary.get("all_generated") else
         "候选包含最佳 Argmax，其余按 Sampling 路线长度排序；")
        + "2-opt 使用项目整数距离矩阵。",
        "`.solution` 仅用于所有路线生成及 2-opt 结束后的评价。", "",
        f"{colored('蓝色', reference_color)}：参考值与优化前 Gap。参考最优距离来自 `.solution`；"
        "原 Gap 是混合搜索距离相对该参考距离的差距，并非论文报告的 Gap；论文 Fβ 来自论文对照表。",
        f"{colored('绿色', refined_color)}：2-opt 后的距离、Gap 及评价指标。", "",
        f"Composite 平均 Gap：{summary['composite_mixed_mean_gap_pct']:.4f}% → "
        f"{summary['composite_two_opt_mean_gap_pct']:.4f}%；最优距离命中 "
        f"{summary['composite_mixed_optimal_cases']} → {summary['composite_two_opt_optimal_cases']} 例。",
        f"本次推理合计 {summary['search_seconds']:.2f}s；2-opt 合计 "
        f"{summary['two_opt_seconds']:.2f}s；批次墙钟 {summary['batch_seconds']:.2f}s。", "",
        '<table border="1" cellspacing="0" cellpadding="6">',
        "<thead><tr>" + "".join(cell(label, color=color, header=True) for label, color in (
            ("算例", None), ("参考最优", reference_color), ("Argmax", None),
            ("Sampling", None), ("混合", None), ("原 Gap %", reference_color),
                ("+2-opt", refined_color), ("新 Gap %", refined_color),
            ("推理 s", None), ("2-opt s", None),
        )) + "</tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        lines.append(
            "<tr>" + "".join((
                cell(row["instance"]),
                cell(row["optimal_length"], color=reference_color),
                cell(row["argmax_best"]), cell(row["sampling_best"]),
                cell(row["mixed_best"]),
                cell(f'{row["mixed_gap_pct"]:.4f}', color=reference_color),
                cell(row["two_opt_best"], color=refined_color),
                cell(f'{row["two_opt_gap_pct"]:.4f}', color=refined_color),
                cell(f'{row["search_seconds"]:.3f}'),
                cell(f'{row["two_opt_seconds"]:.3f}'),
            )) + "</tr>"
        )
    lines.extend((
        "</tbody>", "</table>", "",
        "下表按回路旋转及反向去重。后处理去重数包含全部原始候选和 2-opt 结果；"
        "最短去重数只统计最终最短距离对应的路线。真最优命中数以 `.solution` 为准。",
        "Precision = 真最优命中数 / 最短去重数；Recall = 真最优命中数 / 真最优总数。", "",
        '<table border="1" cellspacing="0" cellpadding="6">',
        "<thead><tr>" + "".join(cell(label, color=color, header=True) for label, color in (
            ("算例", None), ("原生成", None), ("2-opt 处理", None),
            ("后处理去重", refined_color), ("最短去重数", refined_color),
            ("真最优命中", refined_color), ("真最优总数", reference_color),
            ("Precision", refined_color), ("Recall", refined_color),
            ("新 Fβ", refined_color), ("论文 Fβ", reference_color),
        )) + "</tr></thead>",
        "<tbody>",
    ))
    for row in rows:
        lines.append(
            "<tr>" + "".join((
                cell(row["instance"]), cell(row["generated_routes"]),
                cell(row["selected_routes"]),
                cell(row["post_opt_unique_routes"], color=refined_color),
                cell(row["two_opt_best_route_count"], color=refined_color),
                cell(row["two_opt_optimal_hits"], color=refined_color),
                cell(row["truth_solution_count"], color=reference_color),
                cell(f'{row["two_opt_precision"]:.4f}', color=refined_color),
                cell(f'{row["two_opt_recall"]:.4f}', color=refined_color),
                cell(f'{row["two_opt_fbeta"]:.4f}', color=refined_color),
                cell(f'{row["paper_fbeta"]:.4f}', color=reference_color),
            )) + "</tr>"
        )
    lines.extend(("</tbody>", "</table>"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Soft POMO routes and bounded 2-opt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--paper-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--samples-per-start", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--all-generated", action="store_true",
                        help="Select from every Argmax and Sampling route before 2-opt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instances", nargs="+", type=int)
    args = parser.parse_args()
    if args.samples_per_start < 1 or args.top_k < 1:
        parser.error("samples-per-start and top-k must be positive")
    indices = sorted(set(args.instances or range(1, 26)))
    if any(index < 1 or index > 25 for index in indices):
        parser.error("instances must be between 1 and 25")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    device = select_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    policy, state = load_policy(args.checkpoint, device)
    if policy.candidate_mode != "soft":
        parser.error("checkpoint must use Soft guidance")
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    paper = {row["instance"]: float(row["paper_fbeta"]) for row in
             csv.DictReader(args.paper_csv.open(encoding="utf-8-sig"))}
    for index in indices:
        stem = BENCHMARKS[index - 1][0]
        for suffix in (".tsp", ".solution"):
            if not (args.benchmark_dir / f"{stem}{suffix}").is_file():
                parser.error(f"missing input: {stem}{suffix}")
    args.output_dir.mkdir(parents=True)
    rows = []
    batch_started = perf_counter()
    for index in indices:
        row = _run_case(index, args.benchmark_dir, policy, args.seed,
                        args.samples_per_start, args.top_k, paper[f"MSTSP{index}"],
                        all_generated=args.all_generated)
        rows.append(row)
        print(f"{row['instance']}: {row['mixed_best']} -> {row['two_opt_best']}, "
              f"2-opt={row['two_opt_seconds']:.2f}s", flush=True)
    with (args.output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    composite = [row for row in rows if int(str(row["instance"])[5:]) >= 13]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": state.get("epoch"), "device": str(device), "seed": args.seed,
        "samples_per_start": args.samples_per_start, "top_k": args.top_k,
        "all_generated": args.all_generated,
        "instances": len(rows), "batch_seconds": perf_counter() - batch_started,
        "search_seconds": sum(float(row["search_seconds"]) for row in rows),
        "two_opt_seconds": sum(float(row["two_opt_seconds"]) for row in rows),
        "composite_count": len(composite),
        "composite_mixed_mean_gap_pct": sum(float(row["mixed_gap_pct"]) for row in composite) / len(composite) if composite else None,
        "composite_two_opt_mean_gap_pct": sum(float(row["two_opt_gap_pct"]) for row in composite) / len(composite) if composite else None,
        "composite_mixed_optimal_cases": sum(float(row["mixed_gap_pct"]) == 0 for row in composite),
        "composite_two_opt_optimal_cases": sum(float(row["two_opt_gap_pct"]) == 0 for row in composite),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(rows, summary, args.output_dir / "comparison.md")
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
