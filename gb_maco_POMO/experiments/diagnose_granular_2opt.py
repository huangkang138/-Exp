"""Compare ordinary and cross-ball-filtered 2-opt on identical Soft POMO tours."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from html import escape
from pathlib import Path
from time import perf_counter

import torch

from ..pomo import canonical_cycle, search_routes
from ..runner import prepare_tsp
from .compare import BENCHMARKS, load_ground_truth, load_policy, read_optimum
from .diagnose_2opt import _metrics, _select_candidates
from .granular_local_search_2opt import granular_two_opt
from .local_search_2opt import two_opt
from .train_loop import select_device

FIELDS = (
    "instance", "city_count", "seed", "optimal_length", "generated_routes",
    "unique_candidates", "mixed_best", "ordinary_best", "granular_best",
    "ordinary_gap_pct", "granular_gap_pct", "ordinary_best_route_count",
    "granular_best_route_count", "ordinary_hits", "granular_hits",
    "truth_solution_count", "ordinary_precision", "granular_precision",
    "ordinary_recall", "granular_recall", "ordinary_fbeta", "granular_fbeta",
    "ordinary_moves", "granular_moves", "ordinary_pair_checks",
    "granular_pair_checks", "granular_filtered_pairs", "prepare_seconds",
    "search_seconds", "ordinary_seconds", "granular_seconds", "paper_fbeta",
)


def _evaluate(
    raw_routes: list[list[int]], raw_lengths: list[int], refined: list[object],
    optimum: int, truth: set[tuple[int, ...]],
) -> tuple[int, float, float, float, int, int]:
    best = min([*raw_lengths, *(item.length for item in refined)])
    if best < optimum:
        raise RuntimeError("route shorter than the reference optimum")
    best_routes = [route for route, length in zip(raw_routes, raw_lengths) if length == best]
    best_routes.extend(item.route for item in refined if item.length == best)
    precision, recall, score, hits, count = _metrics(best_routes, truth)
    return best, precision, recall, score, hits, count


def _run_case(
    index: int, benchmark_dir: Path, policy: torch.nn.Module, seed: int,
    samples_per_start: int, paper_fbeta: float,
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
        raise RuntimeError(f"{name}: unexpected number of generated routes")
    candidates = _select_candidates(search.routes, search.lengths, city_count,
                                    len(search.routes), all_generated=True)
    if len(candidates) != len({canonical_cycle(route) for route in search.routes}):
        raise RuntimeError(f"{name}: candidate set omitted a distinct generated route")

    # Alternate order across cases to reduce timing bias from running second.
    results: dict[str, tuple[list[object], float]] = {}
    for method in (("ordinary", "granular") if index % 2 else ("granular", "ordinary")):
        local_started = perf_counter()
        if method == "ordinary":
            refined = [two_opt(route, prepared.distance_matrix) for route in candidates]
        else:
            refined = [granular_two_opt(route, prepared.distance_matrix,
                                        prepared.city_to_ball) for route in candidates]
        results[method] = (refined, perf_counter() - local_started)

    # The truth set participates only in evaluation, after both searches finish.
    solution = benchmark_dir / f"{stem}.solution"
    optimum = read_optimum(solution)
    truth = load_ground_truth(solution, city_count)
    ordinary, ordinary_seconds = results["ordinary"]
    granular, granular_seconds = results["granular"]
    ob, op, or_, of, oh, oc = _evaluate(search.routes, search.lengths,
                                        ordinary, optimum, truth)
    gb, gp, gr, gf, gh, gc = _evaluate(search.routes, search.lengths,
                                       granular, optimum, truth)
    return {
        "instance": name, "city_count": city_count, "seed": seed + index,
        "optimal_length": optimum, "generated_routes": len(search.routes),
        "unique_candidates": len(candidates), "mixed_best": search.best_length,
        "ordinary_best": ob, "granular_best": gb,
        "ordinary_gap_pct": 100 * (ob / optimum - 1),
        "granular_gap_pct": 100 * (gb / optimum - 1),
        "ordinary_best_route_count": oc, "granular_best_route_count": gc,
        "ordinary_hits": oh, "granular_hits": gh,
        "truth_solution_count": len(truth),
        "ordinary_precision": op, "granular_precision": gp,
        "ordinary_recall": or_, "granular_recall": gr,
        "ordinary_fbeta": of, "granular_fbeta": gf,
        "ordinary_moves": sum(item.moves for item in ordinary),
        "granular_moves": sum(item.moves for item in granular),
        "ordinary_pair_checks": sum(item.pair_checks for item in ordinary),
        "granular_pair_checks": sum(item.pair_checks for item in granular),
        "granular_filtered_pairs": sum(item.filtered_pairs for item in granular),
        "prepare_seconds": prepare_seconds, "search_seconds": search.elapsed_seconds,
        "ordinary_seconds": ordinary_seconds, "granular_seconds": granular_seconds,
        "paper_fbeta": paper_fbeta,
    }


def _check_existing(row: dict[str, object], previous: dict[str, str]) -> None:
    exact = {
        "optimal_length": "optimal_length", "generated_routes": "generated_routes",
        "unique_candidates": "selected_routes", "mixed_best": "mixed_best",
        "ordinary_best": "two_opt_best", "ordinary_hits": "two_opt_optimal_hits",
        "ordinary_best_route_count": "two_opt_best_route_count",
    }
    for current, old in exact.items():
        if int(row[current]) != int(previous[old]):
            raise RuntimeError(f"{row['instance']}: ordinary baseline changed ({current})")
    for current, old in (("ordinary_fbeta", "two_opt_fbeta"),
                         ("ordinary_precision", "two_opt_precision"),
                         ("ordinary_recall", "two_opt_recall")):
        if abs(float(row[current]) - float(previous[old])) > 1e-12:
            raise RuntimeError(f"{row['instance']}: ordinary baseline changed ({current})")


def _write_report(rows: list[dict[str, object]], summary: dict[str, object], path: Path) -> None:
    def cell(value: object, color: str | None = None, head: bool = False) -> str:
        tag = "th" if head else "td"
        style = f' style="color:{color}"' if color else ""
        return f"<{tag}{style}>{escape(str(value))}</{tag}>"

    def table(headers: list[str], values: list[list[object]], colors: list[str | None]) -> list[str]:
        lines = ['<table border="1" cellspacing="0" cellpadding="6">',
                 '<thead><tr>' + ''.join(cell(h, colors[i], True) for i, h in enumerate(headers)) + '</tr></thead>',
                 '<tbody>']
        for value in values:
            lines.append('<tr>' + ''.join(cell(v, colors[i]) for i, v in enumerate(value)) + '</tr>')
        return [*lines, '</tbody>', '</table>', '']

    lines = [
        '# Soft POMO：普通 2-opt 与粒球边筛选 2-opt', '',
        '同一检查点、CPU、种子 42；每起点 1 条 Argmax + 8 条 Sampling。'
        '每例只生成一次路线，全部去重候选分别交给两种后处理。',
        '粒球版仅检查“待断开的两条边中至少一条跨球”的交换；'
        '普通版检查全部交换。两者均采用严格缩短的最佳改进，直到无法继续缩短。',
        '**这是粒球边筛选消融，并非论文完整 GB-MACO 局部搜索**：'
        '它不包含蚁群演化中的动态关键边、FES 和候选队列。',
        '`.solution` 只在两种搜索结束后用于评价。蓝色为参考值，绿色为粒球版结果。', '',
        f"25 例搜索路线耗时 {summary['search_seconds']:.2f}s；普通 2-opt "
        f"{summary['ordinary_seconds']:.2f}s；粒球版 {summary['granular_seconds']:.2f}s。",
        f"后 13 例达到参考最优距离：普通版 {summary['ordinary_composite_optimal_cases']}/13，"
        f"粒球版 {summary['granular_composite_optimal_cases']}/13；"
        f"25 例平均 Fβ：{summary['ordinary_mean_fbeta']:.4f} → "
        f"{summary['granular_mean_fbeta']:.4f}。", '',
        '## 距离、交换检查与耗时', '',
    ]
    first = []
    second = []
    for row in rows:
        first.append([
            row['instance'], row['optimal_length'], row['generated_routes'],
            row['unique_candidates'], row['ordinary_best'], row['granular_best'],
            f"{row['ordinary_gap_pct']:.4f}", f"{row['granular_gap_pct']:.4f}",
            row['ordinary_pair_checks'], row['granular_pair_checks'],
            row['granular_filtered_pairs'], f"{row['ordinary_seconds']:.3f}",
            f"{row['granular_seconds']:.3f}",
        ])
        second.append([
            row['instance'], row['ordinary_best_route_count'], row['granular_best_route_count'],
            row['ordinary_hits'], row['granular_hits'], row['truth_solution_count'],
            f"{row['ordinary_precision']:.4f}", f"{row['granular_precision']:.4f}",
            f"{row['ordinary_recall']:.4f}", f"{row['granular_recall']:.4f}",
            f"{row['ordinary_fbeta']:.4f}", f"{row['granular_fbeta']:.4f}",
            f"{row['paper_fbeta']:.4f}",
        ])
    lines.extend(table(
        ['算例', '参考最优', '原生成', '实际处理', '普通距离', '粒球距离',
         '普通Gap %', '粒球Gap %', '普通检查', '粒球检查', '粒球跳过',
         '普通2-opt s', '粒球2-opt s'], first,
        [None, '#38BDF8', None, None, None, '#34D399', None, '#34D399',
         None, '#34D399', '#34D399', None, '#34D399'],
    ))
    lines.extend(['## 最优回路覆盖', '',
                  '“最短去重”只统计最终最短距离对应的回路。'
                  'Precision = 真最优命中 / 最短去重；'
                  'Recall = 真最优命中 / 真实最优总数。', ''])
    lines.extend(table(
        ['算例', '普通最短去重', '粒球最短去重', '普通命中', '粒球命中',
         '真最优总数', '普通P', '粒球P', '普通R', '粒球R',
         '普通Fβ', '粒球Fβ', '论文Fβ'], second,
        [None, None, '#34D399', None, '#34D399', '#38BDF8',
         None, '#34D399', None, '#34D399', None, '#34D399', '#38BDF8'],
    ))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare granular edge filtering with ordinary 2-opt")
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--benchmark-dir', required=True, type=Path)
    parser.add_argument('--paper-csv', required=True, type=Path)
    parser.add_argument('--baseline-csv', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--samples-per-start', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.samples_per_start < 1 or args.output_dir.exists():
        parser.error('invalid samples-per-start or output directory already exists')
    device = select_device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(1)
    policy, state = load_policy(args.checkpoint, device)
    if policy.candidate_mode != 'soft':
        parser.error('checkpoint must use Soft guidance')
    paper = {row['instance']: float(row['paper_fbeta']) for row in
             csv.DictReader(args.paper_csv.open(encoding='utf-8-sig'))}
    baseline = {row['instance']: row for row in
                csv.DictReader(args.baseline_csv.open(encoding='utf-8-sig'))}
    if len(baseline) != 25:
        parser.error('baseline must contain 25 cases')
    for stem, _, _ in BENCHMARKS:
        for suffix in ('.tsp', '.solution'):
            if not (args.benchmark_dir / f'{stem}{suffix}').is_file():
                parser.error(f'missing benchmark input: {stem}{suffix}')

    args.output_dir.mkdir(parents=True)
    started = perf_counter()
    rows = []
    for index in range(1, 26):
        row = _run_case(index, args.benchmark_dir, policy, args.seed,
                        args.samples_per_start, paper[f'MSTSP{index}'])
        _check_existing(row, baseline[row['instance']])
        rows.append(row)
        print(f"{row['instance']}: {row['ordinary_best']} -> {row['granular_best']}, "
              f"checks {row['ordinary_pair_checks']} -> {row['granular_pair_checks']}", flush=True)

    with (args.output_dir / 'per_instance.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    composite = rows[12:]
    summary = {
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'checkpoint_epoch': state.get('epoch'), 'device': str(device),
        'seed': args.seed, 'samples_per_start': args.samples_per_start,
        'instances': len(rows), 'batch_seconds': perf_counter() - started,
        'search_seconds': sum(float(r['search_seconds']) for r in rows),
        'ordinary_seconds': sum(float(r['ordinary_seconds']) for r in rows),
        'granular_seconds': sum(float(r['granular_seconds']) for r in rows),
        'ordinary_pair_checks': sum(int(r['ordinary_pair_checks']) for r in rows),
        'granular_pair_checks': sum(int(r['granular_pair_checks']) for r in rows),
        'granular_filtered_pairs': sum(int(r['granular_filtered_pairs']) for r in rows),
        'ordinary_composite_optimal_cases': sum(r['ordinary_best'] == r['optimal_length'] for r in composite),
        'granular_composite_optimal_cases': sum(r['granular_best'] == r['optimal_length'] for r in composite),
        'ordinary_mean_fbeta': sum(float(r['ordinary_fbeta']) for r in rows) / len(rows),
        'granular_mean_fbeta': sum(float(r['granular_fbeta']) for r in rows) / len(rows),
        'ordinary_optimal_hits': sum(int(r['ordinary_hits']) for r in rows),
        'granular_optimal_hits': sum(int(r['granular_hits']) for r in rows),
        'method': 'cross-ball edge filter; same best-improvement and integer distances',
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    _write_report(rows, summary, args.output_dir / 'comparison.md')
    print(f'Results: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
