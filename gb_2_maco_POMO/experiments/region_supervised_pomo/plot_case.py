"""Plot saved MSTSP geometry and route stages without running inference again."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from gb_maco_POMO.experiments.compare import load_ground_truth
from gb_2_maco_POMO.experiments.hierarchical import original25
from gb_2_maco_POMO.experiments.shared.opt_b_search import tour_length
from gb_2_maco_POMO.runner import prepare_tsp

from .config import RESULTS

COLORS = plt.get_cmap("tab10").colors
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def _edges(route):
    return {tuple(sorted((route[i], route[(i + 1) % len(route)])))
            for i in range(len(route))}


def _paths(mode: str, train_seed: int | None, inference_seed: int, case: int):
    run = (RESULTS / "tests" / mode /
           (f"train_seed_{train_seed}" if train_seed is not None else "initial") /
           f"infer_seed_{inference_seed}")
    name = f"MSTSP{case}"
    return (run / f"case_{case:02d}.json",
            run / mode / "raw_routes" / f"{name}.json",
            run / mode / "best_routes" / f"{name}.json")


def _points(ax, problem, *, labels=False, circles=False):
    for ball_id in sorted(set(problem.city_to_ball)):
        cities = [city for city in problem.cities
                  if problem.city_to_ball[city.number] == ball_id]
        color = COLORS[ball_id % len(COLORS)]
        ax.scatter([city.x for city in cities], [city.y for city in cities],
                   c=[color], s=28, edgecolors="white", linewidths=0.45,
                   zorder=3, label=f"粒球 {ball_id}（{len(cities)} 城市）")
        if circles:
            ball = problem.granular_balls[ball_id]
            ax.add_patch(Circle((ball.center.x, ball.center.y), ball.radius,
                                fill=False, edgecolor=color, linestyle="--",
                                linewidth=1, alpha=0.75))
            ax.text(ball.center.x, ball.center.y, f"球{ball_id}", color=color,
                    fontsize=11, fontweight="bold", ha="center", va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75})
        if labels:
            for city in cities:
                ax.annotate(str(city.number), (city.x, city.y), xytext=(3, 3),
                            textcoords="offset points", fontsize=6)


def _setup_axes(axes, problem):
    xs = [city.x for city in problem.cities]
    ys = [city.y for city in problem.cities]
    pad = 0.06 * max(max(xs) - min(xs), max(ys) - min(ys))
    for ax in axes:
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.13)
        ax.tick_params(labelsize=7)


def _route(ax, problem, route, *, labels=False):
    mapping = problem.city_to_ball
    cities = problem.cities
    for a, b in zip(route, route[1:] + route[:1]):
        cross = mapping[a] != mapping[b]
        ax.plot([cities[a].x, cities[b].x], [cities[a].y, cities[b].y],
                color="#E76F51" if cross else "#52667A",
                linewidth=1.8 if cross else 0.9, alpha=0.85, zorder=1)
    _points(ax, problem, labels=labels)
    return sum(mapping[a] != mapping[b]
               for a, b in zip(route, route[1:] + route[:1]))


def _difference(ax, problem, obtained, truth, *, labels=False):
    cities = problem.cities
    obtained_edges, truth_edges = _edges(obtained), _edges(truth)
    for edges, color, width, style in (
        (obtained_edges & truth_edges, "#B6BEC8", 0.8, "-"),
        (obtained_edges - truth_edges, "#CE4257", 2.2, "-"),
        (truth_edges - obtained_edges, "#2A9D8F", 2.2, "--"),
    ):
        for a, b in edges:
            ax.plot([cities[a].x, cities[b].x], [cities[a].y, cities[b].y],
                    color=color, linewidth=width, linestyle=style, zorder=1)
    differences = (obtained_edges - truth_edges) | (truth_edges - obtained_edges)
    affected = Counter(problem.city_to_ball[city]
                       for edge in differences for city in edge)
    zoom_ball = affected.most_common(1)[0][0] if affected else None
    _points(ax, problem, labels=labels or zoom_ball is not None)
    if zoom_ball is not None:
        local = [city for city in cities
                 if problem.city_to_ball[city.number] == zoom_ball]
        xs, ys = [city.x for city in local], [city.y for city in local]
        pad = max(15, 0.22 * max(max(xs) - min(xs), max(ys) - min(ys)))
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.legend(handles=[
        Line2D([0], [0], color="#CE4257", lw=2, label="当前路线独有的边"),
        Line2D([0], [0], color="#2A9D8F", lw=2, ls="--", label="已知最优路线独有的边"),
    ], loc="upper right", fontsize=8)
    return len(obtained_edges - truth_edges), zoom_ball


def plot_case(mode: str, train_seed: int | None, inference_seed: int,
              case: int, output_dir: Path, *, label_cities=False) -> Path:
    case_path, raw_path, best_path = _paths(mode, train_seed, inference_seed, case)
    for path in (case_path, raw_path, best_path):
        if not path.is_file():
            raise FileNotFoundError(f"saved test data missing: {path}")
    row = json.loads(case_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    saved = json.loads(best_path.read_text(encoding="utf-8"))
    stem, city_count, _ = original25.BENCHMARKS[case - 1]
    source = original25.DATA / f"{stem}.tsp"
    problem = prepare_tsp(source)
    if len(problem.cities) != city_count or len(raw) != city_count * 9:
        raise RuntimeError(f"case size or route budget mismatch: MSTSP{case}")
    raw_lengths = [tour_length(route, problem.distance_matrix) for route in raw]
    raw_index = min(range(len(raw)), key=lambda i: (raw_lengths[i], i))
    a_routes, b_routes = saved["A"]["routes"], saved["B"]["routes"]
    if (raw_lengths[raw_index] != row["raw_best"] or
            saved["A"]["best_length"] != row["a_best"] or
            saved["B"]["best_length"] != row["b_best"] or
            not a_routes or not b_routes):
        raise RuntimeError(f"saved stage lengths disagree: MSTSP{case}")
    if (tour_length(a_routes[0], problem.distance_matrix) != row["a_best"] or
            tour_length(b_routes[0], problem.distance_matrix) != row["b_best"]):
        raise RuntimeError(f"representative route length disagrees: MSTSP{case}")
    truth = load_ground_truth(source.with_suffix(".solution"), city_count)
    representative = b_routes[0]
    representative_edges = _edges(representative)
    reference = min(truth, key=lambda route: (
        len(representative_edges - _edges(route)), route))
    if tour_length(reference, problem.distance_matrix) != row["reference_length"]:
        raise RuntimeError(f"reference distance mismatch: MSTSP{case}")

    fig, grid = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    axes = grid.ravel()
    _setup_axes(axes, problem)
    train_label = str(train_seed) if train_seed is not None else "初始模型"
    fig.suptitle(f"MSTSP{case}  |  模型 {mode}  |  训练种子 {train_label}  "
                 f"|  测试种子 {inference_seed}  |  {city_count} 个城市  |  "
                 f"{len(set(problem.city_to_ball))} 个粒球", fontsize=16)
    _points(axes[0], problem, labels=label_cities, circles=True)
    axes[0].set_title("① 城市分布：同色属于同一粒球")
    axes[0].legend(loc="upper right", fontsize=8)
    stages = (
        (1, raw[raw_index], f"② 模型直接生成的最短路线（{'贪心' if raw_index < city_count else '采样'}）",
         row["raw_best"], 1),
        (2, a_routes[0], "③ 2-opt 调整后的最短路线（展示其中一条）", row["a_best"], len(a_routes)),
        (3, representative, "④ Opt-B 处理后的最短路线（展示其中一条）", row["b_best"], len(b_routes)),
        (4, list(reference), "⑤ 最接近④的已知最优路线", row["reference_length"], len(truth)),
    )
    for panel, route, title, length, count in stages:
        crossings = _route(axes[panel], problem, list(route), labels=label_cities)
        gap = 100 * (length / row["reference_length"] - 1)
        axes[panel].set_title(f"{title}\n距离={length}  比最优长={gap:.4f}%  "
                              f"跨粒球次数={crossings}  同距离路线数={count}")
    difference_count, zoom_ball = _difference(
        axes[5], problem, representative, reference, labels=label_cities)
    location = f"放大粒球 {zoom_ball}" if zoom_ball is not None else "全图"
    axes[5].set_title(f"⑥ ④与⑤的差别：{location}\n"
                      f"当前路线独有 {difference_count} 条边；红色实线=当前，绿色虚线=最优")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"MSTSP{case:02d}.png"
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fixed_soft", "A0", "A", "B", "C"), required=True)
    parser.add_argument("--train-seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--inference-seed", type=int, choices=(42, 43, 44, 45, 46), required=True)
    cases = parser.add_mutually_exclusive_group(required=True)
    cases.add_argument("--cases", type=int, nargs="+", metavar="1..25")
    cases.add_argument("--all-cases", action="store_true")
    parser.add_argument("--label-cities", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if (args.model in ("A", "B", "C")) != (args.train_seed is not None):
        parser.error("A/B/C require --train-seed; fixed_soft/A0 use the initial run")
    selected = range(1, 26) if args.all_cases else args.cases
    if any(not 1 <= case <= 25 for case in selected):
        parser.error("case number must be from 1 to 25")
    target = args.output_dir or (RESULTS / "plots" / args.model /
             (f"train_seed_{args.train_seed}" if args.train_seed else "initial") /
             f"infer_seed_{args.inference_seed}")
    for case in selected:
        path = plot_case(args.model, args.train_seed, args.inference_seed,
                         case, target, label_cities=args.label_cities)
        print(path, flush=True)


if __name__ == "__main__":
    main()
