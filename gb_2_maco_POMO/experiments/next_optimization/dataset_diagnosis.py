"""Compare original A0 train/validation geometry with MSTSP21–25; no training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from gb_maco_POMO.experiments.compare import load_ground_truth
from gb_2_maco_POMO.experiments.hierarchical.original25 import BENCHMARKS, DATA
from gb_2_maco_POMO.experiments.shared.data import load_manifest
from gb_2_maco_POMO.runner import prepare_tsp

PROJECT = Path(__file__).resolve().parents[3]
ORIGINAL = PROJECT / "benachmark_Train_MATSP_500"
OUTPUT = PROJECT / "gb_2_maco_POMO/results/next_optimization/dataset_diagnosis"


def _write_csv(path, rows, fieldnames=None):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _spatial_clusters(cities):
    """A conservative, label-free MST gap heuristic; not granular balls."""
    n = len(cities)
    d = [[math.hypot(a.x - b.x, a.y - b.y) for b in cities] for a in cities]
    # Prim's MST, with deterministic index tie-breaking.
    connected = {0}
    mst = []
    nearest = [(d[0][j], 0, j) for j in range(n)]
    while len(connected) < n:
        edge = min((nearest[j] for j in range(n) if j not in connected),
                   key=lambda x: (x[0], x[1], x[2]))
        length, left, right = edge
        connected.add(right)
        mst.append(edge)
        for j in range(n):
            if j not in connected and (d[right][j], right, j) < nearest[j]:
                nearest[j] = (d[right][j], right, j)
    threshold = 3 * median(edge[0] for edge in mst)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for length, a, b in mst:
        if length <= threshold:
            parent[find(a)] = find(b)
    components = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)
    # A lone outlier is counted as noise, not as an additional spatial cluster.
    clusters = sorted((v for v in components.values() if len(v) >= 3),
                      key=lambda group: min(group))
    noise = n - sum(map(len, clusters))
    assignment = {city: group for group, cities_ in enumerate(clusters) for city in cities_}
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if i not in assignment or j not in assignment:
                continue
            (within if assignment[i] == assignment[j] else between).append(d[i][j])
    within_median = median(within) if within else None
    between_median = median(between) if between else None
    ratio = (between_median / within_median if within_median and between_median else None)
    return len(clusters), [len(c) for c in clusters], noise, threshold, within_median, between_median, ratio


def _measure(path, split, identity, distribution):
    problem = prepare_tsp(path)
    k, sizes, noise, threshold, within, between, ratio = _spatial_clusters(problem.cities)
    balls = problem.granular_balls
    ball_sizes = sorted(Counter(problem.city_to_ball).values())
    adjacency = sum(len(v) for v in problem.ball_adjacency.values()) // 2
    overlap = sum(math.hypot(a.center.x-b.center.x, a.center.y-b.center.y) < a.radius+b.radius
                  for i, a in enumerate(balls) for b in balls[i+1:])
    return {"split": split, "instance": identity, "distribution": distribution,
        "city_count": len(problem.cities), "ball_count": len(balls),
        "ball_sizes": json.dumps(ball_sizes), "ball_adjacency_edges": adjacency,
        "ball_overlap_pairs": overlap, "spatial_cluster_count": k,
        "spatial_cluster_sizes": json.dumps(sizes), "spatial_noise_cities": noise,
        "mst_gap_threshold": threshold, "within_cluster_median_distance": within or "",
        "between_cluster_median_distance": between or "",
        "between_within_ratio": ratio or "",
        "known_optimal_tours_at_least": "", "optimal_proof_method": "",
        "full_optimal_set_certified": "",
        "tsp_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run(output: Path = OUTPUT):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    train, validation = load_manifest({"root": str(ORIGINAL),
                                        "train_dir": "train", "validation_dir": "validation"})
    if len(train) != 500 or len(validation) != 100:
        raise RuntimeError("A0 original split is not 500/100")
    rows = []
    for split, partition in (("train", train), ("validation", validation)):
        for item in partition:
            rows.append(_measure(item.path, split, item.path.stem, item.distribution))
    for index in range(21, 26):
        stem, n, _ = BENCHMARKS[index - 1]
        tsp = DATA / f"{stem}.tsp"
        truth = load_ground_truth(DATA / f"{stem}.solution", n)
        row = _measure(tsp, "test", f"MSTSP{index}", "benchmark")
        row["known_optimal_tours_at_least"] = len(truth)
        row["optimal_proof_method"] = "benchmark .solution (known set)"
        rows.append(row)
    output.mkdir(parents=True)
    _write_csv(output / "structural_features.csv", rows)
    def target(r):
        sizes = json.loads(r["spatial_cluster_sizes"])
        return (50 <= r["city_count"] <= 70 and 4 <= r["spatial_cluster_count"] <= 6
                and r["spatial_noise_cities"] <= 2 and len(set(sizes)) > 1
                and r["between_within_ratio"] != "" and r["between_within_ratio"] >= 3)
    distribution = []
    for split in ("train", "validation", "test"):
        part = [r for r in rows if r["split"] == split]
        relevant = [r for r in part if 50 <= r["city_count"] <= 70]
        matched = [r for r in part if target(r)]
        distribution.append({"split": split, "total": len(part),
            "n50_to70": len(relevant), "clear_4to6_cluster_unequal": len(matched),
            "matched_with_known_multiple_optima": sum(
                r["known_optimal_tours_at_least"] != "" and
                r["known_optimal_tours_at_least"] >= 2 for r in matched),
            "known_multiple_optima_any_structure": sum(
                r["known_optimal_tours_at_least"] != "" and
                r["known_optimal_tours_at_least"] >= 2 for r in part),
            "median_balls_n50_to70": median(r["ball_count"] for r in relevant) if relevant else "",
            "median_spatial_clusters_n50_to70": median(r["spatial_cluster_count"] for r in relevant) if relevant else ""})
    _write_csv(output / "distribution_summary.csv", distribution)
    _write_csv(output / "target_like_cases.csv", [r for r in rows if target(r)], list(rows[0]))
    _write_report(output, rows, distribution)
    return distribution


def _write_report(output, rows, distribution):
    lines = ["# A0 训练集与 MSTSP21–25 结构诊断", "",
        "A0 源码指定 `benachmark_Train_MATSP_500`，本报告使用其实际 manifest 的 500/100 划分。"
        "空间簇由欧氏最小生成树断边启发式定义：断开长度大于 MST 边中位数 3 倍的边，"
        "只把至少 3 城市的连通块计作簇。它与项目粒球划分是两个独立对象，簇数是可复核的近似指标。",
        "簇间/簇内距离比为不同簇城市对中位距离除以同簇城市对中位距离。"
        "“目标相近”要求 50–70 城、4–6 簇、簇大小不等、噪声城市不超过 2、距离比至少 3。",
        "A0 原 500/100 数据没有可直接关联的已认证多解标注；已有区域监督标注包与其坐标 SHA256 无重合。"
        "因此空白表示未知，不能当作只有一解；MSTSP 的 `.solution` 仅提供已知回路集合。", "",
        "## 分布", "",
        "| 划分 | 总例数 | 50–70 城 | 目标相近结构 | 其中已有多最优标注 | 50–70 城粒球数中位 | 空间簇数中位 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in distribution:
        lines.append(f"| {r['split']} | {r['total']} | {r['n50_to70']} | "
                     f"{r['clear_4to6_cluster_unequal']} | {r['matched_with_known_multiple_optima']} | "
                     f"{r['median_balls_n50_to70']} | {r['median_spatial_clusters_n50_to70']} |")
    lines += ["", "## 困难算例", "",
              "| 算例 | 城市 | 粒球数/大小 | 空间簇数/大小 | 簇间/簇内比 | 球邻接/重叠边 | 已知最优回路 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["split"] != "test":
            continue
        lines.append(f"| {r['instance']} | {r['city_count']} | {r['ball_count']}/{r['ball_sizes']} | "
                     f"{r['spatial_cluster_count']}/{r['spatial_cluster_sizes']} | "
                     f"{r['between_within_ratio'] if r['between_within_ratio'] != '' else '—'} | "
                     f"{r['ball_adjacency_edges']}/{r['ball_overlap_pairs']} | "
                     f"{r['known_optimal_tours_at_least']} |")
    lines += ["", "完整逐例数据见 `structural_features.csv`；目标相近实例见 `target_like_cases.csv`。"
              "这些结构统计不能证明单个训练例存在复杂跨簇最优连接。", ""]
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for row in run(args.output_dir):
        print(row)


if __name__ == "__main__":
    main()
