"""计算论文指标并汇总多次实验。

脚本读取 ``all_runs.csv`` 与基准数据集的真实最优解 ``.solution``，
按论文 Eq.(13)、Eq.(20)-(23) 计算每次运行的 Fβ 和 DI，再按实例统计
均值、样本标准差、最好/最差值、路线代价及运行时间。
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable

from common import instance_sort_key, load_config, project_root, result_root
from parse_results import FIELDS, parse_solution


BETA_SQUARED = 0.3
SUMMARY_FIELDS = [
    "instance", "type", "city_count", "runs", "Fbeta_mean", "Fbeta_std",
    "Fbeta_best", "Fbeta_worst", "DI_mean", "DI_std", "DI_best", "DI_worst",
    "route_cost_mean", "route_cost_best", "elapsed_mean",
]


def canonical_cycle(route: Iterable[int]) -> tuple[int, ...]:
    """把闭合路线转换为不受起点和行进方向影响的唯一表示。

    同一条对称 TSP 路线即使从不同城市开始，或按相反方向书写，也会
    得到相同结果，避免重复统计 TP、FP 和 FN。
    """
    values = tuple(route)
    if len(values) > 1 and values[0] == values[-1]:
        values = values[:-1]
    if not values:
        return ()
    rotations = [values[i:] + values[:i] for i in range(len(values))]
    reversed_values = tuple(reversed(values))
    rotations.extend(reversed_values[i:] + reversed_values[:i] for i in range(len(values)))
    return min(rotations)


def load_ground_truth(path: Path, city_count: int) -> set[tuple[int, ...]]:
    """读取一个实例的全部真实最优路线，并规范化、去重。"""
    routes, _ = parse_solution(path, city_count)
    return {canonical_cycle(route) for route in routes}


def route_set_from_output(path: str, city_count: int) -> set[tuple[int, ...]]:
    """读取一次算法输出，将返回路线规范化为集合。"""
    if not path:
        return set()
    routes, _ = parse_solution(Path(path), city_count)
    return {canonical_cycle(route) for route in routes}


def edge_set(route: tuple[int, ...]) -> set[tuple[int, int]]:
    """把一条闭合路线转换成无向边集合，末尾自动连接回起点。"""
    return {
        tuple(sorted((route[index], route[(index + 1) % len(route)])))
        for index in range(len(route))
    }


def similarity(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    """按论文 Eq.(13) 计算两条路线的无向共享边比例。"""
    if not first or not second or len(first) != len(second):
        return 0.0
    return len(edge_set(first) & edge_set(second)) / len(first)


def fbeta(returned: set[tuple[int, ...]], truth: set[tuple[int, ...]]) -> float | None:
    """按论文 Eq.(20)-(22) 计算 Fβ，其中 β² 固定为 0.3。

    ``returned`` 是算法返回路线集合，``truth`` 是真实最优路线集合；
    两者交集为 TP，仅 returned 中存在的是 FP，仅 truth 中存在的是 FN。
    """
    if not truth:
        return None
    tp = len(returned & truth)
    fp = len(returned - truth)
    fn = len(truth - returned)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    denominator = BETA_SQUARED * precision + recall
    return (1.0 + BETA_SQUARED) * precision * recall / denominator if denominator else 0.0


def diversity_indicator(returned: set[tuple[int, ...]], truth: set[tuple[int, ...]]) -> float | None:
    """按论文 Eq.(23) 和 Eq.(13) 计算多样性指标 DI。

    对每条真实最优路线，找出算法返回集合中与它最相似的路线，再对这些
    最大相似度取平均。它等价于先按共享边数除以 N，再除以真值路线数，
    因而与论文 Fig.4 使用的 [0, 1] 范围一致。
    """
    if not truth:
        return None
    if not returned:
        return 0.0
    return sum(max(similarity(target, candidate) for candidate in returned) for target in truth) / len(truth)


def numeric(rows: list[dict[str, str]], key: str) -> list[float]:
    """从 CSV 行中提取指定字段的有效有限数值，自动忽略空值。"""
    return [float(row[key]) for row in rows if row.get(key) not in (None, "") and math.isfinite(float(row[key]))]


def stats(values: list[float]) -> tuple[object, object, object, object]:
    """返回均值、样本标准差、最大值和最小值；单样本标准差留空。"""
    if not values:
        return "", "", "", ""
    standard_deviation: object = statistics.stdev(values) if len(values) > 1 else ""
    return statistics.mean(values), standard_deviation, max(values), min(values)


def main() -> int:
    """回填逐次指标，并生成每个实例一行的 instance_summary.csv。"""
    config = load_config()
    parsed_path = result_root(config) / "parsed" / "all_runs.csv"
    if not parsed_path.is_file():
        raise FileNotFoundError(f"Run parse_results.py first: {parsed_path}")
    with parsed_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    truth_cache: dict[str, set[tuple[int, ...]]] = {}
    benchmark_root = project_root(config) / config["benchmark_dir"]
    for row in rows:
        instance = row["instance"]
        instance_data = config["instances"][instance]
        city_count = int(instance_data["city_count"])
        if instance not in truth_cache:
            truth_cache[instance] = load_ground_truth(
                benchmark_root / instance_data["solution_file"], city_count
            )
        # 每次 run 必须使用本实例对应的 ground truth，不能跨实例比较。
        returned = route_set_from_output(row["output_file"], city_count)
        fbeta_value = fbeta(returned, truth_cache[instance])
        di_value = diversity_indicator(returned, truth_cache[instance])
        row["Fbeta"] = "" if fbeta_value is None else fbeta_value
        row["DI"] = "" if di_value is None else di_value

    with parsed_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # 只有退出码为 0 的真实成功运行才进入均值和标准差统计。
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("exit_code") == "0":
            grouped.setdefault(row["instance"], []).append(row)

    summary_rows: list[dict[str, object]] = []
    for instance in sorted(grouped, key=instance_sort_key):
        instance_rows = grouped[instance]
        info = config["instances"][instance]
        f_mean, f_std, f_best, f_worst = stats(numeric(instance_rows, "Fbeta"))
        d_mean, d_std, d_best, d_worst = stats(numeric(instance_rows, "DI"))
        costs = numeric(instance_rows, "route_cost")
        elapsed = numeric(instance_rows, "elapsed_seconds")
        summary_rows.append({
            "instance": instance,
            "type": info["type"],
            "city_count": info["city_count"],
            "runs": len(instance_rows),
            "Fbeta_mean": f_mean,
            "Fbeta_std": f_std,
            "Fbeta_best": f_best,
            "Fbeta_worst": f_worst,
            "DI_mean": d_mean,
            "DI_std": d_std,
            "DI_best": d_best,
            "DI_worst": d_worst,
            "route_cost_mean": statistics.mean(costs) if costs else "",
            "route_cost_best": min(costs) if costs else "",
            "elapsed_mean": statistics.mean(elapsed) if elapsed else "",
        })

    summary_path = result_root(config) / "summary" / "instance_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Metrics updated for {len(rows)} run(s): {parsed_path}")
    print(f"Summarized {len(summary_rows)} instance(s): {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
