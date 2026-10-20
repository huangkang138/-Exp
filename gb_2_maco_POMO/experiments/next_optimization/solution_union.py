"""Union existing seed-42 Opt-B route sets without inference or local search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from gb_maco_POMO.experiments.compare import fbeta, load_ground_truth
from gb_2_maco_POMO.experiments.hierarchical.original25 import BENCHMARKS, DATA
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical

PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "gb_2_maco_POMO/results/decision_fork_sampling/seed_42"
HISTORY = PROJECT / "gb_2_maco_POMO/results/learnable_ball_test25_20261015"
OUTPUT = PROJECT / "gb_2_maco_POMO/results/next_optimization/solution_union"
COMBINATIONS = {
    "S0": ("S0",), "S0+S2": ("S0", "S2"),
    "S0+S3": ("S0", "S3"), "S0+S4": ("S0", "S4"),
    "ALL": ("S0", "S2", "S3", "S4"),
}


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(output: Path = OUTPUT):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    if config["seed"] != 42 or config["routes_per_start"] != 9:
        raise RuntimeError("source seed or route count changed")
    rows = _read_csv(SOURCE / "per_instance.csv")
    if len(rows) != 100:
        raise RuntimeError("expected 25 cases in each of four groups")
    saved = {(r["instance"], r["group"]): r for r in rows}
    if len(saved) != 100:
        raise RuntimeError("duplicate case/group records")
    # Region IDs were already computed during the original experiment.
    # Each group's optimal 2-opt seeds must cover exactly its saved Opt-B regions.
    region_sets = defaultdict(set)
    for r in _read_csv(SOURCE / "route_endpoints.csv"):
        if r["two_opt_region"]:
            region_sets[r["instance"], r["group"]].add(int(r["two_opt_region"]))
    output.mkdir(parents=True)
    (output / "merged_routes").mkdir()
    _save_json(output / "config.json", {
        "source": str(SOURCE), "source_config_sha256": hashlib.sha256(
            (SOURCE / "config.json").read_bytes()).hexdigest(),
        "historical_s0": str(HISTORY), "seed": 42,
        "method": "canonical union of saved Opt-B final B routes; no inference, 2-opt, or Opt-B",
        "region_labels": "saved route_endpoints.csv two_opt_region IDs; validated against each group's final regions_hit",
        "combinations": COMBINATIONS, "beta_squared": 0.3,
        "budget_note": "union combines multiple complete searches and exceeds the single-group budget",
    })
    per_instance = []
    for index, (stem, n, _) in enumerate(BENCHMARKS, 1):
        name = f"MSTSP{index}"
        truth = load_ground_truth(DATA / f"{stem}.solution", n)
        sets = {}
        for group in ("S0", "S2", "S3", "S4"):
            path = (HISTORY / "learned_ball/best_routes" / f"{name}.json") if group == "S0" else (
                SOURCE / group / "best_routes" / f"{name}.json")
            saved_routes = json.loads(path.read_text(encoding="utf-8"))
            result = {canonical(route) for route in saved_routes["B"]["routes"]}
            if len(result) != len(saved_routes["B"]["routes"]):
                raise RuntimeError(f"{name} {group}: duplicate canonical final tour")
            original = saved[name, group]
            if (len(result) != int(original["b_routes"]) or
                    len(result & truth) != int(original["b_hits"]) or
                    abs(fbeta(result, truth) - float(original["b_fbeta"])) > 1e-12 or
                    len(region_sets[name, group]) != int(original["regions_hit"])):
                raise RuntimeError(f"{name} {group}: saved route/region/metric mismatch")
            sets[group] = result
        base = sets["S0"] & truth
        base_regions = region_sets[name, "S0"]
        merged = {}
        for label, groups in COMBINATIONS.items():
            tours = set().union(*(sets[group] for group in groups))
            regions = set().union(*(region_sets[name, group] for group in groups))
            hits = tours & truth
            precision = len(hits) / len(tours) if tours else 0.0
            recall = len(hits) / len(truth)
            score = fbeta(tours, truth)
            per_instance.append({"instance": name, "combination": label,
                "truth_tours": len(truth), "returned_tours": len(tours),
                "true_optimal_tours": len(hits), "precision": precision,
                "recall": recall, "fbeta": score,
                "regions_hit": len(regions), "new_true_tours_vs_s0": len(hits - base),
                "new_regions_vs_s0": len(regions - base_regions)})
            merged[label] = {"groups": groups, "routes": [list(r) for r in sorted(tours)],
                             "true_optimal_routes": [list(r) for r in sorted(hits)],
                             "region_ids": sorted(regions)}
        _save_json(output / "merged_routes" / f"{name}.json", merged)
    summary = []
    for label in COMBINATIONS:
        group = [r for r in per_instance if r["combination"] == label]
        summary.append({"combination": label, "cases": len(group),
            "returned_tours": sum(r["returned_tours"] for r in group),
            "true_optimal_tours": sum(r["true_optimal_tours"] for r in group),
            "regions_hit": sum(r["regions_hit"] for r in group),
            "mean_precision": sum(r["precision"] for r in group) / 25,
            "mean_recall": sum(r["recall"] for r in group) / 25,
            "mean_fbeta": sum(r["fbeta"] for r in group) / 25,
            "new_true_tours_vs_s0": sum(r["new_true_tours_vs_s0"] for r in group),
            "new_regions_vs_s0": sum(r["new_regions_vs_s0"] for r in group)})
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "per_instance.csv", per_instance)
    write_report(output, summary, per_instance)
    return summary


def write_report(output, summary, rows):
    by = {(r["instance"], r["combination"]): r for r in rows}
    lines = ["# seed=42 已有最终回路集合合并", "",
        "仅读取四组保存的 Opt-B 最终 B 回路，按原 canonical 规则去重；真值来自各算例 `.solution`。"
        "没有重新推理、训练、运行 2-opt 或 Opt-B。区域编号取自原实验保存的 2-opt 终点记录，"
        "逐组与原结果区域数交叉核对。Fβ 继续使用原项目 β²=0.3 公式。", "",
        "**这些组合叠加了完整搜索结果，总搜索预算高于任一单组；不能作为同预算模型比较。**", "",
        "## 汇总", "",
        "| 组合 | 不同真最优回路 | 区域覆盖 | 平均 Precision | 平均 Recall | 平均 Fβ | 较 S0 新回路/区域 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['combination']} | {r['true_optimal_tours']} | {r['regions_hit']} | "
                     f"{r['mean_precision']:.6f} | {r['mean_recall']:.6f} | {r['mean_fbeta']:.6f} | "
                     f"{r['new_true_tours_vs_s0']}/{r['new_regions_vs_s0']} |")
    lines += ["", "## MSTSP1–25 逐例覆盖", "",
              "每格依次为真最优回路/区域/Fβ。", "",
              "| 算例 | S0 | S0∪S2 | S0∪S3 | S0∪S4 | 全部并集 |",
              "|---|---:|---:|---:|---:|---:|"]
    for i in range(1, 26):
        name = f"MSTSP{i}"
        def cell(label):
            r = by[name, label]
            return f"{r['true_optimal_tours']}/{r['regions_hit']}/{r['fbeta']:.4f}"
        lines.append(f"| {name} | " + " | ".join(cell(label) for label in COMBINATIONS) + " |")
    best = max(summary, key=lambda row: row["mean_fbeta"])
    lines += ["", "## 结论", "",
              f"最高平均 Fβ：{best['combination']}，{best['mean_fbeta']:.6f}；"
              f"比 S0 新增 {best['new_true_tours_vs_s0']} 条真最优回路、"
              f"{best['new_regions_vs_s0']} 个已知最优区域。", "",
              "`merged_routes/MSTSP*.json` 保存每例每种组合的全部最终回路和真最优回路。", ""]
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for row in run(args.output_dir):
        print(row["combination"], row["true_optimal_tours"], row["regions_hit"],
              f"Fbeta={row['mean_fbeta']:.6f}")


if __name__ == "__main__":
    main()
