"""Write a paired report using only metrics justified by this dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path


FIELDS = (
    "name", "tier", "shape", "n", "reference_length",
    "fixed_raw_best", "learned_raw_best", "fixed_a_best", "learned_a_best",
    "fixed_b_best", "learned_b_best", "fixed_b_gap_pct", "learned_b_gap_pct",
    "fixed_a_proven_optimal_routes", "learned_a_proven_optimal_routes",
    "fixed_b_proven_optimal_routes", "learned_b_proven_optimal_routes",
    "fixed_total_seconds", "learned_total_seconds",
)


def write_comparison(output: Path, cases: list[dict[str, str]]) -> None:
    saved = {}
    for model in ("fixed_soft", "learnable_bias"):
        saved[model] = {}
        for case in cases:
            path = output / model / "cases" / f"{case['name']}.json"
            if not path.is_file():
                raise RuntimeError(f"Cannot make a 130-case comparison; missing {path}")
            saved[model][case["name"]] = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    for case in cases:
        a = saved["fixed_soft"][case["name"]]
        b = saved["learnable_bias"][case["name"]]
        for key in ("file", "n", "reference_length", "starts", "samples_per_start",
                    "raw_routes"):
            if a[key] != b[key]:
                raise ValueError(f"{case['name']}: models used different inputs/budgets ({key})")
        rows.append({
            "name": case["name"], "tier": case["tier"],
            "shape": case["shape"], "n": a["n"],
            "reference_length": a["reference_length"],
            "fixed_raw_best": a["raw_best"], "learned_raw_best": b["raw_best"],
            "fixed_a_best": a["a_best"], "learned_a_best": b["a_best"],
            "fixed_b_best": a["b_best"], "learned_b_best": b["b_best"],
            "fixed_b_gap_pct": a["b_gap_pct"], "learned_b_gap_pct": b["b_gap_pct"],
            "fixed_a_proven_optimal_routes": a["a_proven_optimal_routes"],
            "learned_a_proven_optimal_routes": b["a_proven_optimal_routes"],
            "fixed_b_proven_optimal_routes": a["b_proven_optimal_routes"],
            "learned_b_proven_optimal_routes": b["b_proven_optimal_routes"],
            "fixed_total_seconds": a["total_seconds"],
            "learned_total_seconds": b["total_seconds"],
        })
    with (output / "paired.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for model in ("fixed_soft", "learnable_bias"):
        values = [saved[model][case["name"]] for case in cases]
        summary[model] = {
            "cases": len(values),
            "raw_optimal_distance_hits": sum(r["raw_best"] == r["reference_length"] for r in values),
            "after_2opt_optimal_distance_hits": sum(r["a_best"] == r["reference_length"] for r in values),
            "after_optb_optimal_distance_hits": sum(r["b_best"] == r["reference_length"] for r in values),
            "mean_raw_gap_pct": sum(r["raw_gap_pct"] for r in values) / len(values),
            "mean_after_2opt_gap_pct": sum(r["a_gap_pct"] for r in values) / len(values),
            "mean_after_optb_gap_pct": sum(r["b_gap_pct"] for r in values) / len(values),
            "after_2opt_proven_optimal_routes": sum(r["a_proven_optimal_routes"] for r in values),
            "after_optb_proven_optimal_routes": sum(r["b_proven_optimal_routes"] for r in values),
            "inference_seconds": sum(r["inference_seconds"] for r in values),
            "two_opt_seconds": sum(r["two_opt_seconds"] for r in values),
            "optb_seconds": sum(r["optb_seconds"] for r in values),
            "total_seconds": sum(r["total_seconds"] for r in values),
        }
    summary["paired_wins"] = {
        "fixed_shorter_after_optb": sum(r["fixed_b_best"] < r["learned_b_best"] for r in rows),
        "same_distance_after_optb": sum(r["fixed_b_best"] == r["learned_b_best"] for r in rows),
        "learned_shorter_after_optb": sum(r["learned_b_best"] < r["fixed_b_best"] for r in rows),
        "fixed_more_optimal_routes_after_optb": sum(
            r["fixed_b_proven_optimal_routes"] > r["learned_b_proven_optimal_routes"] for r in rows),
        "same_optimal_routes_after_optb": sum(
            r["fixed_b_proven_optimal_routes"] == r["learned_b_proven_optimal_routes"] for r in rows),
        "learned_more_optimal_routes_after_optb": sum(
            r["learned_b_proven_optimal_routes"] > r["fixed_b_proven_optimal_routes"] for r in rows),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    f, l = summary["fixed_soft"], summary["learnable_bias"]
    lines = [
        "# 新验证集：固定 Soft-POMO 与可学习 Bias-POMO 对照", "",
        f"同一批 {len(rows)} 例，数据清单前 {len(rows)} 条；设备 {protocol['device']}，种子 {protocol['seed']}。",
        f"每例最多 {protocol['max_starts']} 个均匀起点，每起点 1 Argmax + "
        f"{protocol['samples_per_start']} Sampling；去重后按原始距离排序，最多 "
        f"{protocol['max_2opt_candidates']} 条做完整普通 2-opt；从最佳距离回路出发做 "
        f"Opt-B（每例最多 {protocol['optb_extra_pair_checks']:,} 次额外换边检查）。",
        "两模型使用相同算例、起点和预算；逐例实际数量、检查次数与耗时保存在各自 `cases/`。", "",
        "| 指标 | 固定 Soft | 可学习 Bias |", "|---|---:|---:|",
        f"| 原始路线命中最短距离 | {f['raw_optimal_distance_hits']}/{len(rows)} "
        f"({f['raw_optimal_distance_hits']/len(rows):.1%}) | "
        f"{l['raw_optimal_distance_hits']}/{len(rows)} ({l['raw_optimal_distance_hits']/len(rows):.1%}) |",
        f"| 普通 2-opt 后命中 | {f['after_2opt_optimal_distance_hits']}/{len(rows)} "
        f"({f['after_2opt_optimal_distance_hits']/len(rows):.1%}) | "
        f"{l['after_2opt_optimal_distance_hits']}/{len(rows)} "
        f"({l['after_2opt_optimal_distance_hits']/len(rows):.1%}) |",
        f"| Opt-B 后命中 | {f['after_optb_optimal_distance_hits']}/{len(rows)} "
        f"({f['after_optb_optimal_distance_hits']/len(rows):.1%}) | "
        f"{l['after_optb_optimal_distance_hits']}/{len(rows)} "
        f"({l['after_optb_optimal_distance_hits']/len(rows):.1%}) |",
        f"| Opt-B 后平均 Gap | {f['mean_after_optb_gap_pct']:.4f}% | {l['mean_after_optb_gap_pct']:.4f}% |",
        f"| 普通 2-opt 后不同真最优回路（逐例求和） | {f['after_2opt_proven_optimal_routes']} | {l['after_2opt_proven_optimal_routes']} |",
        f"| Opt-B 后不同真最优回路（逐例求和） | {f['after_optb_proven_optimal_routes']} | {l['after_optb_proven_optimal_routes']} |",
        f"| 推理耗时 | {f['inference_seconds']:.1f}s | {l['inference_seconds']:.1f}s |",
        f"| 普通 2-opt 耗时 | {f['two_opt_seconds']:.1f}s | {l['two_opt_seconds']:.1f}s |",
        f"| Opt-B 额外耗时 | {f['optb_seconds']:.1f}s | {l['optb_seconds']:.1f}s |",
        f"| 总耗时（逐例计时求和） | {f['total_seconds']:.1f}s | {l['total_seconds']:.1f}s |",
        "", "## 模型逐例胜负", "",
        f"Opt-B 后最佳距离：固定更短 {summary['paired_wins']['fixed_shorter_after_optb']} 例、"
        f"相同 {summary['paired_wins']['same_distance_after_optb']} 例、"
        f"可学习更短 {summary['paired_wins']['learned_shorter_after_optb']} 例。",
        f"Opt-B 后找到的不同真最优回路数：固定更多 "
        f"{summary['paired_wins']['fixed_more_optimal_routes_after_optb']} 例、"
        f"相同 {summary['paired_wins']['same_optimal_routes_after_optb']} 例、"
        f"可学习更多 {summary['paired_wins']['learned_more_optimal_routes_after_optb']} 例。",
        "", "## 逐例距离和多解", "",
        "A = 普通 2-opt；B = A 后接 Opt-B。真最优回路按旋转/反向等价去重，"
        "并用已证明的最短距离判定，不要求出现在部分证明回路中。", "",
        "| 算例 | 城市 | 已证最短 | 固定 A/B | 可学习 A/B | 固定 B Gap | 可学习 B Gap | 固定 B 最优回路 | 可学习 B 最优回路 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['n']} | {row['reference_length']} | "
            f"{row['fixed_a_best']}/{row['fixed_b_best']} | "
            f"{row['learned_a_best']}/{row['learned_b_best']} | "
            f"{row['fixed_b_gap_pct']:.3f}% | {row['learned_b_gap_pct']:.3f}% | "
            f"{row['fixed_b_proven_optimal_routes']} | {row['learned_b_proven_optimal_routes']} |"
        )
    lines += [
        "", "## 指标边界", "",
        "这套数据的 `manifest.csv` 给出精确全局最短距离，但 `proofs/` 仅列出部分最优回路，"
        "没有完整的最优回路全集。因此无法按旧 25 例报告的定义计算完整 Recall、Fβ。",
        "旧 25 例论文表的 Fβ 与本数据集不是同一批算例，不能作逐例论文对比。",
        "候选筛选采用相同上限；两个模型去重后的实际 2-opt 条数可以不同，逐例记录在 `cases/`。",
        "每个找到的、长度等于已证最短距离的合法不同回路都计为真最优；"
        "其数量是本次预算实际找到的数量，不是全体最优解数量。", "",
        "详情：`paired.csv`、`summary.json`、两模型的 `cases/`、`raw_routes/` 和 `best_routes/`。", "",
    ]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
