"""Small Markdown report for the standalone Opt-B run."""

from __future__ import annotations

from pathlib import Path


def write_report(rows: list[dict[str, object]], summary: dict[str, object], path: Path) -> None:
    a, b = summary["groups"]["a"], summary["groups"]["b"]
    lines = [
        "# Soft POMO + 普通 2-opt + Opt-B", "",
        "每例一次 Soft POMO 混合搜索（每起点 1 Argmax + 8 Sampling），"
        "全部去重原始路线做普通 2-opt，随后从最佳回路出发广度优先探索等长换边。",
        f"设备：CPU；种子：42；每例 Opt-B 换边检查上限："
        f"{summary['extra_pair_check_ceiling_per_instance']:,}；"
        f"输入方式：{summary['input_mode']}。",
        "`.solution` 只在搜索结束后用于评价。参考距离来自 `.solution`，论文对照列仅为 Fβ。", "",
        "| 指标 | 普通 2-opt（A） | + Opt-B（B） | 论文对照 |",
        "|---|---:|---:|---:|",
        f"| 达参考最优距离 | {a['optimal_distance_cases']}/{len(rows)} | "
        f"{b['optimal_distance_cases']}/{len(rows)} | — |",
        f"| 真最优回路命中总数 | {a['optimal_hits']} | {b['optimal_hits']} | — |",
        f"| 平均 Fβ | {a['mean_fbeta']:.4f} | {b['mean_fbeta']:.4f} | "
        f"{summary['paper_mean_fbeta']:.4f} |",
        f"| 换边检查 | {a['pair_checks']:,} | 额外 {b['pair_checks']:,} | — |",
        f"| 局部搜索耗时 | {a['seconds']:.2f}s | 额外 {b['seconds']:.2f}s | — |",
        "", "## 距离与最优回路覆盖", "",
        "下表各组只保留最佳距离的不同回路，按旋转与反向等价去重。", "",
        "| 算例 | 参考距离 | A 距离 | B 距离 | A/B 最短回路数 | A/B 真最优命中 | 真实最优总数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['instance']} | {row['reference_length']} | {row['a_best']} | "
            f"{row['b_best']} | {row['a_routes']}/{row['b_routes']} | "
            f"{row['a_hits']}/{row['b_hits']} | {row['truth_count']} |"
        )
    lines.extend((
        "", "## Fβ 对照", "",
        "提升 = B − A；相对论文 = B − 论文 Fβ，负值表示仍低于论文对照表。", "",
        "| 算例 | A Fβ | B Fβ | 提升 | 论文 Fβ | 相对论文 | B 检查数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ))
    for row in rows:
        a_score, b_score, paper = (float(row["a_fbeta"]), float(row["b_fbeta"]),
                                   float(row["paper_fbeta"]))
        lines.append(
            f"| {row['instance']} | {a_score:.4f} | {b_score:.4f} | "
            f"{b_score - a_score:+.4f} | {paper:.4f} | {b_score - paper:+.4f} | "
            f"{row['b_checks']} |"
        )
    lines.extend(("", "原始路线和 A/B 最佳回路分别保存在 `raw_routes/`、`best_routes/`；"
                  "逐例完整数值在 `per_instance.csv`。", ""))
    path.write_text("\n".join(lines), encoding="utf-8")
