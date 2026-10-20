"""Chinese report assembled only from completed seed-42 measurements."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


FOCUS = (13, 16, 17, 19, 21, 22, 23, 24, 25)
GROUPS = ("S0", "S2", "S3", "S4")


def write_report(output: Path, summaries, cases, comparisons, timing, forks):
    by_case = defaultdict(dict)
    for row in cases:
        by_case[row["instance"]][row["group"]] = row
    by_comparison = {(r["instance"], r["group"]): r for r in comparisons}
    successful = {group: 0 for group in GROUPS}
    fallback = {group: 0 for group in GROUPS}
    for row in forks:
        kind = row["kind"]
        for group, use in (("S2", kind == "city"), ("S3", kind == "ball"),
                           ("S4", kind == "city" and row["slot"] < 2 or
                            kind == "ball" and row["slot"] < 2)):
            if use:
                target = successful if row["status"] == "forked" else fallback
                target[group] += 1
    summary = {r["group"]: r for r in summaries}
    lines = ["# 关键连接决策分叉采样：seed=42", "",
        "## 协议与边界", "",
        "A0 为同一短训练 checkpoint；CPU、全部城市起点、每起点 9 条完整路线。"
        "S2/S3/S4 共享 Argmax 和前 4 条普通 Sampling 参考路线。"
        "分叉只在原始路线的一个决策位置替换合法动作，后续仍按两级模型采样。",
        "普通 2-opt、Opt-B 每例 50,000 次边对检查及 canonical 去重直接调用项目原实现。"
        "测试最优解只在路线生成和后处理结束后用于统计。",
        "相同的是生成路线数量和后处理预算；分叉记录与额外模型计算会增加推理成本。"
        "仅一个推理种子，不能据此推断跨种子稳定性。", "",
        "## 总体结果", "",
        "| 组别 | 最优距离命中 | 2-opt 真最优回路 | Opt-B 真最优回路 | 最优区域覆盖 | 平均 Fβ | 推理+后处理秒 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        r = summary[group]
        lines.append(f"| {group} | {r['distance_hits']}/25 | {r['two_opt_tours']} | "
                     f"{r['optb_tours']} | {r['regions_hit']} | {r['mean_fbeta']:.6f} | "
                     f"{r['total_seconds']:.2f} |")
    lines += ["", "## 逐例：距离与 Gap", "",
              "每格依次为原始 / 2-opt / Opt-B 最短距离；下一表给出各阶段 Gap。",
              "Gap = (本组距离 / 参考最短距离 − 1) × 100%。", "",
              "| 算例 | 参考距离 | S0 | S2 | S3 | S4 |",
              "|---|---:|---:|---:|---:|---:|"]
    for i in range(1, 26):
        name = f"MSTSP{i}"
        rows = by_case[name]
        def distances(group):
            r = rows[group]
            return f"{r['raw_best']}/{r['a_best']}/{r['b_best']}"
        lines.append(f"| {name} | {rows['S0']['reference_length']} | " +
                     " | ".join(distances(group) for group in GROUPS) + " |")
    lines += ["", "| 算例 | S0 Gap % | S2 Gap % | S3 Gap % | S4 Gap % |",
              "|---|---:|---:|---:|---:|"]
    for i in range(1, 26):
        name = f"MSTSP{i}"
        def gaps(group):
            r = by_case[name][group]
            return f"{r['raw_gap_pct']:.2f}/{r['a_gap_pct']:.2f}/{r['b_gap_pct']:.2f}"
        lines.append(f"| {name} | " + " | ".join(gaps(g) for g in GROUPS) + " |")
    lines += ["", "## 逐例：真最优回路与区域", "",
              "每格依次为 2-opt 真最优回路 / Opt-B 真最优回路 / Opt-B 覆盖区域。"
              "回路经旋转和反向规范化；区域按已知真最优回路的等长换边连通分量计。", "",
              "| 算例 | 真最优回路/区域 | S0 | S2 | S3 | S4 |",
              "|---|---:|---:|---:|---:|---:|"]
    for i in range(1, 26):
        name = f"MSTSP{i}"
        rows = by_case[name]
        def coverage(group):
            r = rows[group]
            return f"{r['a_hits']}/{r['b_hits']}/{r['regions_hit']}"
        lines.append(f"| {name} | {rows['S0']['truth_count']}/{rows['S0']['truth_regions']} | " +
                     " | ".join(coverage(g) for g in GROUPS) + " |")
    lines += ["", "## 逐例：Fβ", "",
              "每格依次为 2-opt / Opt-B Fβ；论文只提供其实际记录的 Fβ，测试预算不完全相同。", "",
              "| 算例 | S0 | S2 | S3 | S4 | 论文 |",
              "|---|---:|---:|---:|---:|---:|"]
    for i in range(1, 26):
        name = f"MSTSP{i}"
        rows = by_case[name]
        def score(group):
            r = rows[group]
            return f"{r['a_fbeta']:.4f}/{r['b_fbeta']:.4f}"
        lines.append(f"| {name} | " + " | ".join(score(g) for g in GROUPS) +
                     f" | {rows['S0']['paper_fbeta']:.4f} |")
    lines += ["", "| 算例 | S0 Opt-B P/R | S2 Opt-B P/R | S3 Opt-B P/R | S4 Opt-B P/R | S0/S2/S3/S4 减论文 Fβ |",
              "|---|---:|---:|---:|---:|---:|"]
    for i in range(1, 26):
        name = f"MSTSP{i}"
        rows = by_case[name]
        pr = [f"{rows[g]['b_precision']:.4f}/{rows[g]['b_recall']:.4f}" for g in GROUPS]
        delta = "/".join(f"{rows[g]['b_fbeta'] - rows[g]['paper_fbeta']:+.4f}"
                         for g in GROUPS)
        lines.append(f"| {name} | " + " | ".join(pr) + f" | {delta} |")
    lines += ["", "## 相对 S0 的覆盖变化", "",
              "以下为逐例集合差异之和，新增和丢失分别计数；不同 2-opt 终点不等于独立吸引域。", "",
              "| 组别 | 新增 2-opt 终点 | 其中非全局最优 | 新增/丢失真最优回路 | 新增/丢失区域 | 分叉成功/退化 |",
              "|---|---:|---:|---:|---:|---:|"]
    for group in GROUPS[1:]:
        rows = [r for r in comparisons if r["group"] == group]
        total = lambda key: sum(r[key] for r in rows)
        lines.append(f"| {group} | {total('new_two_opt_endpoints')} | "
                     f"{total('new_nonoptimal_two_opt_endpoints')} | "
                     f"{total('new_true_optimal_tours')}/{total('lost_true_optimal_tours')} | "
                     f"{total('new_regions')}/{total('lost_regions')} | "
                     f"{successful[group]}/{fallback[group]} |")
    lines += ["", "## 重点算例", "",
              "| 算例 | S0/S2/S3/S4 最终距离 | S0/S2/S3/S4 真最优回路 | S0/S2/S3/S4 区域 |",
              "|---|---:|---:|---:|"]
    for i in FOCUS:
        name = f"MSTSP{i}"
        rows = by_case[name]
        lines.append(f"| {name} | " + " / ".join(str(rows[g]["b_best"]) for g in GROUPS) +
                     " | " + " / ".join(str(rows[g]["b_hits"]) for g in GROUPS) +
                     " | " + " / ".join(str(rows[g]["regions_hit"]) for g in GROUPS) + " |")
    s0_inference = sum(r["s0_inference_seconds"] for r in timing)
    fork_seconds = sum(r["all_forks_seconds"] for r in timing)
    best_fbeta = max(GROUPS, key=lambda g: summary[g]["mean_fbeta"])
    best_routes = max(GROUPS, key=lambda g: summary[g]["optb_tours"])
    best_regions = max(GROUPS, key=lambda g: summary[g]["regions_hit"])
    lines += ["", "## 结论", "",
              f"本次平均 Fβ 最高：{best_fbeta}；真最优回路最多：{best_routes}；区域覆盖最多：{best_regions}。",
              f"本次 S0 推理约 {s0_inference:.2f} 秒；全部城市和粒球分叉生成约 {fork_seconds:.2f} 秒。"
              "各组推理时间由共同参考路线生成时间与所用分叉时段组成；"
              "耗时包含分叉记录成本，但不含为诊断重算的全部 2-opt 终点和区域归类。",
              "S0 历史结果已按逐条原始路线字节与 checkpoint SHA256 核对。"
              "路线终点、分叉位置与退化原因见同目录 CSV；本报告只对 seed=42 成立。", ""]
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
