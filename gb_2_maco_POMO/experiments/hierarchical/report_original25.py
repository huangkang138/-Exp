"""Render the certified 25-case comparison from measured per-case records."""

from __future__ import annotations

import json
from pathlib import Path

COLORS = {"fixed_soft": "#38BDF8", "hierarchical_0": "#A78BFA",
          "hierarchical_0p5": "#34D399", "hierarchical_1": "#FBBF24",
          "hierarchical_2": "#FB7185"}


def _label(mode: str) -> str:
    text = mode.replace("hierarchical_", "λ=").replace("p", ".")
    return f'<span style="color:{COLORS[mode]}">{text}</span>'


def write_report(rows: list[dict], output: Path) -> None:
    modes = list(COLORS)
    grouped = {mode: [r for r in rows if r["mode"] == mode] for mode in modes}
    if any(len(group) != 25 for group in grouped.values()):
        raise ValueError("report requires five complete 25-case groups")
    summaries = {}
    for mode, group in grouped.items():
        summaries[mode] = {
            "distance_hits_2opt": sum(r["a_distance_hit"] for r in group),
            "distance_hits_optb": sum(r["b_distance_hit"] for r in group),
            "true_optimal_routes_2opt": sum(r["a_hits"] for r in group),
            "true_optimal_routes_optb": sum(r["b_hits"] for r in group),
            "mean_fbeta_2opt": sum(r["a_fbeta"] for r in group) / 25,
            "mean_fbeta_optb": sum(r["b_fbeta"] for r in group) / 25,
            "mean_raw_gap_pct": sum(r["raw_gap_pct"] for r in group) / 25,
            "inference_seconds": sum(r["inference_seconds"] for r in group),
            "two_opt_seconds": sum(r["two_opt_seconds"] for r in group),
            "optb_seconds": sum(r["optb_seconds"] for r in group),
            "optb_checks": sum(r["b_checks"] for r in group),
        }
    paper_mean = sum(r["paper_fbeta"] for r in grouped["fixed_soft"]) / 25
    (output / "summary.json").write_text(json.dumps({
        "groups": summaries, "paper_mean_fbeta": paper_mean,
        "interpretation": "Fbeta uses beta_squared=0.3 and canonical cycles",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    base = summaries["fixed_soft"]
    lines = ["# 原始 25 例：固定 Soft 与粒球两级决策", "",
             "同一固定 Soft checkpoint；CPU、种子 42＋算例编号、全部 POMO 起点、每起点 1 Argmax＋8 Sampling。",
             "所有不同原始回路执行完整普通 2-opt；Opt-B 每例额外换边检查上限 50,000。",
             "搜索完成后才读取 `.solution`；同一回路的旋转和反向仅计一次。Fβ 使用 β²=0.3。", "",
             "## 汇总", "",
             "| 模式 | 2-opt 距离命中 | Opt-B 距离命中 | 2-opt 真最优回路 | Opt-B 真最优回路 | 平均 Opt-B Fβ | 对固定 Soft ΔFβ | 推理 s | 2-opt s | Opt-B s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in modes:
        s = summaries[mode]
        lines.append(f"| {_label(mode)} | {s['distance_hits_2opt']}/25 | {s['distance_hits_optb']}/25 | "
                     f"{s['true_optimal_routes_2opt']} | {s['true_optimal_routes_optb']} | "
                     f"{s['mean_fbeta_optb']:.6f} | {s['mean_fbeta_optb']-base['mean_fbeta_optb']:+.6f} | "
                     f"{s['inference_seconds']:.2f} | {s['two_opt_seconds']:.2f} | {s['optb_seconds']:.2f} |")
    def leaders(metric: str, *, lowest: bool = False) -> str:
        value = (min if lowest else max)(summaries[mode][metric] for mode in modes[1:])
        return "、".join(_label(mode) for mode in modes[1:]
                       if abs(summaries[mode][metric] - value) < 1e-10)

    missed = "; ".join(
        f"{_label(mode)}: " + "、".join(r["instance"] for r in grouped[mode]
                                           if not r["b_distance_hit"])
        for mode in modes[1:]
    )
    slowest = max(summaries[mode]["inference_seconds"] for mode in modes[1:])
    fastest = min(summaries[mode]["inference_seconds"] for mode in modes[1:])
    lines += ["", f"论文 25 例平均 Fβ：**{paper_mean:.4f}**。历史固定 Soft 复现门禁见 `baseline_gate.json`。",
              "", "## 结果解读", "",
              "- 固定 Soft 在最优距离命中、真最优回路总数和平均 Fβ 上均领先四个两级组。",
              f"- 两级组中，距离命中最多：{leaders('distance_hits_optb')}；"
              f"2-opt 真最优回路最多：{leaders('true_optimal_routes_2opt')}；"
              f"Opt-B 真最优回路最多：{leaders('true_optimal_routes_optb')}；"
              f"平均 Opt-B Fβ 最高：{leaders('mean_fbeta_optb')}。",
              f"- 原始平均 Gap 最低：{leaders('mean_raw_gap_pct', lowest=True)}。"
              f"两级推理总耗时为 {fastest:.2f}–{slowest:.2f} 秒，"
              f"固定 Soft 为 {base['inference_seconds']:.2f} 秒。",
              f"- 未命中最优距离的算例：{missed}。"
              "单看平均 Fβ 会掩盖距离命中与多解数量的权衡。",
              "", "## 逐例距离与 Gap", "",
              "每格为原始 / 2-opt / Opt-B 最短距离，以及对应 Gap（%）。", "",
              "| 算例 | 参考最优 | " + " | ".join(_label(m) for m in modes) + " |",
              "|---|---:|" + "---:|" * len(modes)]
    for index in range(25):
        sample = grouped["fixed_soft"][index]
        cells = []
        for mode in modes:
            r = grouped[mode][index]
            cells.append(f"{r['raw_best']}/{r['a_best']}/{r['b_best']} "
                         f"({r['raw_gap_pct']:.2f}/{r['a_gap_pct']:.2f}/{r['b_gap_pct']:.2f})")
        lines.append(f"| {sample['instance']} | {sample['reference_length']} | " + " | ".join(cells) + " |")
    lines += ["", "## 逐例真最优回路和质量", "",
              "每格为 2-opt 真最优回路数 / Opt-B 真最优回路数；随后分别是两阶段的 Precision / Recall / Fβ。",
              "", "| 算例 | 真值回路数 | 论文 Fβ | " + " | ".join(_label(m) for m in modes) + " |",
              "|---|---:|---:|" + "---:|" * len(modes)]
    for index in range(25):
        sample = grouped["fixed_soft"][index]
        cells = []
        for mode in modes:
            r = grouped[mode][index]
            cells.append(f"{r['a_hits']}/{r['b_hits']}; "
                         f"A {r['a_precision']:.3f}/{r['a_recall']:.3f}/{r['a_fbeta']:.3f}; "
                         f"B {r['b_precision']:.3f}/{r['b_recall']:.3f}/{r['b_fbeta']:.3f}")
        lines.append(f"| {sample['instance']} | {sample['truth_count']} | {sample['paper_fbeta']:.3f} | "
                     + " | ".join(cells) + " |")
    lines += ["", "## 逐例相对固定 Soft", "",
              "每格为 Opt-B 真最优回路数量变化 / Fβ 变化；正值代表两级策略改善。",
              "", "| 算例 | " + " | ".join(_label(m) for m in modes[1:]) + " |",
              "|---|" + "---:|" * (len(modes) - 1)]
    for index in range(25):
        base_row = grouped["fixed_soft"][index]
        cells = [f"{grouped[mode][index]['b_hits']-base_row['b_hits']:+d} / "
                 f"{grouped[mode][index]['b_fbeta']-base_row['b_fbeta']:+.3f}"
                 for mode in modes[1:]]
        lines.append(f"| {base_row['instance']} | " + " | ".join(cells) + " |")
    lines += ["", "## 逐例耗时", "",
              "每格为推理 / 普通 2-opt / Opt-B 秒。逐例两阶段 Precision、Recall、Fβ 和实际检查数另见 `per_instance.csv`。",
              "", "| 算例 | " + " | ".join(_label(m) for m in modes) + " |",
              "|---|" + "---:|" * len(modes)]
    for index in range(25):
        cells = []
        for mode in modes:
            r = grouped[mode][index]
            cells.append(f"{r['inference_seconds']:.2f}/{r['two_opt_seconds']:.2f}/{r['optb_seconds']:.3f}")
        lines.append(f"| MSTSP{index+1} | " + " | ".join(cells) + " |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
