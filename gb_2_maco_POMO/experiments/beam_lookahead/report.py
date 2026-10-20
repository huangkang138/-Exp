"""Full paired formal-test tables from saved metrics, with no invented results."""

from __future__ import annotations

from pathlib import Path


def write_report(output: Path, summaries: dict, paired: list[dict]) -> None:
    lines = ["# B1 城市 Beam 与 B2 粒球前瞻 Beam 对照", "",
        "## 协议与可比性", "",
        "两组固定 A0 权重、全部城市起点，最多 9 条路线/起点；相同普通 2-opt 与每例 50,000 次额外边对检查 Opt-B。",
        "仅汇报实际运行的对照条件。解码量包含局部 Beam 和贪心 Rollout；B1 在同预算条件下允许增加 Beam Width。",
        "验证集仅锁定解码预算，MSTSP1–25 不参与调参。Opt-B 新增路线可追溯到本例最佳 2-opt 起点集合，不能假定每条新增路线有唯一父路线。",
        "A0 是已有历史正式结果，B1/B2 是本次新运行结果。", "",
        "## 总体结果", "",
        "| 条件/模型 | 最优距离命中 | 2-opt 真最优回路 | Opt-B 真最优回路 | 最优区域 | 平均 Fβ | 解码步 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for label in ("A0", "equal_output_B1", "equal_output_B2",
                  "equal_decode_budget_B1", "equal_decode_budget_B2"):
        if label not in summaries:
            continue
        r = summaries[label]
        lines.append(f"| {label} | {r['distance_hits']}/25 | {r['two_opt_hits']} | "
            f"{r['opt_b_hits']} | {r['regions_hit']} | {r['mean_opt_b_fbeta']:.6f} | "
            f"{r.get('decode_steps', '历史未统计')} |")
    for condition in ("equal_output", "equal_decode_budget"):
        part = [row for row in paired if row["condition"] == condition]
        if not part:
            continue
        lines += ["", f"## {condition}：逐例距离与 Gap", "",
            "Gap = (本组距离 / 已知最优距离 − 1) × 100%。每阶段并排 B1、B2。", "",
            "| 算例 | 参考 | 原始 B1 | 原始 B2 | 2-opt B1 | 2-opt B2 | Opt-B B1 | Opt-B B2 | 原始 Gap B1 | 原始 Gap B2 | Opt-B Gap B1 | Opt-B Gap B2 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in part:
            lines.append("| " + " | ".join(str(r[key]) for key in (
                "instance", "reference_length", "B1_raw_best", "B2_raw_best",
                "B1_a_best", "B2_a_best", "B1_b_best", "B2_b_best",
                "B1_raw_gap_pct", "B2_raw_gap_pct", "B1_b_gap_pct", "B2_b_gap_pct")) + " |")
        lines += ["", f"## {condition}：逐例真最优覆盖与 Fβ", "",
            "| 算例 | 2-opt B1 | 2-opt B2 | Opt-B B1 | Opt-B B2 | 区域 B1 | 区域 B2 | Opt-B P B1 | Opt-B P B2 | Opt-B R B1 | Opt-B R B2 | 2-opt Fβ B1 | 2-opt Fβ B2 | Opt-B Fβ B1 | Opt-B Fβ B2 | 论文 Fβ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in part:
            lines.append("| " + " | ".join(str(r[key]) for key in (
                "instance", "B1_a_hits", "B2_a_hits", "B1_b_hits", "B2_b_hits",
                "B1_regions_hit", "B2_regions_hit", "B1_b_precision", "B2_b_precision",
                "B1_b_recall", "B2_b_recall", "B1_a_fbeta", "B2_a_fbeta",
                "B1_b_fbeta", "B2_b_fbeta", "paper_fbeta")) + " |")
    lines += ["", "## 数据与边界", "",
        "`protocol.json` 记录配置、checkpoint SHA 与预算；`validation.csv` 记录预算选择。",
        "`per_instance.csv` 与 `paired_per_instance.csv` 含完整逐例数值。各实例目录保存原始路线、2-opt 终点、Opt-B 新增回路及计数、每起点解码与耗时。",
        "单个固定权重的确定性搜索不构成跨训练种子稳定性证据；B2 同时引入结构前瞻和贪心补全，本实验不能单独归因二者。", ""]
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
