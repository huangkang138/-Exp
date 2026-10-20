"""Rebuild the paired S0/S1 tables from all 250 saved case records."""

from __future__ import annotations

import csv
import json
import statistics

from gb_2_maco_POMO.experiments.region_supervised_pomo.config import (
    INFERENCE_SEEDS, INITIAL_SHA256, RESULTS,
)

from .run import OUTPUT, _s0_folder

FIELDS = ("raw_best", "raw_gap_pct", "a_best", "a_gap_pct", "b_best", "b_gap_pct",
          "a_distance_hit", "b_distance_hit", "a_hits", "b_hits", "b_routes",
          "b_precision", "b_recall", "a_fbeta", "b_fbeta", "b_regions_hit",
          "raw_unique", "raw_duplicate_rate")


def _write_csv(name, rows):
    with (OUTPUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows, key):
    return statistics.mean(float(row[key]) for row in rows)


def _fmt(value, digits=2):
    return f"{value:.{digits}f}"


def main():
    paired = []
    by_case = {i: {"S0": [], "S1": []} for i in range(1, 26)}
    by_seed = []
    for seed in INFERENCE_SEEDS:
        s0 = _s0_folder(seed)
        s1 = OUTPUT / f"infer_seed_{seed}"
        config = json.loads((s1 / "run_config.json").read_text(encoding="utf-8"))
        summary = json.loads((s1 / "summary.json").read_text(encoding="utf-8"))
        if (config["checkpoint_sha256"] != INITIAL_SHA256 or
                summary["cases"] != 25 or not summary["argmax_matches_s0"]):
            raise RuntimeError(f"S1 seed {seed} is incomplete or inconsistent")
        runs = {"S0": [], "S1": []}
        for case in range(1, 26):
            old = json.loads((s0 / f"case_{case:02d}.json").read_text(encoding="utf-8"))
            new = json.loads((s1 / f"case_{case:02d}.json").read_text(encoding="utf-8"))
            if not new["argmax_matches_s0"] or any(old[k] != new[k] for k in
                    ("instance", "reference_length", "truth_count", "truth_regions", "raw_routes")):
                raise RuntimeError(f"unpaired result: seed {seed}, case {case}")
            runs["S0"].append(old)
            runs["S1"].append(new)
            by_case[case]["S0"].append(old)
            by_case[case]["S1"].append(new)
            item = {"seed": seed, "instance": old["instance"],
                    "reference_length": old["reference_length"],
                    "truth_count": old["truth_count"], "truth_regions": old["truth_regions"]}
            for field in FIELDS:
                item[f"S0_{field}"] = old[field]
                item[f"S1_{field}"] = new[field]
                item[f"delta_{field}"] = float(new[field]) - float(old[field])
            paired.append(item)
        row = {"seed": seed}
        for mode in ("S0", "S1"):
            records = runs[mode]
            for field, value in (
                ("raw_distance_hits", sum(r["raw_gap_pct"] == 0 for r in records)),
                ("two_opt_distance_hits", sum(r["a_distance_hit"] for r in records)),
                ("optb_distance_hits", sum(r["b_distance_hit"] for r in records)),
                ("two_opt_true_tours", sum(r["a_hits"] for r in records)),
                ("optb_true_tours", sum(r["b_hits"] for r in records)),
                ("regions_hit", sum(r["b_regions_hit"] for r in records)),
                ("mean_fbeta", _mean(records, "b_fbeta")),
                ("mean_b_gap_pct", _mean(records, "b_gap_pct")),
            ):
                row[f"{mode}_{field}"] = value
        for field in ("raw_distance_hits", "two_opt_distance_hits", "optb_distance_hits",
                      "two_opt_true_tours", "optb_true_tours", "regions_hit",
                      "mean_fbeta", "mean_b_gap_pct"):
            row[f"delta_{field}"] = row[f"S1_{field}"] - row[f"S0_{field}"]
        by_seed.append(row)
    _write_csv("paired_per_instance.csv", paired)
    _write_csv("summary_by_seed.csv", by_seed)
    cases = []
    for case, modes in by_case.items():
        first = modes["S0"][0]
        row = {"instance": f"MSTSP{case}", "reference_length": first["reference_length"],
               "truth_count": first["truth_count"], "truth_regions": first["truth_regions"],
               "paper_fbeta": first["paper_fbeta"]}
        for mode in ("S0", "S1"):
            records = modes[mode]
            for field in FIELDS:
                row[f"{mode}_{field}_mean"] = _mean(records, field)
            row[f"{mode}_distance_hits"] = sum(r["b_distance_hit"] for r in records)
            row[f"{mode}_regions_hit_mean"] = _mean(records, "b_regions_hit")
        for field in FIELDS:
            row[f"delta_{field}_mean"] = row[f"S1_{field}_mean"] - row[f"S0_{field}_mean"]
        cases.append(row)
    _write_csv("per_case_comparison.csv", cases)
    keys = ("optb_distance_hits", "optb_true_tours", "regions_hit", "mean_fbeta")
    summary = {mode: {key: {"mean": statistics.mean(row[f"{mode}_{key}"] for row in by_seed),
                             "sd": statistics.stdev(row[f"{mode}_{key}"] for row in by_seed)}
                      for key in keys} for mode in ("S0", "S1")}
    summary["paired_delta"] = {key: {"mean": statistics.mean(row[f"delta_{key}"] for row in by_seed),
                                     "sd": statistics.stdev(row[f"delta_{key}"] for row in by_seed)}
                               for key in keys}
    (OUTPUT / "summary_statistics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# S0 与 S1 选球采样对照", "",
             "## 协议与可比性", "",
             f"- 模型：同一短训练可学习选球 checkpoint，SHA256 `{INITIAL_SHA256}`；S1 不训练。",
             "- 训练/验证：沿用 checkpoint 原有流程；本实验只更改正式测试采样，未用 MSTSP1–25 选模。",
             "- 算例：MSTSP1–25；推理种子 42–46。每城作为起点，1 次两级 Argmax + 8 次采样。",
             "- S0：后 8 次选球和选城均采样；S1：后 8 次每步选球 Argmax、球内城市采样。",
             "- 两组同用全量不同原始路线普通 2-opt、每例 50,000 次 Opt-B 边对检查、同一 canonical 去重和 Fβ 公式。",
             "- 设备均为 CPU。125 对逐例 Argmax 路线完全一致。同种子不代表后 8 条路线相同。",
             "", "## 五个种子汇总", "",
             "|组别|最优距离命中/25|不同真最优回路总数|区域命中总数|平均 Fβ|", "|---|---:|---:|---:|---:|"]
    for mode in ("S0", "S1"):
        values = summary[mode]
        lines.append("|" + mode + "|" + "|".join(
            f"{_fmt(values[key]['mean'],4 if key == 'mean_fbeta' else 2)} ± "
            f"{_fmt(values[key]['sd'],4 if key == 'mean_fbeta' else 2)}"
            for key in keys) + "|")
    lines.extend(["", "|推理种子|S0 命中|S1 命中|S0 真最优回路|S1 真最优回路|S0 区域|S1 区域|S0 Fβ|S1 Fβ|",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in by_seed:
        lines.append("|" + str(row["seed"]) + "|" + "|".join(
            _fmt(row[f"{mode}_{key}"], 4 if key == "mean_fbeta" else 0)
            for key in ("optb_distance_hits", "optb_true_tours", "regions_hit", "mean_fbeta")
            for mode in ("S0", "S1")) + "|")
    case22, case25 = cases[21], cases[24]
    lines.extend(["", "## 核心诊断", "",
                  f"- S1 最优距离命中从每种子平均 {_fmt(summary['S0']['optb_distance_hits']['mean'])}/25 "
                  f"变为 {_fmt(summary['S1']['optb_distance_hits']['mean'])}/25，但真最优回路平均少 "
                  f"{_fmt(-summary['paired_delta']['optb_true_tours']['mean'])} 条、区域少 "
                  f"{_fmt(-summary['paired_delta']['regions_hit']['mean'])} 个，平均 Fβ 下降 "
                  f"{_fmt(-summary['paired_delta']['mean_fbeta']['mean'],4)}。不能据命中率单项认定 S1 更好。",
                  f"- MSTSP22：S0/S1 命中 {case22['S0_distance_hits']}/{case22['S1_distance_hits']} 次；"
                  f"MSTSP25：{case25['S0_distance_hits']}/{case25['S1_distance_hits']} 次。"
                  "S1 改善 22，却没有改善 25。其他 23 例中，MSTSP21 少命中一次，MSTSP24 多命中一次。",
                  "- MSTSP22、25 的原始路线在两组都是逐条不重复的；两组最短原始距离也相同。"
                  "差异主要出现在不同初始路线进入 2-opt 后到达的最优回路与区域，而非原始路线重复率。",
                  "- MSTSP16、17 的最终距离均保持命中，但 S1 的不同真最优回路及区域覆盖明显减少，"
                  "说明在本次预算下只采样城市会损失这些算例的多解覆盖。",
                  ""])
    lines.extend(["", "## 逐算例距离与 Gap：五种子均值", "",
                  "Gap = 100 × (路线距离 / 已知最优距离 − 1)；三个阶段均为最短距离。", "",
                  "|算例|最优|S0 原始/2-opt/Opt-B|S1 原始/2-opt/Opt-B|S0/S1 最终 Gap %|S0/S1 命中次数|",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in cases:
        def distances(mode):
            return "/".join(_fmt(row[f"{mode}_{key}_mean"]) for key in ("raw_best", "a_best", "b_best"))
        lines.append(f"|{row['instance']}|{row['reference_length']}|{distances('S0')}|{distances('S1')}|"
                     f"{_fmt(row['S0_b_gap_pct_mean'],4)}/{_fmt(row['S1_b_gap_pct_mean'],4)}|"
                     f"{row['S0_distance_hits']}/{row['S1_distance_hits']}|")
    lines.extend(["", "## 逐算例最优回路与区域：五种子均值", "",
                  "回路按起点旋转和反向等价去重；真最优命中以提供的 `.solution` 为准。区域按项目 Opt-B 等长连通规则。",
                  "|算例|真最优数/区域数|S0/S1 2-opt 真最优|S0/S1 Opt-B 去重回路|S0/S1 Opt-B 真最优|S0/S1 区域命中|S0/S1 Precision|S0/S1 Recall|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in cases:
        pair = lambda key, digits=2: f"{_fmt(row[f'S0_{key}_mean'],digits)}/{_fmt(row[f'S1_{key}_mean'],digits)}"
        lines.append(f"|{row['instance']}|{row['truth_count']}/{row['truth_regions']}|"
                     f"{pair('a_hits')}|{pair('b_routes')}|{pair('b_hits')}|"
                     f"{_fmt(row['S0_regions_hit_mean'])}/{_fmt(row['S1_regions_hit_mean'])}|"
                     f"{pair('b_precision',4)}|{pair('b_recall',4)}|")
    lines.extend(["", "## 逐算例 Fβ：五种子均值", "",
                  "|算例|S0/S1 2-opt Fβ|S0/S1 Opt-B Fβ|论文 Fβ|S0/S1 Opt-B 减论文|",
                  "|---|---:|---:|---:|---:|"])
    for row in cases:
        p=float(row["paper_fbeta"])
        lines.append(f"|{row['instance']}|{_fmt(row['S0_a_fbeta_mean'],4)}/{_fmt(row['S1_a_fbeta_mean'],4)}|"
                     f"{_fmt(row['S0_b_fbeta_mean'],4)}/{_fmt(row['S1_b_fbeta_mean'],4)}|{_fmt(p,4)}|"
                     f"{_fmt(row['S0_b_fbeta_mean']-p,4)}/{_fmt(row['S1_b_fbeta_mean']-p,4)}|")
    lines.extend(["", "## 结论边界", "",
                  "S0/S1 的因果对照只改变后 8 次选球方式；同种子采样随机数流不同，不能逐条路线配对解释。",
                  "论文结果只对照其实际提供的 Fβ，评价预算未统一，不宣称超过论文。",
                  "逐次完整数据见 `paired_per_instance.csv`、`summary_by_seed.csv`；各 S1 运行的 `case_XX.json`、路线和 `summary.json` 保留原始证据。", ""])
    (OUTPUT / "诊断报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT / "诊断报告.md")


if __name__ == "__main__":
    main()
