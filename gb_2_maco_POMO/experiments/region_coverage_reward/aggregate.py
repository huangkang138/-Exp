"""Aggregate complete final tests without selecting on test outcomes."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .train import GROUPS, PROJECT


ROOT = PROJECT / "gb_2_maco_POMO/results/region_coverage_reward"
METRICS = ("distance_hits", "true_optimal_routes", "mean_fbeta", "truth_regions_hit",
           "mean_raw_duplicate_rate", "inference_seconds", "two_opt_seconds", "optb_seconds")
CONTRASTS = (("G1", "G0"), ("G1", "Q"), ("G2", "G0"), ("G3", "G1"), ("G3", "G2"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stat(values):
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def collect():
    summaries, cases, regions = [], [], []
    for mode in ("fixed_soft", "short", *GROUPS):
        seeds = (None,) if mode in ("fixed_soft", "short") else (42, 43, 44)
        for train_seed in seeds:
            for inference_seed in (42, 43, 44, 45, 46):
                base = (ROOT / mode / f"train_seed_{train_seed}" if train_seed is not None
                        else ROOT / mode)
                output = base / f"test_seed_{inference_seed}"
                summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                if summary["model"] != mode or summary["inference_seed"] != inference_seed:
                    raise RuntimeError(f"final test identity mismatch: {output}")
                summaries.append(summary)
                case_rows = read_csv(output / "per_instance.csv")
                region_rows = read_csv(output / "region_coverage.csv")
                if len(case_rows) != 25:
                    raise RuntimeError(f"incomplete cases: {output}")
                cases.extend(case_rows)
                regions.extend({"model": mode, "train_seed": train_seed or "",
                                "inference_seed": inference_seed, **row} for row in region_rows)
    if len(summaries) != 85 or len(cases) != 2125:
        raise RuntimeError("all model/train/inference combinations are required")
    return summaries, cases, regions


def aggregate():
    summaries, cases, regions = collect()
    write_csv(ROOT / "all_final_runs.csv", summaries)
    write_csv(ROOT / "all_final_cases.csv", cases)
    write_csv(ROOT / "all_final_regions.csv", regions)
    grouped = defaultdict(list)
    for row in summaries:
        grouped[row["model"]].append(row)
    model_rows = []
    for mode, rows in grouped.items():
        result = {"model": mode, "runs": len(rows)}
        for metric in METRICS:
            for suffix, value in stat([float(r[metric]) for r in rows]).items():
                result[f"{metric}_{suffix}"] = value
        model_rows.append(result)
    write_csv(ROOT / "model_summary.csv", model_rows)
    seed_rows = []
    for mode in GROUPS:
        for seed in (42, 43, 44):
            rows = [r for r in grouped[mode] if r["train_seed"] == seed]
            seed_rows.append({"model": mode, "train_seed": seed,
                              **{f"{metric}_{suffix}": value
                                 for metric in METRICS
                                 for suffix, value in stat([float(r[metric]) for r in rows]).items()}})
    write_csv(ROOT / "training_seed_summary.csv", seed_rows)
    lookup = {(r["model"], r["train_seed"], r["inference_seed"]): r for r in summaries}
    comparison_rows = []
    for left, right in CONTRASTS:
        for train_seed in (42, 43, 44):
            for inference_seed in (42, 43, 44, 45, 46):
                a, b = lookup[left, train_seed, inference_seed], lookup[right, train_seed, inference_seed]
                comparison_rows.append({"contrast": f"{left}-{right}", "train_seed": train_seed,
                                        "inference_seed": inference_seed,
                                        "delta_hits": a["distance_hits"] - b["distance_hits"],
                                        "delta_routes": a["true_optimal_routes"] - b["true_optimal_routes"],
                                        "delta_fbeta": a["mean_fbeta"] - b["mean_fbeta"],
                                        "delta_regions": a["truth_regions_hit"] - b["truth_regions_hit"]})
    write_csv(ROOT / "paired_comparisons.csv", comparison_rows)
    case_group = defaultdict(list)
    for row in cases:
        case_group[row["model"], row["instance"]].append(row)
    case_summary = []
    for (mode, instance), rows in case_group.items():
        case_summary.append({"model": mode, "instance": instance, "runs": len(rows),
                             "hit_rate": statistics.mean(int(r["b_distance_hit"]) for r in rows),
                             "mean_b_gap_pct": statistics.mean(float(r["b_gap_pct"]) for r in rows),
                             "mean_fbeta": statistics.mean(float(r["b_fbeta"]) for r in rows),
                             "mean_b_true_routes": statistics.mean(int(r["b_hits"]) for r in rows),
                             "mean_regions_hit": statistics.mean(int(r["b_regions_hit"]) for r in rows),
                             "mean_raw_duplicate_rate": statistics.mean(float(r["raw_duplicate_rate"]) for r in rows)})
    write_csv(ROOT / "per_instance_model_summary.csv", case_summary)
    training_cost = []
    augmentation_complete = True
    for mode in GROUPS:
        for seed in (42, 43, 44):
            directory = ROOT / mode / f"train_seed_{seed}"
            rows = read_csv(directory / "training_history.csv")
            for epoch in range(int(json.loads((directory / "train_result.json").read_text(encoding="utf-8"))["completed_epochs"]) + 1):
                metrics_file = directory / f"validation_augmented_epoch_{epoch:03d}.json"
                augmentation_complete &= (metrics_file.is_file() and json.loads(
                    metrics_file.read_text(encoding="utf-8"))["raw_replay_best_matched_all_cases"])
            selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
            result = json.loads((directory / "train_result.json").read_text(encoding="utf-8"))
            training_cost.append({"model": mode, "train_seed": seed,
                                  "epochs": result["completed_epochs"], "updates": result["updates"],
                                  "selected_epoch": selection["selected_epoch"],
                                  "train_seconds": sum(float(r["train_seconds"]) for r in rows),
                                  "reward_seconds": sum(float(r["reward_seconds"]) for r in rows),
                                  "actual_two_opt_checks": sum(int(r.get("actual_two_opt_checks") or 0) for r in rows),
                                  "actual_optb_checks": sum(int(r.get("actual_optb_checks") or 0) for r in rows),
                                  "coverage_advantage_routes": sum(int(r["nonzero_coverage_advantages"]) for r in rows),
                                  "reference_updates": sum(int(r["next_epoch_reference_updates"]) for r in rows)})
    write_csv(ROOT / "training_cost.csv", training_cost)
    report = ["# 区域覆盖奖励与可学习 Bias 联合实验", "",
              "## 协议与证据边界", "",
              "五组均从同一短训练 checkpoint 初始化；训练种子 42/43/44，推理种子 42–46。",
              "每例全部城市起点，1 次贪心与 8 次采样；所有不同原始路线完整普通 2-opt，Opt-B 50,000 次边对检查。",
              "模型只依独立验证集选择，MSTSP1–25 真值仅在正式测试后读取。",
              ("验证集原始采样路线平均距离的重生估计见各轮 `validation_augmented_epoch_*.csv`；"
               "重生路线的最短距离逐例与原记录一致，但原始全体路线未保存，无法证明逐条一致；未参与选模。"
               if augmentation_complete else
               "验证集原始采样路线平均距离未全部可靠恢复；缺失轮次不得用推测值替代。"),
              "区域是 `.solution` 中列举真最优路线诱导的等长换边连通分量；未列出的中间路线可能连接这些分量。",
              "论文 Fβ=0.917600 的推理和训练预算未在此重建，数值比较不等于公平预算复现。", "",
              "三训练种子各自的五次推理均值与波动见 `training_seed_summary.csv`；下表 SD 混合了两类波动。", "",
              "## 最终结果", "",
              "| 模型 | 运行数 | 命中/25 均值±SD | 真最优路线均值±SD | Fβ 均值±SD | 命中区域均值±SD |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in model_rows:
        report.append(f"| {row['model']} | {row['runs']} | "
                      f"{row['distance_hits_mean']:.2f}±{row['distance_hits_sd']:.2f} | "
                      f"{row['true_optimal_routes_mean']:.1f}±{row['true_optimal_routes_sd']:.1f} | "
                      f"{row['mean_fbeta_mean']:.6f}±{row['mean_fbeta_sd']:.6f} | "
                      f"{row['truth_regions_hit_mean']:.1f}±{row['truth_regions_hit_sd']:.1f} |")
    by_model = {row["model"]: row for row in model_rows}
    for mode in ("G1", "G3"):
        value = by_model[mode]["mean_fbeta_mean"]
        report.append(f"\n{mode} 的 15 次正式测试平均 Fβ={value:.6f}，"
                      f"{'高于' if value > 0.917600 else '未高于'}论文记录的 0.917600；"
                      "由于预算差异，这只作数值参考。")
    most_hits = max(model_rows, key=lambda r: (r["distance_hits_mean"],
                                                -r["distance_hits_sd"]))
    most_routes = max(model_rows, key=lambda r: r["true_optimal_routes_mean"])
    most_fbeta = max(model_rows, key=lambda r: r["mean_fbeta_mean"])
    report.append(f"\n按本次预算的均值，最优距离命中最多的是 {most_hits['model']}；"
                  f"不同真最优回路数最多的是 {most_routes['model']}；"
                  f"平均 Fβ 最高的是 {most_fbeta['model']}。"
                  "命中数和多解数分别度量不同能力，训练种子波动见逐种子表。")
    report.extend(["", "## 配对对照", "",
                   "以下差值在相同训练种子、推理种子下计算；训练种子仅 3 个，不作显著性断言。", "",
                   "| 对照 | ΔFβ 均值 | Δ命中数 | Δ真最优路线 | Δ区域 |", "|---|---:|---:|---:|---:|"])
    for left, right in CONTRASTS:
        rows = [r for r in comparison_rows if r["contrast"] == f"{left}-{right}"]
        report.append(f"| {left}−{right} | "
                      f"{statistics.mean(r['delta_fbeta'] for r in rows):+.6f} | "
                      f"{statistics.mean(r['delta_hits'] for r in rows):+.2f} | "
                      f"{statistics.mean(r['delta_routes'] for r in rows):+.1f} | "
                      f"{statistics.mean(r['delta_regions'] for r in rows):+.1f} |")
    report.extend(["", "### 逐项判断", ""])
    labels = {("G1", "G0"): "区域组相对原始长度组",
              ("G1", "Q"): "区域贡献相对单纯 2-opt 质量反馈",
              ("G2", "G0"): "Bias 单独作用",
              ("G3", "G1"): "Bias 与区域奖励互补",
              ("G3", "G2"): "联合奖励相对单独 Bias"}
    for left, right in CONTRASTS:
        rows = [r for r in comparison_rows if r["contrast"] == f"{left}-{right}"]
        per_train = [statistics.mean(r["delta_fbeta"] for r in rows if r["train_seed"] == seed)
                     for seed in (42, 43, 44)]
        mean_delta = statistics.mean(r["delta_fbeta"] for r in rows)
        if all(value > 0 for value in per_train):
            judgment = "三个训练种子方向一致，支持有收益"
        elif all(value < 0 for value in per_train):
            judgment = "三个训练种子均退化"
        else:
            judgment = "训练种子间方向不一致，未证明稳定收益"
        report.append(f"- {labels[left, right]}：配对平均 ΔFβ={mean_delta:+.6f}；{judgment}。")
    report.extend(["", "### MSTSP21–25", "",
                   "| 算例 | 模型 | 命中率 | 平均 Gap % | 平均 Fβ | 平均真最优路线 |",
                   "|---|---|---:|---:|---:|---:|"])
    case_lookup = {(r["model"], r["instance"]): r for r in case_summary}
    for index in range(21, 26):
        for mode in ("fixed_soft", "short", "G0", "Q", "G1", "G2", "G3"):
            row = case_lookup[mode, f"MSTSP{index}"]
            report.append(f"| MSTSP{index} | {mode} | {row['hit_rate']:.3f} | "
                          f"{row['mean_b_gap_pct']:.3f} | {row['mean_fbeta']:.6f} | "
                          f"{row['mean_b_true_routes']:.1f} |")
    report.extend(["", "### MSTSP7/22/25 区域总览", "",
                   "| 算例 | 模型 | 平均命中区域 | 平均真最优路线 |",
                   "|---|---|---:|---:|"])
    for index in (7, 22, 25):
        for mode in ("fixed_soft", "short", "G0", "Q", "G1", "G2", "G3"):
            row = case_lookup[mode, f"MSTSP{index}"]
            report.append(f"| MSTSP{index} | {mode} | {row['mean_regions_hit']:.2f} | "
                          f"{row['mean_b_true_routes']:.1f} |")
    report.extend(["", "## 困难算例与区域", "",
                   "逐例命中率、Gap、Fβ 与多解数量见 `per_instance_model_summary.csv`；",
                   "MSTSP7/22/25 的每一区域入口与 Opt-B 贡献见 `all_final_regions.csv`，",
                   "MSTSP21–25 的逐次结果见 `all_final_cases.csv`。", "",
                   "## 训练代价", "",
                   "| 模型 | 平均更新数 | 平均训练秒 | 平均奖励评估秒 | 平均 2-opt 检查 | 平均 Opt-B 检查 | 非零覆盖优势路线 |",
                   "|---|---:|---:|---:|---:|---:|---:|"])
    for mode in GROUPS:
        rows = [r for r in training_cost if r["model"] == mode]
        report.append(f"| {mode} | {statistics.mean(r['updates'] for r in rows):.1f} | "
                      f"{statistics.mean(r['train_seconds'] for r in rows):.1f} | "
                      f"{statistics.mean(r['reward_seconds'] for r in rows):.1f} | "
                      f"{statistics.mean(r['actual_two_opt_checks'] for r in rows):.0f} | "
                      f"{statistics.mean(r['actual_optb_checks'] for r in rows):.0f} | "
                      f"{statistics.mean(r['coverage_advantage_routes'] for r in rows):.0f} |")
    report.extend(["", "`training_cost.csv` 保留逐训练种子的更新和计算成本；上表训练秒不含验证。",
                   "训练集离线预算校准见 `training_only_budget_calibration.json`。", "",
                   "## 判断范围", "",
                   "质量门槛和固定预算使 50–66 城市训练样本的区域优势常为零；",
                   "不能把未观察到的连通关系解释为不存在，也不能凭单次最高 Fβ 宣称稳定超越论文。", ""])
    (ROOT / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    aggregate()
