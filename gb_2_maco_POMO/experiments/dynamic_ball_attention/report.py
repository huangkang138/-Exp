"""Build the Chinese A0/A/B report only after all formal tests finish."""

from __future__ import annotations

import csv
import json
import statistics

import torch

from gb_2_maco_POMO.experiments.region_supervised_pomo.config import RESULTS as OLD_RESULTS

from .config import INFERENCE_SEEDS, INITIAL_SHA256, RESULTS, TRAIN_SEEDS


def _csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _mean(rows, key):
    return statistics.mean(float(row[key]) for row in rows)


def _fmt(value, digits=2):
    return f"{value:.{digits}f}"


def main():
    all_cases, runs, training = [], [], []
    for group in ("A0", "A", "B"):
        seeds = (None,) if group == "A0" else TRAIN_SEEDS
        for seed in seeds:
            if seed is not None:
                directory = RESULTS / f"train_seed_{seed}" / group
                result = json.loads((directory / "train_result.json").read_text(encoding="utf-8"))
                selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
                if (result["completed_epochs"] != 3 or result["total_optimizer_updates"] != 33
                        or selection["test_set_used_for_selection"] is not False):
                    raise RuntimeError(f"incomplete or invalid training: {directory}")
                state = torch.load(selection["checkpoint"], map_location="cpu", weights_only=True)
                training.append({"group": group, "train_seed": seed,
                                 "selected_epoch": selection["selected_epoch"],
                                 "checkpoint_sha256": result["selected_checkpoint_sha256"],
                                 "parameter_count": result["parameter_count"],
                                 "train_validation_seconds": result["training_and_validation_seconds"],
                                 "selected_alpha": (float(state["model_state_dict"]["ball_attention_alpha"])
                                                    if group == "B" else ""),
                                 "validation_2opt_reference_gap_pct":
                                 selection["metrics"]["mean_two_opt_relative_reference_pct"],
                                 "validation_high_quality_tours":
                                 selection["metrics"]["mean_high_quality_tours"]})
            for inference in INFERENCE_SEEDS:
                folder = (OLD_RESULTS / "tests/A0/initial" / f"infer_seed_{inference}"
                          if group == "A0" else RESULTS / "tests" / group /
                          f"train_seed_{seed}" / f"infer_seed_{inference}")
                config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
                summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                if (config["checkpoint_sha256"] != (INITIAL_SHA256 if group == "A0" else
                     next(t["checkpoint_sha256"] for t in training
                          if t["group"] == group and t["train_seed"] == seed))
                        or summary["cases"] != 25):
                    raise RuntimeError(f"test checkpoint or case count differs: {folder}")
                cases = [json.loads((folder / f"case_{i:02d}.json").read_text(encoding="utf-8"))
                         for i in range(1, 26)]
                for case in cases:
                    all_cases.append({"group": group, "train_seed": seed if seed is not None else "",
                                      "inference_seed": inference, **case})
                runs.append({"group": group, "train_seed": seed if seed is not None else "",
                             "inference_seed": inference,
                             "checkpoint_sha256": config["checkpoint_sha256"],
                             "raw_distance_hits": sum(r["raw_gap_pct"] == 0 for r in cases),
                             "two_opt_distance_hits": sum(r["a_distance_hit"] for r in cases),
                             "optb_distance_hits": sum(r["b_distance_hit"] for r in cases),
                             "raw_true_tours": sum(r["raw_true_optimal"] for r in cases),
                             "two_opt_true_tours": sum(r["a_hits"] for r in cases),
                             "optb_true_tours": sum(r["b_hits"] for r in cases),
                             "regions_hit": sum(r["b_regions_hit"] for r in cases),
                             "mean_fbeta": _mean(cases, "b_fbeta"),
                             "inference_seconds": sum(r["inference_seconds"] for r in cases),
                             "two_opt_seconds": sum(r["two_opt_seconds"] for r in cases),
                             "optb_seconds": sum(r["optb_seconds"] for r in cases)})
    RESULTS.mkdir(parents=True, exist_ok=True)
    _csv(RESULTS / "all_test_instances.csv", all_cases)
    _csv(RESULTS / "all_test_runs.csv", runs)
    _csv(RESULTS / "training_selection.csv", training)
    per_case = []
    for index in range(1, 26):
        groups = {group: [r for r in all_cases if r["group"] == group
                          and r["instance"] == f"MSTSP{index}"] for group in ("A0", "A", "B")}
        original = groups["A0"][0]
        if any(r["reference_length"] != original["reference_length"] or
               r["truth_count"] != original["truth_count"] or
               r["truth_regions"] != original["truth_regions"]
               for records in groups.values() for r in records):
            raise RuntimeError(f"test ground truth differs: MSTSP{index}")
        row = {"instance": f"MSTSP{index}", "reference_length": original["reference_length"],
               "truth_count": original["truth_count"], "truth_regions": original["truth_regions"],
               "paper_fbeta": original["paper_fbeta"]}
        for group, records in groups.items():
            row[f"{group}_runs"] = len(records)
            row[f"{group}_distance_hits"] = sum(r["b_distance_hit"] for r in records)
            for key in ("raw_best", "raw_gap_pct", "a_best", "a_gap_pct", "b_best",
                        "b_gap_pct", "raw_true_optimal", "a_hits", "b_hits", "b_routes",
                        "b_regions_hit", "b_precision", "b_recall", "a_fbeta", "b_fbeta",
                        "raw_unique", "inference_seconds"):
                row[f"{group}_{key}_mean"] = _mean(records, key)
        row["B_minus_A_b_hits"] = row["B_b_hits_mean"] - row["A_b_hits_mean"]
        row["B_minus_A_b_fbeta"] = row["B_b_fbeta_mean"] - row["A_b_fbeta_mean"]
        per_case.append(row)
    _csv(RESULTS / "per_case_comparison.csv", per_case)
    lines = ["# 动态 Ball Attention 对照实验", "", "## 协议", "",
             "A0 是不继续训练的短训练 checkpoint；A 保留平均球表示；B 用当前状态 Attention 汇总球内未访问城市，并学习零初始化残差 α。",
             f"三组初始权重 SHA256 `{INITIAL_SHA256}`；A/B 每训练种子同用原 Multiopt 500/100 划分、3 epoch×11 更新、Adam 1e-4、相同 POMO 长度奖励。",
             "验证只用独立 100 例，1 Argmax + 8 Sampling、普通 2-opt；以平均 2-opt 参考相对 Gap、原始参考相对 Gap、较早 epoch 顺序选模。",
             "无认证验证例的参考为最近邻+2-opt 启发式上界，其相对差值不是最优 Gap。MSTSP1–25 不参与训练或选模。",
             "正式测试为每城起点 1 Argmax + 8 Sampling、全部不同原始路线完整普通 2-opt、Opt-B 50,000 对边；旋转和反向等价去重。推理种子 42–46，设备 CPU。",
             "A/B 各 3 个训练种子，每个 checkpoint 5 个推理种子；A0 共 5 个推理种子。",
             "", "## 训练与 checkpoint", "",
             "|组|训练种子|选中 epoch|参数量|α|验证 2-opt 参考差 %|验证高质量路线均值|训练+验证秒|",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in training:
        lines.append(f"|{row['group']}|{row['train_seed']}|{row['selected_epoch']}|"
                     f"{row['parameter_count']}|{row['selected_alpha']}|"
                     f"{_fmt(row['validation_2opt_reference_gap_pct'],4)}|"
                     f"{_fmt(row['validation_high_quality_tours'])}|"
                     f"{_fmt(row['train_validation_seconds'],1)}|")
    lines.extend(["", "## 正式测试汇总：运行均值 ± 标准差", "",
                  "|组|运行数|原始/2-opt/Opt-B 距离命中|原始/2-opt/Opt-B 真最优回路|区域命中|Fβ|推理秒/25例|",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for group in ("A0", "A", "B"):
        group_runs = [r for r in runs if r["group"] == group]
        value = lambda key, digits=2: (f"{_fmt(_mean(group_runs,key),digits)} ± "
                                      f"{_fmt(statistics.stdev(float(r[key]) for r in group_runs),digits)}")
        lines.append(f"|{group}|{len(group_runs)}|"
                     f"{' / '.join(value(k) for k in ('raw_distance_hits','two_opt_distance_hits','optb_distance_hits'))}|"
                     f"{' / '.join(value(k) for k in ('raw_true_tours','two_opt_true_tours','optb_true_tours'))}|"
                     f"{value('regions_hit')}|{value('mean_fbeta',4)}|{value('inference_seconds',1)}|")
    grouped_runs = {group: [r for r in runs if r["group"] == group]
                    for group in ("A0", "A", "B")}
    paired_seed_wins = sum(
        statistics.mean(r["mean_fbeta"] for r in grouped_runs["B"]
                        if r["train_seed"] == seed) >
        statistics.mean(r["mean_fbeta"] for r in grouped_runs["A"]
                        if r["train_seed"] == seed)
        for seed in TRAIN_SEEDS)
    case22, case25 = per_case[21], per_case[24]
    alpha_values = [abs(float(row["selected_alpha"])) for row in training
                    if row["group"] == "B"]
    lines.extend(["", "## 核心诊断", "",
                  f"- B 相对 A：平均 Fβ 差 "
                  f"{_fmt(_mean(grouped_runs['B'],'mean_fbeta')-_mean(grouped_runs['A'],'mean_fbeta'),4)}；"
                  f"距离命中差 {_fmt(_mean(grouped_runs['B'],'optb_distance_hits')-_mean(grouped_runs['A'],'optb_distance_hits'))} /25；"
                  f"不同真最优回路差 {_fmt(_mean(grouped_runs['B'],'optb_true_tours')-_mean(grouped_runs['A'],'optb_true_tours'))}。"
                  f"三个训练种子中，B 的平均 Fβ 高于对应 A 的次数为 {paired_seed_wins}/3。",
                  f"- B 相对初始 A0：平均 Fβ 差 "
                  f"{_fmt(_mean(grouped_runs['B'],'mean_fbeta')-_mean(grouped_runs['A0'],'mean_fbeta'),4)}。",
                  f"- MSTSP22 最优距离命中 A0/A/B："
                  f"{case22['A0_distance_hits']}/5、{case22['A_distance_hits']}/15、{case22['B_distance_hits']}/15；"
                  f"MSTSP25：{case25['A0_distance_hits']}/5、{case25['A_distance_hits']}/15、{case25['B_distance_hits']}/15。",
                  f"- B 选中 checkpoint 的 |α| 为 "
                  f"{', '.join(_fmt(value,6) for value in alpha_values)}；"
                  "只有 α 非零且权重随状态变化时，动态表示才实际影响选球。",
                  f"- B 相对 A 的平均推理耗时差 "
                  f"{_fmt(_mean(grouped_runs['B'],'inference_seconds')-_mean(grouped_runs['A'],'inference_seconds'),2)} 秒/25例；"
                  "两组后处理预算相同。", ""])
    lines.extend(["", "## 逐例距离与 Gap：运行均值", "",
                  "最优 Gap = 100 × (本组路线距离 / 最优距离 − 1)。A0 为 5 次推理均值；A/B 各为 15 次。",
                  "|算例|最优|A0/A/B 原始距离|A0/A/B 2-opt 距离|A0/A/B Opt-B 距离|A0/A/B 最终 Gap %|A0/A/B 命中次数|",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for row in per_case:
        triple = lambda key, digits=2: "/".join(_fmt(row[f"{g}_{key}_mean"],digits)
                                             for g in ("A0","A","B"))
        lines.append(f"|{row['instance']}|{row['reference_length']}|{triple('raw_best')}|"
                     f"{triple('a_best')}|{triple('b_best')}|{triple('b_gap_pct',4)}|"
                     f"{'/'.join(str(row[f'{g}_distance_hits']) for g in ('A0','A','B'))}|")
    lines.extend(["", "## 逐例最优回路与区域：运行均值", "",
                  "|算例|真最优/区域总数|A0/A/B 原始真最优|A0/A/B 2-opt 真最优|A0/A/B Opt-B 去重回路|A0/A/B Opt-B 真最优|A0/A/B 区域|A0/A/B Precision|A0/A/B Recall|B−A 真最优|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in per_case:
        triple = lambda key, digits=2: "/".join(_fmt(row[f"{g}_{key}_mean"],digits)
                                             for g in ("A0","A","B"))
        lines.append(f"|{row['instance']}|{row['truth_count']}/{row['truth_regions']}|"
                     f"{triple('raw_true_optimal')}|{triple('a_hits')}|{triple('b_routes')}|"
                     f"{triple('b_hits')}|{triple('b_regions_hit')}|"
                     f"{triple('b_precision',4)}|{triple('b_recall',4)}|"
                     f"{_fmt(row['B_minus_A_b_hits'])}|")
    lines.extend(["", "## 逐例 Fβ：运行均值", "",
                  "|算例|A0/A/B 2-opt Fβ|A0/A/B Opt-B Fβ|论文 Fβ|A0/A/B Opt-B 减论文|B−A Opt-B Fβ|",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in per_case:
        triple = lambda key: "/".join(_fmt(row[f"{g}_{key}_mean"],4)
                                      for g in ("A0","A","B"))
        paper = float(row["paper_fbeta"])
        lines.append(f"|{row['instance']}|{triple('a_fbeta')}|{triple('b_fbeta')}|"
                     f"{_fmt(paper,4)}|"
                     f"{'/'.join(_fmt(row[f'{g}_b_fbeta_mean']-paper,4) for g in ('A0','A','B'))}|"
                     f"{_fmt(row['B_minus_A_b_fbeta'],4)}|")
    lines.extend(["", "## 结论边界", "",
                  "训练种子和推理种子分别统计；共享的初始 checkpoint 和同组训练种子导致运行并非全部独立。",
                  "相同推理 seed 不代表不同模型生成相同采样路线。Attention 是否影响最终策略还需结合选中 α、训练日志中的权重变化和路线结果判断。",
                  "论文 Fβ 仅作记录对照；未统一其评价预算，不据此宣称超过论文。",
                  "逐次原始数据在 `all_test_instances.csv`，运行汇总在 `all_test_runs.csv`，逐例配对表在 `per_case_comparison.csv`。", ""])
    (RESULTS / "诊断报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(RESULTS / "诊断报告.md")


if __name__ == "__main__":
    main()
