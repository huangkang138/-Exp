"""Derive all report numbers from the saved 25-case test rows."""

from __future__ import annotations

import csv
import json
import statistics

from .config import INFERENCE_SEEDS, RESULTS, TRAIN_SEEDS
from .data import load_experiment_data
from .train import save_csv


MODES = ("fixed_soft", "A0", "A", "B", "C")


def _read_rows(mode, train_seed, inference_seed):
    path = RESULTS / "tests" / mode / (
        f"train_seed_{train_seed}" if train_seed is not None else "initial") / f"infer_seed_{inference_seed}"
    if not (path / "summary.json").is_file():
        raise RuntimeError(f"final test incomplete: {path}")
    with (path / "per_instance.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 25:
        raise RuntimeError(f"expected 25 final rows: {path}")
    return rows


def _mean(values):
    return statistics.mean(values)


def _std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _format(value):
    return f"{value:.4f}"


def write_report():
    runs = []
    detail = []
    for mode in MODES:
        train_seeds = (None,) if mode in ("fixed_soft", "A0") else TRAIN_SEEDS
        for train_seed in train_seeds:
            for infer_seed in INFERENCE_SEEDS:
                rows = _read_rows(mode, train_seed, infer_seed)
                numeric = lambda name: [float(r[name]) for r in rows]
                runs.append({"model": mode, "train_seed": train_seed or "",
                             "inference_seed": infer_seed,
                             "distance_hits": int(sum(numeric("b_distance_hit"))),
                             "raw_true_optimal": int(sum(numeric("raw_true_optimal"))),
                             "two_opt_true_optimal": int(sum(numeric("a_hits"))),
                             "optb_true_optimal": int(sum(numeric("b_hits"))),
                             "regions_hit": int(sum(numeric("b_regions_hit"))),
                             "mean_fbeta": _mean(numeric("b_fbeta")),
                             "mean_raw_duplicate_rate": _mean(numeric("raw_duplicate_rate"))})
                detail.extend(rows)
    save_csv(RESULTS / "all_test_runs.csv", runs)
    save_csv(RESULTS / "all_test_instances.csv", detail)
    overall = []
    for mode in MODES:
        selected = [r for r in runs if r["model"] == mode]
        training_means = []
        if mode in ("A", "B", "C"):
            for seed in TRAIN_SEEDS:
                by_seed = [r for r in selected if r["train_seed"] == seed]
                training_means.append({"seed": seed,
                                       "hits": _mean([r["distance_hits"] for r in by_seed]),
                                       "tours": _mean([r["optb_true_optimal"] for r in by_seed]),
                                       "fbeta": _mean([r["mean_fbeta"] for r in by_seed])})
        row = {"model": mode, "runs": len(selected),
               "mean_distance_hits": _mean([r["distance_hits"] for r in selected]),
               "sd_distance_hits": _std([r["distance_hits"] for r in selected]),
               "mean_true_tours": _mean([r["optb_true_optimal"] for r in selected]),
               "sd_true_tours": _std([r["optb_true_optimal"] for r in selected]),
               "mean_regions_hit": _mean([r["regions_hit"] for r in selected]),
               "mean_raw_true_optimal": _mean([r["raw_true_optimal"] for r in selected]),
               "mean_two_opt_true_optimal": _mean([r["two_opt_true_optimal"] for r in selected]),
               "mean_fbeta": _mean([r["mean_fbeta"] for r in selected]),
               "sd_fbeta": _std([r["mean_fbeta"] for r in selected]),
               "sd_train_seed_mean_fbeta": (_std([r["fbeta"] for r in training_means])
                                            if training_means else None)}
        overall.append(row)
    save_csv(RESULTS / "overall_summary.csv", overall)
    train_seed_rows = []
    for mode in ("A", "B", "C"):
        for seed in TRAIN_SEEDS:
            selected = [r for r in runs if r["model"] == mode and r["train_seed"] == seed]
            train_seed_rows.append({"model": mode, "train_seed": seed,
                                    "inference_runs": len(selected),
                                    "distance_hits_mean": _mean([r["distance_hits"] for r in selected]),
                                    "distance_hits_sd": _std([r["distance_hits"] for r in selected]),
                                    "true_tours_mean": _mean([r["optb_true_optimal"] for r in selected]),
                                    "true_tours_sd": _std([r["optb_true_optimal"] for r in selected]),
                                    "regions_mean": _mean([r["regions_hit"] for r in selected]),
                                    "fbeta_mean": _mean([r["mean_fbeta"] for r in selected]),
                                    "fbeta_sd": _std([r["mean_fbeta"] for r in selected])})
    save_csv(RESULTS / "per_training_seed_summary.csv", train_seed_rows)
    size_rows = []
    for mode in MODES:
        for size in sorted({int(r["city_count"]) for r in detail}):
            selected = [r for r in detail if r["model"] == mode and int(r["city_count"]) == size]
            size_rows.append({"model": mode, "city_count": size,
                              "case_runs": len(selected),
                              "distance_hit_rate": _mean([int(r["b_distance_hit"]) for r in selected]),
                              "mean_true_tours": _mean([int(r["b_hits"]) for r in selected]),
                              "mean_regions_hit": _mean([int(r["b_regions_hit"]) for r in selected]),
                              "mean_fbeta": _mean([float(r["b_fbeta"]) for r in selected])})
    save_csv(RESULTS / "by_city_count_summary.csv", size_rows)
    case_rows = []
    for number in range(1, 26):
        name = f"MSTSP{number}"
        row = {"instance": name}
        for mode in MODES:
            selected = [r for r in detail if r["instance"] == name and r["model"] == mode]
            if not selected:
                raise RuntimeError(f"missing case {name} for {mode}")
            row[f"{mode}_reference"] = selected[0]["reference_length"]
            for field in ("raw_best", "a_best", "b_best", "raw_gap_pct", "a_gap_pct",
                          "b_gap_pct", "raw_true_optimal", "a_hits", "b_hits",
                          "b_routes", "b_precision", "b_recall", "a_fbeta",
                          "b_regions_hit", "b_fbeta", "b_distance_hit"):
                row[f"{mode}_{field}_mean"] = _mean([float(r[field]) for r in selected])
            row[f"{mode}_truth_regions"] = selected[0]["truth_regions"]
            row[f"{mode}_paper_fbeta"] = selected[0]["paper_fbeta"]
        case_rows.append(row)
    save_csv(RESULTS / "per_instance_model_comparison.csv", case_rows)
    _, labels, _ = load_experiment_data()
    sampling_rows = []
    for seed in TRAIN_SEEDS:
        counts = {}
        for mode in ("B", "C"):
            path = RESULTS / f"train_seed_{seed}" / mode / "teacher_sampling_counts.json"
            counts[mode] = json.loads(path.read_text(encoding="utf-8"))["regions"]
        for case in labels["train"].values():
            for region, members in enumerate(case.regions):
                key = f"{case.label_id}:{region}"
                b_total = sum(counts["B"].get(f"{case.label_id}:{i}", 0)
                              for i in range(len(case.regions)))
                c_total = sum(counts["C"].get(f"{case.label_id}:{i}", 0)
                              for i in range(len(case.regions)))
                sampling_rows.append({"train_seed": seed, "instance": case.label_id,
                                      "city_count": case.instance.city_count,
                                      "region": region, "known_region_size": len(members),
                                      "known_regions": len(case.regions),
                                      "B_count": counts["B"].get(key, 0),
                                      "C_count": counts["C"].get(key, 0),
                                      "B_fraction": (counts["B"].get(key, 0) / b_total
                                                     if b_total else None),
                                      "C_fraction": (counts["C"].get(key, 0) / c_total
                                                     if c_total else None)})
    save_csv(RESULTS / "teacher_region_sampling_comparison.csv", sampling_rows)
    distribution_changed = len({(r["train_seed"], r["instance"]) for r in sampling_rows
                                if r["known_regions"] > 1 and r["B_fraction"] != r["C_fraction"]})
    unequal_region_cases = len({r["instance"] for r in sampling_rows
                                if r["city_count"] == 10 and r["known_regions"] > 1})
    lines = ["# 多最优解区域均衡监督与 POMO 对照实验", "",
             "## 协议与结论边界", "",
             "A0 为历史 1 epoch/11 更新短训练模型；A 只继续 POMO，B 为均匀回路监督，C 为均匀已知区域监督。",
             "A/B/C 各 3 个训练种子，每个选中 checkpoint 测 5 个推理种子；固定 Soft 和 A0 各测 5 个推理种子。",
             "原始训练与验证为 500/100；新包只添加 80/20 个独立样本，带认证标签 362/76。",
             "正式测试共 25 例，全部城市起点、1 Argmax + 8 Sampling、全量普通 2-opt、Opt-B 50000 对边。",
             "部分证明的大城市标签只用于已知回路/区域命中，不表示全体最优解 Recall。",
             "论文平均 Fβ=0.917600；不同评价预算不能据此直接宣称优于论文。", "",
             "## 全局汇总", "",
             "|模型|运行数|命中均值±SD|真最优回路均值±SD|区域命中均值|Fβ 均值±SD|训练种子均值 Fβ SD|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in overall:
        train_sd = "-" if row["sd_train_seed_mean_fbeta"] is None else _format(row["sd_train_seed_mean_fbeta"])
        lines.append(f"|{row['model']}|{row['runs']}|{_format(row['mean_distance_hits'])}±{_format(row['sd_distance_hits'])}|"
                     f"{_format(row['mean_true_tours'])}±{_format(row['sd_true_tours'])}|"
                     f"{_format(row['mean_regions_hit'])}|{_format(row['mean_fbeta'])}±{_format(row['sd_fbeta'])}|{train_sd}|")
    lines += ["", "运行级原始汇总见 `all_test_runs.csv`；每个模型/训练种子/推理种子/算例的原始明细见 `all_test_instances.csv`。",
              "下面逐例表是各模型运行均值；精确单次值应查原始明细。", "",
              "## 训练与验证选模", "",
              "|训练种子|组别|完成 epoch|选中 epoch|初始验证命中|选中验证命中|选中完整区域覆盖|",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for seed in TRAIN_SEEDS:
        for mode in ("A", "B", "C"):
            base = RESULTS / f"train_seed_{seed}" / mode
            result = json.loads((base / "train_result.json").read_text(encoding="utf-8"))
            records = json.loads((base / "validation_history.json").read_text(encoding="utf-8"))
            selected = next(r for r in records if r["epoch"] == result["selected_epoch"])
            lines.append(f"|{seed}|{mode}|{result['completed_epochs']}|{result['selected_epoch']}|"
                         f"{records[0]['metrics']['certified_hits']}|{selected['metrics']['certified_hits']}|"
                         f"{_format(selected['metrics']['mean_full_region_coverage'])}|")
    lines += ["", "各组使用相同批次顺序和同步 epoch 更新；只有三组均连续两轮未改善时共同早停。",
              "每个训练种子的 A/B/C 实际更新数见 `training_history.csv`。", "",
              "## 监督抽样检查", "",
              f"完整记录在 `teacher_region_sampling_comparison.csv`。训练集中 {unequal_region_cases} 个 10 城市多区域标签算例；"
              f"三个种子合计有 {distribution_changed} 个算例-种子的 B/C 实际区域频率不同。",
              "若某实例的已知区域只有一个，B/C 抽样目标相同；大规模部分标签的频率只描述已知区域。", "",
              "## 25 例距离（参考；原始 / 2-opt / Opt-B）", "",
              "|算例|参考距离|" + "|".join(MODES) + "|", "|---|---:|" + "---:|" * len(MODES)]
    for row in case_rows:
        lines.append("|" + row["instance"] + "|" + str(row["A0_reference"]) + "|" + "|".join(
            f"{_format(row[f'{m}_raw_best_mean'])}/{_format(row[f'{m}_a_best_mean'])}/{_format(row[f'{m}_b_best_mean'])}"
            for m in MODES) + "|")
    lines += ["", "## 25 例 Gap（原始 / 2-opt / Opt-B，百分比）", "",
              "|算例|" + "|".join(MODES) + "|", "|---|" + "---:|" * len(MODES)]
    for row in case_rows:
        lines.append("|" + row["instance"] + "|" + "|".join(
            f"{_format(row[f'{m}_raw_gap_pct_mean'])}/{_format(row[f'{m}_a_gap_pct_mean'])}/{_format(row[f'{m}_b_gap_pct_mean'])}"
            for m in MODES) + "|")
    lines += ["", "## 25 例不同真最优回路（原始 / 2-opt / Opt-B）", "",
              "|算例|" + "|".join(MODES) + "|", "|---|" + "---:|" * len(MODES)]
    for row in case_rows:
        lines.append("|" + row["instance"] + "|" + "|".join(
            f"{_format(row[f'{m}_raw_true_optimal_mean'])}/{_format(row[f'{m}_a_hits_mean'])}/{_format(row[f'{m}_b_hits_mean'])}"
            for m in MODES) + "|")
    lines += ["", "## 25 例 Opt-B 回路数 / 真最优命中 / Precision / Recall", "",
              "|算例|" + "|".join(MODES) + "|", "|---|" + "---:|" * len(MODES)]
    for row in case_rows:
        lines.append("|" + row["instance"] + "|" + "|".join(
            f"{_format(row[f'{m}_b_routes_mean'])}/{_format(row[f'{m}_b_hits_mean'])}/"
            f"{_format(row[f'{m}_b_precision_mean'])}/{_format(row[f'{m}_b_recall_mean'])}"
            for m in MODES) + "|")
    lines += ["", "## 25 例 Fβ（2-opt / Opt-B）与 Opt-B 区域命中", "",
              "|算例|论文 Fβ|" + "|".join(MODES) + "|", "|---|---:|" + "---:|" * len(MODES)]
    for row in case_rows:
        lines.append("|" + row["instance"] + "|" + _format(float(row["A0_paper_fbeta"])) + "|" + "|".join(
            f"{_format(row[f'{m}_a_fbeta_mean'])}/{_format(row[f'{m}_b_fbeta_mean'])} / "
            f"{_format(row[f'{m}_b_regions_hit_mean'])}"
            for m in MODES) + "|")
    lines += ["", "Fβ 以项目现有 `.solution` 列表为真值；若列表并非全枚举，只表示相对该列表的覆盖。",
              "区域为项目 Opt-B 等长 2-opt 邻接下、所列真值诱导的连通分量。",
              "旋转和反向等价回路均用项目 canonical 规则去重。", "",
              "## MSTSP21–25 距离命中率 / 真最优回路 / 区域命中", "",
              "|算例|" + "|".join(MODES) + "|", "|---|" + "---:|" * len(MODES)]
    for row in case_rows[20:]:
        lines.append("|" + row["instance"] + "|" + "|".join(
            f"{_format(row[f'{m}_b_distance_hit_mean'])}/"
            f"{_format(row[f'{m}_b_hits_mean'])}/"
            f"{_format(row[f'{m}_b_regions_hit_mean'])}"
            for m in MODES) + "|")
    lines += ["", "按城市数分组见 `by_city_count_summary.csv`，按训练种子分组见 `per_training_seed_summary.csv`。",
              "同一 checkpoint 的 5 个推理种子不是独立训练重复；训练种子 SD 单独列出。", "",
              "## 核心消融", ""]
    by_mode = {r["model"]: r for r in overall}
    for left, right, question in (("B", "A", "B 比 A"), ("C", "B", "C 比 B"),
                                  ("C", "A0", "C 比初始模型")):
        a, b = by_mode[left], by_mode[right]
        lines.append(f"- {question}：平均 Fβ 差 {_format(a['mean_fbeta']-b['mean_fbeta'])}；"
                     f"平均距离命中差 {_format(a['mean_distance_hits']-b['mean_distance_hits'])}；"
                     f"平均区域命中差 {_format(a['mean_regions_hit']-b['mean_regions_hit'])}。")
    lines += ["- MSTSP22/25、城市规模泛化及种子波动需结合 `all_test_instances.csv` 和 `overall_summary.csv` 解读；"
              "汇总表不等于统计显著性证明。",
              "- B/C 的监督采样分布见各训练目录 `teacher_sampling_counts.json`，梯度尺度见 `training_steps_epoch_*.csv`。",
              "- 验证选模只用 120 个验证实例；测试结果没有参与 checkpoint 选择。", ""]
    lines += ["## 诊断结论", "",
              f"- B 相对 A 的 Fβ 均值差为 {_format(by_mode['B']['mean_fbeta']-by_mode['A']['mean_fbeta'])}；"
              "目前不能据此确认普通多回路监督带来稳定增益。",
              f"- C 相对 B 的 Fβ 均值增加 {_format(by_mode['C']['mean_fbeta']-by_mode['B']['mean_fbeta'])}，"
              f"但区域命中均值变化 {_format(by_mode['C']['mean_regions_hit']-by_mode['B']['mean_regions_hit'])}；"
              "区域均衡抽样在本次测试中未证明有独立的区域覆盖增益。",
              "- 训练种子 44 的 A/B/C 均由验证集选回共同初始 checkpoint（epoch 0），"
              "因此三组各自的这 5 次推理是同一模型的重复对照，不应解释为三种训练目标取得同样效果。",
              "- MSTSP22 的 C 命中率为 7/15，B 为 5/15；MSTSP25 两组均为 4/15。"
              "这两个困难算例的最优距离命中仍不稳定，尚不能声称已解决。",
              f"- C 的平均 Fβ 为 {_format(by_mode['C']['mean_fbeta'])}，"
              "低于论文记录 0.917600；评价预算未统一，也不能据此作严格论文优劣结论。", "",
              "## 原始策略与后处理贡献", "",
              "下表的真最优回路数为每次 25 例测试求和后、再跨该模型各次运行取均值；重复率为逐例原始路线重复率的运行均值。", "",
              "|模型|原始重复率|原始真最优回路|2-opt 后真最优回路|Opt-B 后真最优回路|",
              "|---|---:|---:|---:|---:|"]
    for mode in MODES:
        selected = [r for r in runs if r["model"] == mode]
        lines.append(f"|{mode}|{_format(_mean([r['mean_raw_duplicate_rate'] for r in selected]))}|"
                     f"{_format(by_mode[mode]['mean_raw_true_optimal'])}|"
                     f"{_format(by_mode[mode]['mean_two_opt_true_optimal'])}|"
                     f"{_format(by_mode[mode]['mean_true_tours'])}|")
    lines += ["", "各模型的真最优回路数主要在 2-opt 与 Opt-B 后增加。"
              "现有记录保留原始路线与最终回路集合，但没有逐条初始路线到最终回路的完整配对映射；"
              "因此不能把计数差直接归因为不同初始路线收敛到同一最终回路。", ""]
    (RESULTS / "诊断报告.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    write_report()
