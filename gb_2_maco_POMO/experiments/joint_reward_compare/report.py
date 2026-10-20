"""Five-column comparison built only after both formal tests finish."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from gb_2_maco_POMO.experiments.dual_reward.train import save_csv, save_json

LABELS = ("A0", "J1_shortest", "J1_diversity", "J2_shortest", "J2_diversity")
METRICS = ("raw_best", "raw_gap_pct", "a_best", "a_gap_pct", "a_routes", "a_hits",
           "a_precision", "a_recall", "a_fbeta", "b_best", "b_gap_pct", "b_routes",
           "b_hits", "b_precision", "b_recall", "b_fbeta", "regions_hit")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build(root: Path, j1_output: Path, j2_output: Path) -> None:
    groups = {"J1": j1_output, "J2": j2_output}
    source = {name: _json(path / "test/summary.json") for name, path in groups.items()}
    if source["J1"]["A0"] != source["J2"]["A0"]:
        raise RuntimeError("A0 formal-test baseline differs between groups")
    summaries = {"A0": source["J1"]["A0"]}
    rows = {}
    for group, path in groups.items():
        for key in ("shortest", "diversity"):
            summaries[f"{group}_{key}"] = source[group][f"Joint_{key}"]
        for row in _csv(path / "test/per_instance.csv"):
            label = "A0" if row["mode"] == "A0" else f"{group}_{row['mode'].split('_',1)[1]}"
            if label == "A0" and label in rows.get(row["instance"], {}):
                continue
            rows.setdefault(row["instance"], {})[label] = row
    if len(rows) != 25 or any(set(record) != set(LABELS) for record in rows.values()):
        raise RuntimeError("five-group formal-test rows are incomplete")
    paired = []
    for index in range(1, 26):
        name = f"MSTSP{index}"
        base = rows[name]["A0"]
        row = {"instance": name, "reference_length": base["reference_length"],
               "truth_count": base["truth_count"], "known_regions": base["known_regions"],
               "paper_fbeta": base["paper_fbeta"]}
        for label in LABELS:
            row.update({f"{label}_{metric}": rows[name][label][metric] for metric in METRICS})
        paired.append(row)
    save_csv(root / "paired_per_instance.csv", paired)
    training = {group: _json(path / "train_result.json") for group, path in groups.items()}
    selections = {group: _json(path / "selection.json") for group, path in groups.items()}
    save_json(root / "five_group_summary.json", {"formal_test": summaries,
              "training": training, "selection": selections})
    lines = ["# Joint Head + 新奖励机制双组对照", "",
             "## 协议", "",
             "A0 为原两级球—城市策略；J1、J2 均从同一 A0 checkpoint 独立初始化，只训练 Joint Head。"
             "两组各训练 10 epoch，seed=42，500 例训练、100 例独立验证，epoch 0/2/4/6/8/10 验证并按"
             "同一规则分别选择最短及多样性 checkpoint。J1 使用原始闭环距离奖励；J2 使用归一化的"
             "0.5 原始距离 + 0.5 有限 2-opt 距离 + 0.1 结构新颖性 Advantage。"
             "正式测试均为 MSTSP1–25、每城市起点 1 Argmax+8 Sampling、完整普通 2-opt、"
             "每例 50,000 次边对检查的 Opt-B；A0 复用经协议和 SHA 校验的历史结果。", "",
             "J2 训练 2-opt 每条原始采样路线最多检查 2048 个边对，仅计算奖励；"
             "策略梯度仍回传到原始路线的 log-probability。结构新颖性在同算例本批次 2-opt 距离"
             "不超过最佳值 1.005 倍的不同结构之间计算。", "",
             "## 核心结果", "",
             "| 模型 | 选中 epoch | 最优距离 | 真最优回路 | 最优区域 | 平均原始 Gap% | 2-opt 真最优回路 | 2-opt Fβ | Opt-B Fβ | 训练秒 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        summary = summaries[label]
        group = label.split("_")[0]
        seconds = "历史基准" if group == "A0" else f"{training[group]['training_seconds']:.1f}"
        lines.append(f"| {label} | {summary['selected_epoch']} | {summary['distance_hits']}/25 | "
                     f"{summary['opt_b_hits']} | {summary['regions_hit']} | "
                     f"{summary['mean_raw_gap_pct']:.4f} | {summary['two_opt_hits']} | "
                     f"{summary['mean_two_opt_fbeta']:.6f} | {summary['mean_opt_b_fbeta']:.6f} | {seconds} |")
    lines += ["", "| 模型 | checkpoint SHA256 |", "|---|---|"]
    for label in LABELS:
        lines.append(f"| {label} | `{summaries[label]['checkpoint_sha256']}` |")
    lines += ["", "J2 有限 2-opt 奖励总耗时："
              f"{training['J2']['two_opt_reward_seconds']:.1f} 秒，"
              f"检查边对 {training['J2']['two_opt_pair_checks']} 次。", "",
              "## MSTSP1–25 全部逐例数据", "",
              "下表每格固定为 A0 / J1 最短 / J1 多样性 / J2 最短 / J2 多样性。"
              "Gap=(实测距离/`.solution` 参考距离−1)×100%；真最优回路按旋转、反向等价去重。", "",
              "| 算例 | 参考距离 | 原始距离 | 2-opt 距离 | Opt-B 距离 | 原始 Gap% | 2-opt Gap% | Opt-B Gap% |",
              "|---|---:|---|---|---|---|---|---|"]
    def cell(row, field, digits=0):
        return " / ".join(f"{float(row[f'{label}_{field}']):.{digits}f}" for label in LABELS)
    for row in paired:
        lines.append(f"| {row['instance']} | {row['reference_length']} | {cell(row,'raw_best')} | "
                     f"{cell(row,'a_best')} | {cell(row,'b_best')} | {cell(row,'raw_gap_pct',2)} | "
                     f"{cell(row,'a_gap_pct',2)} | {cell(row,'b_gap_pct',2)} |")
    lines += ["", "## MSTSP1–25 回路覆盖", "",
              "| 算例 | 已知真最优数 | 2-opt 真最优命中 | Opt-B 去重回路 | Opt-B 真最优命中 | Opt-B Precision | Opt-B Recall | 最优区域 |",
              "|---|---:|---|---|---|---|---|---|"]
    for row in paired:
        lines.append(f"| {row['instance']} | {row['truth_count']} | {cell(row,'a_hits')} | "
                     f"{cell(row,'b_routes')} | {cell(row,'b_hits')} | "
                     f"{cell(row,'b_precision',3)} | {cell(row,'b_recall',3)} | "
                     f"{cell(row,'regions_hit')} |")
    lines += ["", "## MSTSP1–25 Fβ", "",
              "| 算例 | 2-opt Fβ | Opt-B Fβ | 论文 Fβ |",
              "|---|---|---|---:|"]
    for row in paired:
        lines.append(f"| {row['instance']} | {cell(row,'a_fbeta',4)} | "
                     f"{cell(row,'b_fbeta',4)} | {float(row['paper_fbeta']):.4f} |")
    lines += ["", "## MSTSP22、25 专项", "",
              "最终路线的差异不能直接证明某一步模型决策出错；本节只对比已保存的采样与后处理结果。", "",
              "| 算例 | 指标 | A0 | J1 最短 | J1 多样性 | J2 最短 | J2 多样性 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name in ("MSTSP22", "MSTSP25"):
        row = next(item for item in paired if item["instance"] == name)
        for field, description, digits in (("raw_best", "原始最短距离",0),
                                           ("a_best", "2-opt 最短距离",0),
                                           ("b_best", "Opt-B 最短距离",0),
                                           ("b_hits", "真最优回路",0),
                                           ("regions_hit", "最优区域",0),
                                           ("b_fbeta", "Opt-B Fβ",4)):
            values = " | ".join(f"{float(row[f'{label}_{field}']):.{digits}f}" for label in LABELS)
            lines.append(f"| {name} | {description} | {values} |")
    structure_rows = _csv(root / "raw_structure_diagnostics.csv")
    lines += ["", "## 原始路线的接口与球内结构", "",
              "下表的不同结构数来自完整原始采样集合，不含 2-opt 或 `.solution` 信息；"
              "贪心跨球边变化按相同起点与 A0 对比，不能单独推断哪一个选择正确。", "",
              "| 算例 | 指标 | A0 | J1 最短 | J1 多样性 | J2 最短 | J2 多样性 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name in ("MSTSP22", "MSTSP25"):
        row = next(item for item in structure_rows if item["instance"] == name)
        for key, title in (("raw_cross_structures", "原始跨球边结构数"),
                           ("raw_within_structures", "原始球内边结构数"),
                           ("greedy_cross_edge_set_changed_vs_A0", "贪心跨球连接变化起点数")):
            values = " | ".join(row[f"{label}_{key}"] for label in LABELS)
            lines.append(f"| {name} | {title} | {values} |")
    a0 = summaries["A0"]["mean_opt_b_fbeta"]
    j1 = max(summaries[f"J1_{kind}"]["mean_opt_b_fbeta"] for kind in ("shortest","diversity"))
    j2 = max(summaries[f"J2_{kind}"]["mean_opt_b_fbeta"] for kind in ("shortest","diversity"))
    qualifying = [label for label in LABELS[1:]
                  if summaries[label]["distance_hits"] == summaries["A0"]["distance_hits"]
                  and summaries[label]["opt_b_hits"] > summaries["A0"]["opt_b_hits"]
                  and summaries[label]["mean_opt_b_fbeta"] > a0]
    lines += ["", "## 结论与边界", "",
              f"J1 最高 Fβ={j1:.6f}，A0={a0:.6f}；J2 最高 Fβ={j2:.6f}。"
              f"J1 相比 A0 {'提高' if j1 > a0 else '未提高'}平均 Fβ；"
              f"J2 相比 J1 {'提高' if j2 > j1 else '未提高'}最高平均 Fβ。"
              f"同时保持 A0 距离命中、增加真最优回路且提高 Fβ 的模型："
              f"{', '.join(qualifying) if qualifying else '无'}。"
              "应同时核对距离命中、回路数与区域覆盖，不能凭单个指标认定总体改善。"
              "本次仅 seed=42、一次独立训练；J2 2-opt 是有限训练奖励，不等同正式测试的完整普通 2-opt。"
              "训练后最优区域及真最优回路只在正式测试评价，未进入训练奖励。", "",
              "可复核：`paired_per_instance.csv`、`five_group_summary.json`、各组 `training_history.csv`、"
              "`validation_history.csv`、`selection.json`、`test/per_instance.csv` 和 "
              "`raw_structure_diagnostics.csv`。", ""]
    (root / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
