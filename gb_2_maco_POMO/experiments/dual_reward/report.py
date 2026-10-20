"""Build full 25-case comparisons from saved validation and final-test records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .train import save_csv, save_json

LABELS = ("A0", "A_length", "A_multi", "B_length", "B_multi")


def _csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(r[key]) for r in rows) / len(rows)


def _cell(row: dict, key: str, digits: int = 0) -> str:
    return " / ".join(f"{float(row[f'{label}_{key}']):.{digits}f}" for label in LABELS)


def create(root: Path) -> None:
    summary = json.loads((root / "test/summary.json").read_text(encoding="utf-8"))
    paired = _csv(root / "test/paired_per_instance.csv")
    if len(paired) != 25 or set(summary) != set(LABELS):
        raise RuntimeError("formal-test comparison is incomplete")
    validation = {}
    for label in LABELS:
        if label == "A0":
            group, epoch = "A", 0
        else:
            group, kind = label.split("_")
            epoch = int(summary[label]["selected_epoch"])
        rows = _csv(root / group / f"validation_epoch_{epoch}.csv")
        if len(rows) != 100:
            raise RuntimeError(f"{label}: expected 100 validation cases")
        validation[label] = {row["instance"]: row for row in rows}
    if any(set(v) != set(validation["A0"]) for v in validation.values()):
        raise RuntimeError("validation instance sets differ")
    paired_val = []
    val_keys = ("raw_mean_length", "raw_best_length", "two_opt_best_length",
                "inter_length_at_raw_best", "intra_length_at_raw_best",
                "crossings_at_raw_best", "distinct_raw_best_routes",
                "distinct_two_opt_best_routes", "certified_optimal_length",
                "raw_optimal_gap_pct", "two_opt_optimal_gap_pct",
                "raw_optimal_hit", "two_opt_optimal_hit",
                "sampled_raw_best_length", "sampled_two_opt_best_length",
                "sampled_optb_best_length", "sampled_optb_routes",
                "sampled_certified_optb_routes", "sampled_two_opt_optimal_hit",
                "sampled_two_opt_optimal_gap_pct")
    for name in sorted(validation["A0"]):
        row = {"instance": name}
        for label in LABELS:
            source = validation[label][name]
            row.update({f"{label}_{key}": source[key] for key in val_keys})
        paired_val.append(row)
    save_csv(root / "validation_comparison.csv", paired_val)
    val_summary = {}
    for label, records in validation.items():
        rows = list(records.values())
        certified = [r for r in rows if r["certified_optimal_length"] != ""]
        if len(certified) != 56:
            raise RuntimeError(f"{label}: certified validation coverage changed")
        val_summary[label] = {
            "greedy_raw_best": _mean(rows, "raw_best_length"),
            "greedy_two_opt_best": _mean(rows, "two_opt_best_length"),
            "sampled_raw_best": _mean(rows, "sampled_raw_best_length"),
            "sampled_two_opt_best": _mean(rows, "sampled_two_opt_best_length"),
            "sampled_optb_routes": _mean(rows, "sampled_optb_routes"),
            "certified_hits": sum(int(r["sampled_two_opt_optimal_hit"]) for r in certified),
            "certified_gap_pct": _mean(certified, "sampled_two_opt_optimal_gap_pct"),
            "certified_optb_routes": _mean(certified, "sampled_certified_optb_routes"),
            "inter_length_at_raw_best": _mean(rows, "inter_length_at_raw_best"),
            "intra_length_at_raw_best": _mean(rows, "intra_length_at_raw_best"),
            "crossings_at_raw_best": _mean(rows, "crossings_at_raw_best"),
        }
    save_json(root / "validation_summary.json", val_summary)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    training = {group: json.loads((root / group / "train_result.json").read_text(encoding="utf-8"))
                for group in ("A", "B")}
    lines = ["# 球间与球内差异化奖励：10 epoch 对照", "",
        "## 协议与可比性", "",
        f"训练数据：`{config['data']}`，500 训练例、100 独立验证例；"
        f"A/B 均由 A0 `{config['initial_sha256']}` 及 Adam 状态续训，Encoder 冻结。"
        f"seed={config['seed']}，每组 {config['epochs']} epoch，batch={config['batch_size']}，"
        f"microbatch={config['micro_batch_size']}，学习率={config['learning_rate']}。"
        "A 使用原完整路线负长度奖励；B 使用球间/球内辅助 Advantage，系数 0.3。", "",
        "验证在 epoch 0/2/4/6/8/10 运行：每起点 1 Argmax + 8 Sampling，普通 2-opt 与每例 50,000 次边对检查的 Opt-B。"
        "最短 checkpoint 按 56 例认证最优 Gap、命中数及启发式距离择优；多样性 checkpoint "
        "在最短模型的认证 Gap +0.5 个百分点、命中数至多减少 1 例内，以认证最优 Opt-B 路线均数择优。"
        "另 44 例无认证最优距离，其启发式对照不叫真实 Gap。", "",
        "正式测试仅用 MSTSP1–25、seed=42、每城市起点 1 Argmax + 8 Sampling、普通 2-opt、"
        "Opt-B 每例 50,000 次边对检查；正式测试不参与选模。"
        "路线按旋转和反向 canonical 去重，真最优集合以 `.solution` 为准。", "",
        "顺序固定为 **A0 / A_length / A_multi / B_length / B_multi**。"
        "后两类 best 可能选中同一 epoch，若如此，它们的正式测试数值应相同。", "",
        "| 模型 | 用途 | epoch | checkpoint SHA256 |", "|---|---|---:|---|"]
    for label in LABELS:
        s = summary[label]
        lines.append(f"| {label} | {'基准' if label == 'A0' else '独立验证选模'} | "
                     f"{s['selected_epoch']} | `{s['checkpoint_sha256']}` |")
    lines += ["", "## 核心结果", "",
        "| 模型 | 原始距离命中 | 2-opt 距离命中 | Opt-B 距离命中 | 2-opt 真最优回路 | Opt-B 真最优回路 | 最优区域 | 2-opt 平均 Fβ | Opt-B 平均 Fβ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        s = summary[label]
        lines.append(f"| {label} | {s['raw_distance_hits']}/25 | "
                     f"{s['two_opt_distance_hits']}/25 | {s['distance_hits']}/25 | "
                     f"{s['two_opt_hits']} | {s['opt_b_hits']} | {s['regions_hit']} | "
                     f"{s['mean_two_opt_fbeta']:.6f} | {s['mean_opt_b_fbeta']:.6f} |")
    lines += ["", "## 独立验证结果", "",
        "| 模型 | 原始距离均值 | 2-opt 距离均值 | 认证距离命中 | 认证 Gap% | 认证 Opt-B 路线均数 | greedy 球间长度 | greedy 球内长度 | greedy 跨球次数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        v = val_summary[label]
        lines.append(f"| {label} | {v['sampled_raw_best']:.3f} | {v['sampled_two_opt_best']:.3f} | "
                     f"{v['certified_hits']}/56 | {v['certified_gap_pct']:.3f} | "
                     f"{v['certified_optb_routes']:.3f} | {v['inter_length_at_raw_best']:.3f} | "
                     f"{v['intra_length_at_raw_best']:.3f} | {v['crossings_at_raw_best']:.3f} |")
    lines += ["", "## MSTSP1–25 距离及 Gap", "",
        "每格依次为 A0 / A_length / A_multi / B_length / B_multi。"
        "Gap = (实测距离 / `.solution` 参考最优距离 − 1) × 100%。", "",
        "| 算例 | 参考距离 | 原始最短距离 | 原始 Gap% | 2-opt 距离 | 2-opt Gap% | Opt-B 距离 | Opt-B Gap% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['instance']} | {r['reference_length']} | {_cell(r, 'raw_best')} | "
                     f"{_cell(r, 'raw_gap_pct', 2)} | {_cell(r, 'a_best')} | "
                     f"{_cell(r, 'a_gap_pct', 2)} | {_cell(r, 'b_best')} | "
                     f"{_cell(r, 'b_gap_pct', 2)} |")
    lines += ["", "## MSTSP1–25 多解覆盖", "",
        "| 算例 | 已知真最优 | 2-opt 真最优命中 | Opt-B 去重回路数 | Opt-B 真最优命中 | Opt-B Precision | Opt-B Recall | 最优区域 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['instance']} | {r['truth_count']} | {_cell(r, 'a_hits')} | "
                     f"{_cell(r, 'b_routes')} | {_cell(r, 'b_hits')} | "
                     f"{_cell(r, 'b_precision', 3)} | {_cell(r, 'b_recall', 3)} | "
                     f"{_cell(r, 'regions_hit')} |")
    lines += ["", "## MSTSP1–25 Fβ", "",
        "论文仅提供 Fβ，不能据此推断其距离或回路数。", "",
        "| 算例 | 2-opt Fβ | Opt-B Fβ | 论文 Fβ | Opt-B 减论文 Fβ |",
        "|---|---:|---:|---:|---:|"]
    for r in paired:
        delta = " / ".join(f"{float(r[f'{label}_b_fbeta']) - float(r['paper_fbeta']):+.4f}"
                           for label in LABELS)
        lines.append(f"| {r['instance']} | {_cell(r, 'a_fbeta', 4)} | "
                     f"{_cell(r, 'b_fbeta', 4)} | {float(r['paper_fbeta']):.4f} | {delta} |")
    lines += ["", "## 判断边界与复核路径", "",
        "A 是普通续训对照，B 与 A 在相同选模目标下的差异才反映本次奖励改动的关联效果。"
        "两组都只训练一次、只用一个种子；不能据此推出普遍效果。"
        "多样性 checkpoint 可能以小幅距离损失换取路线覆盖，需同时看距离、回路数和 Fβ。", "",
        f"参数变化 L1：A Ball Head={training['A']['ball_parameter_delta_l1']:.6f}、"
        f"Decoder={training['A']['decoder_parameter_delta_l1']:.6f}；"
        f"B Ball Head={training['B']['ball_parameter_delta_l1']:.6f}、"
        f"Decoder={training['B']['decoder_parameter_delta_l1']:.6f}。", "",
        "原始明细：`test/per_instance.csv`；完整配对：`test/paired_per_instance.csv`；"
        "汇总：`test/summary.json`；验证逐例：`validation_comparison.csv`；"
        "选模记录：`A/selection.json`、`B/selection.json`；实验配置：`config.toml`。", ""]
    (root / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
