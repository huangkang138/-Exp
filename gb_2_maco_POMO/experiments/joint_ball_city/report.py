"""Full 25-case Chinese report from saved validation and formal-test records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

LABELS = ("A0", "Joint_shortest", "Joint_diversity")


def _csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _cell(row: dict, key: str, digits: int = 0) -> str:
    return " / ".join(f"{float(row[f'{label}_{key}']):.{digits}f}" for label in LABELS)


def create(root: Path) -> None:
    summary = json.loads((root / "test/summary.json").read_text(encoding="utf-8"))
    paired = _csv(root / "test/paired_per_instance.csv")
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    costs = json.loads((root / "inference_cost.json").read_text(encoding="utf-8"))
    history = _csv(root / "validation_history.csv")
    validation_epochs = list(range(0, config["epochs"] + 1, config["validation_interval"]))
    if len(paired) != 25 or set(summary) != set(LABELS) or len(history) != len(validation_epochs):
        raise RuntimeError("report source records are incomplete")
    for label in LABELS:
        selected = 0 if label == "A0" else summary[label]["selected_epoch"]
        rows = _csv(root / f"validation_epoch_{selected}.csv")
        if len(rows) != 100:
            raise RuntimeError(f"{label}: independent validation count differs")
    lines = ["# 球—城市跨层联合决策实验", "",
        "## 协议与可比性", "",
        f"数据：`{config['data_root']}`，500 训练例、100 独立验证例；验证集其中 56 例有认证最优值。"
        f"A0 checkpoint SHA256：`{config['initial_checkpoint_sha256']}`。"
        f"仅 Joint Head 可训练，Encoder、Ball Head、城市 Decoder 冻结；"
        f"seed={config['seed']}，CUDA，训练 {config['epochs']} epoch，"
        f"batch={config['batch_size']}，microbatch={config['micro_batch_size']}，"
        f"Adam 学习率={config['learning_rate']}。", "",
        f"验证时机：epoch {'/'.join(map(str, validation_epochs))}。最短 best 复用认证 Gap、命中数及启发式距离的既有选模顺序；"
        "多样性 best 在最短 best 的认证 Gap +0.5 个百分点、命中数最多减少 1 例的范围内，"
        "按认证最优 Opt-B 回路均数择优。无认证的 44 例不计算真实最优 Gap。", "",
        "正式测试统一在 CPU：MSTSP1–25，seed=42，每城市起点 1 Argmax + 8 Sampling，"
        "不同原始回路均运行完整普通 2-opt，Opt-B 每例最多 50,000 次边对检查。"
        "旋转与反向按原 canonical 规则去重；`.solution` 仅用于结果评价。"
        "A0 复用相同协议的已保存正式测试记录，Joint 两个 checkpoint 分别测试，未合并路线。", "",
        "**下列表格每格顺序固定为 A0 / Joint 最短 best / Joint 多样性 best。**", "",
        "| 模型 | epoch | checkpoint SHA256 |", "|---|---:|---|"]
    for label in LABELS:
        s = summary[label]
        lines.append(f"| {label} | {s['selected_epoch']} | `{s['checkpoint_sha256']}` |")
    lines += ["", "## 正确性与决策变化", "",
        f"epoch 0：Joint Head 输出恒为零，联合概率最大误差 "
        f"{preflight['maximum_joint_probability_error']:.3g}；全部 25 例的 Argmax 与 Sampling 路线"
        "在相同种子下与 A0 一致。原模型参数在探针更新和正式训练中保持不变。"
        f"Joint 输出层初始梯度范数={preflight['joint_output_gradient_norm']:.6g}。", "",
        "| epoch | 认证最优命中/56 | 认证平均 Gap% | 认证 Opt-B 回路均数 | 球 argmax 变化 | 城市 argmax 变化 | 球概率 L1 | 城市概率 L1 | 联合概率 L1 | 修正均值 | 修正标准差均值 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in history:
        lines.append(f"| {r['epoch']} | {r['certified_hits']}/56 | "
                     f"{float(r['mean_certified_gap_pct']):.4f} | "
                     f"{float(r['mean_certified_optb_routes']):.4f} | "
                     f"{r['ball_argmax_changes']} | {r['city_argmax_changes']} | "
                     f"{float(r['ball_probability_l1_mean']):.5f} | "
                     f"{float(r['city_probability_l1_mean']):.5f} | "
                     f"{float(r['joint_probability_l1_mean']):.5f} | "
                     f"{float(r['score_mean']):.5f} | {float(r['score_std_mean']):.5f} |")
    lines += ["", "选模记录：`selection.json`；逐例验证原始距离、2-opt、Opt-B、唯一回路数、"
              "球间/球内距离、跨球次数与决策变化见 `validation_epoch_*.csv`。", "",
              "## 核心正式测试结果", "",
              "| 模型 | 原始距离命中 | 2-opt 距离命中 | Opt-B 距离命中 | 2-opt 真最优回路 | Opt-B 真最优回路 | 最优区域 | 原始平均 Gap% | 2-opt 平均 Gap% | Opt-B 平均 Gap% | 平均 Fβ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        s = summary[label]
        lines.append(f"| {label} | {s['raw_distance_hits']}/25 | "
                     f"{s['two_opt_distance_hits']}/25 | {s['distance_hits']}/25 | "
                     f"{s['two_opt_hits']} | {s['opt_b_hits']} | {s['regions_hit']} | "
                     f"{s['mean_raw_gap_pct']:.4f} | {s['mean_two_opt_gap_pct']:.4f} | "
                     f"{s['mean_opt_b_gap_pct']:.4f} | {s['mean_opt_b_fbeta']:.6f} |")
    lines += ["", "## 参数量与相同设备推理成本", "",
              "推理成本在 CUDA 上按相同 25 例、相同起点和采样次数单独计时；"
              "不含 TSP 预处理和后处理。显存为 PyTorch 峰值已分配显存。", "",
              "| 模型 | 参数量 | 新增参数 | 25 例推理秒 | 相对 A0 耗时 | 峰值显存 MiB |",
              "|---|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        c = costs[label]
        ratio = 1.0 if label == "A0" else c["time_ratio_to_A0"]
        lines.append(f"| {label} | {c['parameters']} | {c['new_joint_parameters']} | "
                     f"{c['full_25_inference_seconds']:.3f} | {ratio:.3f}× | "
                     f"{c['cuda_peak_allocated_bytes'] / 1048576:.1f} |")
    lines += ["", "## MSTSP1–25 距离与 Gap", "",
              "Gap = (实测距离 / `.solution` 参考最优距离 − 1) × 100%。", "",
              "| 算例 | 参考距离 | 原始最短距离 | 原始 Gap% | 2-opt 距离 | 2-opt Gap% | Opt-B 距离 | Opt-B Gap% |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['instance']} | {r['reference_length']} | {_cell(r, 'raw_best')} | "
                     f"{_cell(r, 'raw_gap_pct', 2)} | {_cell(r, 'a_best')} | "
                     f"{_cell(r, 'a_gap_pct', 2)} | {_cell(r, 'b_best')} | "
                     f"{_cell(r, 'b_gap_pct', 2)} |")
    lines += ["", "## MSTSP1–25 最优回路覆盖", "",
              "| 算例 | 已知真最优数 | 2-opt 真最优命中 | Opt-B 去重回路数 | Opt-B 真最优命中 | Precision | Recall | 最优区域 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['instance']} | {r['truth_count']} | {_cell(r, 'a_hits')} | "
                     f"{_cell(r, 'b_routes')} | {_cell(r, 'b_hits')} | "
                     f"{_cell(r, 'b_precision', 3)} | {_cell(r, 'b_recall', 3)} | "
                     f"{_cell(r, 'regions_hit')} |")
    lines += ["", "## MSTSP1–25 Fβ", "",
              "论文只提供 Fβ，不能据此推断论文的距离或路线覆盖。", "",
              "| 算例 | 2-opt Fβ | Opt-B Fβ | 论文 Fβ | Opt-B 减论文 Fβ |",
              "|---|---:|---:|---:|---:|"]
    for r in paired:
        delta = " / ".join(f"{float(r[f'{label}_b_fbeta']) - float(r['paper_fbeta']):+.4f}"
                           for label in LABELS)
        lines.append(f"| {r['instance']} | {_cell(r, 'a_fbeta', 4)} | "
                     f"{_cell(r, 'b_fbeta', 4)} | {float(r['paper_fbeta']):.4f} | {delta} |")
    lines += ["", "## 关键算例与判断边界", "",
              "下表列出事先指定的重点算例，顺序仍为 A0 / 最短 / 多样性。", "",
              "| 算例 | Opt-B 距离 | 真最优回路 | 区域 | Opt-B Fβ |",
              "|---|---:|---:|---:|---:|"]
    for r in paired:
        if int(r["instance"].replace("MSTSP", "")) in (13, 16, 17, 19, 21, 22, 23, 24, 25):
            lines.append(f"| {r['instance']} | {_cell(r, 'b_best')} | "
                         f"{_cell(r, 'b_hits')} | {_cell(r, 'regions_hit')} | "
                         f"{_cell(r, 'b_fbeta', 4)} |")
    better = [label for label in LABELS[1:]
              if summary[label]["mean_opt_b_fbeta"] > summary["A0"]["mean_opt_b_fbeta"]]
    lines += ["", f"超过 A0 平均 Fβ 的新 checkpoint：{', '.join(better) if better else '无'}。"
              "路线质量与多解覆盖必须同时比较；单个种子和一次训练不支持普遍效果断言。"
              "决策诊断比较的是同一贪心路线状态的概率变化；若距离或 Fβ 下降，"
              "只有结合逐例球/城市概率、路线唯一数与跨球指标才能定位原因，不能单凭最终分数推断。", "",
              "可复核数据：`test/per_instance.csv`、`test/paired_per_instance.csv`、"
              "`test/summary.json`、`validation_epoch_*.csv`、`validation_history.csv`、"
              "`inference_cost_per_instance.csv`、`inference_cost.json`。", ""]
    (root / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
