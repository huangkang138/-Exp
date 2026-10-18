"""Evaluate trained ball policy on the original 25 cases; reuse certified Soft baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import (
    generate_learned_routes, load_checkpoint,
)
from gb_2_maco_POMO.experiments.hierarchical.original25 import PAPER, _csv, run_case

PROJECT = Path(__file__).resolve().parents[3]
BASELINE = (PROJECT / "gb_2_maco_POMO/results/original25_hierarchical_20261015")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train_dir, output = args.train_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        parser.error("output directory must be new")
    if not _read(BASELINE / "baseline_gate.json")["passed"]:
        raise RuntimeError("fixed Soft baseline was not reproduced")
    training = _read(train_dir / "train_result.json")
    checkpoint = train_dir / "checkpoints" / "best.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != training["best_checkpoint_sha256"] or training["train_instances"] != 500:
        raise RuntimeError("best.pt does not match completed training")
    policy, state = load_checkpoint(checkpoint)
    if state["step"] <= 0 or state["train_instances_seen"] <= 0:
        raise RuntimeError("selected checkpoint has not trained")
    paper = _csv(PAPER)
    output.mkdir(parents=True)
    for folder in ("cases", "raw_routes", "best_routes"):
        (output / "learned_ball" / folder).mkdir(parents=True, exist_ok=True)
    _write(output / "protocol.json", {
        "trained_checkpoint": str(checkpoint), "trained_checkpoint_sha256": digest,
        "selected_training_step": state["step"],
        "validation_mean_best_length": state["validation_mean_best_length"],
        "baseline_results": str(BASELINE),
        "baseline_gate": "all 25 historical raw hashes and A/B metrics passed",
        "device": "cpu", "seed": 42, "starts": "all", "argmax": 1,
        "evaluation_torch_version": torch.__version__,
        "sampling": 8, "two_opt": "all unique raw cycles, full ordinary",
        "optb_extra_pair_checks_per_case": 50000, "beta_squared": 0.3,
    })
    rows = []
    for index in range(1, 26):
        name = f"MSTSP{index}"
        learned = run_case(policy, index, "learned_ball", 0.0, output,
                           float(paper[name]["paper_fbeta"]),
                           route_generator=generate_learned_routes)
        _write(output / "learned_ball" / "cases" / f"{name}.json", learned)
        fixed = _read(BASELINE / "fixed_soft" / "cases" / f"{name}.json")
        rows.append((fixed, learned))
        print(f"{name}: fixed B={fixed['b_best']} hits={fixed['b_hits']}; "
              f"learned B={learned['b_best']} hits={learned['b_hits']}", flush=True)
    paired = []
    for fixed, learned in rows:
        item = {"instance": learned["instance"], "n": learned["city_count"],
                "reference": learned["reference_length"], "paper_fbeta": learned["paper_fbeta"]}
        for label, row in (("fixed", fixed), ("learned", learned)):
            for field in ("raw_best", "raw_gap_pct", "a_best", "a_gap_pct", "a_hits",
                          "a_precision", "a_recall", "a_fbeta", "b_best", "b_gap_pct",
                          "b_hits", "b_precision", "b_recall", "b_fbeta",
                          "inference_seconds", "two_opt_seconds", "optb_seconds", "b_checks"):
                item[f"{label}_{field}"] = row[field]
        item["delta_b_hits"] = learned["b_hits"] - fixed["b_hits"]
        item["delta_b_fbeta"] = learned["b_fbeta"] - fixed["b_fbeta"]
        paired.append(item)
    with (output / "per_instance.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    summary = {}
    for label, group in (("fixed", [a for a, _ in rows]),
                         ("learned", [b for _, b in rows])):
        summary[label] = {
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
        }
    summary["paper_mean_fbeta"] = sum(r["paper_fbeta"] for r in summary_input(rows)) / 25
    summary["training"] = training
    summary["probe"] = _read(train_dir / "probe.json")
    _write(output / "summary.json", summary)
    write_report(output, paired, summary)
    print(f"Complete: {output}", flush=True)


def summary_input(rows):
    return [learned for _, learned in rows]


def write_report(output: Path, paired: list[dict], summary: dict) -> None:
    fixed, learned = summary["fixed"], summary["learned"]
    better = [r["instance"] for r in paired if r["delta_b_fbeta"] > 1e-12]
    worse = [r["instance"] for r in paired if r["delta_b_fbeta"] < -1e-12]
    same = [r["instance"] for r in paired if abs(r["delta_b_fbeta"]) <= 1e-12]
    lines = ["# 可学习粒球策略与固定 Soft：原始 25 例", "",
             "粒球层与城市 Attention 使用联合 logP 的 POMO 策略梯度训练一轮；奖励仅为闭环路线距离的相反数。",
             "训练 500 例，独立验证 100 例从已训练的中途/结束 checkpoint 选择 `best.pt`。",
             "测试：CPU、种子 42＋算例编号、全部起点、每起点 1 Argmax＋8 Sampling、全部不同原始回路完整 2-opt、Opt-B 每例至多 50,000 次额外检查。",
             "固定 Soft 读取已精确复现的历史结果，没有重复运行。旋转和反向等价路线只计一次；Fβ 的 β²=0.3。", "",
             "## 汇总", "",
             "| 指标 | 固定 Soft | 可学习粒球 | 差值 |",
             "|---|---:|---:|---:|"]
    for title, key, digits in (("2-opt 最优距离命中", "distance_hits_2opt", 0),
                               ("Opt-B 最优距离命中", "distance_hits_optb", 0),
                               ("2-opt 真最优回路", "true_optimal_routes_2opt", 0),
                               ("Opt-B 真最优回路", "true_optimal_routes_optb", 0),
                               ("平均 Opt-B Fβ", "mean_fbeta_optb", 6),
                               ("平均原始 Gap %", "mean_raw_gap_pct", 3),
                               ("推理秒", "inference_seconds", 2),
                               ("2-opt 秒", "two_opt_seconds", 2),
                               ("Opt-B 秒", "optb_seconds", 3)):
        fmt = f"{{:.{digits}f}}"
        lines.append(f"| {title} | {fmt.format(fixed[key])} | {fmt.format(learned[key])} | "
                     f"{learned[key]-fixed[key]:+.{digits}f} |")
    lines += ["", f"论文平均 Fβ：{summary['paper_mean_fbeta']:.4f}。",
              f"训练耗时：{summary['training']['training_seconds']:.2f} 秒；"
              f"best.pt 来自第 {summary['training']['best_step']} 次更新，"
              f"验证集平均最佳距离 {summary['training']['best_validation_mean_best_length']:.2f}。",
              f"多球训练前检查：粒球层梯度范数 {summary['probe']['ball_grad_norm']:.3f}，"
              f"单次更新参数变化 L1={summary['probe']['ball_parameter_delta_l1']:.6f}；"
              f"整轮训练粒球参数变化 L1={summary['training']['ball_parameter_delta_l1']:.6f}。",
              f"Opt-B Fβ 改善：{', '.join(better) or '无'}；"
              f"退化：{', '.join(worse) or '无'}；持平：{len(same)} 例。",
              "平均原始 Gap 增加，说明本次收益主要发生在完整 2-opt 与 Opt-B 后；"
              "2-opt 总耗时也增加。固定 Soft 耗时来自先前复现实验，"
              "两组使用的 PyTorch 环境不同，耗时数值仅供观察，不能据此认定推理加速。", "",
              "## 逐例对照", "",
              "距离列为原始 / 2-opt / Opt-B；最优回路列为 2-opt / Opt-B。"
              "最后两列给出 Opt-B 的 Precision / Recall / Fβ 和三个阶段耗时。", "",
              "| 算例 | 最优距离 | 固定距离 | 学习距离 | 学习原始/2-opt Gap % | 固定真最优回路 | 学习真最优回路 | 固定 P/R/Fβ | 学习 P/R/Fβ | 固定推理/2-opt/Opt-B 秒 | 学习推理/2-opt/Opt-B 秒 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in paired:
        def dist(label):
            return "/".join(str(r[f"{label}_{key}"]) for key in ("raw_best", "a_best", "b_best"))
        def hits(label):
            return f"{r[f'{label}_a_hits']}/{r[f'{label}_b_hits']}"
        def prf(label):
            return "/".join(f"{r[f'{label}_{key}']:.3f}"
                            for key in ("b_precision", "b_recall", "b_fbeta"))
        def seconds(label):
            return "/".join(f"{r[f'{label}_{key}']:.2f}"
                            for key in ("inference_seconds", "two_opt_seconds", "optb_seconds"))
        gaps = f"{r['learned_raw_gap_pct']:.2f}/{r['learned_a_gap_pct']:.2f}"
        lines.append(f"| {r['instance']} | {r['reference']} | {dist('fixed')} | {dist('learned')} | {gaps} | "
                     f"{hits('fixed')} | {hits('learned')} | {prf('fixed')} | {prf('learned')} | "
                     f"{seconds('fixed')} | {seconds('learned')} |")
    lines += ["", "`per_instance.csv` 另含逐例两阶段 Gap、Precision、Recall、Fβ、实际 Opt-B 检查数和相对基线变化。"
              "`learned_ball/raw_routes` 与 `best_routes` 保留全部生成及最佳路线。", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
