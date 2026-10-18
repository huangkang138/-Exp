"""Evaluate the formal best checkpoint; compare saved fixed and short-run results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import generate_learned_routes, load_checkpoint
from gb_2_maco_POMO.experiments.hierarchical.original25 import PAPER, _csv, run_case

PROJECT = Path(__file__).resolve().parents[3]
FIXED_RESULTS = PROJECT / "gb_2_maco_POMO/results/original25_hierarchical_20261015"
SHORT_RESULTS = PROJECT / "gb_2_maco_POMO/results/learnable_ball_test25_20261015"
FIELDS = ("raw_best", "raw_gap_pct", "a_best", "a_gap_pct", "a_hits", "a_precision",
          "a_recall", "a_fbeta", "b_best", "b_gap_pct", "b_hits", "b_precision",
          "b_recall", "b_fbeta", "inference_seconds", "two_opt_seconds",
          "optb_seconds", "b_checks")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summary(rows):
    return {"distance_hits_2opt": sum(r["a_distance_hit"] for r in rows),
            "distance_hits_optb": sum(r["b_distance_hit"] for r in rows),
            "true_optimal_routes_2opt": sum(r["a_hits"] for r in rows),
            "true_optimal_routes_optb": sum(r["b_hits"] for r in rows),
            "mean_fbeta_2opt": sum(r["a_fbeta"] for r in rows) / 25,
            "mean_fbeta_optb": sum(r["b_fbeta"] for r in rows) / 25,
            "mean_raw_gap_pct": sum(r["raw_gap_pct"] for r in rows) / 25,
            "inference_seconds": sum(r["inference_seconds"] for r in rows),
            "two_opt_seconds": sum(r["two_opt_seconds"] for r in rows),
            "optb_seconds": sum(r["optb_seconds"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output, train_dir = args.output_dir.resolve(), args.train_dir.resolve()
    if output.exists():
        parser.error("output directory must be new")
    training = _read(train_dir / "train_result.json")
    if training["completed_epochs"] != 35 or training["total_optimizer_updates"] != 385:
        raise RuntimeError("formal training incomplete")
    checkpoint = train_dir / "checkpoints/best.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != training["best_checkpoint_sha256"]:
        raise RuntimeError("best checkpoint hash differs from training record")
    if not _read(FIXED_RESULTS / "baseline_gate.json")["passed"]:
        raise RuntimeError("fixed Soft historical replay gate failed")
    short_summary = _read(SHORT_RESULTS / "summary.json")
    if (short_summary["learned"]["true_optimal_routes_2opt"] != 283
            or short_summary["learned"]["true_optimal_routes_optb"] != 669
            or abs(short_summary["learned"]["mean_fbeta_optb"] - 0.901198457678) > 1e-10):
        raise RuntimeError("short-run comparison source differs")
    policy, state = load_checkpoint(checkpoint)
    if state["epoch"] != training["best_epoch"] or state["step"] != training["best_step"]:
        raise RuntimeError("best checkpoint epoch/step mismatch")
    output.mkdir(parents=True)
    for folder in ("cases", "raw_routes", "best_routes"):
        (output / "learned_ball" / folder).mkdir(parents=True, exist_ok=True)
    _write(output / "protocol.json", {
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "best_epoch": state["epoch"], "best_step": state["step"],
        "fixed_results": str(FIXED_RESULTS), "short_results": str(SHORT_RESULTS),
        "device": "cpu", "seed": "42 + case index", "starts": "all cities",
        "argmax_per_start": 1, "sampling_per_start": 8,
        "two_opt": "full ordinary 2-opt on all canonical-unique raw cycles",
        "optb_extra_pair_checks_per_case": 50000, "beta_squared": 0.3,
    })
    paper = _csv(PAPER)
    groups = {"fixed": [], "short": [], "full": []}
    paired = []
    for index in range(1, 26):
        name = f"MSTSP{index}"
        full_path = output / "learned_ball/cases" / f"{name}.json"
        full = run_case(policy, index, "learned_ball", 0.0, output,
                        float(paper[name]["paper_fbeta"]),
                        route_generator=generate_learned_routes)
        _write(full_path, full)
        fixed = _read(FIXED_RESULTS / "fixed_soft/cases" / f"{name}.json")
        short = _read(SHORT_RESULTS / "learned_ball/cases" / f"{name}.json")
        if not (full["reference_length"] == fixed["reference_length"] == short["reference_length"]
                and full["truth_count"] == fixed["truth_count"] == short["truth_count"]):
            raise RuntimeError(f"{name}: inconsistent benchmark truth")
        rows = {"fixed": fixed, "short": short, "full": full}
        for label, row in rows.items():
            groups[label].append(row)
        item = {"instance": name, "city_count": full["city_count"],
                "reference_length": full["reference_length"],
                "truth_count": full["truth_count"], "paper_fbeta": full["paper_fbeta"]}
        for label, row in rows.items():
            item.update({f"{label}_{field}": row[field] for field in FIELDS})
        item["full_minus_fixed_b_hits"] = full["b_hits"] - fixed["b_hits"]
        item["full_minus_short_b_hits"] = full["b_hits"] - short["b_hits"]
        item["full_minus_fixed_b_fbeta"] = full["b_fbeta"] - fixed["b_fbeta"]
        item["full_minus_short_b_fbeta"] = full["b_fbeta"] - short["b_fbeta"]
        paired.append(item)
        print(f"{name} full A={full['a_hits']} B={full['b_hits']} "
              f"Fβ={full['b_fbeta']:.6f}", flush=True)
    with (output / "per_instance.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    summary = {name: _summary(rows) for name, rows in groups.items()}
    summary["paper_mean_fbeta"] = sum(r["paper_fbeta"] for r in groups["full"]) / 25
    summary["training"] = training
    _write(output / "summary.json", summary)
    _report(output, paired, summary, train_dir)
    print(f"Complete: {output}", flush=True)


def _report(output, paired, summary, train_dir):
    history = list(csv.DictReader((train_dir / "training_history.csv").open(encoding="utf-8-sig")))
    val = list(csv.DictReader((train_dir / "validation_history.csv").open(encoding="utf-8-sig")))
    best_epoch = summary["training"]["best_epoch"]
    best_val = summary["training"]["best_validation_mean_best_length"]
    final_val = float(val[-1]["mean_best_length"])
    lines = ["# 可学习粒球完整训练：原始 25 例对照", "",
             "训练按历史正式配置：500 例、35 轮、每轮 11 次 Adam 更新，共 385 次；批量 64、微批量 8、学习率 1e-4、种子 42。",
             "从固定 Soft 权重重新初始化，粒球头与城市 Attention 联合策略梯度训练，奖励为闭环距离的相反数。",
             "独立的 100 例验证集选 best.pt；验证 Gap 是相对冻结的最近邻＋2-opt 参考路线，**不是最优 Gap**。",
             "测试按历史 25 例协议：CPU、全部起点、每起点 1 Argmax＋8 Sampling、所有不同原始回路完整 2-opt、Opt-B 每例最多 50,000 次额外检查；旋转和反向等价回路只计一次。",
             "固定 Soft 和一轮短训练均读取已有逐例原始结果，不重新测试。", "",
             "## 汇总", "",
             "| 指标 | 固定 Soft | 一轮短训 | 完整训练 |",
             "|---|---:|---:|---:|"]
    metrics = (("2-opt 最优距离命中", "distance_hits_2opt", 0),
               ("Opt-B 最优距离命中", "distance_hits_optb", 0),
               ("2-opt 真最优回路", "true_optimal_routes_2opt", 0),
               ("Opt-B 真最优回路", "true_optimal_routes_optb", 0),
               ("平均 Opt-B Fβ", "mean_fbeta_optb", 6),
               ("平均原始 Gap %", "mean_raw_gap_pct", 3),
               ("推理秒", "inference_seconds", 2),
               ("2-opt 秒", "two_opt_seconds", 2),
               ("Opt-B 秒", "optb_seconds", 2))
    for title, key, digits in metrics:
        fmt = f"{{:.{digits}f}}"
        lines.append("| " + title + " | " + " | ".join(fmt.format(summary[g][key])
                      for g in ("fixed", "short", "full")) + " |")
    missed = [r["instance"] for r in paired if r["full_b_best"] > r["reference_length"]]
    lines += ["", f"论文平均 Fβ：{summary['paper_mean_fbeta']:.4f}。",
              f"best.pt：第 {best_epoch} 轮、第 {summary['training']['best_step']} 次更新；"
              f"验证平均最佳距离 {best_val:.2f}，末轮 {final_val:.2f}。",
              f"训练耗时 {summary['training']['training_seconds']:.1f} 秒；"
              f"末轮粒球头相对初始化的参数变化 L1={summary['training']['ball_parameter_delta_l1']:.6f}。",
              f"完整训练的最优距离未命中：{', '.join(missed) or '无'}。"
              f"Opt-B 真最优回路比固定 Soft 多 {summary['full']['true_optimal_routes_optb']-summary['fixed']['true_optimal_routes_optb']} 条，"
              f"比短训多 {summary['full']['true_optimal_routes_optb']-summary['short']['true_optimal_routes_optb']} 条；"
              "但命中率与平均 Fβ 都下降，未实现同时改善。",
              f"第 {best_epoch} 轮以后训练平均奖励继续改善，末轮验证平均最佳距离反而比最佳轮高 {final_val-best_val:.2f}。"
              "这是验证性能退化，符合过拟合迹象；仅凭这些数据不能把原因唯一归为过拟合。",
              "训练曲线见训练目录的 `training_curve.png`；三组测试耗时来自不同运行，"
              "硬件负载和 PyTorch 环境可能不同，仅作为记录。", "",
              "## 逐例", "",
              "距离为原始/2-opt/Opt-B；最优回路为 2-opt/Opt-B；P/R/Fβ 为 Opt-B。", "",
              "| 算例 | 真最优距离 | 固定距离 | 短训距离 | 完训距离 | 固定回路 | 短训回路 | 完训回路 | 固定 P/R/Fβ | 短训 P/R/Fβ | 完训 P/R/Fβ | 论文 Fβ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in paired:
        distance = lambda g: "/".join(str(row[f"{g}_{k}"]) for k in ("raw_best", "a_best", "b_best"))
        hits = lambda g: f"{row[f'{g}_a_hits']}/{row[f'{g}_b_hits']}"
        prf = lambda g: "/".join(f"{row[f'{g}_{k}']:.3f}" for k in ("b_precision", "b_recall", "b_fbeta"))
        lines.append("| " + " | ".join([row["instance"], str(row["reference_length"]),
                    *(distance(g) for g in ("fixed", "short", "full")),
                    *(hits(g) for g in ("fixed", "short", "full")),
                    *(prf(g) for g in ("fixed", "short", "full")),
                    f"{row['paper_fbeta']:.4f}"]) + " |")
    better = [r["instance"] for r in paired if r["full_minus_short_b_fbeta"] > 1e-12]
    worse = [r["instance"] for r in paired if r["full_minus_short_b_fbeta"] < -1e-12]
    lines += ["", f"相对短训 Opt-B Fβ 改善：{', '.join(better) or '无'}；退化：{', '.join(worse) or '无'}。",
              "`per_instance.csv` 包含全部逐例 Gap、Precision、Recall、Fβ、耗时和 Opt-B 检查数；"
              "`learned_ball` 保存原始路线、最佳路线及逐例统计。", "",
              "## 逐轮训练", "",
              "| 轮次 | 累计更新 | 平均奖励 | 训练最短距离 | 验证最短距离 | 验证参考 Gap % | 秒 |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in history:
        lines.append(f"| {r['epoch']} | {r['updates']} | {float(r['train_mean_reward']):.2f} | "
                     f"{float(r['train_mean_best_length']):.2f} | "
                     f"{float(r['validation_mean_best_length']):.2f} | "
                     f"{float(r['validation_reference_gap_pct']):.2f} | {float(r['seconds']):.1f} |"
                     if r["validation_mean_best_length"] else
                     f"| {r['epoch']} | {r['updates']} | {float(r['train_mean_reward']):.2f} | "
                     f"{float(r['train_mean_best_length']):.2f} | — | — | {float(r['seconds']):.1f} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
