"""Compare the new-data Bias run, historical runs, and published F-beta values."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .evaluate_new_dataset import OLD_SUMMARY, PROJECT
from .report_new_dataset import _rows


PAPER = PROJECT / "reproduction/config/paper_results.csv"


def _paper() -> dict[str, float]:
    with PAPER.open(newline="", encoding="utf-8-sig") as stream:
        values = {row["instance"]: float(row["paper_fbeta"])
                  for row in csv.DictReader(stream)}
    if set(values) != {f"MSTSP{i}" for i in range(1, 26)}:
        raise ValueError("Paper F-beta table must contain exactly MSTSP1..25")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.comparison_dir
    new = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    old = json.loads(OLD_SUMMARY.read_text(encoding="utf-8"))
    if (new["device"], new["seed"], new["cases"]) != ("cpu", 42, 25):
        raise ValueError("New comparison protocol does not match seed-42 CPU run")
    if new["fixed_baseline_sha256"] != old["fixed_baseline_sha256"]:
        raise ValueError("Historical fixed baseline does not match")
    latest = _rows(directory / "per_instance.csv")
    previous = _rows(OLD_SUMMARY.with_name("per_instance.csv"))
    paper = _paper()
    for name in paper:
        if latest[name]["reference_length"] != previous[name]["reference_length"]:
            raise ValueError(f"{name}: reference distance differs")
        if (latest[name]["fixed_a_best"] != previous[name]["fixed_a_best"]
                or latest[name]["fixed_b_hits"] != previous[name]["fixed_b_hits"]):
            raise ValueError(f"{name}: historical fixed result differs")

    paper_mean = sum(paper.values()) / 25
    lines = [
        "# 新数据可学习 Bias 与固定 Bias、旧可学习 Bias、论文对照", "",
        "CPU、seed 42；每起点 1 Argmax + 8 Sampling；对全部去重原始路线做完整普通 2-opt；"
        "从最佳距离回路运行原版 Opt-B，每例最多 50,000 次额外换边检查。",
        "参考最短距离和真最优回路来自 `.solution`，仅在推理完成后用于评价；"
        "论文数据来自 `reproduction/config/paper_results.csv`，仅包含 Fβ。"
        "参考距离和模型 Gap 均不是论文运行结果。", "",
        f"新模型：从头训练，旧验证集选出的 best.pt 为 epoch {new['learnable_best_epoch']}；"
        f"checkpoint SHA256 `{new['learnable_checkpoint_sha256']}`。", "",
        "## 汇总", "",
        "| 指标 | 固定 Bias | 旧数据可学习 Bias | 新数据可学习 Bias | 论文 |",
        "|---|---:|---:|---:|---:|",
        f"| 参考最短距离命中 | {old['fixed_distance_hits']}/25 | "
        f"{old['learnable_distance_hits']}/25 | {new['learnable_distance_hits']}/25 | 未提供 |",
        f"| 平均 Gap | {old['fixed_mean_gap_pct']:.4f}% | "
        f"{old['learnable_mean_gap_pct']:.4f}% | {new['learnable_mean_gap_pct']:.4f}% | 未提供 |",
        f"| 普通 2-opt 真最优回路 | {old['fixed_a_hits']} | {old['learnable_a_hits']} | "
        f"{new['learnable_a_hits']} | 未提供 |",
        f"| Opt-B 真最优回路 | {old['fixed_b_hits']} | {old['learnable_b_hits']} | "
        f"{new['learnable_b_hits']} | 未提供 |",
        f"| Opt-B 平均 Fβ | {old['fixed_mean_fbeta']:.6f} | "
        f"{old['learnable_mean_fbeta']:.6f} | {new['learnable_mean_fbeta']:.6f} | {paper_mean:.6f} |",
        f"| 相对论文平均 Fβ | {old['fixed_mean_fbeta'] - paper_mean:+.6f} | "
        f"{old['learnable_mean_fbeta'] - paper_mean:+.6f} | "
        f"{new['learnable_mean_fbeta'] - paper_mean:+.6f} | 基准 |",
        "", "## 距离与 Gap", "",
        "Gap = (Opt-B 最佳距离 / `.solution` 参考最短距离 - 1) × 100%。"
        "普通 2-opt 与 Opt-B 的最佳距离在这次运行中逐例相同。", "",
        "| 算例 | 参考最短距离 | 固定 Opt-B 距离/Gap | 旧可学 Opt-B 距离/Gap | 新可学 Opt-B 距离/Gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for index in range(1, 26):
        name = f"MSTSP{index}"
        now, before = latest[name], previous[name]
        ref = int(now["reference_length"])
        def cell(distance: int) -> str:
            return f"{distance} / {(distance / ref - 1) * 100:.4f}%"
        lines.append(f"| {name} | {ref} | {cell(int(now['fixed_b_best']))} | "
                     f"{cell(int(before['b_best']))} | {cell(int(now['b_best']))} |")

    lines += ["", "## 最优回路覆盖", "",
              "A 为普通完整 2-opt，B 为 Opt-B。真最优命中以 `.solution` 中的不同最优回路计数。"
              "B 去重数包含距离最佳但不在已列真值中的回路；MSTSP25 新模型 B 命中为零。", "",
              "| 算例 | 固定 A/B 命中 | 旧可学 A/B 命中 | 新可学 A/B 去重 | 新可学 A/B 命中 |",
              "|---|---:|---:|---:|---:|"]
    for index in range(1, 26):
        name = f"MSTSP{index}"
        now, before = latest[name], previous[name]
        lines.append(f"| {name} | {now['fixed_a_hits']}/{now['fixed_b_hits']} | "
                     f"{before['a_hits']}/{before['b_hits']} | {now['a_routes']}/{now['b_routes']} | "
                     f"{now['a_hits']}/{now['b_hits']} |")

    lines += ["", "## Fβ 与论文对照", "",
              "以下三个模型均为各自 Opt-B 后的 Fβ；差值 = 新数据可学习 Bias Fβ − 论文 Fβ。", "",
              "| 算例 | 固定 Bias | 旧可学习 Bias | 新可学习 Bias | 论文 Fβ | 新模型相对论文 |",
              "|---|---:|---:|---:|---:|---:|"]
    for index in range(1, 26):
        name = f"MSTSP{index}"
        now, before = latest[name], previous[name]
        score = float(now["b_fbeta"])
        lines.append(f"| {name} | {float(now['fixed_b_fbeta']):.4f} | "
                     f"{float(before['b_fbeta']):.4f} | {score:.4f} | "
                     f"{paper[name]:.4f} | {score - paper[name]:+.4f} |")
    lines += ["", "## 结论", "",
              f"新模型相对固定模型：最短距离命中 {new['learnable_distance_hits']}/25 对 25/25；"
              f"Opt-B 真最优回路 {new['learnable_b_hits']} 对 {old['fixed_b_hits']}；"
              f"平均 Fβ {new['learnable_mean_fbeta']:.6f} 对 {old['fixed_mean_fbeta']:.6f}。",
              "MSTSP22 恢复参考最短距离；MSTSP25 仍为 9527，对参考值 9521，"
              "因此新模型的 25/25 距离目标未达到。"
              "MSTSP23 的 Opt-B 命中为 12（固定 6、旧可学 14），"
              "MSTSP20 为 12（固定 12、旧可学 20）。",
              f"论文表平均 Fβ {paper_mean:.6f}；新模型低 "
              f"{paper_mean - new['learnable_mean_fbeta']:.6f}。"
              "论文未提供同预算距离、命中数或 Opt-B 结果，不能据此比较这些指标。",
              "本次结果仅为 seed 42；seed 123、2026 的稳定性评估尚未加入本报告。"
              "新对照没有单独计时，故不复用历史文档的耗时数据。", ""]
    output = directory / "comparison_新数据可学习Bias_含论文.md"
    if output.exists():
        raise FileExistsError(output)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
