"""Build the three-model report from completed, separately saved evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .evaluate_new_dataset import OLD_SUMMARY


def _rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    result = {row["instance"]: row for row in rows}
    if len(result) != 25:
        raise ValueError(f"Expected 25 distinct MSTSP cases: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--stability-dir", type=Path)
    args = parser.parse_args()
    directory = args.comparison_dir
    new = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    old = json.loads(OLD_SUMMARY.read_text(encoding="utf-8"))
    if new["device"] != "cpu" or new["seed"] != 42 or new["cases"] != 25:
        raise ValueError("New comparison does not use the formal seed-42 CPU protocol")
    if new["fixed_baseline_sha256"] != old["fixed_baseline_sha256"]:
        raise ValueError("Fixed baseline differs between historical and new comparison")
    new_rows = _rows(directory / "per_instance.csv")
    old_rows = _rows(OLD_SUMMARY.with_name("per_instance.csv"))
    if any(new_rows[name]["reference_length"] != old_rows[name]["reference_length"]
           for name in new_rows):
        raise ValueError("Reference distances differ between the two comparisons")
    lines = ["# 新多最优解数据训练：三模型对照", "",
             "CPU、seed 42、每起点 1 Argmax + 8 Sampling、完整普通 2-opt、原版 Opt-B 50,000 次检查。",
             f"新模型为独立从头训练；旧验证集选择的 best.pt 位于 epoch {new['learnable_best_epoch']}。",
             f"新 checkpoint SHA256：`{new['learnable_checkpoint_sha256']}`。",
             "", "| 指标 | 固定 Bias | 旧数据可学习 Bias | 新数据可学习 Bias |",
             "|---|---:|---:|---:|",
             f"| 最短距离命中 | {old['fixed_distance_hits']}/25 | {old['learnable_distance_hits']}/25 | {new['learnable_distance_hits']}/25 |",
             f"| 平均 Gap | {old['fixed_mean_gap_pct']:.4f}% | {old['learnable_mean_gap_pct']:.4f}% | {new['learnable_mean_gap_pct']:.4f}% |",
             f"| 普通 2-opt 真最优回路 | {old['fixed_a_hits']} | {old['learnable_a_hits']} | {new['learnable_a_hits']} |",
             f"| Opt-B 真最优回路 | {old['fixed_b_hits']} | {old['learnable_b_hits']} | {new['learnable_b_hits']} |",
             f"| 平均 Fβ | {old['fixed_mean_fbeta']:.6f} | {old['learnable_mean_fbeta']:.6f} | {new['learnable_mean_fbeta']:.6f} |",
             f"| 同球 Bias 均值 | 2.000 | {old['same_bias_mean']:.3f} | {new['same_bias_mean']:.3f} |",
             f"| 邻球 Bias 均值 | 1.000 | {old['adjacent_bias_mean']:.3f} | {new['adjacent_bias_mean']:.3f} |",
             "", "## 全部 25 例", "",
             "| 算例 | 参考距离 | 固定距离/Opt-B/Fβ | 旧可学距离/Opt-B/Fβ | 新可学距离/Opt-B/Fβ |",
             "|---|---:|---:|---:|---:|"]
    for index in range(1, 26):
        name = f"MSTSP{index}"
        before, after = old_rows[name], new_rows[name]
        lines.append(f"| {name} | {after['reference_length']} | "
                     f"{after['fixed_a_best']}/{after['fixed_b_hits']}/{float(after['fixed_b_fbeta']):.4f} | "
                     f"{before['a_best']}/{before['b_hits']}/{float(before['b_fbeta']):.4f} | "
                     f"{after['a_best']}/{after['b_hits']}/{float(after['b_fbeta']):.4f} |")
    lines += ["", "## MSTSP20～25", "",
              "| 算例 | 参考距离 | 固定距离/Opt-B | 旧可学距离/Opt-B | 新可学距离/Opt-B |",
              "|---|---:|---:|---:|---:|"]
    for index in range(20, 26):
        name = f"MSTSP{index}"
        before, after = old_rows[name], new_rows[name]
        lines.append(f"| {name} | {after['reference_length']} | "
                     f"{after['fixed_a_best']}/{after['fixed_b_hits']} | "
                     f"{before['a_best']}/{before['b_hits']} | "
                     f"{after['a_best']}/{after['b_hits']} |")
    if args.validation_dir:
        val = json.loads((args.validation_dir / "summary.json").read_text(encoding="utf-8"))
        if Path(val["checkpoint"]).resolve() != Path(new["learnable_checkpoint"]).resolve():
            raise ValueError("Extra validation used a different checkpoint")
        lines += ["", "## 新数据集验证", "",
                  f"新集 100 例额外验证平均最短路径长度：{val['mean_best_length']:.2f}。"
                  "该指标不参与 best.pt 选择。"]
    if args.stability_dir:
        with (args.stability_dir / "summary.csv").open(newline="", encoding="utf-8-sig") as stream:
            stability = list(csv.DictReader(stream))
        source = json.loads((args.stability_dir / "sources.json").read_text(encoding="utf-8"))
        if Path(source["checkpoints"]["new_learned"]).resolve() != Path(new["learnable_checkpoint"]).resolve():
            raise ValueError("Stability test used a different new checkpoint")
        lines += ["", "## 种子稳定性", "",
                  "| seed | 模型 | 距离命中 | Opt-B 真最优回路 | 平均 Fβ |",
                  "|---:|---|---:|---:|---:|"]
        for row in stability:
            lines.append(f"| {row['seed']} | {row['model']} | {row['distance_hits']}/25 | "
                         f"{row['b_hits']} | {float(row['mean_fbeta']):.6f} |")
    recovered = all(int(new_rows[f"MSTSP{i}"]["a_best"]) ==
                    int(new_rows[f"MSTSP{i}"]["reference_length"]) for i in (22, 25))
    retained = all(int(new_rows[f"MSTSP{i}"]["b_hits"]) >=
                   int(old_rows[f"MSTSP{i}"]["b_hits"]) for i in (20, 23))
    beats_fixed = (new["learnable_distance_hits"] == 25
                   and new["learnable_b_hits"] > old["fixed_b_hits"]
                   and new["learnable_mean_fbeta"] > old["fixed_mean_fbeta"])
    length_25 = new_rows["MSTSP25"]
    lines += ["", "## 判断", "",
              f"MSTSP22/25 均恢复最短距离命中：{'是' if recovered else '否'}。",
              f"MSTSP25 实际距离 {length_25['a_best']}，参考 {length_25['reference_length']}，"
              f"差 {int(length_25['a_best']) - int(length_25['reference_length'])}；"
              "由于入场路线未达到最短距离，Opt-B 真最优回路为 0。",
              f"MSTSP20/23 的 Opt-B 覆盖均不低于旧可学习模型：{'是' if retained else '否'}。",
              f"新模型同时达到 25/25、超过固定模型 663 条 Opt-B 真最优回路和"
              f" 0.896843 平均 Fβ：{'是' if beats_fixed else '否'}。",
              f"新模型较固定模型 Opt-B 命中 {new['learnable_b_hits'] - old['fixed_b_hits']:+d}，"
              f"平均 Fβ {new['learnable_mean_fbeta'] - old['fixed_mean_fbeta']:+.6f}；"
              f"较旧可学习模型 Opt-B 命中 {new['learnable_b_hits'] - old['learnable_b_hits']:+d}，"
              f"平均 Fβ {new['learnable_mean_fbeta'] - old['learnable_mean_fbeta']:+.6f}。"
              "现有数据不能把变化单独归因于 Bias 强度或训练数据中的多最优证明。",
              "本报告目前只包含 seed 42；seed 123、2026 尚未复核。"
              "额外验证只说明新数据分布上的路径质量，不替代原验证集选模。", ""]
    output = directory / "three_model_report.md"
    if output.exists():
        raise FileExistsError(output)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
