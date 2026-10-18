"""Compare two completed 25-case learnable-ball evaluations by training dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-eval", type=Path, required=True)
    parser.add_argument("--new-eval", type=Path, required=True)
    parser.add_argument("--old-train", type=Path, required=True)
    parser.add_argument("--new-train", type=Path, required=True)
    args = parser.parse_args()
    old_rows = _read_csv(args.old_eval / "per_instance.csv")
    new_rows = _read_csv(args.new_eval / "per_instance.csv")
    if len(old_rows) != 25 or len(new_rows) != 25:
        raise RuntimeError("both 25-case evaluations must be complete")
    old_config, new_config = (_read_json(path / "train_config.json")
                              for path in (args.old_train, args.new_train))
    old_training, new_training = (_read_json(path / "train_result.json")
                                  for path in (args.old_train, args.new_train))
    old_summary, new_summary = (_read_json(path / "summary.json")["full"]
                                for path in (args.old_eval, args.new_eval))
    invariant = ("epochs", "batch_size", "micro_batch_size", "optimizer",
                 "learning_rate", "seed", "validation_interval_epochs",
                 "expected_updates", "fixed_checkpoint_sha256")
    if any(old_config[key] != new_config[key] for key in invariant):
        raise RuntimeError("training settings other than dataset differ")
    old_val_root = old_config.get("validation_dataset", old_config.get("dataset"))
    if old_val_root != new_config["validation_dataset"]:
        raise RuntimeError("checkpoint selection used different validation datasets")
    if not (old_training["total_optimizer_updates"]
            == new_training["total_optimizer_updates"] == 385):
        raise RuntimeError("optimizer update counts differ")
    table = []
    for old, new in zip(old_rows, new_rows):
        if any(old[key] != new[key] for key in ("instance", "reference_length", "truth_count")):
            raise RuntimeError("different benchmark case or ground truth")
        row = {"instance": old["instance"], "reference_length": old["reference_length"],
               "old_raw_gap_pct": old["full_raw_gap_pct"],
               "new_raw_gap_pct": new["full_raw_gap_pct"],
               "old_2opt_hits": old["full_a_hits"], "new_2opt_hits": new["full_a_hits"],
               "old_optb_hits": old["full_b_hits"], "new_optb_hits": new["full_b_hits"],
               "old_optb_fbeta": old["full_b_fbeta"],
               "new_optb_fbeta": new["full_b_fbeta"],
               "old_optb_best": old["full_b_best"], "new_optb_best": new["full_b_best"]}
        table.append(row)
    output = args.new_eval / "dataset_comparison.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    names = (("最优距离命中", "distance_hits_optb", 0),
             ("2-opt 真最优回路", "true_optimal_routes_2opt", 0),
             ("Opt-B 真最优回路", "true_optimal_routes_optb", 0),
             ("平均 Opt-B Fβ", "mean_fbeta_optb", 6),
             ("平均原始 Gap %", "mean_raw_gap_pct", 3))
    lines = ["# 可学习粒球：两种 500 例训练集对照", "",
             f"原训练集：`{old_config.get('train_dataset', old_config.get('dataset'))}`。",
             f"新训练集：`{new_config['train_dataset']}`。",
             f"两次都用 `{old_val_root}` 的独立 100 例选择 checkpoint。"
             "训练设置均为 35 轮、385 次 Adam 更新、批量 64、微批量 8、学习率 1e-4、种子 42；"
             "模型、奖励、25 例测试规则一致。", "",
             "| 指标 | 原训练集 | Multiopt 训练集 | 差值 |",
             "|---|---:|---:|---:|"]
    for title, key, digits in names:
        fmt = f"{{:.{digits}f}}"
        lines.append(f"| {title} | {fmt.format(old_summary[key])} | "
                     f"{fmt.format(new_summary[key])} | "
                     f"{new_summary[key]-old_summary[key]:+.{digits}f} |")
    lines += ["", f"最佳验证轮次：原训练集 {old_training['best_epoch']}；"
              f"Multiopt 训练集 {new_training['best_epoch']}。",
              "两次训练使用相同验证集，但 checkpoint 选中轮次可能不同；"
              "这份结果只证明当前训练及选择流程下的数据集效果，不能单独归因于数据几何。", "",
              "## 逐例", "",
              "| 算例 | 真最优距离 | 原原始 Gap % | 新原始 Gap % | 原 2-opt 回路 | 新 2-opt 回路 | 原 Opt-B 回路 | 新 Opt-B 回路 | 原 Opt-B Fβ | 新 Opt-B Fβ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append("| " + " | ".join([
            row["instance"], row["reference_length"],
            f"{float(row['old_raw_gap_pct']):.3f}", f"{float(row['new_raw_gap_pct']):.3f}",
            row["old_2opt_hits"], row["new_2opt_hits"],
            row["old_optb_hits"], row["new_optb_hits"],
            f"{float(row['old_optb_fbeta']):.6f}",
            f"{float(row['new_optb_fbeta']):.6f}",
        ]) + " |")
    (args.new_eval / "dataset_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
