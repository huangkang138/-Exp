"""Summarize completed A/B/C final tests without influencing checkpoint selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .train import write_csv, write_json

PROJECT = Path(__file__).resolve().parents[3]
FIXED = PROJECT / "最优结果的实验/目前最优/runs/run_20261014_220649_876728"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for group in "ABC":
        parser.add_argument(f"--{group.lower()}-results", type=Path, required=True,
                            help="directory containing best_length/ and best_multi/ results")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output directory must be new")
    paths = {g: getattr(args, f"{g.lower()}_results") for g in "ABC"}
    summaries, details = {}, {}
    for group, root in paths.items():
        for selector in ("best_length", "best_multi"):
            folder = root / selector
            key = f"{group}_{selector}"
            summaries[key] = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            details[key] = {row["instance"]: row for row in read_csv(folder / "per_instance.csv")}
            if summaries[key]["group"] != group or summaries[key]["selector"] != selector \
                    or len(details[key]) != 25:
                raise RuntimeError(f"incomplete or mismatched result: {folder}")
    fixed = {r["instance"]: r for r in read_csv(FIXED / "per_instance.csv")}
    if len(fixed) != 25 or any(set(rows) != set(fixed) for rows in details.values()):
        raise RuntimeError("test instances differ across groups")
    output.mkdir(parents=True)
    rows = []
    for key, summary in summaries.items():
        rows.append({"model": key, "checkpoint_epoch": summary["epoch"],
                     "distance_hits": summary["distance_hits"],
                     "mean_raw_gap_pct": summary["mean_raw_gap_pct"],
                     "mean_2opt_gap_pct": summary["mean_2opt_gap_pct"],
                     "A": summary["a_true_optimal_routes"],
                     "B": summary["b_true_optimal_routes"],
                     "mean_fbeta": summary["mean_fbeta"],
                     "elapsed_seconds": summary["inference_seconds"] +
                     summary["two_opt_seconds"] + summary["optb_seconds"]})
    write_csv(output / "comparison.csv", rows)
    per_case = []
    for name in fixed:
        row = {"instance": name, "fixed_b_best": fixed[name]["b_best"],
               "fixed_b_hits": fixed[name]["b_hits"],
               "fixed_b_fbeta": fixed[name]["b_fbeta"]}
        for key, group in details.items():
            item = group[name]
            row.update({f"{key}_b_best": item["b_best"],
                        f"{key}_b_hits": item["b_hits"],
                        f"{key}_b_fbeta": item["b_fbeta"]})
        per_case.append(row)
    write_csv(output / "per_instance_comparison.csv", per_case)
    def better(left, right, field):
        return [name for name in fixed if float(details[left][name][field]) >
                float(details[right][name][field])]
    observations = {
        selector: {"B_more_true_tours_than_A": better(f"B_{selector}", f"A_{selector}", "b_hits"),
                   "C_more_true_tours_than_B": better(f"C_{selector}", f"B_{selector}", "b_hits")}
        for selector in ("best_length", "best_multi")}
    write_json(output / "observations.json", observations)
    lines = ["# A/B/C two-level policy comparison", "",
             "All models used the same original 25-case test protocol. Checkpoints were "
             "selected on the independent Multiopt validation split before this test.", "",
             "| Model | Epoch | Hits | Raw Gap % | 2-opt Gap % | A | B | Fβ | Seconds |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['checkpoint_epoch']} | "
                     f"{row['distance_hits']}/25 | {row['mean_raw_gap_pct']:.4f} | "
                     f"{row['mean_2opt_gap_pct']:.4f} | {row['A']} | {row['B']} | "
                     f"{row['mean_fbeta']:.6f} | {row['elapsed_seconds']:.2f} |")
    lines += ["", "Historical fixed Soft: 25/25 distance hits, A=250, B=663, "
              "mean Fβ=0.896843. Per-case comparisons are in `per_instance_comparison.csv`.",
              "The table alone does not establish statistical significance or explain causality."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
