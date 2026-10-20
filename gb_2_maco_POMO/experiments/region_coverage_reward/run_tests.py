"""Run the fixed 25-case test protocol after all training selections finish."""

from __future__ import annotations

import csv
import json

from gb_2_maco_POMO.experiments.hierarchical.original25 import CHECKPOINT as SOFT_CHECKPOINT

from .evaluate25 import evaluate
from .train import GROUPS, INITIAL, PROJECT


ROOT = PROJECT / "gb_2_maco_POMO/results/region_coverage_reward"
PRIOR_SHORT = (PROJECT / "gb_2_maco_POMO/results/"
               "multi_solution_diagnosis_20261017_seeds42_46/per_seed_summary.csv")


def verified_training():
    for group in GROUPS:
        for seed in (42, 43, 44):
            directory = ROOT / group / f"train_seed_{seed}"
            result = json.loads((directory / "train_result.json").read_text(encoding="utf-8"))
            if result["completed_epochs"] < 5 and not result["stopped_early"]:
                raise RuntimeError(f"training incomplete: {directory}")
            selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
            if selection["test_set_used_for_selection"] is not False:
                raise RuntimeError("checkpoint selection consulted the test set")


def main():
    verified_training()
    with PRIOR_SHORT.open(encoding="utf-8-sig", newline="") as stream:
        prior = {int(r["seed"]): r for r in csv.DictReader(stream) if r["model"] == "short"}
    if len(prior) != 5:
        raise RuntimeError("existing five-seed short baseline missing")
    for mode, checkpoint in (("fixed_soft", SOFT_CHECKPOINT), ("short", INITIAL)):
        for inference_seed in (42, 43, 44, 45, 46):
            output = ROOT / mode / f"test_seed_{inference_seed}"
            if (output / "summary.json").is_file():
                print("skip complete", mode, inference_seed, flush=True)
                continue
            evaluate(mode, None, inference_seed, checkpoint, output, resume=output.exists())
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            if mode == "fixed_soft" and inference_seed == 42:
                if (summary["distance_hits"] != 25
                        or summary["true_optimal_routes"] != 663
                        or abs(summary["mean_fbeta"] - 0.8968434047253819) > 1e-12):
                    raise RuntimeError("fixed Soft seed-42 historical baseline mismatch")
            if mode == "short":
                old = prior[inference_seed]
                if (summary["distance_hits"] != int(old["optb_distance_hits"])
                        or summary["true_optimal_routes"] != int(old["optb_true_optimal_routes"])
                        or abs(summary["mean_fbeta"] - float(old["mean_fbeta"])) > 1e-12):
                    raise RuntimeError(f"existing short baseline mismatch: seed {inference_seed}")
    for group in GROUPS:
        for train_seed in (42, 43, 44):
            directory = ROOT / group / f"train_seed_{train_seed}"
            selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
            checkpoint = directory / "checkpoints" / f"epoch_{selection['selected_epoch']:03d}.pt"
            for inference_seed in (42, 43, 44, 45, 46):
                output = directory / f"test_seed_{inference_seed}"
                if (output / "summary.json").is_file():
                    print("skip complete", group, train_seed, inference_seed, flush=True)
                    continue
                evaluate(group, train_seed, inference_seed, checkpoint, output,
                         resume=output.exists())


if __name__ == "__main__":
    main()
