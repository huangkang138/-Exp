"""Render the saved formal training/validation history without rerunning training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.train_dir.resolve()
    with (root / "training_history.csv").open(encoding="utf-8-sig", newline="") as stream:
        history = list(csv.DictReader(stream))
    with (root / "validation_history.csv").open(encoding="utf-8-sig", newline="") as stream:
        validation = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in history]
    val_epochs = [int(row["epoch"]) for row in validation]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    left.plot(epochs, [float(row["train_mean_reward"]) for row in history])
    left.set(xlabel="Epoch", ylabel="Mean reward (-closed distance)", title="Training reward")
    right.plot(val_epochs, [float(row["mean_best_length"]) for row in validation], marker="o")
    best = min(validation, key=lambda row: float(row["mean_best_length"]))
    right.scatter([int(best["epoch"])], [float(best["mean_best_length"])], c="red", zorder=3)
    right.set(xlabel="Epoch", ylabel="Mean greedy POMO best length", title="Disjoint validation")
    for axis in (left, right):
        axis.grid(alpha=0.25)
    fig.savefig(root / "training_curve.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
