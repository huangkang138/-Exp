"""Run the original 25-case protocol for one preselected experiment checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from gb_2_maco_POMO.experiments.hierarchical.original25 import PAPER, _csv, run_case
from .model import BallBiasPolicy
from .rollout import generate_routes
from .train import write_csv, write_json


def load_selected_checkpoint(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    arch = state["architecture"]
    policy = BallBiasPolicy(int(arch["embedding_dim"]), int(arch["num_heads"]),
                            int(arch["encoder_layers"]),
                            use_bias=bool(state["use_bias"]),
                            bias_bound=float(state["bias_bound"]))
    policy.load_state_dict(state["model_state_dict"], strict=True)
    policy.eval()
    return policy, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--selector", choices=("best_length", "best_multi"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output directory must be new")
    checkpoint = args.train_dir / "checkpoints" / f"{args.selector}.pt"
    policy, state = load_selected_checkpoint(checkpoint)
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    label = f"{state['group']}_{args.selector}"
    for folder in ("raw_routes", "best_routes"):
        (output / label / folder).mkdir(parents=True)
    def route_generator(model, prepared, starts, samples, seed):
        if starts != list(range(len(prepared.cities))) or samples != 8:
            raise RuntimeError("historical 25-case budget changed")
        return generate_routes(model, prepared, samples=samples, seed=seed,
                               device=torch.device("cpu")), None
    rows = []
    paper = _csv(PAPER)
    for index in range(1, 26):
        row = run_case(policy, index, label, 0.0, output,
                       float(paper[f"MSTSP{index}"]["paper_fbeta"]),
                       route_generator=route_generator)
        rows.append(row)
        print(f"MSTSP{index}: A={row['a_hits']} B={row['b_hits']}", flush=True)
    write_csv(output / "per_instance.csv", rows)
    summary = {"group": state["group"], "selector": args.selector,
               "epoch": state["epoch"], "checkpoint": str(checkpoint),
               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "distance_hits": sum(row["b_distance_hit"] for row in rows),
               "mean_raw_gap_pct": sum(row["raw_gap_pct"] for row in rows) / 25,
               "mean_2opt_gap_pct": sum(row["a_gap_pct"] for row in rows) / 25,
               "a_true_optimal_routes": sum(row["a_hits"] for row in rows),
               "b_true_optimal_routes": sum(row["b_hits"] for row in rows),
               "mean_fbeta": sum(row["b_fbeta"] for row in rows) / 25,
               "inference_seconds": sum(row["inference_seconds"] for row in rows),
               "two_opt_seconds": sum(row["two_opt_seconds"] for row in rows),
               "optb_seconds": sum(row["optb_seconds"] for row in rows)}
    write_json(output / "summary.json", summary)
    (output / "comparison.md").write_text(
        "# Original 25-case evaluation\n\n"
        "Checkpoint selection used only the independent validation split of "
        f"`{state['config']['data_root']}`. "
        "This file reports final-test measurements; compare A/B/C only after all "
        "runs finish with the same protocol.\n\n"
        "| Model | Hits | Raw Gap % | 2-opt Gap % | A | B | Fβ | Inference / 2-opt / Opt-B s |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| {label} | {summary['distance_hits']}/25 | {summary['mean_raw_gap_pct']:.4f} | "
        f"{summary['mean_2opt_gap_pct']:.4f} | {summary['a_true_optimal_routes']} | "
        f"{summary['b_true_optimal_routes']} | {summary['mean_fbeta']:.6f} | "
        f"{summary['inference_seconds']:.2f} / {summary['two_opt_seconds']:.2f} / "
        f"{summary['optb_seconds']:.3f} |\n\n"
        "Per-instance results are in `per_instance.csv`; raw, A and B routes are "
        f"in `{label}/`. A and B count only known true optimal tours.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
