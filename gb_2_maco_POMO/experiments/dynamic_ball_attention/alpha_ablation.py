"""Paired validation diagnosis using one frozen epoch-3 B checkpoint.

Only alpha changes in memory. Route generation and 2-opt call the project's
existing implementations; this script never updates model parameters.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import torch

from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import generate_learned_routes
from gb_2_maco_POMO.experiments.learnable_ball_bias.validation import certified_lengths, heuristic_reference
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, tour_length
from gb_2_maco_POMO.runner import prepare_tsp

from .common import load_data, load_selected, sha256
from .config import DATA, RESULTS
from .validation import VALIDATION_SEED


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _routes(model, problem, index: int, alpha: float, probe: bool):
    """Use the historical 1-greedy + 8-sampling generator for every city start."""
    n = len(problem.cities)
    model.ball_attention_alpha.fill_(alpha)
    original = model.action_logits
    stats = {"states": 0, "top1_changes": 0, "tv_sum": 0.0, "tv_max": 0.0,
             "probability_delta_max": 0.0}
    calls = 0

    def observed_logits(encoded, first, current, available, city_to_ball):
        nonlocal calls
        ball_off, city = original(encoded, first, current, available, city_to_ball)
        if calls < n - 1:  # The generator produces all greedy starts first.
            model.ball_attention_alpha.fill_(trained_alpha)
            try:
                ball_on, _ = original(encoded, first, current, available, city_to_ball)
            finally:
                model.ball_attention_alpha.fill_(0.0)
            p0, p1 = ball_off.softmax(-1), ball_on.softmax(-1)
            eligible = torch.isfinite(ball_off).sum(-1) > 1
            differences = (p0 - p1).abs()
            tv = differences.sum(-1) / 2
            stats["states"] += int(eligible.sum().item())
            stats["top1_changes"] += int(((ball_off.argmax(-1) != ball_on.argmax(-1))
                                          & eligible).sum().item())
            if eligible.any():
                stats["tv_sum"] += float(tv[eligible].sum().item())
                stats["tv_max"] = max(stats["tv_max"], float(tv[eligible].max().item()))
                stats["probability_delta_max"] = max(
                    stats["probability_delta_max"], float(differences[eligible].max().item()))
        calls += 1
        return ball_off, city

    if probe:
        model.action_logits = observed_logits
    try:
        routes, _ = generate_learned_routes(
            model, problem, list(range(n)), 8, VALIDATION_SEED + index)
    finally:
        if probe:
            model.action_logits = original
        model.ball_attention_alpha.fill_(trained_alpha)
    return routes, stats


@torch.inference_mode()
def main() -> None:
    global trained_alpha
    torch.set_num_threads(1)
    checkpoint = RESULTS / "train_seed_42/B/checkpoints/epoch_003.pt"
    model = load_selected(checkpoint, "B", torch.device("cuda"))
    trained_alpha = float(model.ball_attention_alpha.item())
    if trained_alpha == 0 or not torch.isfinite(model.ball_attention_alpha).item():
        raise RuntimeError("epoch-3 alpha is zero or nonfinite")
    _, validation = load_data()
    if len(validation) != 100:
        raise RuntimeError("validation split changed")
    known = certified_lengths(DATA)
    output = RESULTS / "alpha_ablation_seed_42_epoch_3"
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "trained_alpha": trained_alpha, "disabled_alpha": 0.0,
        "validation_cases": 100, "inference_seed_rule": "20261018 + case index",
        "starts": "all cities", "greedy_rounds": 1, "sampling_rounds": 8,
        "two_opt": "ordinary complete 2-opt on every unique raw cycle",
        "ball_probability_states": "alpha=0 greedy trajectories, same states for both alphas",
        "device": "cuda", "checkpoint_selection": "not performed"})
    comparisons = []
    for index, item in enumerate(validation, 1):
        problem = prepare_tsp(item.path)
        distance = problem.distance_matrix
        reference = known.get(item.path.name)
        certified = reference is not None
        if reference is None:
            reference = heuristic_reference(distance)
        generated = {}
        metrics = {}
        for label, alpha in (("off", 0.0), ("on", trained_alpha)):
            routes, probability = _routes(model, problem, index, alpha, label == "off")
            if label == "off":
                probability_stats = probability
            unique = {canonical(route): route for route in routes}
            refined = {key: two_opt(route, distance).length for key, route in unique.items()}
            raw_best = min(tour_length(route, distance) for route in routes)
            best = min(refined.values())
            generated[label] = set(unique)
            metrics[label] = {"raw_routes": len(routes), "raw_unique": len(unique),
                              "raw_best": raw_best, "two_opt_best": best,
                              "raw_gap_pct": 100 * (raw_best / reference - 1),
                              "two_opt_gap_pct": 100 * (best / reference - 1),
                              "certified_hit": int(best == reference) if certified else None}
            with gzip.open(output / f"{index:03d}_{label}_raw_routes.json.gz", "wt",
                           encoding="utf-8") as stream:
                json.dump(routes, stream, separators=(",", ":"))
            if label == "off":
                greedy_off = routes[:item.city_count]
            else:
                greedy_equal = sum(a == b for a, b in zip(greedy_off, routes[:item.city_count]))
        row = {"instance": item.path.name, "city_count": item.city_count,
               "reference_kind": "certified" if certified else "heuristic_upper_bound",
               "reference_length": reference, "greedy_routes_equal": greedy_equal,
               "raw_unique_overlap": len(generated["off"] & generated["on"]),
               "raw_unique_union": len(generated["off"] | generated["on"]),
               "ball_states_compared": probability_stats["states"],
               "ball_top1_changes": probability_stats["top1_changes"],
               "ball_mean_tv": (probability_stats["tv_sum"] / probability_stats["states"]
                                if probability_stats["states"] else 0.0),
               "ball_max_tv": probability_stats["tv_max"],
               "ball_max_probability_delta": probability_stats["probability_delta_max"],
               **{f"{label}_{key}": value for label, data in metrics.items()
                  for key, value in data.items()}}
        comparisons.append(row)
        print(f"{index}/100 {item.path.name} alpha_off={metrics['off']['two_opt_best']} "
              f"alpha_on={metrics['on']['two_opt_best']}", flush=True)
    with (output / "per_case.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    summary = {"cases": len(comparisons), "certified_cases": sum(
        r["reference_kind"] == "certified" for r in comparisons),
        "trained_alpha": trained_alpha,
        "ball_states_compared": sum(r["ball_states_compared"] for r in comparisons),
        "ball_top1_changes": sum(r["ball_top1_changes"] for r in comparisons),
        "ball_mean_tv": (sum(r["ball_mean_tv"] * r["ball_states_compared"] for r in comparisons)
                         / max(1, sum(r["ball_states_compared"] for r in comparisons))),
        "ball_max_tv": max(r["ball_max_tv"] for r in comparisons),
        "ball_max_probability_delta": max(r["ball_max_probability_delta"] for r in comparisons),
        "changed_greedy_routes": sum(r["city_count"] - r["greedy_routes_equal"] for r in comparisons),
        "changed_raw_route_sets": sum(r["off_raw_unique"] != r["raw_unique_overlap"] or
                                      r["on_raw_unique"] != r["raw_unique_overlap"]
                                      for r in comparisons),
        "two_opt_better_on": sum(r["on_two_opt_best"] < r["off_two_opt_best"] for r in comparisons),
        "two_opt_worse_on": sum(r["on_two_opt_best"] > r["off_two_opt_best"] for r in comparisons),
        "two_opt_equal": sum(r["on_two_opt_best"] == r["off_two_opt_best"] for r in comparisons)}
    for label in ("off", "on"):
        summary[f"{label}_mean_raw_gap_pct"] = sum(r[f"{label}_raw_gap_pct"] for r in comparisons) / 100
        summary[f"{label}_mean_two_opt_gap_pct"] = sum(r[f"{label}_two_opt_gap_pct"] for r in comparisons) / 100
        summary[f"{label}_certified_hits"] = sum(r[f"{label}_certified_hit"] or 0 for r in comparisons)
    _write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
