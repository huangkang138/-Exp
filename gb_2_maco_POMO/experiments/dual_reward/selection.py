"""Independent validation metrics and the existing two-checkpoint selection rule."""

from __future__ import annotations

from gb_2_maco_POMO.experiments.learnable_ball_bias.validation import select_checkpoints


def metrics(rows: list[dict]) -> dict:
    certified = [row for row in rows if row["certified_optimal_length"] != ""]
    if len(rows) != 100 or len(certified) != 56:
        raise RuntimeError("expected 100 validation cases, including 56 certified cases")
    mean = lambda values: sum(values) / len(values)
    return {
        "cases": len(rows), "certified_cases": len(certified),
        "certified_hits": sum(row["sampled_two_opt_optimal_hit"] for row in certified),
        "mean_certified_gap_pct": mean([row["sampled_two_opt_optimal_gap_pct"] for row in certified]),
        "mean_certified_optb_routes": mean([row["sampled_certified_optb_routes"] for row in certified]),
        "mean_heuristic_relative_2opt_pct": mean([row["sampled_heuristic_relative_2opt_pct"] for row in rows]),
        "mean_heuristic_relative_raw_pct": mean([row["sampled_heuristic_relative_raw_pct"] for row in rows]),
    }
