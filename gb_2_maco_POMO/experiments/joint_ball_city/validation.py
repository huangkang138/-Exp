"""Independent validation with the dual-reward selection metrics and budget."""

from __future__ import annotations

import torch

from gb_2_maco_POMO.attention import build_city_features
from gb_2_maco_POMO.experiments.learnable_ball_bias.validation import heuristic_reference
from gb_2_maco_POMO.experiments.shared.local_search_2opt import two_opt
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, equal_plateau, tour_length
from gb_2_maco_POMO.runner import prepare_tsp

from .inference import generate_routes
from .model import normalized_distance


@torch.inference_mode()
def decision_diagnostics(policy, prepared, greedy_routes) -> dict:
    """Compare old/new distributions on the same greedy route states."""
    device = next(policy.parameters()).device
    n = len(prepared.cities)
    features = build_city_features(prepared.cities, prepared.granular_balls,
                                   prepared.city_to_ball).to(device)
    _, encoded = policy.encode(features)
    mapping = torch.tensor(prepared.city_to_ball, dtype=torch.long, device=device)[None, :]
    distance = normalized_distance(prepared, device)
    first = torch.arange(n, device=device)[None, :]
    current = first
    route_tensor = torch.tensor(greedy_routes, dtype=torch.long, device=device)[None, :, :]
    available = torch.ones((1, n, n), dtype=torch.bool, device=device)
    available.scatter_(2, first[..., None], False)
    aggregates = {key: 0.0 for key in ("ball_probability_l1", "city_probability_l1",
                                          "joint_probability_l1", "score_sum", "score_square_sum")}
    ball_changes = city_changes = score_count = 0
    score_min, score_max = float("inf"), float("-inf")
    for step in range(1, n):
        old_ball_logits, old_city_logits = policy.action_logits(
            encoded, first, current, available, mapping)
        new_ball_logits, new_city_logp, scores, eligible = policy.joint_components(
            encoded, first, current, available, mapping, distance)
        old_ball = torch.softmax(old_ball_logits, dim=-1)
        new_ball = torch.softmax(new_ball_logits, dim=-1)
        old_city_logits = old_city_logits[:, :, None, :].masked_fill(~eligible, -torch.inf)
        old_city = torch.softmax(old_city_logits, dim=-1).nan_to_num(0)
        new_city = new_city_logp.exp()
        joint_old = old_ball[..., None] * old_city
        joint_new = new_ball[..., None] * new_city
        aggregates["ball_probability_l1"] += (old_ball - new_ball).abs().sum(dim=-1).mean().item()
        aggregates["city_probability_l1"] += ((old_city - new_city).abs().sum(dim=-1) *
                                                old_ball).sum(dim=-1).mean().item()
        aggregates["joint_probability_l1"] += (joint_old - joint_new).abs().sum(dim=(-1, -2)).mean().item()
        legal_balls = eligible.any(dim=-1)
        ball_changes += int((old_ball.argmax(dim=-1) != new_ball.argmax(dim=-1)).sum())
        city_changes += int(((old_city.argmax(dim=-1) != new_city.argmax(dim=-1)) &
                             legal_balls).sum())
        legal_scores = scores[available]
        score_count += legal_scores.numel()
        aggregates["score_sum"] += legal_scores.sum().item()
        aggregates["score_square_sum"] += legal_scores.square().sum().item()
        score_min = min(score_min, legal_scores.min().item())
        score_max = max(score_max, legal_scores.max().item())
        current = route_tensor[:, :, step]
        available.scatter_(2, current[..., None], False)
    states = n * (n - 1)
    score_mean = aggregates["score_sum"] / score_count
    score_variance = max(0., aggregates["score_square_sum"] / score_count - score_mean ** 2)
    return {"decision_states": states,
            "ball_probability_l1_mean": aggregates["ball_probability_l1"] / (n - 1),
            "city_probability_l1_mean": aggregates["city_probability_l1"] / (n - 1),
            "joint_probability_l1_mean": aggregates["joint_probability_l1"] / (n - 1),
            "ball_argmax_changes": ball_changes, "city_argmax_changes": city_changes,
            "score_mean": score_mean, "score_std": score_variance ** .5,
            "score_min": score_min, "score_max": score_max,
            "score_count": score_count}


@torch.inference_mode()
def evaluate(policy, instances, optima: dict[str, int], config) -> tuple[list[dict], dict]:
    policy.eval()
    rows = []
    for index, item in enumerate(instances, 1):
        prepared = prepare_tsp(item.path)
        matrix = prepared.distance_matrix
        n = item.city_count
        raw, _ = generate_routes(policy, prepared, list(range(n)),
                                 config.validation_samples_per_start, config.seed + index)
        raw_lengths = [tour_length(route, matrix) for route in raw]
        unique = {}
        for j in sorted(range(len(raw)), key=lambda j: (raw_lengths[j], j)):
            unique.setdefault(canonical(raw[j]), raw[j])
        refined = [two_opt(route, matrix) for route in unique.values()]
        best = min([*raw_lengths, *(r.length for r in refined)])
        plateau = {canonical(route): route for route, length in zip(raw, raw_lengths)
                   if length == best}
        plateau.update({canonical(r.route): r.route for r in refined if r.length == best})
        optb = equal_plateau(list(plateau.values()), matrix, config.validation_optb_checks)
        optimum = optima.get(item.path.name)
        if optimum is not None and (min(raw_lengths) < optimum or best < optimum or
                                    optb.best_length < optimum):
            raise RuntimeError(f"{item.path.name}: route below certified optimum")
        heuristic = heuristic_reference(matrix)
        raw_best_route = raw[raw_lengths.index(min(raw_lengths))]
        crossing = sum(prepared.city_to_ball[raw_best_route[j]] !=
                       prepared.city_to_ball[raw_best_route[(j + 1) % n]] for j in range(n))
        inter = sum(matrix[raw_best_route[j]][raw_best_route[(j + 1) % n]]
                    for j in range(n) if prepared.city_to_ball[raw_best_route[j]] !=
                    prepared.city_to_ball[raw_best_route[(j + 1) % n]])
        diagnostics = decision_diagnostics(policy, prepared, raw[:n])
        rows.append({"instance": item.path.name, "distribution": item.distribution,
            "city_count": n, "raw_mean_length": sum(raw_lengths) / len(raw_lengths),
            "raw_best_length": min(raw_lengths), "two_opt_best_length": best,
            "optb_best_length": optb.best_length,
            "raw_unique_routes": len(unique), "two_opt_best_routes": len(plateau),
            "optb_routes": len(optb.routes), "inter_length_at_raw_best": inter,
            "intra_length_at_raw_best": min(raw_lengths) - inter,
            "crossings_at_raw_best": crossing,
            "certified_optimal_length": optimum if optimum is not None else "",
            "raw_optimal_gap_pct": 100 * (min(raw_lengths) / optimum - 1) if optimum else "",
            "two_opt_optimal_gap_pct": 100 * (best / optimum - 1) if optimum else "",
            "optb_optimal_gap_pct": 100 * (optb.best_length / optimum - 1) if optimum else "",
            "sampled_two_opt_optimal_hit": int(best == optimum) if optimum else "",
            "sampled_two_opt_optimal_gap_pct": 100 * (best / optimum - 1) if optimum else "",
            "sampled_certified_optb_routes": len(optb.routes) if optimum is not None
                and optb.best_length == optimum else 0,
            "sampled_heuristic_relative_2opt_pct": 100 * (best / heuristic - 1),
            "sampled_heuristic_relative_raw_pct": 100 * (min(raw_lengths) / heuristic - 1),
            **diagnostics})
        print(f"validation {index}/{len(instances)}: {item.path.name}", flush=True)
    if len(rows) != 100:
        raise RuntimeError("independent validation set changed")
    from gb_2_maco_POMO.experiments.dual_reward.selection import metrics
    selection = metrics(rows)
    return rows, selection
