"""Shared starts, random stream and route validation for both inference modes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from gb_2_maco_POMO.attention import POMOPolicy, build_city_features, candidate_guidance
from gb_2_maco_POMO.experiments.hierarchical.hierarchical import (
    distance_scale, select_city, two_stage_probabilities,
)


def load_policy(checkpoint: Path, device_name: str) -> tuple[POMOPolicy, str]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = state["config"]["model"]
    policy = POMOPolicy(
        embedding_dim=int(model["embedding_dim"]), head_num=int(model["num_heads"]),
        encoder_layer_num=int(model["encoder_layers"]),
        candidate_mode=str(model["candidate_mode"]),
        same_ball_bias=float(model["same_ball_bias"]),
        adjacent_ball_bias=float(model["adjacent_ball_bias"]),
    ).to(device_name)
    policy.load_state_dict(state["model_state_dict"], strict=True)
    policy.eval()
    if policy.candidate_mode != "soft" or policy.bias_mode != "fixed":
        raise ValueError("fixed Soft-POMO checkpoint required")
    torch.set_num_threads(1)
    return policy, digest


def selected_starts(n: int, maximum: int) -> list[int]:
    count = min(n, maximum)
    return [i * n // count for i in range(count)]


def generate_routes(policy, prepared, starts, samples, seed, *, mode, penalty=0.0):
    if mode not in ("fixed_soft", "hierarchical"):
        raise ValueError(mode)
    n = len(prepared.cities)
    device = next(policy.parameters()).device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    mapping = torch.tensor(prepared.city_to_ball, device=device, dtype=torch.long)
    distance = torch.tensor(prepared.distance_matrix, device=device, dtype=torch.float32)
    scale = distance_scale(prepared.distance_matrix)
    generated = []
    with torch.inference_mode():
        features = build_city_features(
            prepared.cities, prepared.granular_balls, prepared.city_to_ball,
        ).to(device)
        _, encoded = policy.encode(features)
        first = torch.tensor(starts, device=device).unsqueeze(0)
        for repeat in range(samples + 1):
            routes = [[start] for start in starts]
            seen = [{start} for start in starts]
            current = first.clone()
            for _ in range(1, n):
                available = torch.zeros((1, len(starts), n), dtype=torch.bool, device=device)
                bias = torch.zeros((1, len(starts), n), device=device) if mode == "fixed_soft" else None
                for index, route in enumerate(routes):
                    if mode == "fixed_soft":
                        candidates, preference = candidate_guidance(
                            route[-1], seen[index], prepared.city_to_ball,
                            prepared.granular_balls, prepared.ball_adjacency, n, policy,
                        )
                        available[0, index, candidates] = True
                        bias[0, index] = torch.tensor(preference, device=device)
                    else:
                        available[0, index] = torch.tensor(
                            [city not in seen[index] for city in range(n)], device=device,
                        )
                probabilities = policy(encoded, first, current, available, bias)
                if mode == "fixed_soft":
                    chosen = (torch.distributions.Categorical(probs=probabilities).sample()[0]
                              if repeat else probabilities.argmax(dim=-1)[0])
                else:
                    # Decoder scores are clipped Attention logits. No +2/+1 bias
                    # was supplied; visited cities are masked by available.
                    logits = policy.decoder.last_logits[0]
                    ball_prob, conditional, _ = two_stage_probabilities(
                        logits, available[0], mapping, current[0], distance,
                        scale, penalty,
                    )
                    chosen = select_city(ball_prob, conditional, mapping, repeat > 0)
                for index, city in enumerate(chosen.tolist()):
                    if city in seen[index] or not available[0, index, city]:
                        raise RuntimeError("invalid city choice")
                    routes[index].append(city)
                    seen[index].add(city)
                current = chosen.unsqueeze(0)
            if any(len(route) != n or len(set(route)) != n for route in routes):
                raise RuntimeError("invalid complete route")
            generated.extend(routes)
    return generated, scale
