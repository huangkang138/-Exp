"""Run the saved POMO policy from the same selected starts for each model.

The caller imports this module in a fresh process for each backup, so the two
historical packages (which share a Python package name) never mix in sys.modules.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def load_policy(backup: Path, device_name: str):
    sys.path.insert(0, str(backup))
    import torch
    from gb_maco_POMO.attention import POMOPolicy

    manifest = json.loads((backup / "BACKUP_MANIFEST.json").read_text(encoding="utf-8"))
    checkpoint = backup / "checkpoints" / "best.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != manifest["checkpoint_sha256"]["best.pt"]:
        raise ValueError(f"checkpoint hash changed: {checkpoint}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]["model"]
    options = dict(
        embedding_dim=int(config["embedding_dim"]),
        head_num=int(config["num_heads"]),
        encoder_layer_num=int(config["encoder_layers"]),
        candidate_mode=str(config["candidate_mode"]),
        same_ball_bias=float(config["same_ball_bias"]),
        adjacent_ball_bias=float(config["adjacent_ball_bias"]),
    )
    if "bias_mode" in config:
        options["bias_mode"] = str(config["bias_mode"])
    policy = POMOPolicy(**options).to(torch.device(device_name))
    policy.load_state_dict(state["model_state_dict"], strict=True)
    policy.eval()
    if policy.candidate_mode != "soft":
        raise ValueError("This comparison requires Soft-POMO checkpoints")
    torch.set_num_threads(1)
    return policy, digest, state.get("epoch")


def selected_starts(city_count: int, max_starts: int) -> list[int]:
    """Evenly spaced, deterministic, unique start cities."""
    count = min(city_count, max_starts)
    return [i * city_count // count for i in range(count)]


def generate_routes(policy, prepared, starts: list[int], samples: int, seed: int):
    import torch
    from gb_maco_POMO.attention import build_city_features, candidate_guidance

    city_count = len(prepared.cities)
    device = next(policy.parameters()).device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generated: list[list[int]] = []
    mapping = prepared.city_to_ball
    learnable = getattr(policy, "bias_mode", "fixed") == "learnable"

    with torch.inference_mode():
        features = build_city_features(
            prepared.cities, prepared.granular_balls, prepared.city_to_ball
        ).to(device)
        _, encoded = policy.encode(features)
        for repeat in range(samples + 1):
            routes = [[start] for start in starts]
            seen = [{start} for start in starts]
            first = torch.tensor(starts, device=device).unsqueeze(0)
            current = first.clone()
            for _ in range(1, city_count):
                mask = torch.zeros((1, len(starts), city_count), dtype=torch.bool, device=device)
                bias = torch.zeros((1, len(starts), city_count), device=device)
                same = torch.zeros_like(mask) if learnable else None
                adjacent = torch.zeros_like(mask) if learnable else None
                for index, route in enumerate(routes):
                    city = route[-1]
                    candidates, preference = candidate_guidance(
                        city, seen[index], mapping, prepared.granular_balls,
                        prepared.ball_adjacency, city_count, policy,
                    )
                    if not candidates:
                        raise RuntimeError("No available city before tour completion")
                    mask[0, index, candidates] = True
                    if learnable:
                        current_ball = mapping[city]
                        neighbors = prepared.ball_adjacency[current_ball]
                        for candidate in candidates:
                            ball = mapping[candidate]
                            same[0, index, candidate] = ball == current_ball
                            adjacent[0, index, candidate] = ball != current_ball and ball in neighbors
                    else:
                        bias[0, index] = torch.tensor(preference, device=device)
                if learnable:
                    probabilities = policy(encoded, first, current, mask, None, same, adjacent)
                else:
                    probabilities = policy(encoded, first, current, mask, bias)
                if repeat == 0:
                    chosen = probabilities.argmax(dim=-1)
                else:
                    chosen = torch.distributions.Categorical(probs=probabilities).sample()
                for index, city in enumerate(chosen[0].tolist()):
                    if city in seen[index] or not mask[0, index, city]:
                        raise RuntimeError("Policy selected an invalid city")
                    routes[index].append(city)
                    seen[index].add(city)
                current = chosen
            if any(len(route) != city_count or len(set(route)) != city_count for route in routes):
                raise RuntimeError("Invalid complete route")
            generated.extend(routes)
    return generated
