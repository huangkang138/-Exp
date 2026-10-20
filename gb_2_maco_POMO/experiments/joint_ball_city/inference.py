"""Seeded two-stage greedy and sampling inference for the joint policy."""

from __future__ import annotations

from pathlib import Path

import torch

from gb_2_maco_POMO.attention import build_city_features

from .model import JointBallCityPolicy, normalized_distance


def load_checkpoint(path: Path) -> tuple[JointBallCityPolicy, dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    architecture = state["architecture"]
    model = JointBallCityPolicy(int(architecture["embedding_dim"]),
                                int(architecture["num_heads"]),
                                int(architecture["encoder_layers"]))
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model, state


@torch.inference_mode()
def generate_routes(policy: JointBallCityPolicy, prepared, starts: list[int],
                    samples: int, seed: int):
    device = next(policy.parameters()).device
    torch.manual_seed(seed)
    city_count = len(prepared.cities)
    features = build_city_features(
        prepared.cities, prepared.granular_balls, prepared.city_to_ball).to(device)
    _, encoded = policy.encode(features)
    mapping = torch.tensor(prepared.city_to_ball, dtype=torch.long, device=device)[None, :]
    distance = normalized_distance(prepared, device)
    first = torch.tensor(starts, dtype=torch.long, device=device)[None, :]
    generated = []
    for repeat in range(samples + 1):
        routes = torch.empty((1, len(starts), city_count), dtype=torch.long, device=device)
        routes[:, :, 0] = first
        current = first
        available = torch.ones((1, len(starts), city_count), dtype=torch.bool, device=device)
        available.scatter_(2, first[..., None], False)
        for step in range(1, city_count):
            current, _, _, _ = policy.choose_joint_step(
                encoded, first, current, available, mapping, distance, sample=repeat > 0)
            routes[:, :, step] = current
            available = available.scatter(2, current[..., None], False)
        if not torch.equal(routes.sort(dim=-1).values,
                           torch.arange(city_count, device=device).expand_as(routes)):
            raise RuntimeError("joint inference produced an invalid tour")
        generated.extend(routes[0].cpu().tolist())
    return generated, None
