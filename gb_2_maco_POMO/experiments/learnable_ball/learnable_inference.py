"""Greedy and sampled two-level routes from the trained ball policy."""

from __future__ import annotations

from pathlib import Path

import torch

from gb_2_maco_POMO.attention import build_city_features
from gb_2_maco_POMO.experiments.learnable_ball.learnable_policy import LearnableBallPolicy


def load_checkpoint(path: Path) -> tuple[LearnableBallPolicy, dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    architecture = state["architecture"]
    policy = LearnableBallPolicy(
        int(architecture["embedding_dim"]), int(architecture["num_heads"]),
        int(architecture["encoder_layers"]),
    )
    policy.load_state_dict(state["model_state_dict"], strict=True)
    policy.eval()
    torch.set_num_threads(1)
    return policy, state


@torch.inference_mode()
def generate_learned_routes(policy, prepared, starts, samples, seed):
    device = next(policy.parameters()).device
    n = len(prepared.cities)
    torch.manual_seed(seed)
    features = build_city_features(
        prepared.cities, prepared.granular_balls, prepared.city_to_ball).to(device)
    _, encoded = policy.encode(features)
    mapping = torch.tensor(prepared.city_to_ball, dtype=torch.long, device=device)[None, :]
    first = torch.tensor(starts, dtype=torch.long, device=device)[None, :]
    generated = []
    for repeat in range(samples + 1):
        route = torch.empty((1, len(starts), n), dtype=torch.long, device=device)
        route[:, :, 0] = first
        current = first
        available = torch.ones((1, len(starts), n), dtype=torch.bool, device=device)
        available.scatter_(2, first[..., None], False)
        for step in range(1, n):
            current, _, _ = policy.choose_step(
                encoded, first, current, available, mapping, sample=repeat > 0)
            if not available.gather(2, current[..., None]).all().item():
                raise RuntimeError("learned policy selected a visited city")
            route[:, :, step] = current
            available = available.scatter(2, current[..., None], False)
        if not torch.equal(route.sort(dim=-1).values,
                           torch.arange(n, device=device).expand_as(route)):
            raise RuntimeError("learned policy generated an illegal route")
        generated.extend(route[0].cpu().tolist())
    return generated, None  # No hand-coded distance scale in this policy.
