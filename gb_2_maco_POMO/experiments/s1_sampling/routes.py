"""S1: recompute the best ball at each step and sample only its city."""

from __future__ import annotations

import torch

from gb_2_maco_POMO.attention import build_city_features


@torch.inference_mode()
def generate_s1_routes(policy, prepared, starts, samples, seed):
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
            # The state changes after every city, so the best ball is recomputed.
            ball_logits, city_logits = policy.action_logits(
                encoded, first, current, available, mapping)
            chosen_ball = ball_logits.argmax(dim=-1)
            city_mask = available & (mapping[:, None, :] == chosen_ball[..., None])
            conditional_logits = city_logits.masked_fill(~city_mask, -torch.inf)
            if repeat == 0:
                current = conditional_logits.argmax(dim=-1)
            else:
                current = torch.distributions.Categorical(logits=conditional_logits).sample()
            if not available.gather(2, current[..., None]).all().item():
                raise RuntimeError("S1 selected an unavailable city")
            route[:, :, step] = current
            available.scatter_(2, current[..., None], False)
        if not torch.equal(route.sort(dim=-1).values,
                           torch.arange(n, device=device).expand_as(route)):
            raise RuntimeError("S1 generated an invalid tour")
        generated.extend(route[0].cpu().tolist())
    return generated, None
