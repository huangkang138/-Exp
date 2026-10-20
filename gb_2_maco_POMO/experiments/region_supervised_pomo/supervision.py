"""Teacher forcing for the unchanged ball-then-city policy."""

from __future__ import annotations

import random

import torch

from gb_2_maco_POMO.attention import build_city_features
from gb_2_maco_POMO.runner import prepare_tsp

from .data import LabeledCase


def draw_route(case: LabeledCase, mode: str, rng: random.Random):
    if mode == "B":
        index = rng.randrange(len(case.tours))
        region = next(i for i, members in enumerate(case.regions) if index in members)
    elif mode == "C":
        region = rng.randrange(len(case.regions))
        index = rng.choice(case.regions[region])
    else:
        raise ValueError(mode)
    tour = case.tours[index]
    offset = rng.randrange(len(tour))
    route = tour[offset:] + tour[:offset]
    if rng.randrange(2):
        route = (route[0],) + tuple(reversed(route[1:]))
    return route, region, index


def teacher_forcing_loss(model, selected: list[tuple[LabeledCase, tuple[int, ...]]], device):
    if not selected:
        raise ValueError("teacher forcing requires labeled instances")
    losses = []
    for case, route in selected:
        problem = prepare_tsp(case.instance.path)
        n = len(problem.cities)
        if len(route) != n or set(route) != set(range(n)):
            raise RuntimeError(f"illegal supervision tour: {case.label_id}")
        features = build_city_features(problem.cities, problem.granular_balls,
                                       problem.city_to_ball).to(device)
        _, encoded = model.encode(features)
        mapping = torch.tensor(problem.city_to_ball, dtype=torch.long, device=device)[None, :]
        first = torch.tensor([[route[0]]], dtype=torch.long, device=device)
        current = first
        available = torch.ones((1, 1, n), dtype=torch.bool, device=device)
        available.scatter_(2, first[..., None], False)
        log_terms = []
        for city in route[1:]:
            target = torch.tensor([[city]], dtype=torch.long, device=device)
            ball = mapping.gather(1, target)
            if not available.gather(2, target[..., None]).all().item():
                raise RuntimeError(f"teacher target masked: {case.label_id}, city={city}")
            ball_logits, city_logits = model.action_logits(
                encoded, first, current, available, mapping)
            if not torch.isfinite(ball_logits.gather(2, ball[..., None])).all().item():
                raise RuntimeError(f"teacher ball masked: {case.label_id}, city={city}")
            city_mask = available & (mapping[:, None, :] == ball[..., None])
            conditional = city_logits.masked_fill(~city_mask, -torch.inf)
            if not torch.isfinite(conditional.gather(2, target[..., None])).all().item():
                raise RuntimeError(f"teacher city masked: {case.label_id}, city={city}")
            joint = torch.log_softmax(ball_logits, -1).gather(2, ball[..., None])
            joint = joint + torch.log_softmax(conditional, -1).gather(2, target[..., None])
            log_terms.append(joint.squeeze())
            available = available.scatter(2, target[..., None], False)
            current = target
        losses.append(-torch.stack(log_terms).mean())
    result = torch.stack(losses).mean()
    if not torch.isfinite(result).item():
        raise RuntimeError("nonfinite supervised loss")
    return result
