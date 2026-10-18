"""Two-stage inference on unchanged, bias-free Attention decoder logits."""

from __future__ import annotations

import torch
from torch import Tensor


def distance_scale(distance: list[list[int]]) -> float:
    """Median positive nearest-neighbor edge, in the dataset's integer distance unit."""
    nearest = [min(value for j, value in enumerate(row) if j != i and value > 0)
               for i, row in enumerate(distance)]
    return max(float(torch.tensor(nearest, dtype=torch.float64).median().item()), 1.0)


def two_stage_probabilities(
    logits: Tensor, available: Tensor, city_to_ball: Tensor,
    current: Tensor, distance: Tensor, scale: float, penalty: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return P(ball), P(city|ball), and their marginal P(city).

    logits/available are [P,N], current [P], mapping [N], distance [N,N].
    Empty balls stay at -inf. The minimum entrance edge is computed only over
    unvisited cities, using the same integer edge weights as tour_length.
    """
    if penalty < 0 or scale <= 0 or logits.shape != available.shape:
        raise ValueError("invalid penalty, scale or tensor shape")
    if not available.any(dim=1).all().item():
        raise ValueError("each route needs an unvisited city")
    p, n = logits.shape
    if city_to_ball.shape != (n,) or distance.shape != (n, n) or current.shape != (p,):
        raise ValueError("mapping/distance/current shape mismatch")
    ball_count = int(city_to_ball.max().item()) + 1
    masked_logits = logits.masked_fill(~available, -torch.inf)
    entrance = distance[current]
    ball_scores = logits.new_full((p, ball_count), -torch.inf)
    conditional = logits.new_zeros((p, n))
    for ball in range(ball_count):
        members = city_to_ball == ball
        eligible = available & members[None, :]
        has_member = eligible.any(dim=1)
        if not has_member.any().item():
            continue
        within = masked_logits.masked_fill(~members[None, :], -torch.inf)
        mass = torch.logsumexp(within[has_member], dim=1)
        cost = entrance.masked_fill(~eligible, torch.inf).amin(dim=1)[has_member]
        ball_scores[has_member, ball] = mass - penalty * cost / scale
        conditional[has_member] += torch.softmax(within[has_member], dim=1)
    ball_prob = torch.softmax(ball_scores, dim=1)
    city_prob = conditional * ball_prob[:, city_to_ball]
    return ball_prob, conditional, city_prob


def select_city(
    ball_prob: Tensor, conditional: Tensor, city_to_ball: Tensor, sampling: bool,
) -> Tensor:
    """Make the two actual choices; argmax is sequential at each level."""
    if sampling:
        balls = torch.distributions.Categorical(probs=ball_prob).sample()
    else:
        balls = ball_prob.argmax(dim=1)
    allowed = city_to_ball[None, :] == balls[:, None]
    city_prob = conditional * allowed
    city_prob = city_prob / city_prob.sum(dim=1, keepdim=True)
    if sampling:
        return torch.distributions.Categorical(probs=city_prob).sample()
    return city_prob.argmax(dim=1)
