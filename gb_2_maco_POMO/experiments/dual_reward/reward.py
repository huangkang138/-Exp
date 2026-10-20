"""Closed-tour distance decomposition and per-instance POMO advantages."""

from __future__ import annotations

import torch
from torch import Tensor

EPS = 1e-8
LAMBDA = 0.3


def decompose(routes: Tensor, distance: Tensor, mapping: Tensor):
    """[B,P,N] routes; closing edge is included by roll(-1)."""
    batch, pomo, n = routes.shape
    if distance.shape != (batch, n, n) or mapping.shape != (batch, n):
        raise ValueError("route, distance and ball mapping shapes differ")
    destination = routes.roll(-1, dims=-1)
    batch_ids = torch.arange(batch, device=routes.device)[:, None, None]
    edge_length = distance[batch_ids, routes, destination]
    source_ball = mapping[batch_ids, routes]
    target_ball = mapping[batch_ids, destination]
    inter_mask = source_ball != target_ball
    total = edge_length.sum(dim=-1)
    inter = (edge_length * inter_mask).sum(dim=-1)
    intra = (edge_length * ~inter_mask).sum(dim=-1)
    if not torch.equal(total, inter + intra):
        raise RuntimeError("closed-tour length is not partitioned exactly")
    return total, inter, intra, inter_mask.sum(dim=-1)


def advantages(total: Tensor, inter: Tensor, intra: Tensor, coefficient: float):
    """Normalize within each case's POMO group; zero-variance auxiliaries vanish."""
    if not 0 <= coefficient <= LAMBDA:
        raise ValueError("coefficient is fixed at 0.3; zero is for the preflight only")
    a_total = -total + total.mean(dim=1, keepdim=True)
    a_inter = -inter + inter.mean(dim=1, keepdim=True)
    a_intra = -intra + intra.mean(dim=1, keepdim=True)
    std_total = a_total.std(dim=1, correction=0, keepdim=True)
    def scaled(value: Tensor):
        std = value.std(dim=1, correction=0, keepdim=True)
        return torch.where(std > EPS, value * std_total / (std + EPS),
                           torch.zeros_like(value))
    inter_scaled, intra_scaled = scaled(a_inter), scaled(a_intra)
    return (a_total, a_inter, a_intra,
            a_total + coefficient * inter_scaled,
            a_total + coefficient * intra_scaled)


def losses(rollout, coefficient: float = LAMBDA):
    if rollout.ball_log_probs is None or rollout.city_log_probs is None:
        raise ValueError("sampled rollout with separate log probabilities required")
    a_total, a_inter, a_intra, a_ball, a_city = advantages(
        rollout.lengths, rollout.inter_lengths, rollout.intra_lengths, coefficient)
    ball = -(a_ball.detach() * rollout.ball_log_probs).mean()
    city = -(a_city.detach() * rollout.city_log_probs).mean()
    if not torch.isfinite(ball + city):
        raise RuntimeError("nonfinite dual reward loss")
    return ball, city, {"total_adv_std": a_total.std(correction=0).item(),
                        "inter_adv_std": a_inter.std(correction=0).item(),
                        "intra_adv_std": a_intra.std(correction=0).item(),
                        "ball_loss": ball.detach().item(), "city_loss": city.detach().item()}
