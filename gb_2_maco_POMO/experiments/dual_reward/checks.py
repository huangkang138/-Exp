"""Fail-closed checks executed by the experiment entry point before training."""

from __future__ import annotations

import copy

import torch

from gb_2_maco_POMO.experiments.learnable_ball.learnable_rollout import rollout_batch as original_rollout
from gb_2_maco_POMO.experiments.learnable_ball.train_learnable import _loss as original_loss
from gb_2_maco_POMO.experiments.shared.data import DatasetInstance
from gb_2_maco_POMO.runner import prepare_tsp

from .reward import LAMBDA, decompose, losses
from .rollout import rollout_batch


def _norm(grads):
    return sum(float(g.detach().square().sum()) for g in grads if g is not None) ** 0.5


def check(policy, train: list[DatasetInstance], device: torch.device) -> dict:
    item = next((x for x in train if x.city_count >= 50 and
                 len(prepare_tsp(x.path).granular_balls) > 1), None)
    if item is None:
        raise RuntimeError("no multi-ball original training case for checks")
    trial = copy.deepcopy(policy).to(device)
    trial.encoder.requires_grad_(False)
    trial.train()
    torch.manual_seed(1729)
    historical = original_rollout(trial, [item], device, sample=True)
    torch.manual_seed(1729)
    current = rollout_batch(trial, [item], device, sample=True)
    if (not torch.equal(current.routes, historical.routes) or
            not torch.equal(current.lengths, historical.lengths) or
            not torch.allclose(current.joint_log_probs, historical.joint_log_probs,
                               atol=1e-4, rtol=1e-5)):
        raise RuntimeError("split-logP rollout changed original action sampling")
    prepared = prepare_tsp(item.path)
    route = current.routes[0, 0].tolist()
    edges = [(route[i], route[(i + 1) % len(route)]) for i in range(len(route))]
    explicit_inter = sum(prepared.distance_matrix[a][b] for a, b in edges
                         if prepared.city_to_ball[a] != prepared.city_to_ball[b])
    explicit_intra = sum(prepared.distance_matrix[a][b] for a, b in edges
                         if prepared.city_to_ball[a] == prepared.city_to_ball[b])
    if (current.lengths[0, 0].item() != explicit_inter + explicit_intra or
            current.inter_lengths[0, 0].item() != explicit_inter or
            current.intra_lengths[0, 0].item() != explicit_intra or
            current.crossings[0, 0].item() != sum(
                prepared.city_to_ball[a] != prepared.city_to_ball[b] for a, b in edges)):
        raise RuntimeError("edge classes or closing edge are incorrect")
    parameters = [p for p in trial.parameters() if p.requires_grad]
    zero_ball, zero_city, _ = losses(current, 0.0)
    zero_loss = zero_ball + zero_city
    reference_loss = original_loss(current)
    grad_zero = torch.autograd.grad(zero_loss, parameters, retain_graph=True)
    grad_ref = torch.autograd.grad(reference_loss, parameters, retain_graph=True)
    max_gradient_error = max((a - b).abs().max().item()
                             for a, b in zip(grad_zero, grad_ref))
    gradients_match = all(torch.allclose(a, b, atol=1e-4, rtol=1e-4)
                          for a, b in zip(grad_zero, grad_ref))
    if (not torch.allclose(zero_loss, reference_loss, atol=1e-5, rtol=1e-5)
            or not gradients_match):
        raise RuntimeError("lambda=0 loss/gradient differs from original POMO")
    ball, city, stats = losses(current, LAMBDA)
    ball_parameters = list(trial.ball_head.parameters())
    decoder_parameters = list(trial.decoder.parameters())
    ball_head_grad = _norm(torch.autograd.grad(
        ball, ball_parameters, retain_graph=True))
    decoder_ball_grad = _norm(torch.autograd.grad(
        ball, decoder_parameters, retain_graph=True, allow_unused=True))
    decoder_city_grad = _norm(torch.autograd.grad(
        city, decoder_parameters, retain_graph=True, allow_unused=True))
    if ball_head_grad <= 0 or decoder_city_grad <= 0:
        raise RuntimeError("ball head or city decoder has no effective gradient")
    encoder_before = [p.detach().clone() for p in trial.encoder.parameters()]
    optimizer = torch.optim.Adam((p for p in trial.parameters() if p.requires_grad), lr=1e-4)
    optimizer.zero_grad()
    (ball + city).backward()
    if any(not torch.isfinite(p.grad).all().item() for p in parameters if p.grad is not None):
        raise RuntimeError("nonfinite reward gradient")
    optimizer.step()
    if any(not torch.equal(before, after) for before, after in
           zip(encoder_before, trial.encoder.parameters())):
        raise RuntimeError("frozen encoder changed in probe update")
    return {"instance": item.path.name, "city_count": item.city_count,
            "ball_count": len(prepared.granular_balls),
            "tested_pomo_routes": current.routes.shape[1],
            "lambda_zero_loss_error": abs((zero_loss - reference_loss).item()),
            "lambda_zero_max_gradient_error": max_gradient_error,
            "ball_head_from_ball_loss_grad_norm": ball_head_grad,
            "decoder_from_ball_loss_grad_norm": decoder_ball_grad,
            "decoder_from_city_loss_grad_norm": decoder_city_grad,
            "shared_trainable_parameters": ["decoder (ball query and city logits)"],
            "encoder_frozen": True, "route_and_edge_checks_passed": True,
            "no_2opt_in_reward": True, **stats}
