"""Fail-closed policy, sampling, gradient and frozen-weight checks."""

from __future__ import annotations

import copy
import hashlib

import torch

from gb_2_maco_POMO.attention import build_city_features
from gb_2_maco_POMO.experiments.hierarchical.original25 import BENCHMARKS, DATA
from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import (
    generate_learned_routes, load_checkpoint as load_original,
)
from gb_2_maco_POMO.experiments.learnable_ball.train_learnable import _loss
from gb_2_maco_POMO.experiments.region_supervised_pomo.config import INITIAL_SHA256
from gb_2_maco_POMO.runner import prepare_tsp

from .inference import generate_routes
from .model import JointBallCityPolicy, normalized_distance
from .rollout import JointRollout, rollout_batch
from .validation import decision_diagnostics


def initialize(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != INITIAL_SHA256:
        raise RuntimeError("A0 checkpoint hash changed")
    old, state = load_original(path)
    architecture = state["architecture"]
    model = JointBallCityPolicy(int(architecture["embedding_dim"]),
                                int(architecture["num_heads"]),
                                int(architecture["encoder_layers"]))
    incompatible = model.load_state_dict(state["model_state_dict"], strict=False)
    expected = {f"joint_head.{suffix}" for suffix in
                ("0.weight", "0.bias", "2.weight", "2.bias")}
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError(f"A0 weight loading mismatch: {incompatible}")
    model.freeze_original()
    return old, model, state


def _probability_check(old, joint, prepared, device):
    n = len(prepared.cities)
    features = build_city_features(prepared.cities, prepared.granular_balls,
                                   prepared.city_to_ball).to(device)
    with torch.no_grad():
        _, encoded = old.encode(features)
    first = torch.arange(n, device=device)[None, :]
    current = first.clone()
    available = torch.ones((1, n, n), dtype=torch.bool, device=device)
    available.scatter_(2, first[..., None], False)
    mapping = torch.tensor(prepared.city_to_ball, dtype=torch.long, device=device)[None, :]
    distance = normalized_distance(prepared, device)
    max_error = 0.0
    for step in range(1, n):
        with torch.no_grad():
            ball_logits, city_logits = old.action_logits(
                encoded, first, current, available, mapping)
            new_ball, new_city, correction, eligible = joint.joint_components(
                encoded, first, current, available, mapping, distance)
            if not torch.equal(correction, torch.zeros_like(correction)):
                raise RuntimeError("epoch-0 Joint Head is not identically zero")
            old_ball = torch.softmax(ball_logits, dim=-1)
            new_ball_prob = torch.softmax(new_ball, dim=-1)
            if not torch.equal(torch.isfinite(ball_logits), torch.isfinite(new_ball)):
                raise RuntimeError("legal ball masks differ")
            if not torch.equal(torch.isfinite(new_city), eligible):
                raise RuntimeError("legal city masks differ")
            old_city_logits = city_logits[:, :, None, :].masked_fill(~eligible, -torch.inf)
            old_city = torch.softmax(old_city_logits, dim=-1).nan_to_num(0)
            new_city_prob = torch.exp(new_city).nan_to_num(0)
            error = (old_ball[..., None] * old_city -
                     new_ball_prob[..., None] * new_city_prob).abs().max().item()
            max_error = max(max_error, error)
            normal = (new_ball_prob[..., None] * new_city_prob).sum(dim=(-1, -2))
            if error > 1e-6 or not torch.allclose(normal, torch.ones_like(normal), atol=1e-6):
                raise RuntimeError(f"epoch-0 joint probabilities differ at step {step}: {error}")
            chosen_ball = ball_logits.argmax(dim=-1)
            conditional = city_logits.masked_fill(
                ~(available & (mapping[:, None, :] == chosen_ball[..., None])), -torch.inf)
            current = conditional.argmax(dim=-1)
            available.scatter_(2, current[..., None], False)
    return max_error


def _accumulation_check() -> float:
    # POMO baseline is per instance. Weighted micro means must equal full mean.
    lengths = torch.tensor([[7., 8., 9.], [11., 14., 13.]])
    logps = torch.tensor([[.2, .3, .4], [.1, .5, .8]], requires_grad=True)
    full = _loss(JointRollout(lengths, logps, torch.empty(0)))
    half = sum(_loss(JointRollout(lengths[i:i + 1], logps[i:i + 1],
                                   torch.empty(0))) / 2 for i in range(2))
    if not torch.allclose(full, half, atol=1e-7):
        raise RuntimeError("batch/microbatch loss scaling differs")
    grad_full = torch.autograd.grad(full, logps, retain_graph=True)[0]
    grad_half = torch.autograd.grad(half, logps)[0]
    error = (grad_full - grad_half).abs().max().item()
    if error > 1e-7:
        raise RuntimeError("batch/microbatch gradient scaling differs")
    return error


def check(config, train_instances, device: torch.device) -> dict:
    old, model, _ = initialize(config.initial_checkpoint)
    old.to(device).eval()
    model.to(device).eval()
    item = next((x for x in train_instances if x.city_count >= 50 and
                 len(prepare_tsp(x.path).granular_balls) > 1), None)
    if item is None:
        raise RuntimeError("no multi-ball training case for gradient probe")
    probe_problem = prepare_tsp(item.path)
    probability_error = _probability_check(old, model, probe_problem, device)
    accumulation_error = _accumulation_check()
    # The output layer gets gradients at zero initialization; the hidden layer
    # starts receiving gradients after the output layer changes.
    trial = copy.deepcopy(model)
    torch.manual_seed(1729)
    outcome = rollout_batch(trial, [item], device, sample=True)
    loss = _loss(outcome)
    loss.backward()
    last_grad = sum(float(p.grad.detach().square().sum()) for p in trial.joint_head[-1].parameters()) ** 0.5
    if last_grad <= 0 or not torch.isfinite(torch.tensor(last_grad)):
        raise RuntimeError("Joint Head output layer has no effective gradient")
    original = {name: p.detach().clone() for name, p in trial.named_parameters()
                if not name.startswith("joint_head.")}
    optimizer = torch.optim.Adam(trial.joint_head.parameters(), lr=config.learning_rate)
    optimizer.step()
    if any(not torch.equal(original[name], p) for name, p in trial.named_parameters()
           if name in original):
        raise RuntimeError("frozen A0 weights changed during probe update")
    if not any(p.detach().abs().sum().item() > 0 for p in trial.joint_head[-1].parameters()):
        raise RuntimeError("Joint Head output layer did not update")
    greedy, _ = generate_routes(model, probe_problem,
                                list(range(item.city_count)), 0, config.seed)
    changed = decision_diagnostics(trial, probe_problem, greedy)
    if (changed["ball_probability_l1_mean"] <= 1e-6 or
            changed["joint_probability_l1_mean"] <= 1e-6):
        raise RuntimeError("Joint Head update did not change ball and joint probabilities")
    trial.zero_grad(set_to_none=True)
    torch.manual_seed(1730)
    second = rollout_batch(trial, [item], device, sample=True)
    _loss(second).backward()
    hidden_grad = sum(float(p.grad.detach().square().sum())
                      for p in trial.joint_head[0].parameters()) ** .5
    if hidden_grad <= 0 or not torch.isfinite(torch.tensor(hidden_grad)):
        raise RuntimeError("Joint Head hidden layer did not receive a second-step gradient")
    # Same starts, same seed and same Categorical sampling order on all 25 cases.
    for index, (stem, n, _) in enumerate(BENCHMARKS, 1):
        prepared = prepare_tsp(DATA / f"{stem}.tsp")
        starts = list(range(n))
        historical, _ = generate_learned_routes(
            old, prepared, starts, config.validation_samples_per_start, config.seed + index)
        joint_routes, _ = generate_routes(
            model, prepared, starts, config.validation_samples_per_start, config.seed + index)
        if historical != joint_routes:
            greedy_count = n
            mismatch = next(j for j, (a, b) in enumerate(zip(historical, joint_routes)) if a != b)
            kind = "Argmax" if mismatch < greedy_count else "Sampling"
            raise RuntimeError(f"epoch-0 {kind} route mismatch on MSTSP{index}, route {mismatch}; "
                               f"first-state joint probability error={probability_error:.3g}")
        print(f"preflight MSTSP{index}/25: greedy and sampled routes match A0", flush=True)
    return {"joint_output_zero": True, "maximum_joint_probability_error": probability_error,
            "all_25_argmax_and_sampling_routes_match": True,
            "legal_masks_match": True, "joint_distribution_normalized": True,
            "joint_output_gradient_norm": last_grad,
            "joint_hidden_gradient_norm_after_update": hidden_grad,
            "ball_probability_l1_after_probe_update": changed["ball_probability_l1_mean"],
            "joint_probability_l1_after_probe_update": changed["joint_probability_l1_mean"],
            "frozen_original_parameters_unchanged": True,
            "microbatch_gradient_error": accumulation_error,
            "training_probe_route_count": int(outcome.routes.shape[1]),
            "training_probe_routes_legal": True}
