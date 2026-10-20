"""Gate training on epoch-0 equivalence, masking and alpha gradient."""

from __future__ import annotations

import argparse
import json

import torch

from gb_2_maco_POMO.attention import build_city_features
from gb_2_maco_POMO.experiments.learnable_ball.learnable_rollout import rollout_batch
from gb_2_maco_POMO.experiments.learnable_ball.train_learnable import _loss
from gb_2_maco_POMO.runner import prepare_tsp

from .common import initial_model, load_data
from .config import INITIAL_SHA256, RESULTS


def _joint(ball_logits, city_logits, available, mapping):
    groups = ball_logits.shape[-1]
    eligible = available[:, :, None, :] & (mapping[:, None, None, :] ==
                 torch.arange(groups, device=mapping.device)[None, None, :, None])
    conditional = torch.softmax(city_logits[:, :, None, :].masked_fill(
        ~eligible, torch.finfo(city_logits.dtype).min), dim=-1)
    conditional = conditional * eligible
    selected = conditional.gather(2, mapping[:, None, None, :].expand(
        -1, conditional.shape[1], -1, -1)).squeeze(2)
    return torch.softmax(ball_logits, dim=-1).gather(
        2, mapping[:, None, :].expand(-1, ball_logits.shape[1], -1)) * selected


def inspect(device: torch.device):
    train, _ = load_data()
    item = next(x for x in train if x.city_count >= 50
                and len(prepare_tsp(x.path).granular_balls) > 1)
    problem = prepare_tsp(item.path)
    a, _ = initial_model("A", device)
    b, _ = initial_model("B", device)
    a.eval(); b.eval()
    n = item.city_count
    features = build_city_features(problem.cities, problem.granular_balls,
                                   problem.city_to_ball).to(device)
    mapping = torch.tensor(problem.city_to_ball, dtype=torch.long, device=device)[None, :]
    with torch.inference_mode():
        _, encoded_a = a.encode(features)
        _, encoded_b = b.encode(features)
        if not torch.equal(encoded_a, encoded_b):
            raise RuntimeError("epoch-0 encoder output differs")
        first = torch.arange(n, device=device)[None, :]
        current = first
        available = torch.ones((1, n, n), dtype=torch.bool, device=device)
        available.scatter_(2, first[..., None], False)
        max_ball = max_joint = 0.0
        for step in range(1, n):
            ab, ac = a.action_logits(encoded_a, first, current, available, mapping)
            bb, bc = b.action_logits(encoded_b, first, current, available, mapping)
            if not torch.equal(ab, bb) or not torch.equal(ac, bc):
                raise RuntimeError(f"epoch-0 action logits differ at step {step}")
            ja = _joint(ab, ac, available, mapping)
            jb = _joint(bb, bc, available, mapping)
            if not torch.equal(ja, jb):
                raise RuntimeError(f"epoch-0 joint city probabilities differ at step {step}")
            max_ball = max(max_ball, float((ab[torch.isfinite(ab)] - bb[torch.isfinite(bb)]).abs().max()))
            max_joint = max(max_joint, float((ja - jb).abs().max()))
            # Follow the same greedy path so all later states are reachable.
            chosen_ball = ab.argmax(dim=-1)
            city_mask = available & (mapping[:, None, :] == chosen_ball[..., None])
            current = ac.masked_fill(~city_mask, -torch.inf).argmax(dim=-1)
            available.scatter_(2, current[..., None], False)
        route_a = rollout_batch(a, [item], device, sample=False).routes
        route_b = rollout_batch(b, [item], device, sample=False).routes
        if not torch.equal(route_a, route_b):
            raise RuntimeError("epoch-0 greedy POMO routes differ")
        empty_ball = problem.city_to_ball[0]
        first_one = torch.zeros((1, 1), dtype=torch.long, device=device)
        some_available = torch.tensor([[[ball != empty_ball for ball in problem.city_to_ball]]],
                                      dtype=torch.bool, device=device)
        logits, _ = b.action_logits(encoded_b, first_one, first_one, some_available, mapping)
        _, _, counts, weights = b.ball_summary(
            encoded_b, first_one, first_one, some_available, mapping)
        if (counts[0, 0, empty_ball] != 0 or
                not torch.isneginf(logits[0, 0, empty_ball]) or
                weights[0, 0, empty_ball].abs().sum() != 0):
            raise RuntimeError("empty ball mask is invalid")
        nonempty = counts > 0
        if (not torch.allclose(weights.sum(-1)[nonempty],
                               torch.ones_like(weights.sum(-1)[nonempty]), atol=1e-6)
                or (weights.masked_select(~(some_available[:, :, None, :] &
                      (mapping[:, None, None, :] == torch.arange(
                          counts.shape[-1], device=device)[None, None, :, None]))) != 0).any()):
            raise RuntimeError("attention weights violate the unvisited-city mask")
        # Hold the available set fixed to isolate the current-city Query effect.
        shared = torch.ones((1, 1, n), dtype=torch.bool, device=device)
        shared[:, :, :3] = False
        first_one = torch.zeros((1, 1), dtype=torch.long, device=device)
        _, _, counts1, w1 = b.ball_summary(encoded_b, first_one,
            torch.ones_like(first_one), shared, mapping)
        _, _, _, w2 = b.ball_summary(encoded_b, first_one,
            torch.full_like(first_one, 2), shared, mapping)
        multi = counts1 >= 2
        variation = float(((w1 - w2).abs() * multi[..., None]).max())
        if variation <= 1e-8:
            raise RuntimeError("attention weights did not respond to current city")
    b.train()
    b.zero_grad(set_to_none=True)
    torch.manual_seed(12345)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(12345)
    loss = _loss(rollout_batch(b, [item], device, sample=True))
    loss.backward()
    alpha_grad = b.ball_attention_alpha.grad
    if alpha_grad is None or not torch.isfinite(alpha_grad).item() or alpha_grad.abs().item() <= 1e-9:
        raise RuntimeError("new attention alpha has no effective POMO gradient")
    decoder_grad = sum(float(p.grad.square().sum()) for p in b.decoder.parameters()
                       if p.grad is not None) ** 0.5
    if decoder_grad <= 0:
        raise RuntimeError("original city decoder lost its POMO gradient")
    return {"passed": True, "device": str(device), "case": item.path.name,
            "checkpoint_sha256": INITIAL_SHA256, "max_ball_logit_delta": max_ball,
            "max_joint_probability_delta": max_joint, "greedy_routes_equal": True,
            "empty_ball_masked": True, "attention_weight_change": variation,
            "alpha_gradient": float(alpha_grad), "decoder_gradient_norm": decoder_grad,
            "old_parameter_count": sum(p.numel() for p in a.parameters()),
            "new_parameter_count": sum(p.numel() for p in b.parameters())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.device == "cpu":
        torch.set_num_threads(1)
    report = inspect(torch.device(args.device))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"preflight_{args.device}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
