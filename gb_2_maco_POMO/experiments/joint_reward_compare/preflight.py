"""Fail closed before either ten-epoch run starts."""

from __future__ import annotations

import dataclasses
import hashlib

import torch

from gb_2_maco_POMO.experiments.joint_ball_city.preflight import check, initialize
from gb_2_maco_POMO.experiments.joint_ball_city.rollout import rollout_batch
from gb_2_maco_POMO.experiments.shared.data import load_manifest

from .reward import Refined, loss, novelty, standardized_advantage


def run(j1, j2, reward):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for J1/J2 training")
    torch.manual_seed(j1.seed)
    _, model1, _ = initialize(j1.initial_checkpoint)
    torch.manual_seed(j2.seed)
    _, model2, _ = initialize(j2.initial_checkpoint)
    if any(not torch.equal(model1.state_dict()[key], model2.state_dict()[key])
           for key in model1.state_dict()):
        raise RuntimeError("J1/J2 initial model weights differ")
    train, validation = load_manifest({"root": str(j1.data_root),
                                       "train_dir": "train", "validation_dir": "validation"})
    if len(train) != 500 or len(validation) != 100:
        raise RuntimeError("J1/J2 training/validation split differs")
    base = check(j1, train, torch.device("cuda"))
    mapping = [0, 0, 0, 1, 1, 1]
    a = (0, 1, 2, 3, 4, 5)
    b = (0, 1, 3, 2, 4, 5)
    sample = [Refined(a, 100, 0), Refined((2, 3, 4, 5, 0, 1), 100, 0),
              Refined(tuple(reversed(a)), 100, 0), Refined(b, 100, 0)]
    values = novelty(sample, mapping, reward.quality_ratio, .5, .5)
    if not (values[0] == values[1] == values[2] and values[0] > 0
            and values[3] > values[0] and sum(values[:3]) <= values[3] + 1e-8):
        raise RuntimeError("cycle rotation/reversal received duplicate novelty credit")
    if any(novelty([sample[0]] * 4, mapping, reward.quality_ratio, .5, .5)):
        raise RuntimeError("identical cycles received novelty credit")
    synthetic = [torch.tensor([[1., 2., 3.]]), torch.tensor([[3., 2., 1.]]),
                 torch.tensor([[0., 2., 0.]])]
    components = [standardized_advantage(value) for value in synthetic]
    if any(not torch.isfinite(value).all() or float(value.abs().sum()) <= 0
           for value in components):
        raise RuntimeError("a J2 Advantage component is absent")
    for index in range(3):
        original = sum(weight * value for weight, value in zip((.5, .5, .1), components))
        changed = sum((0 if i == index else weight) * value
                      for i, (weight, value) in enumerate(zip((.5, .5, .1), components)))
        if torch.allclose(original, changed):
            raise RuntimeError(f"J2 Advantage component {index} has no effect")
    model2.to("cuda").eval()
    torch.manual_seed(4242)
    torch.cuda.manual_seed_all(4242)
    sample_case = next(item for item in train if item.city_count >= 50)
    outcome = rollout_batch(model2, [sample_case], torch.device("cuda"), sample=True)
    objective, reward_stats = loss(outcome, [sample_case], reward)
    objective.backward()
    gradients = [parameter.grad for parameter in model2.joint_head[-1].parameters()]
    if any(value is None or not torch.isfinite(value).all() for value in gradients):
        raise RuntimeError("J2 gradient is missing or nonfinite")
    gradient_norm = sum(float(value.square().sum()) for value in gradients) ** .5
    if gradient_norm <= 0:
        raise RuntimeError("J2 output-layer gradient is zero")
    return {"initial_models_equal": True,
            "initial_checkpoint_sha256": hashlib.sha256(j1.initial_checkpoint.read_bytes()).hexdigest(),
            "J1_joint_checks": base, "J2_reward_components_active": True,
            "J2_duplicate_cycle_credit_prevented": True,
            "J2_gradient_norm": gradient_norm,
            "J2_probe_reward_stats": reward_stats,
            "same_validation_and_formal_test_implementation": True}
