"""Focused checks for compatibility, policy gradients and checkpoint rules."""

from __future__ import annotations

import random
import os
import unittest
from pathlib import Path

import torch

from gb_2_maco_POMO.runner import prepare_tsp
from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import generate_learned_routes
from gb_2_maco_POMO.experiments.learnable_ball.learnable_policy import LearnableBallPolicy
from gb_2_maco_POMO.experiments.shared.data import DatasetInstance
from .model import BallBiasPolicy
from .reward import diversity_signal, policy_loss
from .rollout import generate_routes, geometry_batch, rollout_batch
from .validation import select_checkpoints

ROOT = Path(os.environ.get("GB_MACO_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
DATA = ROOT / "benachmark_Train_MATSP_Multiopt_500/train/n050"


def multiball_case():
    for path in DATA.rglob("*.tsp"):
        problem = prepare_tsp(path)
        if len(problem.granular_balls) > 1:
            return path, problem
    raise RuntimeError("no multi-ball test case")


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.path, cls.problem = multiball_case()

    def test_zero_bias_matches_original_and_changes_ball_logits_when_trained(self):
        torch.manual_seed(7)
        old = LearnableBallPolicy()
        new = BallBiasPolicy(use_bias=True)
        missing = new.load_state_dict(old.state_dict(), strict=False)
        self.assertEqual(set(missing.missing_keys), {
            "bias_head.0.weight", "bias_head.0.bias",
            "bias_head.2.weight", "bias_head.2.bias"})
        self.assertEqual(missing.unexpected_keys, [])
        features, mapping, geometry = geometry_batch([self.problem], torch.device("cpu"))
        _, encoded = new.encode(features)
        n = len(self.problem.cities)
        first = torch.arange(n)[None, :]
        available = torch.ones((1, n, n), dtype=torch.bool)
        available.scatter_(2, first[..., None], False)
        original = old.action_logits(encoded, first, first, available, mapping)
        unchanged = new.action_logits(encoded, first, first, available, mapping, geometry)
        self.assertTrue(torch.equal(original[0], unchanged[0]))
        self.assertTrue(torch.equal(original[1], unchanged[1]))
        self.assertEqual(new.bias_parameter_count, 97)
        with torch.no_grad():
            new.bias_head[-1].weight[0, 0] = 1.0
        changed = new.action_logits(encoded, first, first, available, mapping, geometry)
        finite = torch.isfinite(original[0])
        self.assertTrue(torch.any((changed[0] - original[0])[finite].abs() > 0))
        self.assertTrue(torch.equal(torch.isfinite(changed[0]), finite))
        self.assertTrue(torch.any(changed[0].argmax(-1) != original[0].argmax(-1)))

    def test_joint_gradient_and_sampled_diversity(self):
        torch.manual_seed(8)
        model = BallBiasPolicy(use_bias=True)
        item = DatasetInstance(self.path, len(self.problem.cities), "test")
        outcome = rollout_batch(model, [item], torch.device("cpu"), sample=True)
        diversity, selected, details = diversity_signal(
            outcome, [item], routes_per_instance=8, tolerance_pct=0,
            rng=random.Random(42))
        self.assertEqual(details[0]["evaluated"], 8)
        self.assertEqual(int(selected.sum()), 8)
        self.assertTrue(((diversity >= 0) & (diversity <= 1)).all())
        loss = policy_loss(outcome, diversity_weight=0.02,
                           diversity=diversity, assessed=selected)
        loss.backward()
        for params in (model.ball_head.parameters(), model.bias_head.parameters(),
                       model.decoder.parameters()):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in params))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                            if p.grad is not None))

    def test_disabled_bias_replays_original_routes(self):
        torch.manual_seed(9)
        old = LearnableBallPolicy()
        new = BallBiasPolicy(use_bias=False)
        new.load_state_dict(old.state_dict(), strict=True)
        starts = list(range(len(self.problem.cities)))
        historical, _ = generate_learned_routes(old, self.problem, starts, 1, 123)
        current = generate_routes(new, self.problem, samples=1, seed=123,
                                  device=torch.device("cpu"))
        self.assertEqual(historical, current)

    def test_checkpoint_quality_gate(self):
        a = {"epoch": 1, "metrics": {"mean_certified_gap_pct": 0.2,
             "certified_hits": 50, "mean_heuristic_relative_2opt_pct": -1,
             "mean_heuristic_relative_raw_pct": 2,
             "mean_certified_optb_routes": 2}}
        b = {"epoch": 2, "metrics": dict(a["metrics"],
             mean_certified_gap_pct=0.4, certified_hits=49,
             mean_certified_optb_routes=5)}
        c = {"epoch": 3, "metrics": dict(a["metrics"],
             mean_certified_gap_pct=1.1, certified_hits=48,
             mean_certified_optb_routes=20)}
        length, multi = select_checkpoints([a, b, c],
            quality_gap_tolerance_pct=0.5, allowed_hit_loss=1)
        self.assertEqual((length["epoch"], multi["epoch"]), (1, 2))


if __name__ == "__main__":
    unittest.main()
