"""Small functional checks; no formal training is started here."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from gb_maco_POMO.attention import POMOPolicy, build_city_features, candidate_guidance
from gb_maco_POMO.pomo import rollout
from gb_maco_POMO.runner import prepare_tsp


PROJECT = Path(__file__).resolve().parents[2]
SAMPLE = PROJECT.parent / "GB-MACO_PPSN2026-main/PPSN_GB-MACO/benchmark_MSTSP/simple1_9.tsp"


class LearnableBiasTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.fixed = POMOPolicy(candidate_mode="soft")
        self.learnable = POMOPolicy(candidate_mode="soft", bias_mode="learnable")
        missing = self.learnable.load_state_dict(self.fixed.state_dict(), strict=False)
        self.assertEqual(missing.unexpected_keys, [])
        self.assertTrue(all(key.startswith("decoder.bias_head.") for key in missing.missing_keys))
        self.problem = prepare_tsp(SAMPLE)

    def test_initial_probabilities_and_pomo_rollout_match_fixed(self) -> None:
        # Shared Encoder/Decoder weights and identical initial +2/+1 must give the same routes.
        features = build_city_features(
            self.problem.cities, self.problem.granular_balls, self.problem.city_to_ball
        )
        _, encoded = self.fixed.encode(features)
        city_count = len(self.problem.cities)
        mask = torch.zeros((1, 1, city_count), dtype=torch.bool)
        candidates, fixed_bias = candidate_guidance(
            0, {0}, self.problem.city_to_ball, self.problem.granular_balls,
            self.problem.ball_adjacency, city_count, self.fixed,
        )
        mask[0, 0, candidates] = True
        same, adjacent = torch.zeros_like(mask), torch.zeros_like(mask)
        current_ball = self.problem.city_to_ball[0]
        for city in candidates:
            ball = self.problem.city_to_ball[city]
            same[0, 0, city] = ball == current_ball
            adjacent[0, 0, city] = ball != current_ball and ball in self.problem.ball_adjacency[current_ball]
        start = torch.tensor([[0]])
        with torch.no_grad():
            original_probs = self.fixed(encoded, start, start, mask, torch.tensor(fixed_bias).reshape(1, 1, -1))
            learned_probs = self.learnable(encoded, start, start, mask, None, same, adjacent)
            fixed = rollout(self.fixed, self.problem, sample_actions=False)
            learned = rollout(self.learnable, self.problem, sample_actions=False)
        self.assertTrue(torch.allclose(original_probs, learned_probs, atol=1e-7))
        self.assertEqual(fixed.routes, learned.routes)
        self.assertTrue(torch.equal(fixed.tour_lengths, learned.tour_lengths))
        self.assertTrue(torch.allclose(fixed.route_log_probs, learned.route_log_probs, atol=1e-6))
        self.assertIsNotNone(learned.bias_values)
        self.assertTrue(torch.equal(learned.bias_values[..., 0], torch.full_like(learned.bias_values[..., 0], 2)))
        self.assertTrue(torch.equal(learned.bias_values[..., 1], torch.full_like(learned.bias_values[..., 1], 1)))

    def test_query_variation_mask_and_gradient(self) -> None:
        # Nonzero head weights expose different existing route queries; the valid-city mask remains exact.
        with torch.no_grad():
            self.learnable.decoder.bias_head[-1].weight.fill_(0.03)
        result = rollout(self.learnable, self.problem, sample_actions=True)
        values = result.bias_values
        self.assertGreater(float(values[..., 0].std()), 0)
        self.assertGreater(float(values[..., 1].std()), 0)
        self.assertTrue(((values[..., 0] > 0) & (values[..., 0] < 4)).all())
        self.assertTrue(((values[..., 1] > 0) & (values[..., 1] < 2)).all())
        self.assertEqual(len(result.routes), len(self.problem.cities))
        self.assertTrue(all(len(set(route)) == len(self.problem.cities) for route in result.routes))
        loss = -(result.advantages.detach() * result.route_log_probs).mean()
        loss.backward()
        self.assertIsNotNone(self.learnable.decoder.bias_head[-1].weight.grad)
        self.assertGreater(float(self.learnable.decoder.bias_head[-1].weight.grad.abs().sum()), 0)
        before = self.learnable.decoder.bias_head[-1].weight.detach().clone()
        optimizer = torch.optim.Adam(self.learnable.parameters(), lr=1e-4)
        optimizer.step()
        self.assertFalse(torch.equal(before, self.learnable.decoder.bias_head[-1].weight))

    def test_mask_keeps_visited_city_at_zero_probability(self) -> None:
        features = build_city_features(
            self.problem.cities, self.problem.granular_balls, self.problem.city_to_ball
        )
        _, encoded = self.learnable.encode(features)
        first = torch.tensor([[0]])
        mask = torch.ones((1, 1, len(self.problem.cities)), dtype=torch.bool)
        mask[0, 0, 0] = False
        same = torch.zeros_like(mask)
        adjacent = torch.zeros_like(mask)
        same[0, 0, 1] = True
        with torch.no_grad():
            probs = self.learnable(encoded, first, first, mask, None, same, adjacent)
        self.assertEqual(probs[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
