"""候选机制和采样搜索的边界测试，不读取论文最终算例。"""

import unittest

import torch

from ..attention import POMOPolicy, candidate_guidance
from ..pomo import canonical_cycle, search_routes
from ..runner import prepare_tsp
from ..tsp import get_candidate_cities
from .config import DEFAULT_CONFIG, load_config
from .data import load_manifest


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        config = load_config(DEFAULT_CONFIG, {"runtime.device": "cpu"})
        train, _ = load_manifest(config["data"])
        cls.problem = prepare_tsp(next(item.path for item in train if item.city_count == 10))

    def test_soft_and_none_keep_all_unvisited_cities(self):
        mapping = [0, 0, 1, 2]
        adjacency = {0: {1}, 1: {0}, 2: set()}
        for mode, expected in (("soft", [0.0, 2.0, 1.0, 0.0]),
                               ("none", [0.0] * 4)):
            policy = POMOPolicy(candidate_mode=mode)
            candidates, bias = candidate_guidance(0, {0}, mapping, [], adjacency, 4, policy)
            self.assertEqual(candidates, [1, 2, 3])
            self.assertEqual(bias, expected)

    def test_hard_keeps_original_candidates(self):
        problem = self.problem
        seen = {0}
        policy = POMOPolicy(candidate_mode="hard")
        candidates, bias = candidate_guidance(
            0, seen, problem.city_to_ball, problem.granular_balls,
            problem.ball_adjacency, len(problem.cities), policy,
        )
        self.assertEqual(candidates, get_candidate_cities(
            0, seen, problem.city_to_ball, problem.granular_balls,
            problem.ball_adjacency, len(problem.cities),
        ))
        self.assertEqual(bias, [0.0] * len(problem.cities))

    def test_decoder_bias_changes_probabilities_and_respects_mask(self):
        torch.manual_seed(3)
        policy = POMOPolicy(candidate_mode="soft")
        nodes = torch.randn(1, 3, 128)
        first = current = torch.tensor([[0]])
        mask = torch.tensor([[[False, True, True]]])
        plain = policy(nodes, first, current, mask)
        guided = policy(nodes, first, current, mask, torch.tensor([[[0.0, 2.0, 0.0]]]))
        self.assertEqual(guided[0, 0, 0].item(), 0)
        self.assertGreater(guided[0, 0, 1].item(), plain[0, 0, 1].item())

    def test_cycle_equivalence(self):
        key = canonical_cycle([0, 1, 2, 3])
        self.assertEqual(key, canonical_cycle([2, 3, 0, 1]))
        self.assertEqual(key, canonical_cycle([0, 3, 2, 1]))
        self.assertNotEqual(key, canonical_cycle([0, 2, 1, 3]))

    def test_mixed_contains_greedy_and_same_seed_samples(self):
        policy = POMOPolicy(candidate_mode="soft")
        greedy = search_routes(policy, self.problem, inference_mode="argmax", seed=7)
        sampled = search_routes(policy, self.problem, inference_mode="sampling",
                                samples_per_start=2, seed=7)
        mixed = search_routes(policy, self.problem, inference_mode="mixed",
                              samples_per_start=2, seed=7)
        self.assertEqual(mixed.routes, greedy.routes + sampled.routes)
        self.assertEqual(mixed.generated_count, 30)
        self.assertEqual(mixed.unique_count, len({canonical_cycle(r) for r in mixed.routes}))
        self.assertEqual(mixed.best_length, min(greedy.best_length, sampled.best_length))

    def test_sampling_is_legal_reproducible_and_budgeted(self):
        torch.set_num_threads(1)
        policy = POMOPolicy(candidate_mode="none")
        first = search_routes(policy, self.problem, inference_mode="sampling",
                              samples_per_start=2, seed=7)
        second = search_routes(policy, self.problem, inference_mode="sampling",
                               samples_per_start=2, seed=7)
        self.assertEqual(first.routes, second.routes)
        self.assertEqual(first.generated_count, 20)
        self.assertEqual(first.unique_count, len({canonical_cycle(r) for r in first.routes}))
        self.assertTrue(all(set(route) == set(range(10)) for route in first.routes))
        self.assertEqual(first.best_length, min(first.lengths))
        self.assertGreaterEqual(first.high_quality_count, 1)
        self.assertEqual(first.high_quality_rate,
                         first.high_quality_count / first.unique_count)


if __name__ == "__main__":
    unittest.main()
