"""Small deterministic correctness checks for full 2-opt POMO feedback."""

from __future__ import annotations

import random
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from ..attention import POMOPolicy
from ..pomo import canonical_cycle
from ..runner import prepare_tsp
from .batch_rollout import rollout_batch
from .data import DatasetInstance
from .local_search_2opt import two_opt
from .opt_guided_feedback import (
    auxiliary_losses, coverage_marginals, heuristic_reference,
    select_indices, selected_feedback,
)
from .train_loop import _train_epoch


PROJECT = Path(__file__).resolve().parents[2]
SAMPLE = PROJECT.parent / "GB-MACO_PPSN2026-main/PPSN_GB-MACO/benchmark_MSTSP/simple1_9.tsp"


class FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.instance = DatasetInstance(SAMPLE, 9, "simple")
        cls.problem = prepare_tsp(SAMPLE)

    def test_full_two_opt_and_canonical(self) -> None:
        route = list(range(9))
        refined = two_opt(route, self.problem.distance_matrix)
        self.assertLessEqual(refined.length, sum(self.problem.distance_matrix[route[i]]
                             [route[(i + 1) % 9]] for i in range(9)))
        self.assertEqual(refined, two_opt(route, self.problem.distance_matrix))
        self.assertEqual(canonical_cycle(route), canonical_cycle(route[3:] + route[:3]))
        self.assertEqual(canonical_cycle(route), canonical_cycle(list(reversed(route))))

    def test_coverage_marginals(self) -> None:
        a, b = tuple(range(4)), (0, 2, 1, 3)
        rewards, passed = coverage_marginals((a, a, b), (100, 100, 101), 100, 0)
        self.assertEqual(rewards, (0, 0, 0))
        self.assertEqual(passed, 2)
        rewards, passed = coverage_marginals((a, b), (100, 100), 100, 0)
        self.assertEqual(rewards, (1, 1))
        self.assertEqual(passed, 2)

    def test_sampling_graph_and_a_update(self) -> None:
        torch.manual_seed(42)
        first = POMOPolicy(candidate_mode="soft", bias_mode="fixed")
        second = POMOPolicy(candidate_mode="soft", bias_mode="fixed")
        second.load_state_dict(first.state_dict(), strict=True)
        opt_a = torch.optim.Adam(first.parameters(), lr=1e-5)
        opt_b = torch.optim.Adam(second.parameters(), lr=1e-5)
        torch.manual_seed(123)
        outcome_a = rollout_batch(first, [self.instance], torch.device("cpu"), sample_actions=True)
        indices = select_indices([self.instance], random.Random(42))  # No torch RNG use.
        torch.manual_seed(123)
        outcome_b = rollout_batch(second, [self.instance], torch.device("cpu"), sample_actions=True)
        self.assertEqual(outcome_a.routes, outcome_b.routes)
        self.assertTrue(torch.equal(outcome_a.route_log_probs, outcome_b.route_log_probs))
        reference = heuristic_reference(self.problem.distance_matrix)
        before = outcome_b.route_log_probs.detach().clone()
        feedback = selected_feedback(outcome_b, [self.instance], indices,
                                     {self.instance.path: reference}, 0.005)
        for position, start in enumerate(indices[0]):
            direct = two_opt(outcome_b.routes[0][start], self.problem.distance_matrix)
            self.assertEqual(feedback[0].refined_lengths[position], direct.length)
            self.assertEqual(feedback[0].refined_cycles[position], canonical_cycle(direct.route))
        self.assertTrue(torch.equal(before, outcome_b.route_log_probs.detach()))
        self.assertEqual(outcome_b.routes, outcome_a.routes)
        varied = [replace(feedback[0], quality_rewards=tuple(
            -1.0 - 0.01 * i for i in range(8)))]
        quality, coverage = auxiliary_losses(outcome_b.route_log_probs, varied)
        self.assertTrue(torch.isfinite(quality) and torch.isfinite(coverage))
        self.assertTrue(any(g.abs().sum() > 0 for g in torch.autograd.grad(
            quality, tuple(second.parameters()), retain_graph=True)))
        injected = [replace(feedback[0], coverage_rewards=(1,) + (0,) * 7)]
        _, coverage_probe = auxiliary_losses(outcome_b.route_log_probs, injected)
        self.assertTrue(any(g.abs().sum() > 0 for g in torch.autograd.grad(
            coverage_probe, tuple(second.parameters()), retain_graph=True)))
        for outcome, policy, optimizer in ((outcome_a, first, opt_a),
                                           (outcome_b, second, opt_b)):
            advantages = -outcome.lengths - (-outcome.lengths).mean(dim=1, keepdim=True)
            loss = -(advantages.detach() * outcome.route_log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in policy.parameters()))
            optimizer.step()
        for left, right in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_auxiliary_gradients_and_identical_terminal(self) -> None:
        torch.manual_seed(7)
        policy = POMOPolicy(candidate_mode="soft", bias_mode="fixed")
        outcome = rollout_batch(policy, [self.instance], torch.device("cpu"), sample_actions=True)
        indices = select_indices([self.instance], random.Random(7))
        reference = heuristic_reference(self.problem.distance_matrix)
        feedback = selected_feedback(outcome, [self.instance], indices,
                                     {self.instance.path: reference}, 0.005)
        varied = [replace(feedback[0], quality_rewards=tuple(
            -1.0 - 0.01 * i for i in range(8)))]
        quality, coverage = auxiliary_losses(outcome.route_log_probs, varied)
        grads = torch.autograd.grad(quality, tuple(policy.parameters()), retain_graph=True)
        self.assertTrue(any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))
        self.assertTrue(torch.isfinite(coverage))
        cycles = (tuple(range(9)),) * 8
        marginals, _ = coverage_marginals(cycles, (reference,) * 8, reference, 0.005)
        self.assertEqual(marginals, (0,) * 8)

    def test_a_update_matches_original_train_loop(self) -> None:
        torch.manual_seed(91)
        original = POMOPolicy(candidate_mode="soft", bias_mode="fixed")
        guided_a = POMOPolicy(candidate_mode="soft", bias_mode="fixed")
        guided_a.load_state_dict(original.state_dict(), strict=True)
        old_optimizer = torch.optim.Adam(original.parameters(), lr=1e-5)
        new_optimizer = torch.optim.Adam(guided_a.parameters(), lr=1e-5)
        torch.manual_seed(92)
        old = _train_epoch(original, old_optimizer, [self.instance], 1, 1,
                           torch.device("cpu"), random.Random(42), 1)
        torch.manual_seed(92)
        new_outcome = rollout_batch(guided_a, [self.instance], torch.device("cpu"),
                                    sample_actions=True)
        new_optimizer.zero_grad()
        advantage = -new_outcome.lengths - (-new_outcome.lengths).mean(dim=1, keepdim=True)
        new_loss = -(advantage.detach() * new_outcome.route_log_probs).mean()
        new_loss.backward()
        new_optimizer.step()
        self.assertAlmostEqual(old["mean_loss"], float(new_loss.detach()), places=5)
        for left, right in zip(original.parameters(), guided_a.parameters()):
            self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
