"""Probability identity and actual two-level selection invariants."""

import unittest

import torch

from gb_2_maco_POMO.experiments.hierarchical.hierarchical import (
    select_city, two_stage_probabilities,
)


class HierarchicalTest(unittest.TestCase):
    def test_zero_lambda_equals_unbiased_city_softmax(self):
        logits = torch.tensor([[1.0, -0.2, 2.4, 0.7, -0.6],
                               [0.3, 1.2, -1.0, 0.8, 2.0]])
        available = torch.tensor([[True, False, True, True, True],
                                  [False, True, True, False, True]])
        mapping = torch.tensor([0, 1, 0, 2, 1])
        current = torch.tensor([1, 0])
        distance = torch.tensor([[0, 3, 5, 7, 8], [3, 0, 4, 2, 6],
                                 [5, 4, 0, 1, 2], [7, 2, 1, 0, 3],
                                 [8, 6, 2, 3, 0]], dtype=torch.float32)
        balls, conditional, city = two_stage_probabilities(
            logits, available, mapping, current, distance, 3.0, 0.0,
        )
        expected = torch.softmax(logits.masked_fill(~available, -torch.inf), dim=1)
        torch.testing.assert_close(city, expected, rtol=1e-6, atol=1e-7)
        self.assertTrue(torch.all(city[~available] == 0))
        chosen = select_city(balls, conditional, mapping, False)
        self.assertTrue(all(available[row, value] for row, value in enumerate(chosen)))

    def test_penalty_changes_ball_preference(self):
        logits = torch.zeros((1, 2))
        balls, _, city = two_stage_probabilities(
            logits, torch.ones((1, 2), dtype=torch.bool), torch.tensor([0, 1]),
            torch.tensor([0]), torch.tensor([[0.0, 10.0], [10.0, 0.0]]),
            2.0, 1.0,
        )
        self.assertGreater(balls[0, 0], balls[0, 1])
        torch.testing.assert_close(city.sum(dim=1), torch.ones(1))


if __name__ == "__main__":
    unittest.main()
