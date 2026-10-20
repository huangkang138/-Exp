"""Focused reward tests; no model training or formal benchmark evaluation."""

from __future__ import annotations

import unittest
import inspect
from pathlib import Path

import torch

from gb_2_maco_POMO.experiments.learnable_ball_bias.model import BallBiasPolicy
from gb_2_maco_POMO.experiments.learnable_ball_bias.rollout import rollout_batch
from gb_2_maco_POMO.experiments.shared.data import DatasetInstance
from gb_2_maco_POMO.experiments.shared.data import load_manifest

from .reward import compute_advantages, coverage_contributions, coverage_loss, standardized_quality
from .references import build_references
from .validation import observed_components
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical
DATA_ROOT = Path(inspect.getfile(BallBiasPolicy)).resolve().parents[3] / "benachmark_Train_MATSP_Multiopt_500"


class RewardTests(unittest.TestCase):
    def test_quality_is_centered_and_constant_group_is_zero(self):
        scores = standardized_quality([100, 110, 120])
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])
        self.assertAlmostEqual(sum(scores), 0)
        self.assertEqual(standardized_quality([100, 100, 100]), [0, 0, 0])

    def test_region_and_solution_overlap_credit(self):
        sets = {0: {(0, 1), (0, 2)}, 1: {(0, 1)}, 2: {(0, 3)}}
        scores, regions = coverage_contributions(sets, {0: 10, 1: 10, 2: 10})
        self.assertEqual(regions, 2)
        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[2], 0)
        same, regions = coverage_contributions({0: {(0, 1)}, 1: {(0, 1)}}, {0: 10, 1: 10})
        self.assertEqual(regions, 1)
        self.assertEqual(same[0], same[1])
        self.assertAlmostEqual(same[0], 0.5)
        distinct, regions = coverage_contributions({0: {(0, 1)}, 1: {(0, 1)}}, {0: 10, 1: 11})
        self.assertEqual(regions, 2)
        self.assertEqual(distinct[0], distinct[1])

    def test_known_equal_swap_region_and_canonical_dedup(self):
        distance = [[0 if i == j else 1 for j in range(5)] for i in range(5)]
        a = canonical([0, 1, 2, 3, 4])
        b = canonical([0, 1, 3, 2, 4])
        self.assertEqual(observed_components([a, b], distance), 1)
        self.assertEqual(canonical([1, 2, 3, 4, 0]), a)
        self.assertEqual(canonical([0, 4, 3, 2, 1]), a)

    def test_joint_gradient_and_training_only_reference(self):
        torch.set_num_threads(1)
        path = next((DATA_ROOT / "train/n050").rglob("*.tsp"))
        item = DatasetInstance(path, 50, "test")
        torch.manual_seed(7)
        model = BallBiasPolicy(use_bias=True)
        outcome = rollout_batch(model, [item], torch.device("cpu"), sample=True)
        references = {str(path): {"length": 0, "certified": False, "source": "test"}}
        advantage, diag, _ = compute_advantages(
            outcome, [item], references, "G3", {"two_opt": {}, "opt_b": {}})
        self.assertLessEqual(diag.selected, 16)
        self.assertEqual(diag.eligible, 0)
        self.assertTrue(torch.isfinite(advantage).all())
        loss = coverage_loss(outcome, advantage)
        loss.backward()
        for parameters in (model.encoder.parameters(), model.ball_head.parameters(),
                           model.decoder.parameters(), model.bias_head.parameters()):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters))

    def test_reference_split_is_disjoint_and_training_only(self):
        train, validation = load_manifest({"root": str(DATA_ROOT), "train_dir": "train",
                                           "validation_dir": "validation"})
        references = build_references(DATA_ROOT, train, validation)
        self.assertEqual(len(references), 500)
        self.assertEqual(sum(r["certified"] for r in references.values()), 282)
        self.assertEqual(set(references), {str(item.path) for item in train})
        self.assertTrue(set(references).isdisjoint(str(item.path) for item in validation))


if __name__ == "__main__":
    unittest.main()
