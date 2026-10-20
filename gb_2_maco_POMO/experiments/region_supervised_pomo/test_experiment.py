"""Focused checks for data isolation, project region rules, and supervision."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import load_checkpoint
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical

from .config import INITIAL
from .data import load_experiment_data
from .supervision import draw_route, teacher_forcing_loss
from .train import _save_checkpoint, grad_cosine, pomo_loss
from .validation import evaluate


class SupervisionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.partitions, cls.labels, cls.audit = load_experiment_data()

    def test_merged_population_and_project_regions(self):
        self.assertEqual((len(self.partitions["train"]), len(self.partitions["validation"])),
                         (580, 120))
        self.assertEqual((len(self.labels["train"]), len(self.labels["validation"])),
                         (362, 76))
        self.assertEqual(sum(row["same_partition"] for row in self.audit), 438)

    def test_region_sampling_and_route_equivalence(self):
        case = next(c for c in self.labels["train"].values()
                    if c.complete and len({len(group) for group in c.regions}) > 1)
        rng_b, rng_c = random.Random(42), random.Random(42)
        for mode, rng in (("B", rng_b), ("C", rng_c)):
            for _ in range(50):
                route, region, index = draw_route(case, mode, rng)
                self.assertIn(index, case.regions[region])
                self.assertEqual(canonical(list(route)), case.tours[index])
        biggest = max(range(len(case.regions)), key=lambda i: len(case.regions[i]))
        b_probability = len(case.regions[biggest]) / len(case.tours)
        c_probability = 1 / len(case.regions)
        self.assertNotEqual(b_probability, c_probability)

    def test_teacher_forcing_reaches_both_heads(self):
        case = next(c for c in self.labels["train"].values()
                    if len(c.tours[0]) >= 50)
        model, _ = load_checkpoint(INITIAL)
        route, _, _ = draw_route(case, "C", random.Random(42))
        loss = teacher_forcing_loss(model, [(case, route)], torch.device("cpu"))
        loss.backward()
        ball = sum(float(p.grad.abs().sum()) for p in model.ball_head.parameters()
                   if p.grad is not None)
        city = sum(float(p.grad.abs().sum()) for p in model.decoder.parameters()
                   if p.grad is not None)
        self.assertGreater(ball, 0)
        self.assertGreater(city, 0)

    def test_pomo_loss_uses_route_sum_and_instance_baseline(self):
        log_probs = torch.tensor([[-2.0, -3.0], [-4.0, -5.0]], requires_grad=True)
        outcome = SimpleNamespace(lengths=torch.tensor([[10.0, 12.0], [20.0, 24.0]]),
                                  joint_log_probs=log_probs)
        loss = pomo_loss(outcome)
        self.assertAlmostEqual(float(loss.detach()), -0.75)
        loss.backward()
        self.assertTrue(torch.allclose(log_probs.grad,
                                       torch.tensor([[-0.25, 0.25], [-0.5, 0.5]])))

    def test_gradient_conflict_measure(self):
        self.assertAlmostEqual(grad_cosine([torch.tensor([1., 0.])],
                                           [torch.tensor([-1., 0.])]), -1.0)
        self.assertIsNone(grad_cosine([torch.zeros(2)], [torch.ones(2)]))

    def test_one_validation_case_uses_existing_search(self):
        case = next(c for c in self.labels["validation"].values() if c.complete)
        model, _ = load_checkpoint(INITIAL)
        with patch("gb_2_maco_POMO.experiments.region_supervised_pomo.validation.summarize",
                   side_effect=lambda rows: {"case_count": len(rows)}):
            rows, metrics = evaluate(model, [case.instance], self.labels["validation"], 42)
        self.assertEqual(metrics["case_count"], 1)
        self.assertEqual(rows[0]["reference_kind"], "certified")
        self.assertIsNotNone(rows[0]["full_fbeta"])
        self.assertLessEqual(rows[0]["optb_checks"], 50_000)

    def test_checkpoint_is_resume_readable(self):
        model, _ = load_checkpoint(INITIAL)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch.pt"
            _save_checkpoint(path, "B", 42, 1, {"dataset": "test"}, model,
                             optimizer, random.Random(42), random.Random(100042),
                             [], [], {}, {}, torch.device("cpu"))
            saved = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual((saved["group"], saved["seed"], saved["epoch"]), ("B", 42, 1))
        self.assertEqual(saved["architecture"]["embedding_dim"], 128)


if __name__ == "__main__":
    unittest.main()
