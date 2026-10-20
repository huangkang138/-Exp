"""Regression: a one-ball batch has no learnable ball choice."""

import unittest

import torch

from .train import _check_step_gradients


class GradientGateTest(unittest.TestCase):
    def test_zero_ball_gradient_is_valid_but_recorded(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ball_head = torch.nn.Linear(1, 1, bias=False)
                self.decoder = torch.nn.Linear(1, 1, bias=False)

        model = TinyModel()
        model.ball_head.weight.grad = torch.zeros_like(model.ball_head.weight)
        model.decoder.weight.grad = torch.ones_like(model.decoder.weight)
        self.assertEqual(_check_step_gradients(model, "A", 10), (0.0, 1.0))

    def test_nonfinite_gradient_still_fails(self):
        model = torch.nn.Module()
        model.ball_head = torch.nn.Linear(1, 1, bias=False)
        model.decoder = torch.nn.Linear(1, 1, bias=False)
        model.ball_head.weight.grad = torch.full_like(model.ball_head.weight, float("nan"))
        with self.assertRaisesRegex(RuntimeError, "nonfinite gradient"):
            _check_step_gradients(model, "A", 10)


if __name__ == "__main__":
    unittest.main()
