"""Functional proof that fixed Attention stays frozen while the Bias Head learns."""

from __future__ import annotations

import unittest
import random
import tempfile
from pathlib import Path

import torch

from .batch_rollout import rollout_batch
from .compare import BENCHMARKS
from .compare_learnable_bias import BENCHMARK_DIR
from .data import DatasetInstance
from .finetune_bias_head import (
    DEFAULT_CONFIG, _assert_frozen_unchanged, _load_frozen_policy,
    _source_path, _verify_initial_equivalence, HISTORY_FIELDS,
)
from .config import load_config
from .train_loop import _train_epoch, _write_csv


class FineTuneBiasHeadTests(unittest.TestCase):
    def test_checkpoint_transfer_and_one_policy_gradient_update(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = load_config(DEFAULT_CONFIG, {})
        fixed, learned, original, _ = _load_frozen_policy(_source_path(config, DEFAULT_CONFIG))
        _verify_initial_equivalence(fixed, learned)
        stem, count, _ = BENCHMARKS[14]
        example = DatasetInstance(BENCHMARK_DIR / f"{stem}.tsp", count, "benchmark")
        optimizer = torch.optim.Adam(learned.decoder.bias_head.parameters(), lr=1e-4)
        before = {key: value.clone() for key, value in learned.decoder.bias_head.state_dict().items()}
        outcome = rollout_batch(learned, [example], torch.device("cpu"), sample_actions=True)
        reward = -outcome.lengths
        advantage = reward - reward.mean(dim=1, keepdim=True)
        loss = -(advantage.detach() * outcome.route_log_probs).mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        norm = sum(float(p.grad.square().sum()) for p in learned.decoder.bias_head.parameters()
                   if p.grad is not None) ** 0.5
        self.assertGreater(norm, 0)
        self.assertTrue(all(p.grad is None for name, p in learned.named_parameters()
                            if not name.startswith("decoder.bias_head.")))
        optimizer.step()
        _assert_frozen_unchanged(learned, original)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in
                            learned.decoder.bias_head.state_dict().items()))
        changed = rollout_batch(learned, [example], torch.device("cpu"), sample_actions=False)
        self.assertGreater(changed.bias_stats["same_square_mean"] -
                           changed.bias_stats["same_mean"] ** 2, 0)
        stats = _train_epoch(learned, optimizer, [example], 1, 1,
                             torch.device("cpu"), random.Random(42), 1)
        self.assertGreater(stats["bias_head_grad_norm_mean"], 0)
        self.assertGreater(stats["same_bias_std"], 0)
        self.assertTrue(torch.isfinite(torch.tensor(stats["mean_loss"])))
        _assert_frozen_unchanged(learned, original)
        with tempfile.TemporaryDirectory() as directory:
            _write_csv(Path(directory) / "history.csv", HISTORY_FIELDS,
                       [{"epoch": 1, **stats, "validation_mean_best_length": None,
                         "attention_unchanged": True}])


if __name__ == "__main__":
    unittest.main()
