"""Numerical equivalence and trainable-gradient checks for the new policy."""

import unittest

import torch

from gb_2_maco_POMO.attention import POMOPolicy, candidate_guidance
from gb_maco_POMO.attention import POMOPolicy as OriginalSoftPolicy
from gb_maco_POMO.attention import build_city_features as original_city_features
from gb_2_maco_POMO.runner import prepare_tsp
from gb_2_maco_POMO.experiments.shared.data import DatasetInstance
from gb_2_maco_POMO.experiments.learnable_ball_bias.rollout import geometry_batch, rollout_batch
from .model import FIXED, load_fixed
from .validation import select_checkpoints


class ResidualPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model, config, _ = load_fixed()
        cls.model.eval()
        state = torch.load(FIXED, map_location='cpu', weights_only=True)
        cls.base = POMOPolicy(int(config['embedding_dim']), int(config['num_heads']),
                              int(config['encoder_layers']), candidate_mode='soft',
                              same_ball_bias=2.0, adjacent_ball_bias=1.0)
        cls.base.load_state_dict(state['model_state_dict'], strict=True)
        cls.base.eval()
        cls.original = OriginalSoftPolicy(int(config['embedding_dim']), int(config['num_heads']),
                                          int(config['encoder_layers']), candidate_mode='soft',
                                          same_ball_bias=2.0, adjacent_ball_bias=1.0)
        cls.original.load_state_dict(state['model_state_dict'], strict=True)
        cls.original.eval()
        from pathlib import Path
        data = FIXED.parents[4] / 'benachmark_Train_MATSP_500/train/n050'
        cls.path = next(p for p in data.rglob('*.tsp') if len(prepare_tsp(p).granular_balls) > 1)
        cls.problem = prepare_tsp(cls.path)

    def _base_logits(self, encoded, first, current, available):
        n = len(self.problem.cities)
        biases = []
        for i in range(first.shape[1]):
            visited = set(torch.where(~available[0, i])[0].tolist())
            cities, bias = candidate_guidance(
                int(current[0, i]), visited, self.problem.city_to_ball,
                self.problem.granular_balls, self.problem.ball_adjacency, n, self.base)
            self.assertEqual(cities, torch.where(available[0, i])[0].tolist())
            biases.append(bias)
        return self.base(encoded, first, current, available, torch.tensor(biases)[None],
                         return_logits=True)

    def test_zero_residual_distribution_and_complete_greedy_tours(self):
        features, mapping, geometry = geometry_batch([self.problem], torch.device('cpu'))
        original_features = original_city_features(self.problem.cities,
            self.problem.granular_balls, self.problem.city_to_ball)
        self.assertTrue(torch.equal(features, original_features))
        _, encoded = self.model.encode(features)
        _, base_encoded = self.base.encode(features)
        _, original_encoded = self.original.encode(original_features)
        self.assertTrue(torch.equal(encoded, base_encoded))
        self.assertTrue(torch.equal(encoded, original_encoded))
        n = len(self.problem.cities)
        first = torch.arange(n)[None]
        current = first.clone()
        available = torch.ones((1, n, n), dtype=torch.bool)
        available.scatter_(2, first[..., None], False)
        for step in range(1, n):
            base_logits = self._base_logits(base_encoded, first, current, available)
            ball_logits, city_logits = self.model.action_logits(
                encoded, first, current, available, mapping, geometry)
            self.assertTrue(torch.allclose(city_logits, base_logits, atol=1e-6))
            joint = self.model.joint_city_log_probs(ball_logits, city_logits, mapping)
            self.assertTrue(torch.isfinite(joint[available]).all())
            self.assertTrue(torch.allclose(joint.exp(), base_logits.softmax(-1), atol=2e-6))
            bias = []
            for i in range(n):
                visited = set(torch.where(~available[0, i])[0].tolist())
                _, values = candidate_guidance(int(current[0, i]), visited,
                    self.problem.city_to_ball, self.problem.granular_balls,
                    self.problem.ball_adjacency, n, self.base)
                bias.append(values)
            original_probs = self.original(original_encoded, first, current, available,
                                           torch.tensor(bias)[None])
            self.assertTrue(torch.allclose(joint.exp(), original_probs, atol=2e-6))
            city, _, ball = self.model.choose_step(encoded, first, current, available,
                                                  mapping, sample=False, geometry=geometry)
            self.assertTrue(torch.equal(city, base_logits.argmax(-1)))
            self.assertTrue(torch.equal(ball, mapping.gather(1, city)))
            current = city
            available.scatter_(2, city[..., None], False)
        self.assertFalse(available.any())

    def test_only_residual_head_receives_gradient(self):
        model, _, _ = load_fixed()
        for p in (*model.encoder.parameters(), *model.decoder.parameters()):
            p.requires_grad_(False)
        fixed_before = {k: v.detach().clone() for k, v in model.state_dict().items()
                        if not k.startswith('ball_head.')}
        head_before = [p.detach().clone() for p in model.ball_head.parameters()]
        item = DatasetInstance(self.path, len(self.problem.cities), 'test')
        torch.manual_seed(42)
        outcome = rollout_batch(model, [item], torch.device('cpu'), sample=True)
        reward = -outcome.lengths.detach()
        loss = -((reward - reward.mean(1, keepdim=True)) * outcome.joint_log_probs).mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.ball_head.parameters()))
        self.assertTrue(all(p.grad is None for p in (*model.encoder.parameters(),
                                                     *model.decoder.parameters())))
        torch.optim.Adam(model.ball_head.parameters(), lr=1e-4).step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in
                            zip(head_before, model.ball_head.parameters())))
        self.assertTrue(all(torch.equal(old, model.state_dict()[key])
                            for key, old in fixed_before.items()))

    def test_nonzero_residual_joint_probability_is_normalized(self):
        model, _, _ = load_fixed()
        with torch.no_grad():
            model.ball_head[-1].weight[0, 0] = 0.4
            model.ball_head[-1].bias[0] = 0.2
        features, mapping, geometry = geometry_batch([self.problem], torch.device('cpu'))
        _, encoded = model.encode(features)
        n = len(self.problem.cities)
        first = torch.arange(n)[None]
        available = torch.ones((1, n, n), dtype=torch.bool)
        available.scatter_(2, first[..., None], False)
        ball_logits, city_logits = model.action_logits(encoded, first, first,
                                                        available, mapping, geometry)
        joint = model.joint_city_log_probs(ball_logits, city_logits, mapping).exp()
        self.assertTrue(torch.allclose(joint.sum(-1), torch.ones_like(first, dtype=joint.dtype), atol=1e-6))
        self.assertTrue((joint[~available] == 0).all())

    def test_checkpoint_selectors_use_validation_proxy_and_quality_gate(self):
        records = [{'epoch': e, 'metrics': {'certified_cases': 0}}
                   for e in (0, 1, 2)]
        rows = {
            0: {'x': {'two_opt_best': 100, 'raw_best': 110, 'b_routes': 2},
                'y': {'two_opt_best': 100, 'raw_best': 110, 'b_routes': 2}},
            1: {'x': {'two_opt_best': 99, 'raw_best': 108, 'b_routes': 1},
                'y': {'two_opt_best': 100, 'raw_best': 108, 'b_routes': 10}},
            2: {'x': {'two_opt_best': 99, 'raw_best': 107, 'b_routes': 1},
                'y': {'two_opt_best': 100, 'raw_best': 107, 'b_routes': 5}},
        }
        length, multi, info = select_checkpoints(records, rows, tolerance_pct=0.05)
        self.assertEqual((length['epoch'], multi['epoch']), (2, 1))
        self.assertEqual(info['distance_hits'], 2)
        self.assertIn('proxy', info['note'].lower())


if __name__ == '__main__':
    unittest.main()
