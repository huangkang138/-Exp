"""A two-stage policy whose zero residual reproduces fixed Soft city decisions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor

from gb_2_maco_POMO.experiments.learnable_ball.learnable_policy import LearnableBallPolicy

PROJECT = Path(__file__).resolve().parents[3]
FIXED = PROJECT / 'gb_maco_POMO/results/checkpoints/run_20261013_230742_828b99/best.pt'
FIXED_SHA = 'dbdfc5cb2faef0d7cbecb7c5c7cc77db16049e5d10e5d584a26e288f1bd88410'


class SoftResidualBallPolicy(LearnableBallPolicy):
    """Reuse the Soft encoder/decoder and the existing Ball Head as a residual.

    All unvisited cities stay eligible. The fixed +2/+1 bias enters city
    logits once; logsumexp makes the ball marginal exact at residual zero.
    """

    def __init__(self, embedding_dim=128, head_num=8, encoder_layer_num=3,
                 *, residual_bound=2.0):
        super().__init__(embedding_dim, head_num, encoder_layer_num)
        if residual_bound <= 0:
            raise ValueError('residual_bound must be positive')
        self.residual_bound = float(residual_bound)
        self.same_ball_bias = 2.0
        self.adjacent_ball_bias = 1.0
        # Parent Ball Head already ends in a zero-initialized Linear layer.

    def action_logits(self, encoded: Tensor, first: Tensor, current: Tensor,
                      available: Tensor, city_to_ball: Tensor,
                      geometry: tuple[Tensor, Tensor, Tensor]):
        batch, n, dim = encoded.shape
        if available.shape != (batch, first.shape[1], n) or city_to_ball.shape != (batch, n):
            raise ValueError('route state or mapping shape mismatch')
        if not available.any(dim=-1).all():
            raise ValueError('each route needs an available city')
        _, _, adjacency = geometry
        groups = adjacency.shape[-1]
        if adjacency.shape != (batch, groups, groups):
            raise ValueError('adjacency shape mismatch')
        if ((city_to_ball < 0) | (city_to_ball >= groups)).any():
            raise ValueError('every city must have one valid ball')
        membership = city_to_ball[:, None, :] == torch.arange(
            groups, device=encoded.device)[None, :, None]
        eligible = available[:, :, None, :] & membership[:, None, :, :]
        counts = eligible.sum(-1)
        current_ball = city_to_ball.gather(1, current)
        same = current_ball[..., None] == city_to_ball[:, None, :]
        neighbor_ball = adjacency[:, None, :, :].expand(-1, first.shape[1], -1, -1).gather(
            2, current_ball[..., None, None].expand(-1, -1, 1, groups)).squeeze(2)
        adjacent = neighbor_ball.gather(2, city_to_ball[:, None, :].expand(-1, first.shape[1], -1))
        bias = (same.to(encoded.dtype) * self.same_ball_bias
                + (adjacent & ~same).to(encoded.dtype) * self.adjacent_ball_bias)
        # The decoder clips its unbiased score, adds exactly one fixed Soft bias,
        # then masks visited cities. It returns differentiable pre-softmax logits.
        city_logits = self(encoded, first, current, available,
                           candidate_bias=bias, return_logits=True)
        base = torch.logsumexp(city_logits[:, :, None, :].masked_fill(~eligible, -torch.inf), -1)

        ball_mean = torch.einsum('bpgn,bnd->bpgd', eligible.to(encoded.dtype), encoded)
        ball_mean = ball_mean / counts.clamp_min(1)[..., None]
        start_vector = encoded.gather(1, first[..., None].expand(-1, -1, dim))
        current_vector = encoded.gather(1, current[..., None].expand(-1, -1, dim))
        query = self.decoder.query_projection(torch.cat((start_vector, current_vector), -1))
        fraction = (counts / n).to(encoded.dtype)[..., None]
        context = torch.cat((query[:, :, None, :].expand_as(ball_mean), ball_mean, fraction), -1)
        residual = self.residual_bound * torch.tanh(self.ball_head(context).squeeze(-1))
        residual = residual.masked_fill(counts == 0, 0.0)
        ball_logits = (base + residual).masked_fill(counts == 0, -torch.inf)
        return ball_logits, city_logits

    def joint_city_log_probs(self, ball_logits: Tensor, city_logits: Tensor,
                             city_to_ball: Tensor) -> Tensor:
        """P(city) = P(ball(city)) * P(city | ball(city))."""
        base = torch.logsumexp(city_logits[:, :, None, :].masked_fill(
            city_to_ball[:, None, None, :] != torch.arange(
                ball_logits.shape[-1], device=city_logits.device)[None, None, :, None],
            -torch.inf), -1)
        valid = torch.isfinite(base)
        adjustment = torch.where(valid, ball_logits, 0.0) - torch.where(valid, base, 0.0)
        adjustment_at_city = adjustment.gather(2, city_to_ball[:, None, :].expand(
            -1, city_logits.shape[1], -1))
        return city_logits + adjustment_at_city - torch.logsumexp(ball_logits, -1)[..., None]

    def choose_step(self, encoded, first, current, available, city_to_ball,
                    *, sample: bool, geometry=None):
        ball_logits, city_logits = self.action_logits(
            encoded, first, current, available, city_to_ball, geometry)
        ball_dist = torch.distributions.Categorical(logits=ball_logits)
        if sample:
            ball = ball_dist.sample()
            city_mask = available & (city_to_ball[:, None, :] == ball[..., None])
            conditional = city_logits.masked_fill(~city_mask, -torch.inf)
            city_dist = torch.distributions.Categorical(logits=conditional)
            city = city_dist.sample()
            logp = ball_dist.log_prob(ball) + city_dist.log_prob(city)
        else:
            # Separate argmax(ball), argmax(city|ball) is not Soft global argmax.
            joint = self.joint_city_log_probs(ball_logits, city_logits, city_to_ball)
            city = joint.argmax(-1)
            ball = city_to_ball.gather(1, city)
            logp = joint.gather(2, city[..., None]).squeeze(-1)
        if not available.gather(2, city[..., None]).all():
            raise RuntimeError('illegal city chosen')
        return city, logp, ball


def load_fixed(residual_bound=2.0, device='cpu'):
    digest = hashlib.sha256(FIXED.read_bytes()).hexdigest()
    if digest != FIXED_SHA:
        raise RuntimeError('fixed Soft checkpoint SHA changed')
    state = torch.load(FIXED, map_location='cpu', weights_only=True)
    config = state['config']['model']
    if (config['candidate_mode'], config['same_ball_bias'], config['adjacent_ball_bias']) != ('soft', 2.0, 1.0):
        raise RuntimeError('fixed checkpoint is not the expected Soft policy')
    model = SoftResidualBallPolicy(int(config['embedding_dim']), int(config['num_heads']),
                                   int(config['encoder_layers']), residual_bound=residual_bound)
    missing = model.load_state_dict(state['model_state_dict'], strict=False)
    expected = {f'ball_head.{layer}.{suffix}' for layer in (0, 2)
                for suffix in ('weight', 'bias')}
    if set(missing.missing_keys) != expected or missing.unexpected_keys:
        raise RuntimeError(f'fixed Soft weights incompatible: {missing}')
    if any(not torch.equal(model.state_dict()[key], value)
           for key, value in state['model_state_dict'].items()):
        raise RuntimeError('loaded city backbone differs from fixed Soft')
    if not torch.count_nonzero(model.ball_head[-1].weight) == 0 or not torch.count_nonzero(model.ball_head[-1].bias) == 0:
        raise RuntimeError('residual output is not zero initialized')
    return model.to(device), config, digest
