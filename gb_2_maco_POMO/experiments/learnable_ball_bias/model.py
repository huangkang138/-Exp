"""A small geometric bias on ball logits; the original city policy is unchanged."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gb_2_maco_POMO.experiments.learnable_ball.learnable_policy import LearnableBallPolicy


class BallBiasPolicy(LearnableBallPolicy):
    def __init__(self, embedding_dim=128, head_num=8, encoder_layer_num=3,
                 *, use_bias=False, bias_bound=2.0):
        super().__init__(embedding_dim, head_num, encoder_layer_num)
        self.use_bias = use_bias
        self.bias_bound = float(bias_bound)
        if self.bias_bound <= 0:
            raise ValueError("bias_bound must be positive")
        # Only four geometric inputs: this branch cannot duplicate the encoded
        # route/ball context already handled by ball_head.
        if use_bias:
            self.bias_head = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
            nn.init.zeros_(self.bias_head[-1].weight)
            nn.init.zeros_(self.bias_head[-1].bias)

    @property
    def bias_parameter_count(self):
        return sum(p.numel() for p in self.bias_head.parameters()) if self.use_bias else 0

    def action_logits(self, encoded: Tensor, first: Tensor, current: Tensor,
                      available: Tensor, city_to_ball: Tensor,
                      geometry: tuple[Tensor, Tensor, Tensor] | None = None):
        ball_logits, city_logits = super().action_logits(
            encoded, first, current, available, city_to_ball)
        if not self.use_bias:
            return ball_logits, city_logits
        if geometry is None:
            raise ValueError("enabled ball bias requires city positions, centers and adjacency")
        xy, centers, adjacency = geometry  # [B,N,2], [B,G,2], [B,G,G]
        b, p, n = available.shape
        g = centers.shape[1]
        if xy.shape != (b, n, 2) or adjacency.shape != (b, g, g):
            raise ValueError("ball bias geometry shape mismatch")
        current_xy = xy[:, None, :, :].expand(-1, p, -1, -1).gather(
            2, current[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        current_ball = city_to_ball[:, None, :].expand(-1, p, -1).gather(
            2, current[..., None]).squeeze(-1)
        same = (current_ball[..., None] == torch.arange(g, device=xy.device)).to(xy.dtype)
        neighboring = adjacency[:, None, :, :].expand(-1, p, -1, -1).gather(
            2, current_ball[..., None, None].expand(-1, -1, 1, g)).squeeze(2).to(xy.dtype)
        center_distance = torch.linalg.vector_norm(
            current_xy[:, :, None, :] - centers[:, None, :, :], dim=-1)
        membership = city_to_ball[:, None, None, :] == torch.arange(
            g, device=xy.device)[None, None, :, None]
        eligible = available[:, :, None, :] & membership
        city_distance = torch.linalg.vector_norm(
            current_xy[:, :, None, :] - xy[:, None, :, :], dim=-1)
        nearest = city_distance[:, :, None, :].expand(-1, -1, g, -1).masked_fill(
            ~eligible, torch.inf).amin(dim=-1)
        nearest = torch.where(torch.isfinite(nearest), nearest, torch.zeros_like(nearest))
        features = torch.stack((same, neighboring, center_distance, nearest), dim=-1)
        bias = self.bias_bound * torch.tanh(self.bias_head(features).squeeze(-1))
        return (ball_logits + bias).masked_fill(~eligible.any(dim=-1), -torch.inf), city_logits

    def choose_step(self, encoded, first, current, available, city_to_ball,
                    *, sample: bool, geometry=None):
        ball_logits, city_logits = self.action_logits(
            encoded, first, current, available, city_to_ball, geometry)
        ball_dist = torch.distributions.Categorical(logits=ball_logits)
        ball = ball_dist.sample() if sample else ball_logits.argmax(dim=-1)
        city_mask = available & (city_to_ball[:, None, :] == ball[..., None])
        conditional = city_logits.masked_fill(~city_mask, -torch.inf)
        city_dist = torch.distributions.Categorical(logits=conditional)
        city = city_dist.sample() if sample else conditional.argmax(dim=-1)
        if not city_mask.gather(-1, city[..., None]).all():
            raise RuntimeError("illegal conditional city choice")
        return city, ball_dist.log_prob(ball) + city_dist.log_prob(city), ball
