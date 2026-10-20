"""Replace only the mean ball summary with a zero-initialized attention residual."""

from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn

from gb_2_maco_POMO.experiments.learnable_ball.learnable_policy import LearnableBallPolicy


class DynamicBallPolicy(LearnableBallPolicy):
    """Keep the old encoder, decoder, ball scorer and action order unchanged.

    Existing decoder.query_projection supplies the route-state Query. Encoded,
    unvisited cities in each ball are both Keys and Values. The only new
    parameter is alpha; zero gives the old mean exactly at epoch 0.
    """

    def __init__(self, embedding_dim=128, head_num=8, encoder_layer_num=3):
        super().__init__(embedding_dim, head_num, encoder_layer_num)
        self.ball_attention_alpha = nn.Parameter(torch.zeros(()))

    def ball_summary(self, encoded: Tensor, first: Tensor, current: Tensor,
                     available: Tensor, city_to_ball: Tensor):
        batch, n, dim = encoded.shape
        if (first.shape != current.shape or available.shape != (batch, first.shape[1], n)
                or city_to_ball.shape != (batch, n) or available.dtype != torch.bool):
            raise ValueError("invalid route state or city-to-ball mapping")
        group_count = int(city_to_ball.max().item()) + 1
        membership = city_to_ball[:, None, :] == torch.arange(
            group_count, device=encoded.device)[None, :, None]
        eligible = available[:, :, None, :] & membership[:, None, :, :]
        counts = eligible.sum(dim=-1)
        ball_mean = torch.einsum("bpgn,bnd->bpgd", eligible.to(encoded.dtype), encoded)
        ball_mean = ball_mean / counts.clamp_min(1)[..., None]
        first_vector = encoded.gather(1, first[..., None].expand(-1, -1, dim))
        current_vector = encoded.gather(1, current[..., None].expand(-1, -1, dim))
        query = self.decoder.query_projection(
            torch.cat((first_vector, current_vector), dim=-1))
        scores = torch.einsum("bpd,bnd->bpn", query, encoded) / sqrt(dim)
        scores = scores[:, :, None, :].masked_fill(~eligible, torch.finfo(encoded.dtype).min)
        weights = torch.softmax(scores, dim=-1) * eligible.to(encoded.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attended = torch.einsum("bpgn,bnd->bpgd", weights, encoded)
        ball_embedding = ball_mean + self.ball_attention_alpha * (attended - ball_mean)
        return ball_embedding, query, counts, weights

    def action_logits(self, encoded: Tensor, first: Tensor, current: Tensor,
                      available: Tensor, city_to_ball: Tensor):
        if not available.any(dim=-1).all().item():
            raise ValueError("each route needs an unvisited city")
        city_logits = self(encoded, first, current, available, return_logits=True)
        ball_embedding, query, counts, _ = self.ball_summary(
            encoded, first, current, available, city_to_ball)
        fraction = (counts / encoded.shape[1]).to(encoded.dtype)[..., None]
        context = torch.cat((query[:, :, None, :].expand_as(ball_embedding),
                             ball_embedding, fraction), dim=-1)
        ball_logits = self.ball_head(context).squeeze(-1)
        return ball_logits.masked_fill(counts == 0, -torch.inf), city_logits
