"""Trainable ball policy sharing the unchanged POMO city encoder/decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gb_2_maco_POMO.attention import POMOPolicy


class LearnableBallPolicy(POMOPolicy):
    """Choose a nonempty ball, then a city within it, at every route step.

    Route state is the existing Decoder query from first/current city.
    Ball state is the mean encoded embedding of its unvisited cities plus the
    unvisited fraction. The ball MLP is the only added parameter group.
    """

    def __init__(self, embedding_dim: int = 128, head_num: int = 8,
                 encoder_layer_num: int = 3) -> None:
        super().__init__(embedding_dim=embedding_dim, head_num=head_num,
                         encoder_layer_num=encoder_layer_num,
                         candidate_mode="soft", same_ball_bias=0.0,
                         adjacent_ball_bias=0.0, bias_mode="fixed")
        self.ball_head = nn.Sequential(
            nn.Linear(2 * embedding_dim + 1, embedding_dim), nn.Tanh(),
            nn.Linear(embedding_dim, 1),
        )
        # Every nonempty ball starts with equal logits; no hand-coded distance
        # or +2/+1 preference. The head learns from the joint REINFORCE loss.
        nn.init.zeros_(self.ball_head[-1].weight)
        nn.init.zeros_(self.ball_head[-1].bias)

    def action_logits(
        self, encoded: Tensor, first: Tensor, current: Tensor,
        available: Tensor, city_to_ball: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Differentiable ball [B,P,G] and unbiased city [B,P,N] logits."""
        batch, n, dim = encoded.shape
        if available.shape != (batch, first.shape[1], n) or city_to_ball.shape != (batch, n):
            raise ValueError("route state or ball mapping shape mismatch")
        if not available.any(dim=-1).all().item():
            raise ValueError("each route needs an unvisited city")
        city_logits = self(encoded, first, current, available, return_logits=True)
        group_count = int(city_to_ball.max().item()) + 1
        membership = city_to_ball[:, None, :] == torch.arange(
            group_count, device=encoded.device)[None, :, None]
        eligible = available[:, :, None, :] & membership[:, None, :, :]
        counts = eligible.sum(dim=-1)
        ball_mean = torch.einsum("bpgn,bnd->bpgd", eligible.to(encoded.dtype), encoded)
        ball_mean = ball_mean / counts.clamp_min(1)[..., None]
        start_vector = encoded.gather(1, first[..., None].expand(-1, -1, dim))
        current_vector = encoded.gather(1, current[..., None].expand(-1, -1, dim))
        query = self.decoder.query_projection(torch.cat((start_vector, current_vector), dim=-1))
        fraction = (counts / n).to(encoded.dtype)[..., None]
        context = torch.cat((query[:, :, None, :].expand_as(ball_mean), ball_mean, fraction), dim=-1)
        ball_logits = self.ball_head(context).squeeze(-1)
        ball_logits = ball_logits.masked_fill(counts == 0, -torch.inf)
        return ball_logits, city_logits

    def choose_step(
        self, encoded: Tensor, first: Tensor, current: Tensor,
        available: Tensor, city_to_ball: Tensor, *, sample: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return city IDs, joint log P(ball)+log P(city|ball), ball IDs."""
        ball_logits, city_logits = self.action_logits(
            encoded, first, current, available, city_to_ball)
        ball_dist = torch.distributions.Categorical(logits=ball_logits)
        chosen_ball = ball_dist.sample() if sample else ball_logits.argmax(dim=-1)
        city_mask = available & (city_to_ball[:, None, :] == chosen_ball[..., None])
        conditional_logits = city_logits.masked_fill(~city_mask, -torch.inf)
        city_dist = torch.distributions.Categorical(logits=conditional_logits)
        chosen_city = city_dist.sample() if sample else conditional_logits.argmax(dim=-1)
        joint_log_prob = ball_dist.log_prob(chosen_ball) + city_dist.log_prob(chosen_city)
        return chosen_city, joint_log_prob, chosen_ball
