"""Joint correction of conditional city scores and the derived ball marginal."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gb_2_maco_POMO.experiments.learnable_ball.learnable_policy import LearnableBallPolicy


class JointBallCityPolicy(LearnableBallPolicy):
    def __init__(self, embedding_dim: int = 128, head_num: int = 8,
                 encoder_layer_num: int = 3) -> None:
        super().__init__(embedding_dim, head_num, encoder_layer_num)
        self.joint_head = nn.Sequential(nn.Linear(3 * embedding_dim + 2, 64),
                                        nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.joint_head[-1].weight)
        nn.init.zeros_(self.joint_head[-1].bias)

    def freeze_original(self) -> None:
        self.encoder.requires_grad_(False)
        self.ball_head.requires_grad_(False)
        self.decoder.requires_grad_(False)

    def joint_components(self, encoded: Tensor, first: Tensor, current: Tensor,
                         available: Tensor, city_to_ball: Tensor,
                         normalized_distance: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return new ball logits, per-ball city logP, corrections and eligibility.

        The existing decoder scores all unvisited cities once; its score does
        not depend on the selected ball. Conditioning it on each legal ball is
        therefore the existing decoder's exact conditional distribution.
        """
        batch, n, dim = encoded.shape
        if (first.shape != current.shape or available.shape != (batch, first.shape[1], n)
                or city_to_ball.shape != (batch, n)
                or normalized_distance.shape != (batch, n, n)
                or not available.any(dim=-1).all().item()):
            raise ValueError("invalid joint route state")
        with torch.no_grad():
            ball_logits, city_logits = self.action_logits(
                encoded, first, current, available, city_to_ball)
        groups = ball_logits.shape[-1]
        membership = city_to_ball[:, None, :] == torch.arange(
            groups, device=encoded.device)[None, :, None]
        eligible = available[:, :, None, :] & membership[:, None, :, :]
        counts = eligible.sum(dim=-1)
        if not (counts > 0).any(dim=-1).all().item():
            raise RuntimeError("no legal ball")
        weights = eligible.to(encoded.dtype)
        ball_mean = torch.einsum("bpgn,bnd->bpgd", weights, encoded.detach())
        ball_mean = ball_mean / counts.clamp_min(1)[..., None]
        mapping = city_to_ball[:, None, :].expand(-1, first.shape[1], -1)
        city_ball_mean = ball_mean.gather(2, mapping[..., None].expand(-1, -1, -1, dim))
        fraction = (counts.to(encoded.dtype) / n).gather(2, mapping)[..., None]
        current_vector = encoded.detach().gather(
            1, current[..., None].expand(-1, -1, dim))
        candidate_vector = encoded.detach()[:, None, :, :].expand(
            -1, first.shape[1], -1, -1)
        distance = normalized_distance.gather(
            1, current[..., None].expand(-1, -1, n))[..., None]
        features = torch.cat((current_vector[:, :, None, :].expand_as(candidate_vector),
                              candidate_vector, city_ball_mean, distance, fraction), dim=-1)
        correction = self.joint_head(features).squeeze(-1)
        city_by_ball = city_logits[:, :, None, :].masked_fill(~eligible, -torch.inf)
        old_denom = torch.logsumexp(city_by_ball, dim=-1, keepdim=True)
        old_city_logp = (city_by_ball - old_denom.masked_fill(counts[..., None] == 0, 0))
        old_city_logp = old_city_logp.masked_fill(~eligible, -torch.inf)
        corrected = old_city_logp + correction[:, :, None, :]
        # logsumexp(all -inf) has undefined backward even if the empty ball is
        # masked later. Give empty balls a detached finite placeholder first.
        safe_corrected = corrected.masked_fill(counts[..., None] == 0, 0)
        log_z = torch.logsumexp(safe_corrected, dim=-1).masked_fill(counts == 0, -torch.inf)
        old_ball_logp = torch.log_softmax(ball_logits, dim=-1)
        new_ball_logits = (old_ball_logp + log_z).masked_fill(counts == 0, -torch.inf)
        new_city_logp = corrected - log_z.masked_fill(counts == 0, 0)[..., None]
        new_city_logp = new_city_logp.masked_fill(~eligible, -torch.inf)
        return new_ball_logits, new_city_logp, correction, eligible

    def choose_joint_step(self, encoded: Tensor, first: Tensor, current: Tensor,
                          available: Tensor, city_to_ball: Tensor,
                          normalized_distance: Tensor, *, sample: bool):
        ball_logits, city_logp, correction, eligible = self.joint_components(
            encoded, first, current, available, city_to_ball, normalized_distance)
        ball_dist = torch.distributions.Categorical(logits=ball_logits)
        ball = ball_dist.sample() if sample else ball_logits.argmax(dim=-1)
        conditional = city_logp.gather(2, ball[..., None, None].expand(
            -1, -1, 1, city_logp.shape[-1])).squeeze(2)
        city_dist = torch.distributions.Categorical(logits=conditional)
        city = city_dist.sample() if sample else conditional.argmax(dim=-1)
        if not eligible.gather(2, ball[..., None, None].expand(
                -1, -1, 1, eligible.shape[-1])).squeeze(2).gather(
                -1, city[..., None]).all().item():
            raise RuntimeError("joint policy selected an illegal city")
        logp = ball_dist.log_prob(ball) + city_dist.log_prob(city)
        return city, logp, ball, correction


def normalized_distance(prepared, device: torch.device) -> Tensor:
    """Instance-scale integer edge distance; no fixed distance bias is added."""
    matrix = torch.tensor(prepared.distance_matrix, dtype=torch.float32, device=device)
    return (matrix / matrix.amax().clamp_min(1))[None, :, :]
