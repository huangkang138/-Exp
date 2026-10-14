"""POMO 路线的无向边多样性奖励；只读取当次采样路线，不读取最优解。"""

from __future__ import annotations

import torch
from torch import Tensor


def edge_diversity(routes: Tensor, eligible: Tensor) -> Tensor:
    """对合格路线找最相似的另一条回路；旋转和反向路线得分均为 0。"""
    if routes.ndim != 3 or eligible.shape != routes.shape[:2]:
        raise ValueError("routes 应为 [B,P,N]，eligible 应为 [B,P]")
    batch, pomo, cities = routes.shape
    if cities < 3 or pomo < 2:
        return torch.zeros_like(eligible, dtype=torch.float32)
    successors = routes.roll(-1, dims=-1)
    lower = torch.minimum(routes, successors)
    upper = torch.maximum(routes, successors)
    edge_ids = lower * cities + upper
    # 无向边在每条闭环路线中只出现一次；稠密指示矩阵保持在当前设备。
    incidence = torch.zeros((batch, pomo, cities * cities), device=routes.device)
    incidence.scatter_(2, edge_ids, 1.0)
    shared = torch.bmm(incidence, incidence.transpose(1, 2)) / cities
    peers = eligible[:, None, :].expand(-1, pomo, -1)
    diagonal = torch.eye(pomo, dtype=torch.bool, device=routes.device)[None]
    valid_peers = peers & ~diagonal
    maximum = shared.masked_fill(~valid_peers, -1.0).max(dim=-1).values
    # 仅一条合格路线时无法证明它具有多样性，定义为 0。
    return torch.where(eligible & valid_peers.any(dim=-1), 1.0 - maximum, 0.0)


def diversity_reward(
    lengths: Tensor, routes: Tensor, *, quality_ratio: float, diversity_lambda: float,
) -> Tensor:
    """R=-L/Lmin+lambda*w*D；w 在最优到阈值间线性降为 0。"""
    if lengths.shape != routes.shape[:2]:
        raise ValueError("lengths 与 routes 的实例/POMO 维度不一致")
    detached = lengths.detach()
    minimum = detached.min(dim=1, keepdim=True).values
    if not torch.isfinite(detached).all() or (minimum <= 0).any():
        raise ValueError("路线长度必须有限且为正")
    relative = detached / minimum
    # λ=0 的对照实验只使用归一化距离；不构造边矩阵或质量权重。
    if diversity_lambda == 0:
        return -relative
    eligible = relative <= quality_ratio
    weight = ((quality_ratio - relative) / (quality_ratio - 1.0)).clamp(0.0, 1.0)
    diversity = edge_diversity(routes.detach(), eligible)
    return -relative + diversity_lambda * weight * diversity


def lambda_at_epoch(epoch: int, *, target: float, warmup_epochs: int, ramp_epochs: int) -> float:
    """前 warmup_epochs 轮沿用原奖励，之后在 ramp_epochs 轮内升至目标 lambda。"""
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return target
    return target * min((epoch - warmup_epochs) / ramp_epochs, 1.0)
