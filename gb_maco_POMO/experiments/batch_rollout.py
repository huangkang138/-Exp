"""多实例 POMO rollout；复用原候选规则，不改变原单实例 pomo.py。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..attention import POMOPolicy, build_city_features, candidate_guidance
from ..runner import prepare_tsp
from .data import DatasetInstance


@dataclass
class BatchRollout:
    """B 个算例各有 N 条路线；routes 为 [B,N,N] 城市编号。"""

    lengths: Tensor
    route_log_probs: Tensor | None
    fallback_count: int
    candidate_total: int
    decision_count: int
    routes: Tensor | None = None


def rollout_batch(
    policy: POMOPolicy, instances: list[DatasetInstance], device: torch.device,
    *, sample_actions: bool,
) -> BatchRollout:
    """一次前向处理多个同规模实例，各实例独立维护路线、visited 和候选 Mask。"""
    if not instances or any(item.city_count != instances[0].city_count for item in instances):
        raise ValueError("同一个 micro batch 必须非空且城市数相同")
    prepared = [prepare_tsp(item.path) for item in instances]
    city_count = instances[0].city_count
    if any(len(problem.cities) != city_count for problem in prepared):
        raise ValueError(".tsp 城市数量与 manifest 的 city_count 不一致")
    batch_size = len(prepared)
    features = torch.cat(
        [build_city_features(problem.cities, problem.granular_balls, problem.city_to_ball)
         for problem in prepared], dim=0,
    ).to(device)
    distances = torch.tensor(
        [problem.distance_matrix for problem in prepared], dtype=torch.float32, device=device
    )
    _, encoded_nodes = policy.encode(features)

    # P=N：每个实例从城市 0..N-1 各出发一条路线；网络参数仍只共享一套。
    first_cities = torch.arange(city_count, device=device).expand(batch_size, -1)
    current_cities = first_cities.clone()
    routes = [[[start] for start in range(city_count)] for _ in range(batch_size)]
    visited = [[{start} for start in range(city_count)] for _ in range(batch_size)]
    selected_log_probs: list[Tensor] = []
    fallback_count = candidate_total = decision_count = 0

    for _step in range(1, city_count):
        # CPU 上生成候选 Mask，整个 micro batch 一次送入 GPU 的 Decoder。
        mask = torch.zeros((batch_size, city_count, city_count), dtype=torch.bool)
        bias = torch.zeros((batch_size, city_count, city_count), dtype=torch.float32)
        for batch_id, problem in enumerate(prepared):
            for pomo_id in range(city_count):
                current = routes[batch_id][pomo_id][-1]
                seen = visited[batch_id][pomo_id]
                candidates, preference = candidate_guidance(
                    current, seen, problem.city_to_ball, problem.granular_balls,
                    problem.ball_adjacency, city_count, policy,
                )
                if not candidates:
                    raise RuntimeError("路线尚未完成，但候选城市为空")
                current_ball = problem.city_to_ball[current]
                local_balls = {current_ball} | problem.ball_adjacency[current_ball]
                fallback_count += not any(
                    point.index not in seen
                    for ball_id in local_balls
                    for point in problem.granular_balls[ball_id].data
                )
                candidate_total += len(candidates)
                decision_count += 1
                mask[batch_id, pomo_id, candidates] = True
                bias[batch_id, pomo_id] = torch.tensor(preference)

        probabilities = policy(encoded_nodes, first_cities, current_cities,
                               mask.to(device), bias.to(device))
        if sample_actions:
            distribution = torch.distributions.Categorical(probs=probabilities)
            next_cities = distribution.sample()
            selected_log_probs.append(distribution.log_prob(next_cities))
        else:
            next_cities = probabilities.argmax(dim=-1)

        # 选中的编号移回 CPU 更新 Python 的路线状态；网络运算本身在指定设备上。
        for batch_id, chosen in enumerate(next_cities.detach().cpu().tolist()):
            for pomo_id, city in enumerate(chosen):
                if city in visited[batch_id][pomo_id] or not mask[batch_id, pomo_id, city]:
                    raise RuntimeError("Policy 选中了已访问或非候选城市")
                routes[batch_id][pomo_id].append(city)
                visited[batch_id][pomo_id].add(city)
        current_cities = next_cities

    route_ids = torch.tensor(routes, dtype=torch.long, device=device)
    batch_ids = torch.arange(batch_size, device=device)[:, None, None]
    # roll(-1) 把路线最后一个城市连回起点，得到闭环路线长度。
    lengths = distances[batch_ids, route_ids, route_ids.roll(-1, dims=-1)].sum(dim=-1)
    route_log_probs = torch.stack(selected_log_probs).sum(dim=0) if sample_actions else None
    return BatchRollout(lengths, route_log_probs, fallback_count, candidate_total, decision_count, route_ids)
