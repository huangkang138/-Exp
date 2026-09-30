"""POMO 的路线构造与训练入口。

流程：prepare_tsp 的粒球数据 → 每条路线独立生成候选城市
→ 共享的 Attention Policy 给候选城市概率 → 采样/选最大概率城市
→ 闭环路线长度与 reward → Policy Gradient 更新。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

if __package__ in (None, ""):
    # 允许在 IDE 中直接运行本文件，也允许 python -m gb_maco_POMO.pomo。
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "gb_maco_POMO"

from .attention import POMOPolicy, build_city_features
from .runner import PreparedTSP, prepare_tsp
from .tsp import get_candidate_cities


@dataclass
class StepRecord:
    """保存 POMO-0 的第一步，便于核对候选城市、概率和选中城市。"""

    current_city: int
    visited: list[int]
    candidate_cities: list[int]
    candidate_probabilities: dict[int, float]
    selected_city: int


@dataclass
class CandidateDecision:
    """一条 POMO 路线在某一步使用候选 Mask 前的统计记录。"""

    pomo_index: int
    step: int  # 起点之后的第几次选城，从 1 开始。
    current_city: int
    candidate_size: int
    fallback_used: bool
    remaining_city_count: int  # 本次选城之前尚未访问的城市数。


@dataclass
class POMORollout:
    """一次 rollout 的结果：N 条路线，以及计算 loss 所需的奖励和 logP。

    routes[0] 是从城市 0 出发的路线；rewards[0] 是该路线的奖励。
    其余列表和 Tensor 也按相同的 POMO 下标对应。
    """

    feature_shape: tuple[int, ...]
    embedding_shape: tuple[int, ...]
    encoded_shape: tuple[int, ...]
    routes: list[list[int]]
    tour_lengths: Tensor  # [N]，含最后一个城市回到起点的距离。
    rewards: Tensor       # [N]，等于 -tour_lengths。
    baseline: Tensor      # 标量，N 条路线的平均 reward。
    advantages: Tensor    # [N]，每条 reward 减去同实例 baseline。
    route_log_probs: Tensor  # [N]，每条路线各次动作 logP 的和。
    first_step: StepRecord
    candidate_decisions: list[CandidateDecision]


@dataclass
class TrainStepResult:
    """一次参数更新的结果；parameter_changed 用于确认确实更新了权重。"""

    rollout: POMORollout
    loss: float
    parameter_changed: bool


def route_length(route: list[int], distance_matrix: list[list[int]]) -> int:
    """按已有距离矩阵求闭环长度，不重新用坐标计算距离。

    例如路线 [0, 2, 1] 的长度是 d(0,2)+d(2,1)+d(1,0)。
    取模运算让最后一个城市的“下一个城市”回到 route[0]。
    """
    if not route:
        raise ValueError("路线不能为空")
    return sum(
        distance_matrix[route[index]][route[(index + 1) % len(route)]]
        for index in range(len(route))
    )


def rollout(
    policy: POMOPolicy, prepared: PreparedTSP, *, sample_actions: bool
) -> POMORollout:
    """使用同一 Policy 构造 N 条不同起点的完整路线。

    prepared 含城市、距离矩阵、粒球、城市归属和球邻接表。
    对 N 个城市分别从 0..N-1 出发；各路线保管自己的 visited 和 route。
    每一步分别调用 get_candidate_cities()，因此各路线的候选集可以不同。
    sample_actions=True 时按概率采样并保留选中动作的 logP，供训练使用；
    False 时选择概率最大的候选城市，供推理使用。
    返回路线、闭环长度、reward、baseline、advantage 以及 logP 之和。
    """
    city_count = len(prepared.cities)
    if city_count < 2:
        raise ValueError("POMO 训练至少需要两个城市")
    if len(prepared.distance_matrix) != city_count or any(
        len(row) != city_count for row in prepared.distance_matrix
    ):
        raise ValueError("距离矩阵必须是 N×N")

    device = next(policy.parameters()).device
    features = build_city_features(
        prepared.cities, prepared.granular_balls, prepared.city_to_ball
    ).to(device)
    # Encoder 只运行一次；后续 N-1 个决策步骤共享同一份 encoded_nodes。
    embedded_nodes, encoded_nodes = policy.encode(features)
    # 第 p 条路线从城市 p 开始；第一步时当前位置也正是它的起点。
    first_cities = torch.arange(city_count, device=device).unsqueeze(0)  # [1,N]
    current_cities = first_cities.clone()

    # 每个 POMO 下标拥有自己的起点、已访问集合和路线；参数仍由同一 Policy 共享。
    routes = [[start] for start in range(city_count)]
    visited = [{start} for start in range(city_count)]
    selected_log_probs: list[Tensor] = []
    first_step: StepRecord | None = None

    # 起点已经访问过，所以还要选择 N-1 次；每一轮每条路线各选一个城市。
    candidate_decisions: list[CandidateDecision] = []
    for step in range(1, city_count):
        # mask[0, 路线编号, 城市编号] 为 True，表示该路线这一步允许选该城市。
        candidate_mask = torch.zeros(
            (1, city_count, city_count), dtype=torch.bool, device=device
        )
        candidates_per_route: list[list[int]] = []
        for pomo_id in range(city_count):
            current_city = routes[pomo_id][-1]
            # 现有函数按“当前球 + 直接邻球”返回未访问城市；局部为空时会 fallback。
            candidates = get_candidate_cities(
                current_city=current_city,
                visited=visited[pomo_id],
                mapping=prepared.city_to_ball,
                balls=prepared.granular_balls,
                ball_adjacency=prepared.ball_adjacency,
                city_count=city_count,
            )
            if not candidates or visited[pomo_id].intersection(candidates):
                raise RuntimeError("候选城市为空，或包含已访问城市")
            # 只观察原候选范围里是否还有未访问城市，不改变候选生成和 fallback。
            current_ball = prepared.city_to_ball[current_city]
            local_balls = {current_ball} | prepared.ball_adjacency[current_ball]
            fallback_used = not any(
                point.index not in visited[pomo_id]
                for ball_id in local_balls
                for point in prepared.granular_balls[ball_id].data
            )
            candidate_decisions.append(
                CandidateDecision(
                    pomo_index=pomo_id,
                    step=step,
                    current_city=current_city,
                    candidate_size=len(candidates),
                    fallback_used=fallback_used,
                    remaining_city_count=city_count - len(visited[pomo_id]),
                )
            )
            candidates_per_route.append(candidates)
            candidate_mask[0, pomo_id, candidates] = True

        probabilities = policy(
            encoded_nodes, first_cities, current_cities, candidate_mask
        )  # [1,N,N]，第二维是路线，第三维是城市。
        # 对每条路线的每一步都检查 Mask 和概率，而不只检查最终路线。
        if not torch.isfinite(probabilities).all().item():
            raise RuntimeError("概率出现 NaN 或 Inf")
        if (probabilities[~candidate_mask] != 0).any().item():
            raise RuntimeError("非候选城市获得了概率")
        if not torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones((1, city_count), device=device),
            atol=1e-6,
        ):
            raise RuntimeError("候选城市概率和不为 1")

        distribution = torch.distributions.Categorical(probs=probabilities)
        if sample_actions:
            next_cities = distribution.sample()  # 训练：按网络概率采样。
        else:
            next_cities = probabilities.argmax(dim=-1)  # 推理：选最大概率。
        # 只记录实际选中的城市的 logP，路线结束后求和用于 Policy Gradient。
        selected_log_probs.append(distribution.log_prob(next_cities).squeeze(0))

        chosen = next_cities.squeeze(0).tolist()
        if first_step is None:
            # 只保存 POMO-0 的第一步作调试样例；detach 仅用于打印，不参与 loss。
            first_step = StepRecord(
                current_city=routes[0][-1],
                visited=sorted(visited[0]),
                candidate_cities=candidates_per_route[0],
                candidate_probabilities={
                    city: probabilities[0, 0, city].detach().item()
                    for city in candidates_per_route[0]
                },
                selected_city=chosen[0],
            )
        # 每条路线只更新自己的 route 和 visited，不影响其他 POMO 路线。
        for pomo_id, city in enumerate(chosen):
            if city in visited[pomo_id] or city not in candidates_per_route[pomo_id]:
                raise RuntimeError("Policy 选中了已访问或非候选城市")
            routes[pomo_id].append(city)
            visited[pomo_id].add(city)
        current_cities = next_cities

    assert first_step is not None
    for route in routes:
        if len(route) != city_count or len(set(route)) != city_count:
            raise RuntimeError("POMO 路线没有恰好访问全部不同城市")
    lengths = torch.tensor(
        [route_length(route, prepared.distance_matrix) for route in routes],
        dtype=torch.float32,
        device=device,
    )
    # 奖励只看完整闭环长度；同一算例所有 POMO 路线的平均奖励作 baseline。
    rewards = -lengths
    baseline = rewards.mean(dim=0)  # 单个算例的 POMO 维是长度 N 的第 0 维。
    advantages = rewards - baseline
    # selected_log_probs 形状是 N-1 个 [N]；先按步骤堆叠，再沿步骤求和。
    route_log_probs = torch.stack(selected_log_probs, dim=0).sum(dim=0)
    return POMORollout(
        tuple(features.shape),
        tuple(embedded_nodes.shape),
        tuple(encoded_nodes.shape),
        routes,
        lengths,
        rewards,
        baseline,
        advantages,
        route_log_probs,
        first_step,
        candidate_decisions,
    )


def train_step(
    policy: POMOPolicy, prepared: PreparedTSP, optimizer: torch.optim.Optimizer
) -> TrainStepResult:
    """对一个已预处理算例做一次真正的 Policy Gradient 参数更新。

    rollout() 用采样构造 N 条路线；优势值 = 各路线 reward - 平均 reward。
    先把一条路线每次选中动作的 logP 相加，再计算
    loss = -mean(advantage.detach() * route_log_prob)。
    detach 表示优势值只当作评价信号，不从 reward/baseline 反传梯度。
    optimizer 由调用者创建，便于多次 train_step 复用同一个 Adam 状态。
    """
    policy.train()
    result = rollout(policy, prepared, sample_actions=True)
    loss = -(result.advantages.detach() * result.route_log_probs).mean()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Policy Gradient loss 不是有限值")

    # 保存更新前的权重，只用于验证 optimizer.step() 后是否真的改变参数。
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    changed = any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, policy.parameters())
    )
    return TrainStepResult(result, loss.detach().item(), changed)


def evaluate(policy: POMOPolicy, prepared: PreparedTSP) -> POMORollout:
    """用训练后的 Policy 构造路线：关闭梯度，每一步取最大概率候选城市。

    不采样，也不调用 optimizer；返回的数据结构与训练 rollout 相同。
    """
    policy.eval()
    with torch.no_grad():
        return rollout(policy, prepared, sample_actions=False)


def main(argv: list[str] | None = None) -> None:
    """命令行演示：读取一个现有 TSP，创建 Policy 和 Adam，执行一次训练。"""
    parser = argparse.ArgumentParser(description="在已有 TSP 算例上运行一次 POMO train step")
    parser.add_argument(
        "input", nargs="?", type=Path,
        default=Path(__file__).resolve().parent.parent / "benchmark_MSTSP" / "simple1_9.tsp",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.learning_rate <= 0:
        parser.error("--learning-rate 必须为正数")

    torch.manual_seed(args.seed)
    prepared = prepare_tsp(args.input)
    policy = POMOPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    step = train_step(policy, prepared, optimizer)
    result = step.rollout
    print(f"Dataset: {args.input.name}")
    print(f"City count: {len(prepared.cities)}")
    print(f"POMO size: {len(result.routes)}")
    print(f"Encoder: input {list(result.feature_shape)}; embedding {list(result.embedding_shape)}; output {list(result.encoded_shape)}")
    print("POMO routes:")
    for pomo_id, route in enumerate(result.routes):
        print(f"{pomo_id}: {route}")
    print(f"Tour lengths: {result.tour_lengths.tolist()}")
    print(f"Rewards: {result.rewards.tolist()}")
    print(f"Baseline: {result.baseline.item():.4f}")
    print(f"Advantages: {result.advantages.tolist()}")
    print(f"Loss: {step.loss:.6f}")
    print(f"Policy parameter changed: {step.parameter_changed}")
    trace = result.first_step
    print("POMO-0 first step:")
    print(f"current_city: {trace.current_city}")
    print(f"visited: {trace.visited}")
    print(f"candidate_cities: {trace.candidate_cities}")
    print(f"candidate probabilities: {trace.candidate_probabilities}")
    print(f"selected_city: {trace.selected_city}")


if __name__ == "__main__":
    main()
