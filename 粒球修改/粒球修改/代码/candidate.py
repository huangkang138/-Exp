# -*- coding: utf-8 -*-
"""候选城市判定 —— 学长要求的第一项改进。

学长原来的做法
--------------------------------------------------------------------------
把蚂蚁的下一步**硬限制**在「当前粒球 + 临近粒球」里的城市，其余城市直接屏蔽。

实测下来这套规则在 benchmark 上其实从没真正生效过（见 probe_cand.py / probe_margin.py）：
composite 系列每个簇都是半径约 40~60 的紧凑小团，簇心之间却相距 1000~3500，
按"圆盘间隙 <= min(半径)"去判"临近"，13 个 composite 算例里**一对临近球都没有**。
于是候选集退化成"只有当前球"——占全部城市约 23%。
一旦当前球里的城市被访问完，候选集空了，蚂蚁就只能在球外乱挑，距离引导完全失效。
这正是学长说的"复杂地形效果比较不好"的来源。

改进一：两个地方一起动
--------------------------------------------------------------------------
**（1）换掉"临近"的判定口径**，不再看圆盘间隙（在紧簇+远簇的数据上失效），
改成按**球心距离的 k 近邻**：
    rule="knn"  -> 每个球连上离它最近的 k 个球（尺度无关，永远有意义）
    rule="mst"  -> 球心建最小生成树，落在树上的球对算邻居（保证整张图连通）
    rule="gap"  -> 保留原来的圆盘间隙口径（gap <= margin * min(r_i, r_j)）做对照

**（2）把硬屏蔽换成软加权 + 退火**：
    outside_scale = 0   -> 完全等价于学长原来的硬屏蔽
    outside_scale = 1   -> 等价于完全不做限制（原论文行为）
    取 0.05 这种小值：候选集内享有完整权重（优先在球内走），
    候选集外的城市仍然保留一条很细的通路（跨簇的探索能力不会丢）。
    anneal_fes > 0 时，折扣在前这段预算里从 0 线性放开到 outside_scale：
    前期收紧加速收敛，后期放开保证解质量。
"""

from math import hypot


# 支持的四档口径，给命令行直接选
CANDIDATE_MODES = ("off", "hard_ball", "hard_neighbor", "soft_neighbor")
# 三种"临近粒球"判定口径
ADJACENCY_RULES = ("gap", "knn", "mst")


def ball_adjacency(
    balls: list,
    rule: str = "knn",
    margin: float = 1.0,
    k: int = 1,
) -> list[set[int]]:
    """返回每个粒球的相邻球集合。

    只按**球心几何位置**判定，不去看 connect_ball_overlap 留下的 label / flag：
    那两个字段在函数末尾被"把非核心球挂到最近核心球"这一步改写过，
    已经不再是干净的连通组编号，拿来当邻接关系会串味。

    balls : list[GranularBall]，需要 center / radius
    rule  : gap / knn / mst，见模块开头说明
    margin: 仅 gap 规则使用，间隙 <= margin * min(r_i, r_j) 视为相邻
    k     : 仅 knn 规则使用，每个球连上最近的 k 个球
    """
    count = len(balls)
    adjacency: list[set[int]] = [set() for _ in range(count)]
    if count < 2:
        return adjacency

    if rule == "gap":
        for i in range(count):
            center_i, radius_i = balls[i].center, balls[i].radius
            for j in range(i + 1, count):
                center_j, radius_j = balls[j].center, balls[j].radius
                gap = hypot(center_i.x - center_j.x, center_i.y - center_j.y) - (
                    radius_i + radius_j
                )
                if gap <= margin * min(radius_i, radius_j):
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        return adjacency

    centers = [(ball.center.x, ball.center.y) for ball in balls]

    if rule == "knn":
        for i in range(count):
            distances = sorted(
                (hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]), j)
                for j in range(count)
                if j != i
            )
            for _, j in distances[: max(1, k)]:
                adjacency[i].add(j)
                adjacency[j].add(i)
        return adjacency

    if rule == "mst":
        # 球心上的朴素 Prim：返回一个连通且无环的邻接图，
        # 和 v3 粒球"结构感知"的思路一致 —— 相邻 = 在最小生成树上直接相连。
        inside = [False] * count
        best = [float("inf")] * count
        parent = [-1] * count
        best[0] = 0.0
        for _ in range(count):
            u, u_best = -1, float("inf")
            for i in range(count):
                if not inside[i] and best[i] < u_best:
                    u, u_best = i, best[i]
            if u == -1:
                break
            inside[u] = True
            if parent[u] >= 0:
                adjacency[u].add(parent[u])
                adjacency[parent[u]].add(u)
            for v in range(count):
                if not inside[v]:
                    d = hypot(centers[u][0] - centers[v][0], centers[u][1] - centers[v][1])
                    if d < best[v]:
                        best[v], parent[v] = d, u
        return adjacency

    raise ValueError(f"未知的临近判定口径：{rule!r}，可选 {ADJACENCY_RULES}")


def candidate_ball_sets(
    balls: list,
    mode: str,
    rule: str = "knn",
    margin: float = 1.0,
    k: int = 1,
) -> list[set[int]]:
    """每个粒球的"候选球集合"（一定含它自己）。"""
    count = len(balls)
    if mode == "hard_ball":
        # 只要当前球，不看邻居 —— 学长方案的最激进版本
        return [{i} for i in range(count)]
    adjacency = ball_adjacency(balls, rule, margin, k)
    return [{i} | adjacency[i] for i in range(count)]


def build_candidate_masks(
    city_to_ball: list[int],
    balls: list,
    mode: str,
    margin: float = 1.0,
    city_count: int | None = None,
    rule: str = "knn",
    k: int = 1,
) -> list[set[int]] | None:
    """把"候选球集合"摊平成"候选城市集合"。

    返回 list[set[int]]，长度 = 城市数；
    mask[c] = 站在城市 c 上时，享有完整权重的城市集合。

    mode == "off" 时返回 None，表示下游不要做任何过滤（原论文行为）。
    """
    if mode == "off":
        return None
    if city_count is None:
        city_count = len(city_to_ball)

    ball_sets = candidate_ball_sets(balls, mode, rule, margin, k)

    # 每个球里装了哪些城市（城市编号 = 距离矩阵下标）
    ball_cities: list[list[int]] = [[] for _ in balls]
    for city, ball in enumerate(city_to_ball):
        if 0 <= ball < len(ball_cities):
            ball_cities[ball].append(city)

    masks: list[set[int]] = []
    for city in range(city_count):
        ball = city_to_ball[city] if city < len(city_to_ball) else -1
        if not (0 <= ball < len(ball_cities)):
            # 兜底：映射异常时不做限制，绝不把蚂蚁困死
            masks.append(set(range(city_count)))
            continue
        allowed: set[int] = set()
        for other in ball_sets[ball]:
            allowed.update(ball_cities[other])
        masks.append(allowed)
    return masks


def mask_summary(masks: list[set[int]] | None, city_count: int) -> dict:
    """给实验报告用的小结：平均候选集大小、平均覆盖率。"""
    if not masks:
        return {
            "avg_size": float(city_count),
            "coverage": 1.0,
            "min_size": city_count,
            "max_size": city_count,
        }
    sizes = [len(mask) for mask in masks]
    average = sum(sizes) / len(sizes)
    return {
        "avg_size": average,
        "coverage": average / city_count if city_count else 1.0,
        "min_size": min(sizes),
        "max_size": max(sizes),
    }


__all__ = [
    "CANDIDATE_MODES",
    "ADJACENCY_RULES",
    "ball_adjacency",
    "candidate_ball_sets",
    "build_candidate_masks",
    "mask_summary",
]
