# -*- coding: utf-8 -*-
"""粒球生成流程 · 结构感知版 v3。

v2 的教训（都是实测，不是推测）
--------------------------------------------------------------------------
v2 拿"球内平方误差 SSE"当分裂判据，一门心思追求"球抱得紧"。结果：
    球内跨度确实降到 v1 的 0.60 倍，
    但 ACO 端到端 F_beta 从 0.686 掉到 0.500（composite 组，3 seed 平均）。
同时"贪心初始路线命中标准答案边的比例"两边几乎一样（0.6765 vs 0.6735），
说明问题不在初始路线，而在搜索内部（球太小 -> 同球连续段变短 ->
aco.py 的 _boost_initial_route 给的初始信息素引导变弱）。

更关键的一个反例：geometry1_10（10 个点，56 条并列最优解）
    原版 v1：初始路线命中率 100%    v2：80%
而 v1 在这批数据上只给 1 个球。
所以"10 个点只分 1 个球"并不是缺陷 —— 这批数据本来就是一个簇，分 1 个球是对的。
真正的问题是我一开始把"球内跨度大"当成了缺陷，而那只是"球里装了一个完整簇"的必然结果。

结论：TSP/MSTSP 里的好粒球 ≠ 几何最紧的球，
      而是"恰好圈住一个自然簇"的球。

v3 的做法
--------------------------------------------------------------------------
不看"球紧不紧"，改看"点和点之间有没有断开"：
  1. 对点集建**最小生成树 MST** —— MST 是"把所有点连成一体的最省边长方式"
  2. 如果数据里有几个簇，那么"簇与簇之间"那条连接边，在 MST 里必然特别长
  3. 按**最大间隙**切 MST：把边权降序排，找相邻比值最大的那个位置；
     比值超过 gap_threshold 就在那里切开
  4. 递归切下去，直到剩下的边权都连续（说明是同一个簇内部，不该再分）

为什么这样更对：
  - 判据是**尺度自适应**的（看比值不看绝对值），不用 min_ball_size 这类硬阈值
  - 均匀分布的点云没有明显间隙 -> 自然只出 1 个球（和 v1 行为一致，且是对的）
  - 真正分簇的数据 -> 切在簇边界上，球数自动等于簇数

关键性质（让实现很便宜）
  从 MST 里删掉一条边，两个连通分量各自的 MST 就是原 MST 落在该分量上的边集。
  所以只需要建一次 MST，递归时挑自己那部分边即可，总复杂度 O(n^2)。
"""

from math import sqrt

from .granular_ball import (
    GranularBall,
    Point2D,
    _UnionFind,
    connect_ball_overlap,
    squared_distance,
)

# 默认参数
GAP_THRESHOLD = 3.5   # 最长边 / 边权中位数 超过它就认为"存在真实的簇间边"（见下）
MIN_SIZE = 3          # 一个球至少几个点（只防孤点球，不再当"停止分裂"的总开关）


# ------------------------------------------------------------------ MST

def build_mst(points: list[Point2D]) -> list[tuple[int, int, float]]:
    """Prim 算法建最小生成树，返回 [(u, v, 欧氏距离), ...]，n-1 条边。

    用 O(n^2) 的朴素 Prim：TSP 算例点数最多几十到几千，
    朴素版常数小、不用额外依赖，比 Kruskal + 并查集更快。
    """
    n = len(points)
    if n < 2:
        return []

    inf = float("inf")
    in_tree = [False] * n
    best = [inf] * n
    parent = [-1] * n
    best[0] = 0.0
    edges: list[tuple[int, int, float]] = []

    for _ in range(n):
        u = -1
        u_best = inf
        for i in range(n):
            if not in_tree[i] and best[i] < u_best:
                u_best = best[i]
                u = i
        if u == -1:
            break
        in_tree[u] = True
        if parent[u] >= 0:
            edges.append((parent[u], u, sqrt(u_best)))

        pu = points[u]
        for v in range(n):
            if not in_tree[v]:
                d = squared_distance(pu, points[v])
                if d < best[v]:
                    best[v] = d
                    parent[v] = u
    return edges


def find_cut(edges: list[tuple[int, int, float]], gap_threshold: float):
    """判断这一块要不要切，要切就返回那条"簇间边"的索引，否则返回 None。

    判据用 **最长边 / 边权中位数**，不用"降序后相邻比值最大"：
    后者会被"最小的一条边"污染 —— 点集里只要有一对特别近的点，
    相邻比值最大的位置就跑到末尾去了，于是把点云切成一堆孤点。
    实测（run_mstdiag.py，每类 20 个算例的平均值）：
        uniform 验证集 n10/n30/n60 ：max/med = 1.85 / 2.04 / 2.58
        clustered 验证集 n10/30/60 ：max/med = 5.10 / 6.45 / 7.78
        benchmark composite       ：max/med = 48 ~ 82
    两类之间有一条干净的缝（2.58 vs 5.10），取 3.5 就能分开。
    """
    if len(edges) < 2:
        return None
    weights = sorted((e[2] for e in edges))
    med = weights[len(weights) // 2]
    if med <= 0.0:
        return None
    top = max(range(len(edges)), key=lambda k: edges[k][2])
    if edges[top][2] / med < gap_threshold:
        return None
    return top


# ------------------------------------------------------------------ 递归切分

def _cluster(
    root_members: list[int],
    root_edges: list[tuple[int, int, float]],
    gap_threshold: float,
    min_size: int,
) -> list[list[int]]:
    """每次只切掉最长的那条边，把一块分成两半，用显式栈迭代（避免递归爆栈）。

    为什么一次只切一条：一次切掉多条"长边"会把点云打散成一堆一两点的碎块。
    逐条切 + 每次检查"分出来的两半都够大"，才能保证切出来的每一块都是一个完整的簇。
    """
    groups: list[list[int]] = []
    stack: list[tuple[list[int], list[tuple[int, int, float]]]] = [
        (root_members, root_edges)]

    while stack:
        members, edges = stack.pop()

        # 太小就停：两半都要够大才允许切
        if len(members) < 2 * min_size:
            groups.append(members)
            continue

        cut = find_cut(edges, gap_threshold)
        if cut is None:
            groups.append(members)
            continue

        # 切掉最长边 -> 用剩下的边求连通分量
        kept = edges[:cut] + edges[cut + 1:]
        index_of = {g: k for k, g in enumerate(members)}
        uf = _UnionFind(len(members))
        for u, v, _ in kept:
            uf.unite(index_of[u], index_of[v])

        buckets: dict[int, list[int]] = {}
        for k, g in enumerate(members):
            buckets.setdefault(uf.find(k), []).append(g)

        parts = list(buckets.values())
        if len(parts) < 2 or any(len(p) < min_size for p in parts):
            groups.append(members)          # 切出来的有碎块，说明这条边不该切
            continue

        for part in parts:
            part_set = set(part)
            sub_edges = [(u, v, w) for u, v, w in kept if u in part_set and v in part_set]
            stack.append((part, sub_edges))

    return groups


# ------------------------------------------------------------------ 对外入口

def gbc_v3(
    data: list[Point2D],
    gap_threshold: float = GAP_THRESHOLD,
    min_size: int = MIN_SIZE,
) -> list[GranularBall]:
    """结构感知版粒球：MST 最大间隙切分 -> 复用原版重叠分组。

    参数
    ----
    data          : 城市坐标点（index 是城市编号）
    gap_threshold : 相邻边权比值超过它就切，越大球越少（默认 3.0）
    min_size      : 一个球至少几个点（默认 3，只防孤点）
    """
    if not data:
        return []

    edges = build_mst(data)
    groups = _cluster(list(range(len(data))), edges, gap_threshold, min_size)

    ball_data = [[data[i] for i in part] for part in groups]
    return connect_ball_overlap(ball_data, c_count=1)


# 方便直接当原版用
gbc = gbc_v3


__all__ = ["GranularBall", "Point2D", "build_mst", "gbc_v3", "gbc"]
