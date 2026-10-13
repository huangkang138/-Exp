# -*- coding: utf-8 -*-
"""粒球生成流程 · 改进版 v2。

和原版 granular_ball.py 的关系
--------------------------------------------------------------------------
输入输出完全一样（list[Point2D] -> list[GranularBall]），
下游 tsp.py / aco.py / runner.py 一行都不用改，把 gbc 换成 gbc_v2 即可。
原版文件原封不动保留，方便对照实验。

原版的三个卡点（都是在这台机器上实测出来的，不是猜的）
--------------------------------------------------------------------------
1. `len(ball) <= 16` 一刀切：球里点数不超过 16 就不再往下分。
   实测后果：10 个城市的算例永远只有 1 个球，20 个城市永远只有 2 个球 ——
   粒球在这两档上等于没起作用。

2. split_ball 只切一刀：选两个相距最远的点当种子，把点分完就结束，
   不再重算中心。切出来常常是一条斜线劈开的"扁球"，不紧致。

3. 判据是"平均到球心的距离"（get_dm）：这个数对"少数点跑得很远"不敏感，
   扁球、拖尾球都测不出来，于是该切的没切。
   实测后果：球的"球内最远两点距 / 数据总跨度"高达 0.90（uniform n20），
   "球间最小空隙"全是负数 —— 球彼此重叠，没有分开。

v2 对应的三处改动
--------------------------------------------------------------------------
1. 停止条件：点数阈值 -> 三重自适应判断
   (a) 球太小（< min_ball_size）就停
   (b) 分裂带来的"球内平方误差下降率"不够（< split_gain）就停
   (c) 分出来的两个子球不满足可分性（中心距 < sep_factor × 两球半径和）就停
2. 分裂方式：一刀切 -> 迭代重算中心的两均值（Lloyd），直到分配稳定
3. 判据：平均到球心距离 -> 球内平方误差 SSE（平方会把"跑远的那几个点"放大，
   扁球一定被切开），配合尺度和点数都无关的相对阈值

另外做了一处纯速度优化：原版算"球心 / 平均距离 / 半径"要分别遍历三遍点集，
v2 用 _profile() 一趟算完。粒球本身不是瓶颈，但在大算例上能把耗时压下来。

球间重叠分组（connect_ball_overlap）直接复用原版，保证和论文口径一致。
"""

from math import hypot, sqrt

from .granular_ball import (
    GranularBall,
    Point2D,
    connect_ball_overlap,
    get_radius,
    squared_distance,
)

# 默认参数（三个阈值都是在本机上扫描出来的，见 run_fine.py / run_sweep.py）
#
# 关于 sep_factor 的取值说明：
#   一开始想当然地设成 1.0（"两个子球不重叠才继续分"），结果在大点集上把所有
#   分裂全拦掉了 —— 一个 2000 点的随机点云最后只剩 1 个球。
#   原因是：把一团点对半切开时，两个子球的半径和本来就大于两个中心的距离
#   （均匀方形区域对半切，中心距约 0.5 倍边长，半径和约 1.1 倍边长）。
#   所以 1.0 相当于"永远不许切凸的点团"。实测 0.2 附近既能拦住退化切分，
#   又不误伤正常分裂。
MIN_BALL_SIZE = 3      # 一个球最少几个点，防止切出"孤点球"
SPLIT_GAIN = 0.10      # 分裂至少要带来 10% 的球内平方误差下降，否则不切
SEP_FACTOR = 0.2       # 两个子球中心距要 >= 这个倍数 × (r1+r2)
MAX_ITER = 8           # 两均值最多迭代几轮（实测 3~5 轮就稳定了）
USE_NORMALIZE = False  # 原版的"半径规范化"：实测在改进分裂之后反而拖后腿，默认关


# ------------------------------------------------------------------ 小工具

def _profile(points: list[Point2D]) -> tuple[float, float, float, float]:
    """一趟遍历同时算出：球心 (cx, cy)、球内平方误差 SSE、半径。

    对应原版要分开跑三遍的三个量：
        mean_point() -> 球心
        get_dm()     -> 平均到球心距离（v2 换成 SSE）
        get_radius() -> 半径
    合成一趟只是省时间，算出来的数和原版一模一样。
    """
    count = len(points)
    cx = sum(p.x for p in points) / count
    cy = sum(p.y for p in points) / count
    sse = 0.0
    farthest = 0.0
    for p in points:
        dx = p.x - cx
        dy = p.y - cy
        d2 = dx * dx + dy * dy
        sse += d2
        if d2 > farthest:
            farthest = d2
    return cx, cy, sse, sqrt(farthest)


def ball_sse(points: list[Point2D]) -> float:
    """球内平方误差：每个点到球心的距离平方之和。

    这是比 get_dm 灵敏得多的紧致度度量：
    一个点跑远一倍，它对 SSE 的贡献变成 4 倍，而对平均距离只变成 2 倍。
    """
    return _profile(points)[2] if len(points) > 1 else 0.0


def ball_diameter(points: list[Point2D]) -> float:
    """球内最远两点之间的距离（球"有多扁"的直接度量）。画图/离线分析用。"""
    best = 0.0
    count = len(points)
    for i in range(count):
        xi = points[i].x
        yi = points[i].y
        for j in range(i + 1, count):
            dx = xi - points[j].x
            dy = yi - points[j].y
            d = dx * dx + dy * dy
            if d > best:
                best = d
    return sqrt(best)


# ------------------------------------------------------------------ 核心：分裂

def two_means(points: list[Point2D], max_iter: int = MAX_ITER):
    """把一组点切成两堆，返回 (第一堆, 第二堆)，切不动时返回 None。

    和原版 split_ball 的区别：
      原版：定两个种子 -> 分完就结束（一刀切）
      v2  ：定两个种子 -> 分堆 -> 重算两堆中心 -> 再分堆…… 直到分配不再变化

    种子沿用原版的思路（球心最远点 + 离它最远的点），这样保持和论文的连续性。
    """
    count = len(points)
    if count < 2:
        return None

    # 种子一：离球心最远的点
    cx = sum(p.x for p in points) / count
    cy = sum(p.y for p in points) / count
    seed_a = points[0]
    best = -1.0
    for p in points:
        dx = p.x - cx
        dy = p.y - cy
        d = dx * dx + dy * dy
        if d > best:
            best = d
            seed_a = p

    # 种子二：离种子一最远的点
    seed_b = points[0]
    best = -1.0
    for p in points:
        d = squared_distance(p, seed_a)
        if d > best:
            best = d
            seed_b = p

    ax, ay = seed_a.x, seed_a.y
    bx, by = seed_b.x, seed_b.y
    first: list[Point2D] = []
    second: list[Point2D] = []

    for _ in range(max_iter):
        # 第一步：按"离哪个中心近"重新分堆
        first = []
        second = []
        for p in points:
            d1x = p.x - ax
            d1y = p.y - ay
            d2x = p.x - bx
            d2y = p.y - by
            if d1x * d1x + d1y * d1y < d2x * d2x + d2y * d2y:
                first.append(p)
            else:
                second.append(p)

        # 某一堆空了，说明这团点切不开
        if not first or not second:
            return None

        # 第二步：重算两个中心
        n = len(first)
        nax = sum(p.x for p in first) / n
        nay = sum(p.y for p in first) / n
        m = len(second)
        nbx = sum(p.x for p in second) / m
        nby = sum(p.y for p in second) / m

        moved = (nax - ax) ** 2 + (nay - ay) ** 2 + (nbx - bx) ** 2 + (nby - by) ** 2
        ax, ay, bx, by = nax, nay, nbx, nby
        if moved < 1e-12:
            break  # 中心不再移动 = 分堆稳定

    return first, second


def _divide_v2(
    pending: list[list[Point2D]],
    finished: list[list[Point2D]],
    min_ball_size: int,
    split_gain: float,
    sep_factor: float,
    max_iter: int,
) -> list[list[Point2D]]:
    """一轮质量分裂：能切的切，不能切的收进 finished。"""
    next_pending: list[list[Point2D]] = []
    for ball in pending:
        # (a) 球已经足够小
        if len(ball) <= min_ball_size:
            finished.append(ball)
            continue

        cx, cy, parent_sse, _ = _profile(ball)
        if parent_sse <= 1e-12:
            finished.append(ball)  # 所有点重合
            continue

        pair = two_means(ball, max_iter)
        if pair is None:
            finished.append(ball)
            continue
        first, second = pair

        # 不允许切出比 min_ball_size 还小的边角料
        if len(first) < min_ball_size or len(second) < min_ball_size:
            finished.append(ball)
            continue

        # (b) 收益不够就不切
        a1, b1, sse1, r1 = _profile(first)
        a2, b2, sse2, r2 = _profile(second)
        if sse1 + sse2 > (1.0 - split_gain) * parent_sse:
            finished.append(ball)
            continue

        # (c) 两个子球分不开就不切：中心距要 >= sep_factor × 两个半径之和
        if hypot(a1 - a2, b1 - b2) < sep_factor * (r1 + r2):
            finished.append(ball)
            continue

        next_pending.extend((first, second))
    return next_pending


def _normalize_v2(
    pending: list[list[Point2D]],
    finished: list[list[Point2D]],
    radius_detect: float,
    max_iter: int,
) -> list[list[Point2D]]:
    """保留原版的"半径规范化"：半径超过检测阈值两倍的球再切一刀。

    和原版唯一的区别是这里也用迭代两均值切，而不是一刀切。
    """
    next_pending: list[list[Point2D]] = []
    for ball in pending:
        if len(ball) < 2 or get_radius(ball) <= 2.0 * radius_detect:
            finished.append(ball)
        else:
            pair = two_means(ball, max_iter)
            if pair is None or not pair[0] or not pair[1]:
                finished.append(ball)
            else:
                next_pending.extend(pair)
    return next_pending


# ------------------------------------------------------------------ 对外入口

def gbc_v2(
    data: list[Point2D],
    min_ball_size: int = MIN_BALL_SIZE,
    split_gain: float = SPLIT_GAIN,
    sep_factor: float = SEP_FACTOR,
    max_iter: int = MAX_ITER,
    use_normalize: bool = USE_NORMALIZE,
) -> list[GranularBall]:
    """改进版粒球生成：质量分裂 -> （可选）半径规范化 -> 重叠分组。

    参数
    ----
    data           : 城市坐标点（index 是城市编号）
    min_ball_size  : 一个球最少几个点，越小球越多（默认 3）
    split_gain     : 分裂至少要带来多少比例的球内平方误差下降，越大球越少（默认 0.10）
    sep_factor     : 两个子球中心距 / 两球半径和 的下限，越大球越少（默认 0.2）
    max_iter       : 两均值最多迭代轮数（默认 8）
    use_normalize  : 是否启用原版的半径规范化补充步骤（默认关，见文件头说明）
    """
    if not data:
        return []

    # ---- 第一步：质量分裂（这里和原版差别最大）----
    pending: list[list[Point2D]] = [data]
    finished: list[list[Point2D]] = []
    while True:
        old_count = len(pending) + len(finished)
        pending = _divide_v2(pending, finished, min_ball_size, split_gain, sep_factor, max_iter)
        if len(pending) + len(finished) == old_count:
            pending = finished
            break

    # ---- 第二步：半径规范化（原版有，实测改进分裂之后不再需要）----
    if use_normalize:
        radii = sorted(get_radius(ball) for ball in pending if len(ball) >= 2)
        if radii:
            radius_detect = max(radii[len(radii) // 2], sum(radii) / len(radii))
        else:
            radius_detect = 0.0

        finished = []
        while True:
            old_count = len(pending) + len(finished)
            pending = _normalize_v2(pending, finished, radius_detect, max_iter)
            if len(pending) + len(finished) == old_count:
                pending = finished
                break

    # ---- 第三步：重叠分组（完全复用原版，保证论文口径一致）----
    return connect_ball_overlap(pending, c_count=1)


# 方便直接当原版用
gbc = gbc_v2
