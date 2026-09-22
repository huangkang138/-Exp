"""将原项目 GranularBall.cpp 中的粒球生成流程转换为 Python。"""

from dataclasses import dataclass, field
from math import sqrt


@dataclass
class Point2D:
    """城市的二维坐标；index 是输入城市列表中的编号。
    原 C++ 结构：GranularBall.h:9 GBPoint2D。
    """

    x: float = 0.0
    y: float = 0.0
    index: int = -1


@dataclass
class GranularBall:
    """对应 C++ 的 GBall，保存点集合及重叠分组的状态。
    原 C++ 结构：GranularBall.h:15 GBall。
    """

    data: list[Point2D] = field(default_factory=list)
    center: Point2D = field(default_factory=Point2D)
    radius: float = 0.0
    flag: int = 0
    label: int = -1
    num: int = 0
    out: int = 0
    size: int = 1
    overlap: int = 0
    hardlapcount: int = 0
    softlapcount: int = 0


def mean_point(points: list[Point2D]) -> Point2D:
    """计算一组点的几何中心；空列表沿用原实现的零坐标。
    原 C++：GranularBall.cpp:3 meanPoint。
    """
    if not points:
        return Point2D()
    count = len(points)
    return Point2D(
        x=sum(point.x for point in points) / count,
        y=sum(point.y for point in points) / count,
    )


def squared_distance(a: Point2D, b: Point2D) -> float:
    """返回平方欧氏距离，供分裂粒球时比较远近。
    原 C++：GranularBall.cpp:15 squaredDist。
    """
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


def distance_2d(a: Point2D, b: Point2D) -> float:
    """返回两点的欧氏距离。
    原 C++：GranularBall.cpp:21 dist2D。
    """
    return sqrt(squared_distance(a, b))


def get_radius(points: list[Point2D]) -> float:
    """取中心到球内各点距离的最大值作为半径。
    原 C++：GranularBall.cpp:25 getRadius。
    """
    if not points:
        return 0.0
    center = mean_point(points)
    return max(distance_2d(center, point) for point in points)


def get_dm(points: list[Point2D]) -> float:
    """计算平均中心距离；原实现对不超过两个点的球固定返回 1。
    原 C++：GranularBall.cpp:35 getDM。
    """
    if len(points) <= 2:
        return 1.0
    center = mean_point(points)
    return sum(distance_2d(center, point) for point in points) / len(points)


def split_ball(data: list[Point2D]) -> tuple[list[Point2D], list[Point2D]]:
    """选两个相距较远的种子点，再将其余点分到距离更近的种子。
    原 C++：GranularBall.cpp:48 splitBall。
    """
    center = mean_point(data)
    first = max(range(len(data)), key=lambda i: squared_distance(data[i], center))
    second = max(range(len(data)), key=lambda i: squared_distance(data[i], data[first]))

    ball1: list[Point2D] = []
    ball2: list[Point2D] = []
    for point in data:
        # 距离相等时进入第二个球，与 C++ 的严格小于判断一致。
        if squared_distance(point, data[first]) < squared_distance(point, data[second]):
            ball1.append(point)
        else:
            ball2.append(point)

    if not ball1 or not ball2:
        # 种子重合时按原顺序平分，防止分裂退化为空球。
        middle = len(data) // 2
        return data[:middle], data[middle:]
    return ball1, ball2


def _divide(
    pending: list[list[Point2D]], finished: list[list[Point2D]]
) -> list[list[Point2D]]:
    """只尝试分裂超过 16 个点、且分裂后平均距离降低的球。
    原 C++：GranularBall.cpp:93 division。
    """
    next_pending: list[list[Point2D]] = []
    for ball in pending:
        if len(ball) <= 16:
            finished.append(ball)
            continue

        first, second = split_ball(ball)
        child_dm = (len(first) * get_dm(first) + len(second) * get_dm(second)) / len(ball)
        if child_dm < get_dm(ball):
            next_pending.extend((first, second))
        else:
            finished.append(ball)
    return next_pending


def _normalize(
    pending: list[list[Point2D]], finished: list[list[Point2D]], radius_detect: float
) -> list[list[Point2D]]:
    """继续分裂半径超过检测阈值两倍的球。
    原 C++：GranularBall.cpp:126 normalized_ball。
    """
    next_pending: list[list[Point2D]] = []
    for ball in pending:
        if len(ball) < 2 or get_radius(ball) <= 2.0 * radius_detect:
            finished.append(ball)
        else:
            next_pending.extend(split_ball(ball))
    return next_pending


class _UnionFind:
    """把满足动态重叠条件的粒球归入同一个连通组。
    原 C++ 类：GranularBall.h:29 UF。
    """

    def __init__(self, count: int) -> None:
        """初始化每个粒球的独立连通组。
        原 C++：GranularBall.h:31 UF 构造函数。
        """
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, item: int) -> int:
        """查找组根节点，并压缩沿途父节点。
        原 C++：GranularBall.h:35 UF::find。
        """
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def unite(self, first: int, second: int) -> None:
        """按组大小合并；相同时沿用 C++ 中第二组为根的规则。
        原 C++：GranularBall.h:43 UF::unite。
        """
        root1 = self.find(first)
        root2 = self.find(second)
        if root1 == root2:
            return
        if self.size[root1] > self.size[root2]:
            self.parent[root2] = root1
            self.size[root1] += self.size[root2]
        else:
            self.parent[root1] = root2
            self.size[root2] += self.size[root1]


def connect_ball_overlap(
    ball_data: list[list[Point2D]], c_count: int = 1
) -> list[GranularBall]:
    """分析球间重叠，生成每个粒球的组标签与重叠状态。
    原 C++：GranularBall.cpp:150 connect_ball_overlap。
    """
    balls = [
        GranularBall(
            data=points,
            center=mean_point(points),
            radius=get_radius(points),
            label=index,
            num=len(points),
        )
        for index, points in enumerate(ball_data)
    ]

    # 第一遍只记录直接重叠；每个球最多参与一次 hardlap 计数。
    for i, first in enumerate(balls):
        if first.out == 1:
            continue
        for second in balls[i + 1 :]:
            if second.out == 1:
                continue
            if (
                distance_2d(first.center, second.center) <= first.radius + second.radius
                and first.hardlapcount == 0
                and second.hardlapcount == 0
            ):
                first.overlap = second.overlap = 1
                first.hardlapcount += 1
                second.hardlapcount += 1

    # 第二遍使用动态阈值合并粒球，同时记录较宽松的软重叠。
    groups = _UnionFind(len(balls))
    for i, first in enumerate(balls):
        if first.out == 1:
            continue
        for j in range(i + 1, len(balls)):
            second = balls[j]
            if second.out == 1:
                continue
            larger = max(first.radius, second.radius)
            smaller = min(first.radius, second.radius)
            gap = distance_2d(first.center, second.center)
            denominator = min(first.hardlapcount, second.hardlapcount) + 1
            extra = (smaller if c_count == 1 else larger) / denominator
            if gap <= first.radius + second.radius + extra and first.num > 2 and second.num > 2:
                first.flag = second.flag = 1
                groups.unite(i, j)
            if gap <= first.radius + second.radius + larger:
                first.softlapcount = second.softlapcount = 1

    for index, ball in enumerate(balls):
        ball.label = groups.find(index)
        ball.size = groups.size[ball.label]

    # 保留原算法的规则：少于四个粒球的连通组不标为核心组。
    label_counts: dict[int, int] = {}
    for ball in balls:
        label_counts[ball.label] = label_counts.get(ball.label, 0) + 1
    for ball in balls:
        if ball.hardlapcount == 0 and ball.softlapcount == 0:
            ball.flag = 0
        if label_counts[ball.label] < 4:
            ball.flag = 0

    # 非核心球寻找最近的核心球；sqrt(2) 是原代码使用的距离上限。
    for ball in balls:
        if ball.flag != 0:
            continue
        best_distance = sqrt(2.0)
        for candidate in balls:
            if candidate.flag == 1:
                gap = distance_2d(ball.center, candidate.center) - (
                    ball.radius + candidate.radius
                )
                if gap < best_distance:
                    best_distance = gap
                    ball.label = candidate.label
                    ball.flag = 2
    return balls


def gbc(data: list[Point2D]) -> list[GranularBall]:
    """执行原 GBC 流程：质量分裂、半径规范化、重叠分组。
    原 C++：GranularBall.cpp:270 GBC。
    """
    pending = [data]
    finished: list[list[Point2D]] = []

    while True:
        old_count = len(pending) + len(finished)
        pending = _divide(pending, finished)
        if len(pending) + len(finished) == old_count:
            pending = finished
            break

    radii = sorted(get_radius(ball) for ball in pending if len(ball) >= 2)
    if radii:
        # C++ 使用排序后的中间下标，偶数个半径时取靠右的那个。
        radius_detect = max(radii[len(radii) // 2], sum(radii) / len(radii))
    else:
        radius_detect = 0.0

    finished = []
    while True:
        old_count = len(pending) + len(finished)
        pending = _normalize(pending, finished, radius_detect)
        if len(pending) + len(finished) == old_count:
            pending = finished
            break

    return connect_ball_overlap(pending, c_count=1)
