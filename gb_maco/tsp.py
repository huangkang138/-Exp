"""城市距离、回路长度及粒球引导的贪心初始路径。"""

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path

from .granular_ball import GranularBall, Point2D


@dataclass
class City:
    """对应 C++ 的 City；number 与距离矩阵下标一致。
    原 C++ 结构：Head.h:62 City。
    """

    number: int
    x: float
    y: float


@dataclass
class Route:
    """保存一次 TSP 回路的城市顺序及包括返程的总长度。
    原 C++ 结构：Head.h:67 Route。
    """

    city_path: list[int] = field(default_factory=list)
    length: int = 0


def read_cities(path: str | Path) -> list[City]:
    """读取原项目的纯坐标 .tsp 文件；每行对应一个城市的 x、y。
    原 C++：main.cpp:294 read_file；Python 改为按文件行数确定城市数量。
    """
    cities: list[City] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"第 {line_number} 行应只有 x、y 两个数值")
        cities.append(City(len(cities), float(fields[0]), float(fields[1])))
    if not cities:
        raise ValueError("输入文件没有城市坐标")
    return cities


def distance_city(first: City, second: City) -> int:
    """计算两城欧氏距离，并按 C++ 的 int(距离 + 0.5) 取整。
    原 C++：main.cpp:312 distance_city。
    """
    dx = first.x - second.x
    dy = first.y - second.y
    return int(sqrt(dx * dx + dy * dy) + 0.5)


def init_distance_matrix(cities: list[City]) -> list[list[int]]:
    """预先计算所有城市两两之间的整数距离。
    原 C++：main.cpp:326 init_distance_matrix。
    """
    if [city.number for city in cities] != list(range(len(cities))):
        raise ValueError("城市编号必须从 0 开始，且与列表位置一致")
    return [[distance_city(first, second) for second in cities] for first in cities]


def path_len(route: Route, distance_matrix: list[list[int]]) -> int:
    """累加相邻城市距离，并加上最后一个城市返回起点的距离。
    原 C++：main.cpp:317 path_len。
    """
    path = route.city_path
    if not path:
        return 0
    return sum(
        distance_matrix[path[index]][path[(index + 1) % len(path)]]
        for index in range(len(path))
    )


def city_to_ball_mapping(
    cities: list[City], granular_balls: list[GranularBall]
) -> list[int]:
    """根据粒球内原始编号建立城市到粒球的映射。
    原 C++ 代码段：main.cpp:1051-1088 城市到粒球映射；原代码没有独立函数。
    """
    mapping = [-1] * len(cities)
    for ball_id, ball in enumerate(granular_balls):
        for point in ball.data:
            mapping[point.index] = ball_id

    # 原代码会把没有被任何球覆盖的城市分给最近的粒球。
    for city in cities:
        if mapping[city.number] == -1:
            mapping[city.number] = min(
                range(len(granular_balls)),
                key=lambda ball_id: (
                    (city.x - granular_balls[ball_id].center.x) ** 2
                    + (city.y - granular_balls[ball_id].center.y) ** 2
                ),
            )
    return mapping


def greed_route(
    cities: list[City],
    distance_matrix: list[list[int]],
    start_city_index: int,
    city_to_ball: list[int],
    granular_balls: list[GranularBall],
) -> Route:
    """按 C++ 的粒球惩罚规则，从指定城市构造贪心初始回路。
    原 C++：main.cpp:334 greed_route。
    """
    city_count = len(cities)
    visited = [False] * city_count
    visited[start_city_index] = True
    current = start_city_index
    route = Route(city_path=[cities[current].number])

    for _ in range(1, city_count):
        current_ball = city_to_ball[current]
        remaining = sum(
            not visited[index] and city_to_ball[index] == current_ball
            for index in range(city_count)
        )
        ball_size = granular_balls[current_ball].num
        ratio = remaining / ball_size if ball_size > 0 else 0.0

        best_score = float("inf")
        next_city = -1
        chosen_distance = 0
        for candidate in range(city_count):
            if visited[candidate]:
                continue
            base_distance = distance_matrix[current][candidate]
            score = base_distance
            if city_to_ball[candidate] != current_ball:
                # 球内还有未访问城市时，暂时提高跨球边的选择成本。
                score = base_distance * (1.0 + 0.8 * ratio)
            if score < best_score:
                best_score = score
                next_city = candidate
                chosen_distance = base_distance

        if next_city == -1:
            break
        visited[next_city] = True
        route.city_path.append(cities[next_city].number)
        route.length += chosen_distance
        current = next_city

    route.length += distance_matrix[current][start_city_index]
    return route
