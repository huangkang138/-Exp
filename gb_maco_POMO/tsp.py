"""TSP 城市读取、距离矩阵及城市到粒球映射。"""

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from .granular_ball import GranularBall


@dataclass
class City:
    """对应 C++ 的 City；number 与距离矩阵下标一致。
    原 C++ 结构：Head.h:62 City。
    """

    number: int
    x: float
    y: float


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


