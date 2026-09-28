"""执行 TSP 输入到粒球预处理；结果供后续 POMO 实验使用。"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from .granular_ball import GranularBall, Point2D, gbc
from .tsp import City, city_to_ball_mapping, init_distance_matrix, read_cities


@dataclass
class PreparedTSP:
    """贪心构造路径之前的全部预处理结果。"""

    cities: list[City]
    distance_matrix: list[list[int]]
    granular_balls: list[GranularBall]
    city_to_ball: list[int]


def prepare_tsp(path: str | Path) -> PreparedTSP:
    """按原 runner 的顺序完成读取、距离计算、粒球生成和映射。"""
    cities = read_cities(path)
    distance_matrix = init_distance_matrix(cities)
    granular_balls = gbc(
        [Point2D(city.x, city.y, city.number) for city in cities]
    )
    city_to_ball = city_to_ball_mapping(cities, granular_balls)
    return PreparedTSP(cities, distance_matrix, granular_balls, city_to_ball)


def main(argv: list[str] | None = None) -> None:
    """处理指定 TSP 文件；未指定时处理数据目录内的全部 TSP 文件。"""
    parser = argparse.ArgumentParser(description="运行 GB-MACO 的 TSP 粒球预处理")
    parser.add_argument("input", nargs="?", type=Path, help="单个纯坐标 .tsp 文件")
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark_MSTSP"))
    args = parser.parse_args(argv)

    paths = [args.input] if args.input is not None else sorted(args.data_dir.glob("*.tsp"))
    if not paths:
        parser.error(f"没有找到 TSP 文件：{args.data_dir}")

    for path in paths:
        prepared = prepare_tsp(path)
        print(
            f"{path.name}: cities={len(prepared.cities)}, "
            f"balls={len(prepared.granular_balls)}, "
            f"mapped={len(prepared.city_to_ball)}"
        )


if __name__ == "__main__":
    main()
