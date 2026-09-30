"""执行 TSP 输入到粒球预处理；结果供后续 POMO 实验使用。"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# 直接点击运行本文件时，补上项目根目录和包名，让下面的相对导入生效。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "gb_maco_POMO"

from .granular_ball import GranularBall, Point2D, build_ball_adjacency, gbc
from .tsp import City, city_to_ball_mapping, init_distance_matrix, read_cities


@dataclass
class PreparedTSP:
    """把读入的城市、生成的粒球及其关系放在一起。

    ball_adjacency[0] 是与第 0 个球直接相连的球编号集合。
    例如 {1, 2} 表示第 0 个球直接连接第 1、2 个球。
    """

    cities: list[City]
    distance_matrix: list[list[int]]
    granular_balls: list[GranularBall]
    city_to_ball: list[int]
    ball_adjacency: dict[int, set[int]]


def prepare_tsp(path: str | Path) -> PreparedTSP:
    """读取城市并完成距离、粒球、归属映射和直接邻接预处理。"""
    cities = read_cities(path)
    distance_matrix = init_distance_matrix(cities)
    granular_balls = gbc(
        [Point2D(city.x, city.y, city.number) for city in cities]
    )
    city_to_ball = city_to_ball_mapping(cities, granular_balls)
    # city_to_ball[城市编号] 查所属球；ball_adjacency[球编号] 查直接邻球。
    # 把邻接表放进返回结果，后续生成候选城市时就能直接使用。
    ball_adjacency = build_ball_adjacency(granular_balls)
    return PreparedTSP(cities, distance_matrix, granular_balls, city_to_ball, ball_adjacency)


def main(argv: list[str] | None = None) -> None:
    """处理指定 TSP 文件；未指定时处理数据目录内的全部 TSP 文件。"""
    parser = argparse.ArgumentParser(description="运行 GB-MACO 的 TSP 粒球预处理")
    parser.add_argument("input", nargs="?", type=Path, help="单个纯坐标 .tsp 文件")
    # 默认从项目内找数据集，不依赖 IDE 当前设置的工作目录。
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "benchmark_MSTSP")
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
        # 集合本身没有固定顺序，打印时排序，方便逐次比较输出。
        print("Ball adjacency:")
        for ball_id, neighbors in prepared.ball_adjacency.items():
            print(f"Ball {ball_id} -> {sorted(neighbors)}")


if __name__ == "__main__":
    main()
